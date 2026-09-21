"""
src/intelligence/macro.py — Macro Economic Indicators via FRED API (Section 5, Item 4).

Macro Regime Rules:
  - Indicators tracked:
      * Federal Funds Rate (FEDFUNDS / DFF)
      * CPI Year-over-Year Inflation (CPIAUCSL)
      * Unemployment Rate (UNRATE)
      * 10-Year Treasury Yield (DGS10)
  - Regimes:
      * FAVORABLE:   Rates falling OR < 3.0%, Inflation < 3.0%
                     -> No adjustment (mult=1.0, buy_bar shift=+0.00)
      * NEUTRAL:     Rates 3.0%-5.0%, Inflation 3.0%-5.0%
                     -> Position size reduced by 20% (mult=0.80)
      * RESTRICTIVE: Rates > 5.0% AND Inflation > 5.0%
                     -> Position size reduced by 40% (mult=0.60)
                     -> Buy bar increased by +3% (+0.03)
  - Caching Policy:
      * Fetches on Mondays (weekday == 0); reuses cached data Tue-Fri.
      * Falls back cleanly to latest database cache when API is down or key unset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.parse

from config.settings import settings
from src.db import repository

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "treasury_10y": "DGS10",
}

# Baseline defaults if neither API nor DB cache is available
BASELINE_MACRO = {
    "fed_funds_rate": 4.50,
    "cpi_yoy": 2.80,
    "unemployment_rate": 4.10,
    "treasury_10y": 4.25,
    "macro_regime": "NEUTRAL",
}


@dataclass
class MacroEnvironmentResult:
    """Current macroeconomic snapshot and regime impact."""
    date: str
    fed_funds_rate: float
    cpi_yoy: float
    unemployment_rate: float
    treasury_10y: float
    macro_regime: str              # FAVORABLE / NEUTRAL / RESTRICTIVE
    position_size_multiplier: float  # 1.0 (FAVORABLE), 0.8 (NEUTRAL), 0.6 (RESTRICTIVE)
    buy_bar_shift: float           # 0.0 (FAVORABLE/NEUTRAL), +0.03 (RESTRICTIVE)
    is_cached: bool
    source_date: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "fed_funds_rate": round(self.fed_funds_rate, 2),
            "cpi_yoy": round(self.cpi_yoy, 2),
            "unemployment_rate": round(self.unemployment_rate, 2),
            "treasury_10y": round(self.treasury_10y, 2),
            "macro_regime": self.macro_regime,
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "buy_bar_shift": round(self.buy_bar_shift, 4),
            "is_cached": self.is_cached,
            "source_date": self.source_date,
        }


def classify_macro_regime(
    fed_funds: float,
    cpi_yoy: float,
    rates_falling: bool = False,
) -> str:
    """
    Classifies the macroeconomic regime based on interest rates and inflation.
    """
    # RESTRICTIVE: rates > 5% AND inflation > 5%
    if fed_funds > 5.0 and cpi_yoy > 5.0:
        return "RESTRICTIVE"

    # FAVORABLE: rates falling OR rates < 3%, AND inflation < 3%
    if (rates_falling or fed_funds < 3.0) and cpi_yoy < 3.0:
        return "FAVORABLE"

    # NEUTRAL: standard transition regime (3-5% rates, 3-5% inflation, etc.)
    return "NEUTRAL"


def fetch_fred_series_latest(series_id: str, api_key: str) -> Optional[float]:
    """
    Fetches the single most recent observation for a given series from FRED REST API.
    """
    if not api_key:
        return None

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "5",
    }
    url = f"https://api.stlouisfed.org/fred/series/observations?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIStockTrader/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            obs = data.get("observations", [])
            for o in obs:
                val_str = o.get("value", "")
                if val_str and val_str != ".":
                    return float(val_str)
    except Exception as exc:
        logger.warning("FRED API fetch failed for %s: %s", series_id, exc)
    return None


def fetch_fred_cpi_yoy(api_key: str) -> Optional[float]:
    """
    Computes Year-over-Year CPI percentage change from the latest 13 monthly observations.
    """
    if not api_key:
        return None
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "15",
    }
    url = f"https://api.stlouisfed.org/fred/series/observations?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIStockTrader/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            obs = [o for o in data.get("observations", []) if o.get("value") and o.get("value") != "."]
            if len(obs) >= 13:
                current_cpi = float(obs[0]["value"])
                prev_year_cpi = float(obs[12]["value"])
                if prev_year_cpi > 0:
                    return ((current_cpi - prev_year_cpi) / prev_year_cpi) * 100.0
    except Exception as exc:
        logger.warning("FRED CPI YoY fetch failed: %s", exc)
    return None


def get_macro_environment(
    as_of_date_str: str,
    force_fetch: bool = False,
    persist: bool = True,
) -> MacroEnvironmentResult:
    """
    Resolves the current macro regime using FRED API or database cache.
    Follows weekly fetch cadence (Mondays only, cached Tue-Fri).
    """
    try:
        dt = datetime.strptime(as_of_date_str[:10], "%Y-%m-%d").date()
    except Exception:
        dt = date.today()

    is_monday = (dt.weekday() == 0)
    api_key = settings.fred_api_key

    # Check cached indicators in database first
    cached = repository.get_latest_macro_indicators(as_of_date=as_of_date_str[:10])

    # Should we fetch live from FRED?
    # Only if forced or if today is Monday and API key is set
    should_fetch = (force_fetch or (is_monday and not cached)) and bool(api_key)

    fed_funds: Optional[float] = None
    cpi: Optional[float] = None
    unemp: Optional[float] = None
    t10: Optional[float] = None
    source_date = as_of_date_str[:10]
    used_cache = False

    if should_fetch:
        logger.info("Fetching weekly macroeconomic data from FRED for %s ...", as_of_date_str)
        fed_funds = fetch_fred_series_latest("FEDFUNDS", api_key)
        cpi = fetch_fred_cpi_yoy(api_key)
        unemp = fetch_fred_series_latest("UNRATE", api_key)
        t10 = fetch_fred_series_latest("DGS10", api_key)

    # If fetch was not attempted or failed, fall back to cached database record
    if fed_funds is None or cpi is None:
        if cached is not None:
            fed_funds = cached.get("fed_funds_rate")
            cpi = cached.get("cpi_yoy")
            unemp = cached.get("unemployment_rate")
            t10 = cached.get("treasury_10y")
            source_date = cached.get("date", as_of_date_str[:10])
            used_cache = True
            logger.info("MACRO_DATA_CACHED — using %s values", source_date)
        else:
            # Fallback to sensible baseline defaults
            fed_funds = BASELINE_MACRO["fed_funds_rate"]
            cpi = BASELINE_MACRO["cpi_yoy"]
            unemp = BASELINE_MACRO["unemployment_rate"]
            t10 = BASELINE_MACRO["treasury_10y"]
            source_date = "BASELINE_DEFAULT"
            used_cache = True
            logger.info("No FRED data or DB cache found. Using baseline default macro values.")

    # Fill any individual missing indicator with baseline
    fed_funds = fed_funds if fed_funds is not None else BASELINE_MACRO["fed_funds_rate"]
    cpi = cpi if cpi is not None else BASELINE_MACRO["cpi_yoy"]
    unemp = unemp if unemp is not None else BASELINE_MACRO["unemployment_rate"]
    t10 = t10 if t10 is not None else BASELINE_MACRO["treasury_10y"]

    # Classify regime
    regime = classify_macro_regime(fed_funds, cpi)

    if regime == "FAVORABLE":
        mult = 1.0
        shift = 0.0
    elif regime == "RESTRICTIVE":
        mult = settings.macro_restrictive_size_multiplier  # 0.60 (-40%)
        shift = settings.macro_restrictive_buy_bar_shift    # +0.03 (+3%)
    else:  # NEUTRAL
        mult = settings.macro_neutral_size_multiplier      # 0.80 (-20%)
        shift = 0.0

    result = MacroEnvironmentResult(
        date=as_of_date_str[:10],
        fed_funds_rate=fed_funds,
        cpi_yoy=cpi,
        unemployment_rate=unemp,
        treasury_10y=t10,
        macro_regime=regime,
        position_size_multiplier=mult,
        buy_bar_shift=shift,
        is_cached=used_cache,
        source_date=source_date,
    )

    # Persist newly fetched record to DB
    if persist and not used_cache:
        try:
            repository.save_macro_indicators({
                "date": as_of_date_str[:10],
                "fed_funds_rate": fed_funds,
                "cpi_yoy": cpi,
                "unemployment_rate": unemp,
                "treasury_10y": t10,
                "macro_regime": regime,
                "fetched_date": as_of_date_str[:10],
            })
            repository.log_event(
                "INFO", "macro_regime",
                f"Macro regime evaluated for {as_of_date_str}: {regime} (Fed Funds: {fed_funds:.2f}%, CPI: {cpi:.2f}%)",
                result.to_dict(),
            )
        except Exception as exc:
            logger.warning("Could not persist macro indicators: %s", exc)

    return result
