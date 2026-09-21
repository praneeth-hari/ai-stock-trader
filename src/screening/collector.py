"""
src/screening/collector.py — Fundamental Data Acquisition & Raw Snapshotting.

Fetches point-in-time fundamental financial metrics via yfinance.
Caches raw JSON snapshots to data/fundamentals/YYYY-MM-DD/{ticker}.json to maintain
complete auditability and prevent redundant external network calls.

CRITICAL SEPARATION:
This data is completely distinct from the OHLCV price series stored in data/raw/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

FUNDAMENTALS_BASE_DIR = Path("data/fundamentals")


@dataclass
class FundamentalData:
    ticker: str
    fetch_date: str
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    profit_margins: Optional[float] = None
    operating_margins: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(val: Any) -> Optional[float]:
    """Helper to safely convert numeric or None values to float."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (ValueError, TypeError):
        return None


def fetch_ticker_fundamentals(
    ticker: str,
    fetch_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> FundamentalData:
    """
    Fetch fundamental financial metrics for a single ticker.

    Checks cache at data/fundamentals/{fetch_date}/{ticker}.json first.
    If absent or force_refresh=True, queries yfinance and writes the raw JSON snapshot.
    """
    clean_ticker = ticker.strip().upper()
    date_str = fetch_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_dir = (cache_dir or FUNDAMENTALS_BASE_DIR) / date_str
    target_file = target_dir / f"{clean_ticker}.json"

    # Check cache
    if not force_refresh and target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return FundamentalData(**data)
        except Exception as exc:
            logger.warning("Failed to read cached fundamentals for %s: %s. Re-fetching.", clean_ticker, exc)

    # Fetch live from yfinance
    logger.info("Fetching live fundamentals for %s via yfinance...", clean_ticker)
    try:
        tk = yf.Ticker(clean_ticker)
        info = tk.info or {}
    except Exception as exc:
        logger.error("yfinance network failure for %s fundamentals: %s", clean_ticker, exc)
        info = {}

    fund = FundamentalData(
        ticker=clean_ticker,
        fetch_date=date_str,
        trailing_pe=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        peg_ratio=_safe_float(info.get("pegRatio")),
        price_to_book=_safe_float(info.get("priceToBook")),
        enterprise_to_ebitda=_safe_float(info.get("enterpriseToEbitda")),
        profit_margins=_safe_float(info.get("profitMargins")),
        operating_margins=_safe_float(info.get("operatingMargins")),
        return_on_equity=_safe_float(info.get("returnOnEquity")),
        return_on_assets=_safe_float(info.get("returnOnAssets")),
        revenue_growth=_safe_float(info.get("revenueGrowth")),
        earnings_growth=_safe_float(info.get("earningsGrowth")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        current_ratio=_safe_float(info.get("currentRatio")),
        free_cash_flow=_safe_float(info.get("freeCashflow")),
        operating_cash_flow=_safe_float(info.get("operatingCashflow")),
        dividend_yield=_safe_float(info.get("dividendYield")),
        payout_ratio=_safe_float(info.get("payoutRatio")),
        market_cap=_safe_float(info.get("marketCap")),
    )

    # Persist raw snapshot
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(fund.to_dict(), f, indent=2)
        logger.info("Saved raw fundamental snapshot: %s", target_file)
    except Exception as exc:
        logger.warning("Could not persist fundamentals cache for %s: %s", clean_ticker, exc)

    return fund


def fetch_universe_fundamentals(
    tickers: Optional[List[str]] = None,
    fetch_date: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, FundamentalData]:
    """
    Fetches fundamentals for a list of tickers (defaults to full screener universe).
    """
    from src.screening.universe import get_screener_tickers

    t_list = tickers or get_screener_tickers()
    results: Dict[str, FundamentalData] = {}
    for t in t_list:
        results[t] = fetch_ticker_fundamentals(
            ticker=t,
            fetch_date=fetch_date,
            force_refresh=force_refresh,
        )
    return results


def load_raw_fundamental_snapshot(
    ticker: str,
    fetch_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[FundamentalData]:
    """
    Reads cached fundamental snapshot from disk without touching yfinance.
    Returns FundamentalData if file exists, else None.
    """
    clean_ticker = ticker.strip().upper()
    date_str = fetch_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_file = (cache_dir or FUNDAMENTALS_BASE_DIR) / date_str / f"{clean_ticker}.json"
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return FundamentalData(**data)
        except Exception as exc:
            logger.warning("Failed to load cached fundamentals for %s: %s", clean_ticker, exc)
    return None

