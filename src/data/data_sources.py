"""
src/data/data_sources.py — Multiple Data Sources & Resilient Failover Architecture (Section 8 Item 1).

PURPOSE:
Provides fetch_with_fallback() which orchestrates multi-provider market data fetching:
  1. Primary Source: yfinance
  2. Backup Source: Alpha Vantage (free tier, 25 calls/day limit)
  3. Fallback Source: SQLite Database Cache & Raw Snapshots

LOGGING & ALERTS:
  - Logs exact source used per fetch:
    "DATA_SOURCE: yfinance (primary)"
    "DATA_SOURCE: alpha_vantage (backup)"
    "DATA_SOURCE: sqlite_cache (fallback)"
  - Persists data source event to repository event_log for audit trail.
  - Triggers Telegram and Email alerts on failover from primary to backup.
  - Enforces daily rate limit tracking for Alpha Vantage with threshold warnings.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Daily Call Tracking for Alpha Vantage ─────────────────────────────────────
_av_call_count: int = 0
_av_call_date: str = ""


def get_alpha_vantage_call_count() -> int:
    """Return the number of Alpha Vantage API calls made today."""
    global _av_call_count, _av_call_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _av_call_date != today:
        _av_call_count = 0
        _av_call_date = today
    return _av_call_count


def increment_alpha_vantage_call_count() -> int:
    """Increment and return daily Alpha Vantage call count."""
    global _av_call_count, _av_call_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _av_call_date != today:
        _av_call_count = 0
        _av_call_date = today
    _av_call_count += 1
    return _av_call_count


def reset_alpha_vantage_call_count() -> None:
    """Reset daily call count (useful for testing)."""
    global _av_call_count, _av_call_date
    _av_call_count = 0
    _av_call_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _log_event_safe(level: str, component: str, message: str, details: Optional[dict] = None) -> None:
    """Safely log event to repository event_log table without raising exceptions."""
    try:
        from src.db import repository
        repository.log_event(level, component, message, details)
    except Exception as exc:
        logger.debug("Failed to log event to repository: %s", exc)


def _trigger_failover_alerts(ticker: str, primary_source: str, backup_source: str) -> None:
    """Send Telegram and Email failover alerts when primary source fails."""
    try:
        from src.alerts.email_alerts import send_failover_alert
        send_failover_alert(ticker, primary_source, backup_source)
    except Exception as exc:
        logger.warning("Failed sending email failover alert for %s: %s", ticker, exc)

    try:
        from src.alerts.telegram_alerts import notify_failover
        notify_failover(ticker, primary_source, backup_source)
    except Exception as exc:
        logger.warning("Failed sending telegram failover alert for %s: %s", ticker, exc)


def normalize_av_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardize Alpha Vantage output into standard schema:
    [date, open, high, low, close, volume, ticker].
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    clean_df = df.copy()

    # Rename Alpha Vantage column patterns like '1. open', '2. high', '4. close'
    col_map = {}
    for col in clean_df.columns:
        c_lower = str(col).lower()
        if "open" in c_lower:
            col_map[col] = "open"
        elif "high" in c_lower:
            col_map[col] = "high"
        elif "low" in c_lower:
            col_map[col] = "low"
        elif "close" in c_lower and "adj" not in c_lower:
            col_map[col] = "close"
        elif "volume" in c_lower:
            col_map[col] = "volume"

    clean_df = clean_df.rename(columns=col_map)

    # Date from index or column
    if "date" not in clean_df.columns:
        clean_df = clean_df.reset_index()
        date_col = next((c for c in clean_df.columns if str(c).lower() in ("date", "index")), None)
        if date_col:
            clean_df = clean_df.rename(columns={date_col: "date"})

    clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.strftime("%Y-%m-%d")

    required = ["date", "open", "high", "low", "close", "volume"]
    for r in required:
        if r not in clean_df.columns:
            raise ValueError(f"Alpha Vantage response for {ticker} missing column: {r}")

    for r in ["open", "high", "low", "close", "volume"]:
        clean_df[r] = pd.to_numeric(clean_df[r], errors="coerce")

    clean_df["ticker"] = ticker.strip().upper()
    return clean_df[required + ["ticker"]].sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


# ── Provider Fetch Implementations ────────────────────────────────────────────

def _fetch_yfinance_primary(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    max_retries: int = 2,
    wait_sec: float = 1.0,
) -> pd.DataFrame:
    """Fetch raw data from yfinance (primary source)."""
    kwargs: dict = dict(tickers=ticker, auto_adjust=False, progress=False)
    if start_date and end_date:
        warmup_calendar_days = int(settings.warmup_bars * 365 / 252) + 15
        extended_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=warmup_calendar_days)
        ).strftime("%Y-%m-%d")
        kwargs["start"] = extended_start
        kwargs["end"] = end_date
    elif period:
        kwargs["period"] = period
    else:
        kwargs["period"] = "2y"

    for attempt in range(1, max_retries + 1):
        try:
            raw_df = yf.download(**kwargs)
            if not raw_df.empty:
                return raw_df
        except Exception as exc:
            logger.warning("yfinance fetch error for %s (attempt %d/%d): %s", ticker, attempt, max_retries, exc)

        if attempt < max_retries and wait_sec > 0:
            time.sleep(wait_sec)

    return pd.DataFrame()


def _fetch_alpha_vantage_backup(ticker: str) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage (backup source)."""
    api_key = (getattr(settings, "alpha_vantage_api_key", "") or "").strip()
    if not api_key:
        logger.warning("Alpha Vantage backup skipped: ALPHA_VANTAGE_API_KEY not configured.")
        return pd.DataFrame()

    current_calls = get_alpha_vantage_call_count()
    limit = getattr(settings, "alpha_vantage_daily_limit", 25)

    if current_calls >= limit:
        logger.warning("Alpha Vantage backup skipped: Daily call limit reached (%d/%d).", current_calls, limit)
        return pd.DataFrame()

    if current_calls >= limit - 5:
        logger.warning("Alpha Vantage daily call count approaching limit: %d/%d", current_calls, limit)

    increment_alpha_vantage_call_count()
    logger.info("Fetching Alpha Vantage backup for %s (Call %d/%d)", ticker, get_alpha_vantage_call_count(), limit)

    # Try alpha_vantage python package first
    try:
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=api_key, output_format="pandas")
        df, _ = ts.get_daily_full(symbol=ticker)
        if not df.empty:
            return normalize_av_df(df, ticker)
    except Exception as exc:
        logger.warning("alpha_vantage SDK error for %s: %s. Trying HTTP REST fallback.", ticker, exc)

    # HTTP REST fallback if SDK fails
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            time_series = data.get("Time Series (Daily)", {})
            if time_series:
                df = pd.DataFrame.from_dict(time_series, orient="index")
                return normalize_av_df(df, ticker)
    except Exception as exc:
        logger.warning("Alpha Vantage REST API error for %s: %s", ticker, exc)

    return pd.DataFrame()


def _fetch_sqlite_cache_fallback(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch market data from SQLite database or raw disk snapshot (fallback source)."""
    # 1. Try SQLite database
    try:
        from src.db import repository
        cached_db = repository.get_market_data(ticker, start_date=start_date, end_date=end_date)
        if cached_db.empty:
            cached_db = repository.get_market_data(ticker)
        if not cached_db.empty:
            std_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
            return cached_db[std_cols].sort_values("date").reset_index(drop=True)
    except Exception as cache_exc:
        logger.warning("SQLite cache query failed for %s: %s", ticker, cache_exc)

    # 2. Try raw CSV snapshots on disk
    cached_files = sorted(Path("data/raw").glob(f"*/{ticker}.csv"), reverse=True)
    for cf in cached_files:
        try:
            from src.data.market_data import normalize_ohlcv
            cached_df = pd.read_csv(cf, header=[0, 1], index_col=0, parse_dates=True)
            if not cached_df.empty:
                return normalize_ohlcv(cached_df, ticker)
        except Exception:
            continue

    return pd.DataFrame()


# ── Public API: fetch_with_fallback ───────────────────────────────────────────

def fetch_with_fallback(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    save_raw: bool = True,
    pull_date: Optional[str] = None,
    max_retries: Optional[int] = None,
    retry_wait_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data using multi-provider fallback strategy:
      1. Primary: yfinance
      2. Backup: Alpha Vantage (if yfinance empty/fails)
      3. Fallback: SQLite cache / snapshot (if both fail)

    Returns:
        Standardized DataFrame [date, open, high, low, close, volume, ticker].
    """
    from src.data.market_data import normalize_ohlcv, save_raw_snapshot

    clean_ticker = ticker.strip().upper()
    retries = max_retries if max_retries is not None else settings.data_fetch_max_retries
    wait_sec = retry_wait_sec if retry_wait_sec is not None else settings.data_fetch_retry_wait_sec

    # ── 1. Try Primary Source: yfinance ───────────────────────────────────────
    logger.info("Trying Primary Data Source (yfinance) for %s...", clean_ticker)
    yf_raw = _fetch_yfinance_primary(
        clean_ticker,
        start_date=start_date,
        end_date=end_date,
        period=period,
        max_retries=retries,
        wait_sec=wait_sec,
    )

    if not yf_raw.empty:
        source_label = "DATA_SOURCE: yfinance (primary)"
        logger.info("%s used for %s (%d rows)", source_label, clean_ticker, len(yf_raw))
        _log_event_safe("INFO", "market_data", source_label, {"ticker": clean_ticker, "source": "yfinance"})

        if save_raw:
            try:
                save_raw_snapshot(yf_raw, clean_ticker, pull_date=pull_date)
            except Exception as exc:
                logger.warning("Failed saving raw snapshot for %s: %s", clean_ticker, exc)

        return normalize_ohlcv(yf_raw, clean_ticker)

    # ── 2. Primary failed → Try Backup Source: Alpha Vantage ──────────────────
    logger.warning("Primary source (yfinance) failed for %s. Initiating failover to Alpha Vantage...", clean_ticker)

    av_df = _fetch_alpha_vantage_backup(clean_ticker)

    if not av_df.empty:
        source_label = "DATA_SOURCE: alpha_vantage (backup)"
        logger.warning("%s used for %s (%d rows)", source_label, clean_ticker, len(av_df))
        _log_event_safe("WARNING", "market_data", source_label, {"ticker": clean_ticker, "source": "alpha_vantage"})

        # Trigger failover alerts (Telegram + Email)
        _trigger_failover_alerts(clean_ticker, "yfinance", "Alpha Vantage")

        return av_df

    # ── 3. Both Primary & Backup failed → Try Fallback Source: SQLite Cache ───
    logger.warning("Primary (yfinance) and Backup (Alpha Vantage) failed for %s. Trying SQLite cache...", clean_ticker)

    cache_df = _fetch_sqlite_cache_fallback(clean_ticker, start_date=start_date, end_date=end_date)

    if not cache_df.empty:
        logger.warning("USING CACHED DATA — LIVE FETCH FAILED")
        source_label = "DATA_SOURCE: sqlite_cache (fallback)"
        logger.warning("%s used for %s (%d rows)", source_label, clean_ticker, len(cache_df))
        _log_event_safe("WARNING", "market_data", source_label, {"ticker": clean_ticker, "source": "sqlite_cache"})
        return cache_df

    # ── All sources failed ────────────────────────────────────────────────────
    err = f"DAILY RUN ABORTED — NO DATA AVAILABLE for {clean_ticker}"
    logger.critical(err)
    _log_event_safe("CRITICAL", "market_data", err, {"ticker": clean_ticker})
    raise RuntimeError(err)
