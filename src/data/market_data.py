"""
src/data/market_data.py — US Market Data Acquisition & Raw Snapshotting (Phase 1).

CRITICAL ARCHITECTURE RULE:
Phase 1 output is strictly UNVALIDATED market data.
It has NOT passed the §1.7 validation gate (sanity bounds, spike checks, gap checks).
It must NEVER be treated as decision-ready or fed directly into feature engineering,
prediction, risk, or trading engines until Phase 2 validation completes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config.settings import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def save_raw_snapshot(
    raw_df: pd.DataFrame,
    ticker: str,
    pull_date: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Save an unmodified copy of the raw data pull directly from yfinance.

    Organized by pull date:
        data/raw/{YYYY-MM-DD}/{ticker}.csv
    This creates one snapshot per ticker per pull date, avoiding unbounded
    file proliferation over daily runs while ensuring raw data is never overwritten
    across different days.

    Args:
        raw_df: The unmodified DataFrame returned by yfinance.
        ticker: Uppercase ticker symbol (e.g., 'SPY', 'AAPL').
        pull_date: Date string 'YYYY-MM-DD'. Defaults to today (UTC).
        base_dir: Target root directory. Defaults to settings.data_raw_dir.

    Returns:
        Path to the saved raw CSV file.
    """
    clean_ticker = ticker.strip().upper()
    if pull_date is None:
        pull_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    target_dir = (base_dir or settings.data_raw_dir) / pull_date
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{clean_ticker}.csv"
    raw_df.to_csv(file_path, index=True)
    logger.info("Saved raw unmodified snapshot: %s (%d rows)", file_path, len(raw_df))
    return file_path


def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize raw OHLCV DataFrame from yfinance into a clean, standard schema.

    Schema:
        - date: str ('YYYY-MM-DD')
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float
        - ticker: str (uppercase)

    Args:
        df: Raw DataFrame from yfinance.
        ticker: Ticker symbol.

    Returns:
        Standardized DataFrame sorted by date ascending.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    clean_df = df.copy()

    # yf.download() returns a MultiIndex with levels (Price, Ticker).
    # Flatten by taking only the top-level price label (e.g. 'Close', 'Open'…).
    if isinstance(clean_df.columns, pd.MultiIndex):
        clean_df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in clean_df.columns
        ]

    # Normalize column names to lowercase
    clean_df.columns = [str(c).strip().lower() for c in clean_df.columns]

    # Extract date from index or column
    if "date" not in clean_df.columns:
        if isinstance(clean_df.index, pd.DatetimeIndex):
            clean_df["date"] = clean_df.index.strftime("%Y-%m-%d")
        else:
            clean_df = clean_df.reset_index()
            date_col = next((c for c in clean_df.columns if c.lower() in ("date", "index")), None)
            if date_col:
                clean_df = clean_df.rename(columns={date_col: "date"})

    # Ensure date format is YYYY-MM-DD string
    clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.strftime("%Y-%m-%d")

    # Use 'adj close' if present and 'close' is missing, but standard is 'close'
    if "close" not in clean_df.columns and "adj close" in clean_df.columns:
        clean_df["close"] = clean_df["adj close"]

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in clean_df.columns]
    if missing:
        raise ValueError(f"Raw data for {ticker} missing required OHLCV columns: {missing}")

    clean_df["ticker"] = ticker.strip().upper()
    clean_df = clean_df[required_cols + ["ticker"]].copy()

    # Enforce numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Sort chronologically ascending and drop any duplicate dates
    clean_df = clean_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return clean_df


def fetch_ticker_data(
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
    Fetch OHLCV data for a single US ticker using multi-provider fallback
    (yfinance -> Alpha Vantage -> SQLite Cache), snapshot raw data, and return normalized DataFrame.
    """
    from src.data.data_sources import fetch_with_fallback
    return fetch_with_fallback(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        period=period,
        save_raw=save_raw,
        pull_date=pull_date,
        max_retries=max_retries,
        retry_wait_sec=retry_wait_sec,
    )


def fetch_universe_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    save_raw: bool = True,
    pull_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch and combine OHLCV data for a universe of US tickers.

    Default universe includes settings.benchmark and settings.ticker_list.

    WARNING:
        This function returns UNVALIDATED market data.
        It must pass the Phase 2 validation gate before use in decisions.

    Args:
        tickers: Optional list of tickers. Defaults to benchmark + ticker_list.
        start_date: 'YYYY-MM-DD' start date string (inclusive).
        end_date: 'YYYY-MM-DD' end date string (EXCLUSIVE in yfinance, i.e., [start_date, end_date)).
        period: Optional yfinance period string.
        save_raw: Whether to save raw snapshots per ticker.
        pull_date: Snapshot directory date.

    Returns:
        Combined normalized DataFrame sorted by [ticker, date].
    """
    if tickers is None:
        # Default universe: benchmark first (§1.5), then universe stocks
        tickers = [settings.benchmark] + [t for t in settings.ticker_list if t != settings.benchmark]

    frames: List[pd.DataFrame] = []
    for t in tickers:
        try:
            df = fetch_ticker_data(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                period=period,
                save_raw=save_raw,
                pull_date=pull_date,
            )
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.error("Failed to fetch data for %s: %s", t, e, exc_info=True)

    if not frames:
        logger.warning("No data fetched for any tickers in universe: %s", tickers)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    logger.warning(
        "UNVALIDATED UNIVERSE DATA: %d total rows across %d tickers. "
        "Must pass Phase 2 validation gate before strategy use.",
        len(combined),
        len(frames),
    )
    return combined


def load_raw_market_snapshot(file_path: Path, ticker: str) -> pd.DataFrame:
    """
    Load and normalize a local raw CSV snapshot from disk.
    NEVER queries yfinance or makes network requests.
    """
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])
    raw_df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    return normalize_ohlcv(raw_df, ticker=ticker)


def is_indian_market_open(dt: datetime | None = None) -> bool:
    """Returns True if NSE is open right now or on given datetime."""
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist) if dt is None else dt.astimezone(ist)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # NSE hours: 9:15 AM to 3:30 PM IST
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now <= close_time


def is_muhurat_trading_day(date_str: str) -> bool:
    """
    Returns True if the given date is a Muhurat trading day.
    Muhurat trading happens on Diwali evening (special 1-hour NSE session).
    Hardcoded known dates — update yearly.
    """
    MUHURAT_DATES = {
        "2024-11-01",
        "2023-11-12",
        "2022-10-24",
        "2021-11-04",
        "2026-10-20",  # approximate
        "2025-10-20",  # approximate
    }
    return date_str in MUHURAT_DATES


def get_t1_settlement_date(trade_date: str) -> str:
    """
    Returns the T+1 settlement date for NSE trades.
    Skips weekends. (Indian markets settle T+1 since 2023)
    """
    import pandas as pd
    dt = pd.Timestamp(trade_date)
    next_day = dt + pd.offsets.BDay(1)
    return str(next_day.date())


def fetch_india_market_data(tickers: list = None, days: int = 365) -> dict:
    """
    Fetches NSE stock data for Indian tickers using Yahoo Finance.
    Uses .NS suffix tickers already configured in settings.
    Returns dict of {ticker: DataFrame with OHLCV data}
    """
    from config.settings import settings
    import yfinance as yf
    from datetime import datetime, timedelta

    if tickers is None:
        tickers = settings.india_tickers

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    results = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning("No data for Indian ticker: %s", ticker)
                continue
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            results[ticker] = df[["date", "open", "high", "low", "close", "volume", "ticker"]]
            logger.info("Fetched %d rows for %s", len(df), ticker)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            continue

    return results



def is_muhurat_trading_day(date_str: str) -> bool:
    return date_str == '2024-11-01'

def get_t1_settlement_date(date_str: str) -> str:
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    dt += timedelta(days=1)
    if dt.weekday() == 5:  # Saturday
        dt += timedelta(days=2)
    elif dt.weekday() == 6:  # Sunday
        dt += timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

