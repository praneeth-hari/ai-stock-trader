"""
tests/test_market_data.py — Tests for Phase 1 Market Data Acquisition.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.market_data import (
    normalize_ohlcv,
    save_raw_snapshot,
    fetch_ticker_data,
    fetch_universe_data,
)


def test_normalize_ohlcv_standard():
    raw_data = {
        "Date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
        "Open": [380.0, 382.0],
        "High": [385.0, 386.0],
        "Low": [378.0, 381.0],
        "Close": [381.0, 384.0],
        "Volume": [1000000, 1200000],
    }
    df_raw = pd.DataFrame(raw_data).set_index("Date")
    normalized = normalize_ohlcv(df_raw, ticker="spy")

    expected_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
    assert list(normalized.columns) == expected_cols
    assert len(normalized) == 2
    assert normalized["ticker"].iloc[0] == "SPY"
    assert normalized["date"].iloc[0] == "2023-01-03"
    assert normalized["date"].iloc[1] == "2023-01-04"
    assert normalized["close"].iloc[1] == 384.0


def test_normalize_ohlcv_empty():
    df_empty = pd.DataFrame()
    normalized = normalize_ohlcv(df_empty, ticker="AAPL")
    expected_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
    assert list(normalized.columns) == expected_cols
    assert len(normalized) == 0


def test_save_raw_snapshot_date_hierarchy(tmp_path: Path):
    df_raw = pd.DataFrame({"Close": [100.0, 101.0], "Volume": [500, 600]})
    pull_date = "2026-09-12"

    saved_file = save_raw_snapshot(
        raw_df=df_raw,
        ticker="AAPL",
        pull_date=pull_date,
        base_dir=tmp_path,
    )

    assert saved_file.exists()
    assert saved_file.name == "AAPL.csv"
    assert saved_file.parent.name == pull_date
    assert saved_file.parent.parent == tmp_path

    # Read back and verify content is preserved
    read_df = pd.read_csv(saved_file)
    assert len(read_df) == 2


def test_fetch_ticker_data_live_spy(tmp_path: Path):
    """
    Test live fetch of SPY for a fixed historical trading window.
    Verifies schema, requested date range presence, and raw snapshot on disk.

    NOTE on end_date exclusivity: yfinance uses a half-open [start, end) interval.
    end_date='2024-01-11' fetches up to and including 2024-01-10.
    Trading days Jan 2,3,4,5,8,9,10 = 7 days.

    NOTE on warmup extension: fetch_ticker_data automatically extends the pull
    backwards by ~305 calendar days so Phase 3 features are not NaN at start_date.
    The returned DataFrame contains the warmup rows plus the requested range,
    so len(df) >> 7. We verify the requested dates are present, not exact count.
    """
    start_date = "2024-01-02"
    end_date = "2024-01-11"    # exclusive — fetches up to Jan 10 inclusive
    pull_date = "2026-09-12"
    expected_dates = {"2024-01-02", "2024-01-03", "2024-01-04",
                      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"}

    df = fetch_ticker_data(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
        save_raw=True,
        pull_date=pull_date,
    )

    if df.empty:
        pytest.skip("yfinance returned empty DataFrame for SPY (transient network issue)")

    expected_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
    assert list(df.columns) == expected_cols
    assert (df["ticker"] == "SPY").all()

    # The 7 requested trading days must all be present.
    present_dates = set(df["date"].values)
    missing = expected_dates - present_dates
    assert not missing, (
        f"Expected trading days missing from result: {missing}. "
        f"Got {len(df)} total rows, max date={df['date'].max()}"
    )

    # Warmup rows extend the pull before start_date — df should be much larger than 7.
    assert len(df) >= 7
    assert df["date"].max() == "2024-01-10"

    # Verify snapshot in data/raw/{pull_date}/SPY.csv
    snapshot_path = Path("data/raw") / pull_date / "SPY.csv"
    assert snapshot_path.exists()
    assert snapshot_path.stat().st_size > 0
