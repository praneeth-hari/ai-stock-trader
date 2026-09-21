"""
tests/test_data_fallback.py — Test Data Feed Retries, SQLite Cache Fallback, and Abort Behavior (Section 1).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.settings import settings
from src.data.market_data import fetch_ticker_data
from src.db import repository


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """Use an isolated SQLite database for each test to avoid test contamination."""
    test_db = f"sqlite:///{tmp_path}/test_fallback.db"
    monkeypatch.setattr(settings, "db_url", test_db)
    # Prevent sleep delays during tests
    monkeypatch.setattr(settings, "data_fetch_retry_wait_sec", 0.001)
    repository.create_all_tables()


def _make_dummy_ohlcv(ticker: str = "AAPL", date_str: str = "2024-01-05") -> pd.DataFrame:
    return pd.DataFrame([{
        "date": date_str,
        "ticker": ticker,
        "open": 180.0,
        "high": 185.0,
        "low": 179.0,
        "close": 182.5,
        "volume": 50_000_000.0,
    }])


def test_retry_success_on_third_attempt(monkeypatch):
    """If yfinance fails twice but succeeds on the 3rd attempt, fetch_ticker_data succeeds."""
    mock_success_df = pd.DataFrame({
        "Open": [180.0], "High": [185.0], "Low": [179.0], "Close": [182.5], "Volume": [50_000_000],
    }, index=pd.to_datetime(["2024-01-05"]))

    attempts = [pd.DataFrame(), pd.DataFrame(), mock_success_df]

    def mock_download(**kwargs):
        return attempts.pop(0)

    monkeypatch.setattr("yfinance.download", mock_download)

    df = fetch_ticker_data("AAPL", start_date="2024-01-05", end_date="2024-01-06", save_raw=False)
    assert not df.empty
    assert len(df) == 1
    assert df["close"].iloc[0] == 182.5
    # Verify all 3 attempts were consumed
    assert len(attempts) == 0


def test_sqlite_fallback_when_all_retries_fail(caplog, monkeypatch):
    """When all retries fail, retrieve cached data from SQLite and log warning."""
    # Pre-populate SQLite with cached data
    cached_df = _make_dummy_ohlcv("MSFT", "2024-01-04")
    repository.save_market_data(cached_df)

    # Mock yfinance to always return empty DataFrame
    monkeypatch.setattr("yfinance.download", lambda **kwargs: pd.DataFrame())

    with caplog.at_level("WARNING"):
        df = fetch_ticker_data("MSFT", start_date="2024-01-04", end_date="2024-01-05", save_raw=False)

    assert not df.empty
    assert df["ticker"].iloc[0] == "MSFT"
    assert df["close"].iloc[0] == 182.5
    assert "USING CACHED DATA — LIVE FETCH FAILED" in caplog.text


def test_abort_when_no_cache_exists(caplog, monkeypatch):
    """When all retries fail and SQLite has no cached data, abort with critical log and raise."""
    monkeypatch.setattr("yfinance.download", lambda **kwargs: pd.DataFrame())

    with caplog.at_level("CRITICAL"):
        with pytest.raises(RuntimeError, match="DAILY RUN ABORTED — NO DATA AVAILABLE"):
            fetch_ticker_data("UNKNOWN_TICKER", save_raw=False)

    assert "DAILY RUN ABORTED — NO DATA AVAILABLE" in caplog.text


def test_market_data_ingestion_stores_data_as_of():
    """Verify that market data rows saved to the repository have data_as_of timestamps."""
    raw_df = _make_dummy_ohlcv("GOOGL", "2024-01-08")
    repository.save_market_data(raw_df)

    retrieved = repository.get_market_data("GOOGL", include_metadata=True)
    assert not retrieved.empty
    assert "data_as_of" in retrieved.columns
    assert retrieved["data_as_of"].iloc[0] is not None
    assert len(str(retrieved["data_as_of"].iloc[0])) > 0
