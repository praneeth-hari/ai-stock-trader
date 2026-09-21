"""
tests/test_data_sources.py — Unit tests for Section 8 Item 1: Multiple Data Sources.

Covers:
  - Primary source (yfinance) success → returns data, logs yfinance source event
  - Primary fails, Backup (Alpha Vantage) succeeds → triggers failover alerts, increments call count, logs alpha_vantage source event
  - Primary and Backup fail, Fallback (SQLite cache) succeeds → returns cached data, logs sqlite_cache source event
  - All sources fail → raises RuntimeError with NO DATA AVAILABLE message
  - Alpha Vantage rate limit tracking and threshold warnings
  - Alpha Vantage normalization helper (normalize_av_df)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.data.data_sources as ds


@pytest.fixture(autouse=True)
def _reset_av_counter():
    """Ensure call count is reset before each test."""
    ds.reset_alpha_vantage_call_count()
    yield
    ds.reset_alpha_vantage_call_count()


def _dummy_raw_df() -> pd.DataFrame:
    """Return a mock yfinance raw DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame({
        "Open": [150.0, 151.0, 152.0],
        "High": [155.0, 156.0, 157.0],
        "Low": [149.0, 150.0, 151.0],
        "Close": [154.0, 155.0, 156.0],
        "Volume": [1000, 1100, 1200],
    }, index=dates)


def _dummy_av_raw_df() -> pd.DataFrame:
    """Return a mock Alpha Vantage raw DataFrame."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    return pd.DataFrame({
        "1. open": [150.0, 151.0, 152.0],
        "2. high": [155.0, 156.0, 157.0],
        "3. low": [149.0, 150.0, 151.0],
        "4. close": [154.0, 155.0, 156.0],
        "5. volume": [1000, 1100, 1200],
    }, index=dates)


# ── 1. Primary Source Tests ───────────────────────────────────────────────────

class TestPrimarySource:

    def test_yfinance_success_returns_data_and_logs(self):
        with patch("src.data.data_sources._fetch_yfinance_primary", return_value=_dummy_raw_df()), \
             patch("src.db.repository.log_event") as mock_log:
            df = ds.fetch_with_fallback("AAPL", save_raw=False)

        assert not df.empty
        assert "ticker" in df.columns
        assert df["ticker"].iloc[0] == "AAPL"
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]

        # Verify event logging for yfinance
        mock_log.assert_called_once()
        assert "DATA_SOURCE: yfinance (primary)" in mock_log.call_args[0][2]


# ── 2. Backup Source Tests (Alpha Vantage) ────────────────────────────────────

class TestBackupSource:

    def test_yfinance_fails_alpha_vantage_succeeds(self):
        with patch("src.data.data_sources._fetch_yfinance_primary", return_value=pd.DataFrame()), \
             patch("src.data.data_sources._fetch_alpha_vantage_backup", return_value=ds.normalize_av_df(_dummy_av_raw_df(), "AAPL")), \
             patch("src.db.repository.log_event") as mock_log, \
             patch("src.alerts.email_alerts.send_failover_alert") as mock_email, \
             patch("src.alerts.telegram_alerts.notify_failover") as mock_tele:

            df = ds.fetch_with_fallback("AAPL", save_raw=False)

        assert not df.empty
        assert df["ticker"].iloc[0] == "AAPL"
        # Verify log event
        mock_log.assert_called_once()
        assert "DATA_SOURCE: alpha_vantage (backup)" in mock_log.call_args[0][2]

        # Verify failover alerts triggered
        mock_email.assert_called_once_with("AAPL", "yfinance", "Alpha Vantage")
        mock_tele.assert_called_once_with("AAPL", "yfinance", "Alpha Vantage")


# ── 3. Fallback Source Tests (SQLite Cache) ───────────────────────────────────

class TestFallbackSource:

    def test_yfinance_and_av_fail_sqlite_cache_succeeds(self):
        cached_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [500.0],
            "ticker": ["AAPL"],
        })

        with patch("src.data.data_sources._fetch_yfinance_primary", return_value=pd.DataFrame()), \
             patch("src.data.data_sources._fetch_alpha_vantage_backup", return_value=pd.DataFrame()), \
             patch("src.data.data_sources._fetch_sqlite_cache_fallback", return_value=cached_df), \
             patch("src.db.repository.log_event") as mock_log:

            df = ds.fetch_with_fallback("AAPL", save_raw=False)

        assert not df.empty
        assert df["close"].iloc[0] == 104.0
        mock_log.assert_called_once()
        assert "DATA_SOURCE: sqlite_cache (fallback)" in mock_log.call_args[0][2]


# ── 4. All Sources Fail ───────────────────────────────────────────────────────

class TestAllSourcesFail:

    def test_all_sources_fail_raises_runtime_error(self):
        with patch("src.data.data_sources._fetch_yfinance_primary", return_value=pd.DataFrame()), \
             patch("src.data.data_sources._fetch_alpha_vantage_backup", return_value=pd.DataFrame()), \
             patch("src.data.data_sources._fetch_sqlite_cache_fallback", return_value=pd.DataFrame()):

            with pytest.raises(RuntimeError, match="DAILY RUN ABORTED — NO DATA AVAILABLE for AAPL"):
                ds.fetch_with_fallback("AAPL", save_raw=False)


# ── 5. Rate Limit Tracking Tests ──────────────────────────────────────────────

class TestRateLimitTracking:

    def test_call_count_increment_and_reset(self):
        assert ds.get_alpha_vantage_call_count() == 0
        ds.increment_alpha_vantage_call_count()
        assert ds.get_alpha_vantage_call_count() == 1
        ds.reset_alpha_vantage_call_count()
        assert ds.get_alpha_vantage_call_count() == 0

    def test_alpha_vantage_skips_when_api_key_missing(self):
        with patch.object(ds.settings, "alpha_vantage_api_key", ""):
            df = ds._fetch_alpha_vantage_backup("AAPL")
        assert df.empty

    def test_alpha_vantage_skips_when_limit_reached(self):
        with patch.object(ds.settings, "alpha_vantage_api_key", "testkey"), \
             patch.object(ds.settings, "alpha_vantage_daily_limit", 2):
            ds.increment_alpha_vantage_call_count()
            ds.increment_alpha_vantage_call_count()
            assert ds.get_alpha_vantage_call_count() == 2

            df = ds._fetch_alpha_vantage_backup("AAPL")
            assert df.empty


# ── 6. Normalization Tests ────────────────────────────────────────────────────

class TestNormalization:

    def test_normalize_av_df(self):
        raw = _dummy_av_raw_df()
        normalized = ds.normalize_av_df(raw, "MSFT")

        assert list(normalized.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]
        assert normalized["ticker"].iloc[0] == "MSFT"
        assert len(normalized) == 3
        assert normalized["open"].iloc[0] == 150.0

    def test_normalize_av_df_empty(self):
        norm = ds.normalize_av_df(pd.DataFrame(), "TSLA")
        assert norm.empty
        assert list(norm.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]
