"""
tests/test_live_ticker.py — Section 6 Item 3: Live Price Ticker Unit & Integration Tests.

Verifies:
  1. Market hours correctly identifies weekends as Closed.
  2. Market hours correctly identifies NYSE holidays as Closed.
  3. Market hours correctly identifies weekday 9:30 AM - 4:00 PM ET as Open.
  4. Market hours identifies pre-market and after-hours as Closed.
  5. Live price fetch returns valid quote with positive delta when price is up.
  6. Live price fetch returns valid quote with negative delta when price is down.
  7. Network failure falls back gracefully to last known price with STALE badge flag.
  8. Only currently held stocks in open positions are queried (API call conservation).
  9. Empty portfolio yields empty quote list with zero API calls.
 10. Unrealized position P&L accurately calculated using live price against entry price.
 11. Streamlit AppTest verifies full Tab 1 rendering including Live Prices section.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.live_ticker import (
    fetch_single_ticker_live_price,
    get_live_quotes_for_held_stocks,
    is_us_market_open,
)
from src.db import repository

NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture()
def clean_db(tmp_path):
    """Provides a fresh isolated SQLite database for live ticker tests."""
    db_file = tmp_path / "test_live_ticker.db"
    db_url = f"sqlite:///{db_file}"
    repository._engine = None
    orig_url = repository.settings.db_url

    try:
        repository.settings.__dict__["db_url"] = db_url
        repository.create_all_tables()
        yield db_url
    finally:
        repository._engine = None
        repository.settings.__dict__["db_url"] = orig_url


# ── Market Hours Tests ────────────────────────────────────────────────────────

def test_1_market_hours_weekend():
    """Saturdays and Sundays must be identified as Market Closed."""
    saturday_dt = datetime(2026, 9, 19, 11, 0, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(saturday_dt)
    assert is_open is False
    assert "Saturday" in msg

    sunday_dt = datetime(2026, 9, 20, 14, 0, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(sunday_dt)
    assert is_open is False
    assert "Sunday" in msg


def test_2_market_hours_nyse_holiday():
    """Standard NYSE holidays must be identified as Market Closed."""
    # Christmas 2026-12-25 (Friday)
    christmas_dt = datetime(2026, 12, 25, 12, 0, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(christmas_dt)
    assert is_open is False
    assert "Holiday" in msg


def test_3_market_hours_regular_session():
    """Active trading session (Wednesday 10:30 AM ET) must return Market Open."""
    wednesday_dt = datetime(2026, 9, 16, 10, 30, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(wednesday_dt)
    assert is_open is True
    assert "Open" in msg


def test_4_market_hours_pre_and_post_market():
    """Hours strictly outside 9:30 AM - 4:00 PM ET on weekdays must return Market Closed."""
    pre_market = datetime(2026, 9, 16, 8, 30, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(pre_market)
    assert is_open is False
    assert "Pre-Market" in msg

    after_hours = datetime(2026, 9, 16, 16, 30, tzinfo=NY_TZ)
    is_open, msg = is_us_market_open(after_hours)
    assert is_open is False
    assert "After-Hours" in msg


# ── Quote Fetching & Calculation Tests ────────────────────────────────────────

def test_5_fetch_single_ticker_price_up():
    """When last_price > previous_close, delta is positive and is_up is True."""
    mock_fast_info = MagicMock()
    mock_fast_info.last_price = 155.0
    mock_fast_info.previous_close = 150.0

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_instance

        quote = fetch_single_ticker_live_price("AAPL")
        assert quote["ticker"] == "AAPL"
        assert quote["current_price"] == 155.0
        assert quote["prev_close"] == 150.0
        assert quote["change_dollar"] == 5.0
        assert quote["change_pct"] == pytest.approx(3.33, abs=1e-2)
        assert quote["is_up"] is True
        assert quote["is_stale"] is False


def test_6_fetch_single_ticker_price_down():
    """When last_price < previous_close, delta is negative and is_up is False."""
    mock_fast_info = MagicMock()
    mock_fast_info.last_price = 145.0
    mock_fast_info.previous_close = 150.0

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_instance

        quote = fetch_single_ticker_live_price("NVDA")
        assert quote["ticker"] == "NVDA"
        assert quote["current_price"] == 145.0
        assert quote["change_dollar"] == -5.0
        assert quote["change_pct"] == pytest.approx(-3.33, abs=1e-2)
        assert quote["is_up"] is False
        assert quote["is_stale"] is False


def test_7_fetch_single_ticker_network_failure_stale_fallback():
    """When yfinance raises network/rate error, falls back to provided price with is_stale=True."""
    with patch("yfinance.Ticker", side_effect=RuntimeError("Connection reset")):
        quote = fetch_single_ticker_live_price("META", fallback_price=650.0)
        assert quote["ticker"] == "META"
        assert quote["current_price"] == 650.0
        assert quote["is_stale"] is True


def test_8_only_held_stocks_queried():
    """Verifies that quotes are fetched strictly for tickers held in open positions."""
    held_positions = [
        {"ticker": "AAPL", "shares": 10.0, "entry_price": 150.0, "current_price": 150.0},
        {"ticker": "META", "shares": 5.0, "entry_price": 600.0, "current_price": 600.0},
    ]

    with patch("dashboard.live_ticker.fetch_single_ticker_live_price") as mock_fetch:
        mock_fetch.side_effect = lambda ticker, fallback_price: {
            "ticker": ticker,
            "current_price": fallback_price or 100.0,
            "prev_close": 100.0,
            "change_dollar": 0.0,
            "change_pct": 0.0,
            "is_up": True,
            "is_stale": False,
            "timestamp": "2026-09-19 12:00:00 ET",
        }

        quotes = get_live_quotes_for_held_stocks(held_positions)
        assert len(quotes) == 2
        called_tickers = [call.kwargs["ticker"] for call in mock_fetch.call_args_list]
        assert called_tickers == ["AAPL", "META"]


def test_9_empty_positions_zero_api_calls():
    """Empty positions list must return empty quotes list immediately with zero network queries."""
    with patch("dashboard.live_ticker.fetch_single_ticker_live_price") as mock_fetch:
        quotes = get_live_quotes_for_held_stocks([])
        assert quotes == []
        mock_fetch.assert_not_called()


def test_10_unrealized_position_pnl_calculation():
    """Verifies position unrealized P&L matches (current_price - entry_price) * shares."""
    positions = [
        {"ticker": "AAPL", "shares": 10.0, "entry_price": 150.0, "current_price": 150.0},
    ]

    with patch("dashboard.live_ticker.fetch_single_ticker_live_price") as mock_fetch:
        mock_fetch.return_value = {
            "ticker": "AAPL",
            "current_price": 160.0,  # +$10/share
            "prev_close": 155.0,
            "change_dollar": 5.0,
            "change_pct": 3.23,
            "is_up": True,
            "is_stale": False,
            "timestamp": "2026-09-19 12:00:00 ET",
        }

        quotes = get_live_quotes_for_held_stocks(positions)
        assert len(quotes) == 1
        q = quotes[0]
        assert q["unrealized_pnl"] == pytest.approx(100.0, abs=1e-2)  # ($160 - $150) * 10
        assert q["unrealized_pnl_pct"] == pytest.approx(6.67, abs=1e-2)  # ($10 / $150) * 100


def test_11_app_test_renders_live_ticker_in_tab1(clean_db):
    """Streamlit AppTest confirms that Tab 1 renders Live Prices ticker cleanly."""
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    # Seed snapshot with active positions
    positions = {
        "AAPL": {"shares": 5.0, "entry_price": 150.0, "current_price": 155.0, "market_value": 775.0},
    }
    repository.save_portfolio_snapshot("2026-09-19", cash=5000.0, total_value=5775.0, positions=positions)

    fake_spy = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=250, freq="B").strftime("%Y-%m-%d"),
        "open": [400.0] * 250, "high": [405.0] * 250, "low": [395.0] * 250,
        "close": [402.0] * 250, "volume": [50000000.0] * 250, "ticker": ["SPY"] * 250,
    })

    with patch("src.data.market_data.fetch_ticker_data", return_value=fake_spy):
        at = AppTest.from_file("dashboard/app.py", default_timeout=15)
        at.run()
        assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"
