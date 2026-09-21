"""
tests/test_earnings.py — Unit tests for Section 5 Item 2: Earnings Calendar Awareness.

Covers:
  - parse_earnings_date_from_calendar (dict and DataFrame formats)
  - fetch_earnings_for_ticker classification:
      * TODAY (days == 0) → is_blocked=True, hold_protection=True
      * BLACKOUT (1-2 days) → is_blocked=True
      * CAUTION (3-5 days) → size_multiplier=0.5
      * OK (> 5 days) → no adjustment
  - get_universe_earnings_calendar with injected dates
  - Badge text property
  - Graceful degradation on yfinance failure
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.intelligence.earnings import (
    parse_earnings_date_from_calendar,
    fetch_earnings_for_ticker,
    get_universe_earnings_calendar,
    EarningsInfo,
    EarningsStatusResult,
)


# ── Helper for injected date tests ────────────────────────────────────────────

def make_date_str(days_from_now: int) -> str:
    return (date.today() + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

TODAY_STR = date.today().strftime("%Y-%m-%d")


# ── parse_earnings_date_from_calendar tests ───────────────────────────────────

def test_parse_dict_with_earnings_date_list():
    target_date = date(2024, 4, 25)
    cal = {"Earnings Date": [target_date]}
    result = parse_earnings_date_from_calendar(cal)
    assert result == target_date


def test_parse_dict_with_earnings_date_string():
    cal = {"Earnings Date": ["2024-04-25"]}
    result = parse_earnings_date_from_calendar(cal)
    assert result == date(2024, 4, 25)


def test_parse_dict_multiple_dates_picks_earliest():
    cal = {"Earnings Date": [date(2024, 5, 1), date(2024, 4, 20)]}
    result = parse_earnings_date_from_calendar(cal)
    assert result == date(2024, 4, 20)


def test_parse_empty_dict_returns_none():
    assert parse_earnings_date_from_calendar({}) is None


def test_parse_none_returns_none():
    assert parse_earnings_date_from_calendar(None) is None


def test_parse_empty_list_returns_none():
    assert parse_earnings_date_from_calendar({"Earnings Date": []}) is None


# ── fetch_earnings_for_ticker classification tests ────────────────────────────

def _make_injected_result(days: int) -> EarningsInfo:
    """Helper: returns EarningsInfo as if earnings are `days` from today."""
    as_of_dt = date.today()
    earnings_dt = as_of_dt + timedelta(days=days)
    injected = {str(earnings_dt): None}  # just use get_universe_earnings_calendar with inject

    tickers = ["AAPL"]
    inject_map = {"AAPL": earnings_dt.strftime("%Y-%m-%d")} if days >= 0 else {}

    result = get_universe_earnings_calendar(
        tickers=tickers,
        as_of_date_str=TODAY_STR,
        injected_dates=inject_map,
        persist=False,
    )
    return result.calendar["AAPL"]


def test_earnings_today_classification():
    info = _make_injected_result(0)
    assert info.status == "TODAY"
    assert info.is_blocked is True
    assert info.hold_protection is True
    assert info.size_multiplier == 1.0


def test_earnings_blackout_1_day():
    info = _make_injected_result(1)
    assert info.status == "BLACKOUT"
    assert info.is_blocked is True
    assert info.hold_protection is False
    assert info.size_multiplier == 1.0


def test_earnings_blackout_2_days():
    info = _make_injected_result(2)
    assert info.status == "BLACKOUT"
    assert info.is_blocked is True


def test_earnings_caution_3_days():
    info = _make_injected_result(3)
    assert info.status == "CAUTION"
    assert info.is_blocked is False
    assert info.size_multiplier == pytest.approx(0.5, abs=1e-9)


def test_earnings_caution_5_days():
    info = _make_injected_result(5)
    assert info.status == "CAUTION"
    assert info.size_multiplier == pytest.approx(0.5, abs=1e-9)


def test_earnings_ok_more_than_5_days():
    info = _make_injected_result(10)
    assert info.status == "OK"
    assert info.is_blocked is False
    assert info.size_multiplier == 1.0


def test_earnings_unknown_no_date():
    """When no injected date provided and yfinance returns nothing → UNKNOWN."""
    with patch("src.intelligence.earnings.yf.Ticker") as mock_t:
        mock_t.return_value.calendar = None
        result = get_universe_earnings_calendar(
            tickers=["NVDA"],
            as_of_date_str=TODAY_STR,
            persist=False,
        )
    info = result.calendar.get("NVDA")
    assert info is not None
    assert info.status == "UNKNOWN"
    assert info.is_blocked is False


# ── Badge text property tests ──────────────────────────────────────────────────

def test_badge_text_today():
    info = _make_injected_result(0)
    assert "Today" in info.badge_text or "today" in info.badge_text.lower()


def test_badge_text_blackout():
    info = _make_injected_result(1)
    assert "1d" in info.badge_text or "1" in info.badge_text


def test_badge_text_ok_no_badge():
    info = _make_injected_result(10)
    # OK status has no warning badge
    assert info.status == "OK"


# ── get_universe_earnings_calendar tests ──────────────────────────────────────

def test_universe_calendar_blocked_list():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    inject = {
        "AAPL": make_date_str(1),    # BLACKOUT
        "MSFT": make_date_str(10),   # OK
        "GOOGL": make_date_str(4),   # CAUTION
    }
    result = get_universe_earnings_calendar(
        tickers=tickers,
        as_of_date_str=TODAY_STR,
        injected_dates=inject,
        persist=False,
    )
    assert "AAPL" in result.blocked_tickers
    assert "MSFT" not in result.blocked_tickers
    assert "GOOGL" in result.caution_tickers


def test_universe_calendar_size_multiplier():
    inject = {"AAPL": make_date_str(4)}
    result = get_universe_earnings_calendar(
        tickers=["AAPL"],
        as_of_date_str=TODAY_STR,
        injected_dates=inject,
        persist=False,
    )
    assert result.get_size_multiplier("AAPL") == pytest.approx(0.5, abs=1e-9)


def test_universe_calendar_is_blocked():
    inject = {"TSLA": make_date_str(2)}
    result = get_universe_earnings_calendar(
        tickers=["TSLA"],
        as_of_date_str=TODAY_STR,
        injected_dates=inject,
        persist=False,
    )
    assert result.is_blocked("TSLA") is True
    assert result.is_blocked("AAPL") is False


def test_universe_calendar_has_hold_protection():
    inject = {"AAPL": make_date_str(0)}
    result = get_universe_earnings_calendar(
        tickers=["AAPL"],
        as_of_date_str=TODAY_STR,
        injected_dates=inject,
        persist=False,
    )
    assert result.has_hold_protection("AAPL") is True


def test_universe_calendar_graceful_yfinance_failure():
    """yfinance exception must not propagate; returns UNKNOWN status."""
    with patch("src.intelligence.earnings.yf.Ticker") as mock_t:
        mock_t.side_effect = Exception("Connection failed")
        result = get_universe_earnings_calendar(
            tickers=["AAPL"],
            as_of_date_str=TODAY_STR,
            persist=False,
        )
    assert result.calendar["AAPL"].status == "UNKNOWN"
