"""
tests/test_weekly_summary.py — Unit tests for Section 7 Item 3: Weekly Summary Report.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.reports.weekly_summary import (
    _get_week_date_range,
    format_weekly_summary_message,
    generate_weekly_summary_data,
    send_weekly_summary,
)


def test_get_week_date_range_sunday():
    # Sunday Sep 20, 2026 -> Trading week Mon Sep 14 to Fri Sep 18
    sun = date(2026, 9, 20)
    mon, fri = _get_week_date_range(sun)
    assert mon == date(2026, 9, 14)
    assert fri == date(2026, 9, 18)


def test_get_week_date_range_weekday():
    # Wednesday Sep 16, 2026 -> Mon Sep 14 to Fri Sep 18
    wed = date(2026, 9, 16)
    mon, fri = _get_week_date_range(wed)
    assert mon == date(2026, 9, 14)
    assert fri == date(2026, 9, 18)


def test_generate_weekly_summary_data_empty_db():
    with patch("src.db.repository.get_portfolio_snapshots", return_value=[]), \
         patch("src.db.repository.get_latest_portfolio_snapshot", return_value=None):
        data = generate_weekly_summary_data(date(2026, 9, 20))
        assert data is None


def test_generate_weekly_summary_data_with_snapshots():
    snaps = [
        {"run_date": "2026-09-14", "total_value": 9950.0, "cash": 5000.0, "positions": {"AAPL": {"shares": 6.73, "unrealized_pnl_pct": 2.41}}},
        {"run_date": "2026-09-18", "total_value": 10247.83, "cash": 5464.48, "positions": {"AAPL": {"shares": 6.73, "unrealized_pnl_pct": 2.41}}},
    ]

    mock_trades = pd.DataFrame([
        {"Date Sold": "2026-09-16", "Ticker": "AAPL", "Profit / Loss ($)": "+$142.50", "Profit / Loss (%)": "+4.8%", "action": "SELL"},
        {"Date Sold": "2026-09-17", "Ticker": "MSFT", "Profit / Loss ($)": "-$65.20", "Profit / Loss (%)": "-2.1%", "action": "SELL"},
    ])

    with patch("src.db.repository.get_portfolio_snapshots", return_value=snaps), \
         patch("dashboard.data_loader.get_recent_trades_df", return_value=mock_trades), \
         patch("src.db.repository.get_events", return_value=[]):
        
        data = generate_weekly_summary_data(date(2026, 9, 20))
        assert data is not None
        assert data["starting_value"] == 9950.0
        assert data["ending_value"] == 10247.83
        assert data["total_trades"] == 2
        assert data["wins"] == 1
        assert data["losses"] == 1
        assert "AAPL" in data["best_trade_str"]
        assert "MSFT" in data["worst_trade_str"]


def test_format_weekly_summary_message():
    data = {
        "week_label": "15 Sep — 19 Sep 2026",
        "starting_value": 9950.0,
        "ending_value": 10247.83,
        "weekly_gain_dollar": 297.83,
        "weekly_gain_pct": 2.99,
        "spy_comp_str": "+1.2 pts ahead ✅",
        "total_trades": 3,
        "wins": 2,
        "losses": 1,
        "win_rate": 66.7,
        "best_trade_str": "AAPL +$142.50 (+4.8%)",
        "worst_trade_str": "MSFT -$65.20 (-2.1%)",
        "holdings_formatted": "- AAPL: 6.73 shares (+2.41%) 🟢\n- META: 3.40 shares (-0.05%) 🔴",
        "cash": 5464.48,
        "cash_pct": 54.7,
        "circuit_breaker_alert_str": "No circuit breaker events ✅",
        "ai_model_alert_str": "AI model healthy ✅",
        "data_feed_alert_str": "All data feeds normal ✅",
        "earnings_watch_str": "MSFT earnings in 3 days ⚠️",
        "regime_str": "Risk-ON ✅",
    }

    msg = format_weekly_summary_message(data)
    assert "📊 WEEKLY SUMMARY" in msg
    assert "💼 PORTFOLIO" in msg
    assert "$10,247.83" in msg
    assert "📈 THIS WEEK'S TRADES" in msg
    assert "AAPL +$142.50 (+4.8%)" in msg
    assert "🎓 Remember: Paper trading only!" in msg


def test_send_weekly_summary_not_configured():
    with patch("src.reports.weekly_summary._is_telegram_configured", return_value=False):
        res = send_weekly_summary()
        assert res is False


def test_send_weekly_summary_success():
    data = {
        "friday_date": "2026-09-18",
        "week_label": "15 Sep — 19 Sep 2026",
        "starting_value": 9950.0,
        "ending_value": 10247.83,
        "weekly_gain_dollar": 297.83,
        "weekly_gain_pct": 2.99,
        "spy_comp_str": "+1.2 pts ahead ✅",
        "total_trades": 3,
        "wins": 2,
        "losses": 1,
        "win_rate": 66.7,
        "best_trade_str": "AAPL +$142.50 (+4.8%)",
        "worst_trade_str": "MSFT -$65.20 (-2.1%)",
        "holdings_formatted": "- AAPL: 6.73 shares (+2.41%) 🟢",
        "cash": 5464.48,
        "cash_pct": 54.7,
        "circuit_breaker_alert_str": "No circuit breaker events ✅",
        "ai_model_alert_str": "AI model healthy ✅",
        "data_feed_alert_str": "All data feeds normal ✅",
        "earnings_watch_str": "No earnings blackouts next week ✅",
        "regime_str": "Risk-ON ✅",
    }

    with patch("src.reports.weekly_summary._is_telegram_configured", return_value=True), \
         patch("src.reports.weekly_summary.generate_weekly_summary_data", return_value=data), \
         patch("src.reports.weekly_summary.notify", return_value=True) as mock_notify:
        
        res = send_weekly_summary(date(2026, 9, 20))
        assert res is True
        assert mock_notify.call_count == 1
