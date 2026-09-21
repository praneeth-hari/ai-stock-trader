"""
tests/test_alert_deduplication.py — Tests for Telegram & Email alert deduplication.
"""

from unittest.mock import patch
import pytest

from config.settings import settings
from src.alerts import (
    is_alert_already_sent,
    mark_alert_sent,
    notify,
    notify_daily_summary,
    notify_trade_bought,
    notify_trade_sold,
    notify_circuit_breaker,
    send_alert,
    send_trade_executed_alert,
    send_stop_loss_alert,
)
from src.db.repository import EventLog, Session, create_all_tables, get_engine, select
from sqlalchemy import delete


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database schema is initialized and clear alert_sent logs before each test."""
    create_all_tables()
    try:
        with Session(get_engine()) as session:
            session.execute(delete(EventLog).where(EventLog.message.like("ALERT_SENT:%")))
            session.commit()
    except Exception:
        pass
    yield
    try:
        with Session(get_engine()) as session:
            session.execute(delete(EventLog).where(EventLog.message.like("ALERT_SENT:%")))
            session.commit()
    except Exception:
        pass


def test_dedup_helper_functions():
    key = "TEST_KEY_2026-09-19"
    assert is_alert_already_sent(key) is False

    mark_alert_sent(key)
    assert is_alert_already_sent(key) is True


def test_suppress_duplicate_alerts_disabled():
    key = "TEST_DISABLED_KEY_2026-09-19"
    mark_alert_sent(key)

    with patch.object(settings, "suppress_duplicate_alerts", False):
        assert is_alert_already_sent(key) is False


def test_telegram_alert_deduplication():
    key = "TEST_TELEGRAM_KEY_2026-09-19"

    with patch("src.alerts.telegram_alerts._is_telegram_configured", return_value=True), \
         patch("src.alerts.telegram_alerts.asyncio.run") as mock_async_run:

        # First call succeeds
        res1 = notify("Test Telegram Message", alert_key=key)
        assert res1 is True
        assert mock_async_run.call_count == 1

        # Second call with same key is skipped silently
        res2 = notify("Test Telegram Message", alert_key=key)
        assert res2 is False
        assert mock_async_run.call_count == 1  # Not called again


def test_telegram_trigger_deduplication():
    run_date = "2026-09-19"

    with patch("src.alerts.telegram_alerts._is_telegram_configured", return_value=True), \
         patch("src.alerts.telegram_alerts.asyncio.run") as mock_async_run:

        # Daily summary first run
        res1 = notify_daily_summary(run_date=run_date, portfolio_value=10000.0)
        assert res1 is True
        assert mock_async_run.call_count == 1

        # Daily summary second run on same date
        res2 = notify_daily_summary(run_date=run_date, portfolio_value=10000.0)
        assert res2 is False
        assert mock_async_run.call_count == 1


def test_email_alert_deduplication():
    key = "TEST_EMAIL_KEY_2026-09-19"

    with patch("src.alerts.email_alerts._is_email_configured", return_value=True), \
         patch("src.alerts.email_alerts._send_smtp") as mock_send_smtp:

        # First call succeeds
        res1 = send_alert("Subject 1", "Body 1", alert_key=key)
        assert res1 is True
        assert mock_send_smtp.call_count == 1

        # Second call with same key skipped
        res2 = send_alert("Subject 2", "Body 2", alert_key=key)
        assert res2 is False
        assert mock_send_smtp.call_count == 1


def test_email_trigger_deduplication():
    with patch("src.alerts.email_alerts._is_email_configured", return_value=True), \
         patch("src.alerts.email_alerts._send_smtp") as mock_send_smtp:

        res1 = send_stop_loss_alert(ticker="AAPL", entry_price=150.0, exit_price=138.0, loss_amount=12.0)
        assert res1 is True

        res2 = send_stop_loss_alert(ticker="AAPL", entry_price=150.0, exit_price=138.0, loss_amount=12.0)
        assert res2 is False
