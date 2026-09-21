"""
tests/test_telegram_alerts.py — Unit tests for Section 7 Item 2: Telegram Notifications.

Covers:
  - Missing credentials (bot token or chat id) → graceful skip (returns False, no exception)
  - Successful notification send (mocked asyncio.run / telegram.Bot)
  - Failure handling (async error caught, returns False, never crashes)
  - All 7 trigger notification helpers: trade bought, trade sold, stop-loss,
    circuit breaker, pipeline failure, PSI drift, daily summary
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.alerts.telegram_alerts as ta


@pytest.fixture(autouse=True)
def disable_dedup_for_telegram_tests():
    with patch("src.alerts.dedup.settings.suppress_duplicate_alerts", False):
        yield


def _mock_settings(token: str = "123456:ABC-DEF1234ghIkl-zyx57W-1v234v", chat_id: str = "987654321", suppress_duplicate_alerts: bool = False):
    """Return a mock settings object with Telegram credentials."""
    mock = MagicMock()
    mock.telegram_bot_token = token
    mock.telegram_chat_id = chat_id
    mock.suppress_duplicate_alerts = suppress_duplicate_alerts
    return mock


# ── 1. Missing credentials tests ──────────────────────────────────────────────

class TestMissingCredentials:

    def test_empty_bot_token_returns_false(self):
        with patch("src.alerts.telegram_alerts.settings", _mock_settings(token="")):
            assert ta.notify("Test message") is False

    def test_empty_chat_id_returns_false(self):
        with patch("src.alerts.telegram_alerts.settings", _mock_settings(chat_id="")):
            assert ta.notify("Test message") is False

    def test_whitespace_credentials_returns_false(self):
        with patch("src.alerts.telegram_alerts.settings", _mock_settings(token="  ", chat_id="  ")):
            assert ta.notify("Test message") is False

    def test_is_telegram_configured_helper(self):
        with patch("src.alerts.telegram_alerts.settings", _mock_settings("token", "123")):
            assert ta._is_telegram_configured() is True
        with patch("src.alerts.telegram_alerts.settings", _mock_settings("", "123")):
            assert ta._is_telegram_configured() is False


# ── 2. Successful Send ────────────────────────────────────────────────────────

class TestSuccessfulSend:

    def test_notify_calls_asyncio_run(self):
        with patch("src.alerts.telegram_alerts.settings", _mock_settings("bot_token_123", "chat_id_456")), \
             patch("asyncio.run") as mock_run:
            res = ta.notify("Hello Telegram")
            assert res is True
            mock_run.assert_called_once()

    def test_async_send_coroutine_called(self):
        captured = []

        def fake_run(coro):
            captured.append(coro)
            # Close coroutine to avoid unawaited coroutine warning
            coro.close()
            return None

        with patch("src.alerts.telegram_alerts.settings", _mock_settings("mytoken", "mychat")), \
             patch("asyncio.run", side_effect=fake_run):
            res = ta.notify("Test async call")
            assert res is True
            assert len(captured) == 1


# ── 3. Failure Handling Safety ────────────────────────────────────────────────

class TestFailureSafety:

    def test_notify_handles_asyncio_run_exception(self):
        def _raise_timeout(coro):
            coro.close()
            raise RuntimeError("Network timeout")

        with patch("src.alerts.telegram_alerts.settings", _mock_settings("token", "chat")), \
             patch("asyncio.run", side_effect=_raise_timeout):
            res = ta.notify("Message")
            assert res is False

    def test_helpers_never_raise_on_error(self):
        def _raise_error(coro):
            coro.close()
            raise Exception("Fatal telegram error")

        with patch("src.alerts.telegram_alerts.settings", _mock_settings("token", "chat")), \
             patch("asyncio.run", side_effect=_raise_error):
            assert ta.notify_trade_bought("AAPL", 150.0, 10.0) is False
            assert ta.notify_trade_sold("MSFT", 200.0, 5.0, pnl=50.0) is False
            assert ta.notify_stop_loss("TSLA", 100.0, 5.0) is False
            assert ta.notify_circuit_breaker(7.5, 5, 3) is False
            assert ta.notify_pipeline_failure("Crash") is False
            assert ta.notify_psi_drift(0.30) is False
            assert ta.notify_daily_summary("2024-01-01", 10000.0) is False


# ── 4. Trigger Message Builders ───────────────────────────────────────────────

class TestTriggerBuilders:

    @pytest.fixture
    def mock_notify(self):
        with patch("src.alerts.telegram_alerts.notify") as m:
            m.return_value = True
            yield m

    def test_notify_trade_bought(self, mock_notify):
        res = ta.notify_trade_bought("aapl", 182.50, 6.73, model_confidence=0.71, portfolio_value=10543.21)
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "✅ BOUGHT A STOCK" in msg
        assert "📌 Stock: AAPL" in msg
        assert "💰 Bought at: $182.50" in msg
        assert "📦 Shares: 6.7300" in msg
        assert "🎯 AI Confidence: 71% (High)" in msg
        assert "💼 Portfolio now: $10,543.21" in msg

    def test_notify_trade_sold_profit(self, mock_notify):
        res = ta.notify_trade_sold("meta", 665.75, 3.4, pnl=84.32, pnl_pct=5.25, exit_reason="Take-Profit", portfolio_value=12000.0)
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "📤 SOLD A STOCK" in msg
        assert "📌 Stock: META" in msg
        assert "💰 Exit Price: $665.75" in msg
        assert "📈 Profit/Loss: $+84.32 (+5.25%)" in msg
        assert "🚪 Reason: Take-Profit" in msg
        assert "💼 Portfolio now: $12,000.00" in msg

    def test_notify_trade_sold_loss(self, mock_notify):
        res = ta.notify_trade_sold("nvda", 100.00, 2.0, pnl=-50.0, pnl_pct=-20.0, exit_reason="Stop-Loss")
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "📉 Profit/Loss: $-50.00 (-20.00%)" in msg

    def test_notify_stop_loss(self, mock_notify):
        res = ta.notify_stop_loss("msft", 150.0, 7.8, days_held=4, portfolio_value=9800.0)
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "🔴 SOLD TO PREVENT BIGGER LOSS" in msg
        assert "📌 Stock: MSFT" in msg
        assert "📉 We lost: -$150.00 (7.8%)" in msg
        assert "📅 Days Held: 4" in msg
        assert "💼 Portfolio now: $9,800.00" in msg

    def test_notify_circuit_breaker(self, mock_notify):
        res = ta.notify_circuit_breaker(7.5, 5, 3, cash_pct=100.0)
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "⚠️ MARKET CRASH PROTECTION ON" in msg
        assert "📉 The overall market dropped 7.5% in the last 5 days" in msg
        assert "🛑 We stopped buying new stocks" in msg
        assert "💼 Cash Reserve: 100.0%" in msg

    def test_notify_pipeline_failure(self, mock_notify):
        res = ta.notify_pipeline_failure("Data connection lost", run_date="2024-03-15")
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "🚨 PIPELINE FAILED" in msg
        assert "📅 Date: 2024-03-15" in msg
        assert "❌ Error: Data connection lost" in msg

    def test_notify_psi_drift(self, mock_notify):
        res = ta.notify_psi_drift(0.32)
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "⚠️ AI MODEL NEEDS UPDATING" in msg
        assert "📊 Model Health (PSI): 0.32" in msg
        assert "🛑 Stopped buying new stocks for now" in msg

    def test_notify_daily_summary(self, mock_notify):
        res = ta.notify_daily_summary(
            run_date="2024-03-15",
            portfolio_value=10543.21,
            daily_pnl=120.50,
            daily_pnl_pct=1.15,
            cash=3000.0,
            cash_pct=28.5,
            positions=["AAPL", "META"],
            max_positions=5,
            status="SUCCESS",
        )
        assert res is True
        msg = mock_notify.call_args[0][0]
        assert "📊 DAILY SUMMARY" in msg
        assert "📅 Date: 2024-03-15" in msg
        assert "📈 Today's Change: $+120.50 (+1.15%)" in msg
        assert "🏦 Cash Reserve: $3,000.00 (28.5%)" in msg
        assert "📌 Open Positions: AAPL, META (2/5)" in msg
        assert "✅ System Status: SUCCESS" in msg
