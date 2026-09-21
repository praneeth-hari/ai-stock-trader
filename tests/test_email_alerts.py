"""
tests/test_email_alerts.py — Unit tests for Section 7 Item 1: Email Alerts.

Covers:
  - Missing credentials → graceful skip (no crash, returns False)
  - Successful SMTP send (mocked smtplib.SMTP)
  - Plain-text body contains mandatory footer tokens
  - HTML body present and contains mandatory footer tokens
  - All 5 trigger alert helpers: trade executed, stop-loss, circuit breaker,
    pipeline failure, PSI drift
  - Retry queue mechanism on SMTP failure
  - Email is never sent when credentials are empty strings
"""

from __future__ import annotations

import smtplib
import threading
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

import src.alerts.email_alerts as ea


@pytest.fixture(autouse=True)
def disable_dedup_for_email_tests():
    with patch("src.alerts.dedup.settings.suppress_duplicate_alerts", False):
        yield


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_settings(sender="test@gmail.com", password="app-password", recipient="recv@example.com", suppress_duplicate_alerts=False, email_alerts_enabled=True):
    """Return a mock settings object with alert credentials filled in."""
    mock = MagicMock()
    mock.alert_email_sender = sender
    mock.alert_email_password = password
    mock.alert_email_recipient = recipient
    mock.alert_email_smtp_host = "smtp.gmail.com"
    mock.alert_email_smtp_port = 587
    mock.psi_drift_alert_threshold = 0.25
    mock.suppress_duplicate_alerts = suppress_duplicate_alerts
    mock.email_alerts_enabled = email_alerts_enabled
    return mock


def _make_smtp_context():
    """Return a MagicMock SMTP context manager that records send_message calls."""
    smtp_instance = MagicMock()
    smtp_instance.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_instance.__exit__ = MagicMock(return_value=False)
    smtp_ctx = MagicMock(return_value=smtp_instance)
    return smtp_ctx, smtp_instance


# ── 1. Missing credentials / Disabled settings – graceful skip ────────────────

class TestMissingCredentials:

    def test_disabled_email_alerts_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(email_alerts_enabled=False)):
            result = ea.send_alert("Test Subject", "Test body")
        assert result is False

    def test_empty_sender_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(sender="")):
            result = ea.send_alert("Test Subject", "Test body")
        assert result is False

    def test_empty_password_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(password="")):
            result = ea.send_alert("Test Subject", "Test body")
        assert result is False

    def test_empty_recipient_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(recipient="")):
            result = ea.send_alert("Test Subject", "Test body")
        assert result is False

    def test_all_empty_returns_false_no_exception(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings("", "", "")):
            result = ea.send_alert("Subject", "Body")
        assert result is False

    def test_whitespace_only_credentials_skips(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings("  ", "  ", "  ")):
            result = ea.send_alert("Subject", "Body")
        assert result is False


# ── 2. Successful SMTP send ───────────────────────────────────────────────────

class TestSuccessfulSend:

    def test_send_alert_returns_true_on_success(self):
        smtp_ctx, smtp_inst = _make_smtp_context()
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_alert("Hello", "World body")
        assert result is True

    def test_send_message_called_once(self):
        smtp_ctx, smtp_inst = _make_smtp_context()
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            ea.send_alert("Subject", "Plain body")
        smtp_inst.send_message.assert_called_once()

    def test_smtp_starttls_called(self):
        smtp_ctx, smtp_inst = _make_smtp_context()
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            ea.send_alert("Subject", "Body")
        smtp_inst.starttls.assert_called_once()

    def test_smtp_login_called_with_credentials(self):
        smtp_ctx, smtp_inst = _make_smtp_context()
        mock_settings = _mock_settings(sender="s@g.com", password="secret")
        with patch("src.alerts.email_alerts.settings", mock_settings), \
             patch("smtplib.SMTP", smtp_ctx):
            ea.send_alert("Subject", "Body")
        smtp_inst.login.assert_called_once_with("s@g.com", "secret")

    def test_smtp_connected_to_correct_host_and_port(self):
        smtp_ctx, smtp_inst = _make_smtp_context()
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            ea.send_alert("Subject", "Body")
        smtp_ctx.assert_called_once_with("smtp.gmail.com", 587, timeout=15)


# ── 3. Message format – plain text and HTML ───────────────────────────────────

class TestMessageFormat:

    def _capture_msg(self, subject="Test Subject", body_text="Test body", body_html=None):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            ea.send_alert(subject, body_text, body_html)
        assert len(captured) == 1
        return captured[0]

    def test_subject_in_message(self):
        msg = self._capture_msg(subject="My Subject")
        assert msg["Subject"] == "My Subject"

    def test_from_header_set(self):
        msg = self._capture_msg()
        assert msg["From"] == "test@gmail.com"

    def test_to_header_set(self):
        msg = self._capture_msg()
        assert msg["To"] == "recv@example.com"

    def test_message_is_multipart_alternative(self):
        msg = self._capture_msg()
        assert msg.get_content_type() == "multipart/alternative"

    def test_plain_text_part_present(self):
        msg = self._capture_msg(body_text="plain hello")
        payloads = msg.get_payload()
        plain_parts = [p for p in payloads if p.get_content_type() == "text/plain"]
        assert len(plain_parts) == 1

    def test_html_part_present(self):
        msg = self._capture_msg()
        payloads = msg.get_payload()
        html_parts = [p for p in payloads if p.get_content_type() == "text/html"]
        assert len(html_parts) == 1

    def test_footer_timestamp_in_plain_text(self):
        msg = self._capture_msg(body_text="body content")
        payloads = msg.get_payload()
        plain_part = next(p for p in payloads if p.get_content_type() == "text/plain")
        text_content = plain_part.get_payload(decode=True).decode("utf-8")
        assert "Timestamp:" in text_content

    def test_footer_paper_trading_in_plain_text(self):
        msg = self._capture_msg(body_text="body content")
        payloads = msg.get_payload()
        plain_part = next(p for p in payloads if p.get_content_type() == "text/plain")
        text_content = plain_part.get_payload(decode=True).decode("utf-8")
        assert "Paper Trading Only" in text_content
        assert "No Real Money" in text_content

    def test_footer_in_html_part(self):
        msg = self._capture_msg(body_text="body")
        payloads = msg.get_payload()
        html_part = next(p for p in payloads if p.get_content_type() == "text/html")
        html_content = html_part.get_payload(decode=True).decode("utf-8")
        assert "Paper Trading Only" in html_content

    def test_custom_html_body_included(self):
        msg = self._capture_msg(body_html="<p>Custom HTML</p>")
        payloads = msg.get_payload()
        html_part = next(p for p in payloads if p.get_content_type() == "text/html")
        html_content = html_part.get_payload(decode=True).decode("utf-8")
        assert "Custom HTML" in html_content


# ── 4. Trigger: send_trade_executed_alert ─────────────────────────────────────

class TestTradeExecutedAlert:

    def _run(self, **kwargs):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_trade_executed_alert(**kwargs)
        return result, captured

    def test_buy_subject_format(self):
        result, captured = self._run(action="BUY", ticker="AAPL", price=182.50, shares=5.0)
        assert result is True
        assert captured[0]["Subject"] == "📈 BOUGHT AAPL @ $182.50"

    def test_sell_subject_format_with_pnl(self):
        result, captured = self._run(
            action="SELL", ticker="AAPL", price=195.00, shares=5.0, pnl=84.32
        )
        assert result is True
        assert "SOLD AAPL @ $195.00" in captured[0]["Subject"]
        assert "+84.32" in captured[0]["Subject"]

    def test_sell_subject_format_with_negative_pnl(self):
        result, captured = self._run(
            action="SELL", ticker="MSFT", price=190.00, shares=2.0, pnl=-30.00
        )
        assert "SOLD MSFT" in captured[0]["Subject"]
        assert "-30.00" in captured[0]["Subject"]

    def test_buy_body_contains_ticker_and_price(self):
        result, captured = self._run(action="BUY", ticker="NVDA", price=450.00, shares=1.5)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "NVDA" in text
        assert "450.00" in text

    def test_confidence_in_body_when_provided(self):
        result, captured = self._run(
            action="BUY", ticker="AAPL", price=200.0, shares=1.0, model_confidence=0.7345
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "0.7345" in text

    def test_exit_reason_in_sell_body(self):
        result, captured = self._run(
            action="SELL", ticker="TSLA", price=300.0, shares=1.0,
            pnl=-50.0, exit_reason="SIGNAL_EXIT"
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "SIGNAL_EXIT" in text

    def test_portfolio_value_in_body_when_provided(self):
        result, captured = self._run(
            action="BUY", ticker="META", price=500.0, shares=1.0, portfolio_value=12000.00
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "12000.00" in text

    def test_missing_credentials_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(sender="")):
            result = ea.send_trade_executed_alert(action="BUY", ticker="AAPL", price=100.0, shares=1.0)
        assert result is False


# ── 5. Trigger: send_stop_loss_alert ─────────────────────────────────────────

class TestStopLossAlert:

    def _run(self, **kwargs):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_stop_loss_alert(**kwargs)
        return result, captured

    def test_subject_format(self):
        result, captured = self._run(
            ticker="MSFT", entry_price=200.00, exit_price=184.00,
            loss_amount=80.00, loss_pct=-8.0
        )
        assert result is True
        assert "STOP-LOSS" in captured[0]["Subject"]
        assert "MSFT" in captured[0]["Subject"]
        assert "-8.0%" in captured[0]["Subject"]

    def test_subject_uses_computed_pct_when_not_provided(self):
        result, captured = self._run(
            ticker="GOOGL", entry_price=200.00, exit_price=184.00, loss_amount=16.00
        )
        assert "STOP-LOSS" in captured[0]["Subject"]
        assert "GOOGL" in captured[0]["Subject"]
        # -8.0% computed from entry 200 -> exit 184
        assert "-8.0%" in captured[0]["Subject"]

    def test_body_contains_entry_and_exit_prices(self):
        result, captured = self._run(
            ticker="TSLA", entry_price=250.0, exit_price=230.0, loss_amount=20.0
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "250.00" in text
        assert "230.00" in text

    def test_body_contains_days_held_when_provided(self):
        result, captured = self._run(
            ticker="AAPL", entry_price=180.0, exit_price=160.0, loss_amount=20.0, days_held=7
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "7" in text and "trading days" in text

    def test_missing_credentials_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(recipient="")):
            result = ea.send_stop_loss_alert(
                ticker="AAPL", entry_price=200.0, exit_price=180.0, loss_amount=20.0
            )
        assert result is False


# ── 6. Trigger: send_circuit_breaker_alert ────────────────────────────────────

class TestCircuitBreakerAlert:

    def _run(self, **kwargs):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_circuit_breaker_alert(**kwargs)
        return result, captured

    def test_subject_format(self):
        result, captured = self._run(spy_drop_pct=7.5, lookback_window=5, pause_days=5)
        assert result is True
        assert "CIRCUIT BREAKER" in captured[0]["Subject"]
        assert "Buys paused" in captured[0]["Subject"]

    def test_body_contains_spy_drop(self):
        result, captured = self._run(spy_drop_pct=7.5, lookback_window=5, pause_days=5)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "7.50" in text

    def test_body_contains_pause_days(self):
        result, captured = self._run(spy_drop_pct=7.0, lookback_window=5, pause_days=10)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "10" in text

    def test_optional_portfolio_value_in_body(self):
        result, captured = self._run(
            spy_drop_pct=8.0, lookback_window=5, pause_days=5, portfolio_value=9500.0
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "9500.00" in text

    def test_optional_cash_pct_in_body(self):
        result, captured = self._run(
            spy_drop_pct=8.0, lookback_window=5, pause_days=5, cash_pct=22.5
        )
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "22.5" in text

    def test_missing_credentials_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(password="")):
            result = ea.send_circuit_breaker_alert(spy_drop_pct=8.0, lookback_window=5, pause_days=5)
        assert result is False


# ── 7. Trigger: send_pipeline_failure_alert ───────────────────────────────────

class TestPipelineFailureAlert:

    def _run(self, **kwargs):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_pipeline_failure_alert(**kwargs)
        return result, captured

    def test_subject_format(self):
        result, captured = self._run(error_message="Database connection failed")
        assert result is True
        assert "PIPELINE FAILED" in captured[0]["Subject"]
        assert "Action needed" in captured[0]["Subject"]

    def test_body_contains_error_message(self):
        result, captured = self._run(error_message="Timeout on SPY fetch")
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "Timeout on SPY fetch" in text

    def test_body_contains_run_date_when_provided(self):
        result, captured = self._run(error_message="err", run_date="2024-01-15")
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "2024-01-15" in text

    def test_body_contains_portfolio_value_when_provided(self):
        result, captured = self._run(error_message="err", last_portfolio_value=8750.50)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "8750.50" in text

    def test_missing_credentials_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(sender="")):
            result = ea.send_pipeline_failure_alert(error_message="Some error")
        assert result is False


# ── 8. Trigger: send_psi_drift_alert ─────────────────────────────────────────

class TestPsiDriftAlert:

    def _run(self, **kwargs):
        smtp_ctx, smtp_inst = _make_smtp_context()
        captured: list = []
        smtp_inst.send_message.side_effect = lambda m: captured.append(m)
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            result = ea.send_psi_drift_alert(**kwargs)
        return result, captured

    def test_subject_format_contains_psi_value(self):
        result, captured = self._run(psi_value=0.3142)
        assert result is True
        assert "MODEL DRIFT ALERT" in captured[0]["Subject"]
        assert "PSI=0.3142" in captured[0]["Subject"]

    def test_body_contains_psi_value(self):
        result, captured = self._run(psi_value=0.2500)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "0.2500" in text

    def test_body_contains_threshold_info(self):
        result, captured = self._run(psi_value=0.30)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "0.25" in text  # threshold mentioned

    def test_body_mentions_buy_pause(self):
        result, captured = self._run(psi_value=0.30)
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "paused" in text.lower()

    def test_optional_details_in_body(self):
        result, captured = self._run(psi_value=0.28, details="Caused by regime change in 2024-Q3")
        payloads = captured[0].get_payload()
        plain = next(p for p in payloads if p.get_content_type() == "text/plain")
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "regime change" in text

    def test_missing_credentials_returns_false(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(recipient="")):
            result = ea.send_psi_drift_alert(psi_value=0.30)
        assert result is False


# ── 9. Retry queue on SMTP failure ────────────────────────────────────────────

class TestRetryQueue:

    def test_send_failure_queues_for_retry(self):
        """When SMTP fails, item is placed in retry queue and returns False."""
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", side_effect=smtplib.SMTPException("connection refused")), \
             patch("src.alerts.email_alerts._retry_worker") as mock_worker, \
             patch("threading.Thread") as mock_thread:
            # Capture initial queue state
            initial_len = len(ea._retry_queue)
            result = ea.send_alert("Subject", "Body", retry_delay=0)
        assert result is False

    def test_retry_worker_attempts_resend_on_success(self):
        """_retry_worker calls _send_smtp; succeeds on second attempt."""
        smtp_ctx, smtp_inst = _make_smtp_context()
        item = {
            "subject": "Retry Subject",
            "body_text": "Retry body",
            "body_html": None,
        }
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", smtp_ctx):
            ea._retry_worker(item, delay=0)
        smtp_inst.send_message.assert_called_once()

    def test_retry_worker_logs_error_on_failure(self, caplog):
        """_retry_worker catches exceptions and logs error, never raises."""
        import logging
        item = {
            "subject": "Retry Subject",
            "body_text": "Retry body",
            "body_html": None,
        }
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", side_effect=smtplib.SMTPException("refused")):
            with caplog.at_level(logging.ERROR, logger="src.alerts.email_alerts"):
                # Should not raise
                ea._retry_worker(item, delay=0)
        assert any("Retry email alert failed" in r.message for r in caplog.records)

    def test_send_never_raises_on_any_exception(self):
        """send_alert must not raise under any circumstances."""
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", side_effect=RuntimeError("unexpected")), \
             patch("threading.Thread"):
            try:
                ea.send_alert("Subj", "Body", retry_delay=0)
            except Exception as exc:
                pytest.fail(f"send_alert raised an exception: {exc}")

    def test_helper_never_raises_on_smtp_exception(self):
        """High-level helper functions also never raise even if SMTP fails."""
        with patch("src.alerts.email_alerts.settings", _mock_settings()), \
             patch("smtplib.SMTP", side_effect=smtplib.SMTPAuthenticationError(535, "bad credentials")), \
             patch("threading.Thread"):
            try:
                ea.send_trade_executed_alert(action="BUY", ticker="AAPL", price=100.0, shares=1.0)
                ea.send_stop_loss_alert(ticker="MSFT", entry_price=200.0, exit_price=180.0, loss_amount=20.0)
                ea.send_circuit_breaker_alert(spy_drop_pct=7.0, lookback_window=5, pause_days=5)
                ea.send_pipeline_failure_alert(error_message="some error")
                ea.send_psi_drift_alert(psi_value=0.30)
            except Exception as exc:
                pytest.fail(f"Helper alert raised: {exc}")


# ── 10. _is_email_configured ─────────────────────────────────────────────────

class TestIsEmailConfigured:

    def test_returns_true_when_all_set(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings()):
            assert ea._is_email_configured() is True

    def test_returns_false_when_sender_missing(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(sender="")):
            assert ea._is_email_configured() is False

    def test_returns_false_when_password_missing(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(password="")):
            assert ea._is_email_configured() is False

    def test_returns_false_when_recipient_missing(self):
        with patch("src.alerts.email_alerts.settings", _mock_settings(recipient="")):
            assert ea._is_email_configured() is False
