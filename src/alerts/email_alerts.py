"""
src/alerts/email_alerts.py — Automated Email Notifications via Gmail SMTP (Section 7 Item 1).

Delivers real-time email notifications for:
  1. Trade executed (BUY or SELL with fill price, shares, PnL, confidence, portfolio value)
  2. Stop-loss triggered (ticker, entry, exit, loss %, loss $, days held, portfolio value)
  3. Circuit breaker activated (SPY drop %, lookback window, pause days, portfolio value, cash %)
  4. Pipeline failure (error message, date, time, last known portfolio value)
  5. PSI drift alarm (red alert, PSI value, explanation, buy pause notice)

SAFETY GUARANTEES:
  - Never crashes or blocks pipeline execution. All operations wrapped in try/except.
  - If credentials missing in .env: logs warning and skips silently.
  - Failed email attempts are queued and retried once after 60 seconds.
  - Plain text + HTML with mandatory 'Paper Trading Only — No Real Money' footer.
"""

from __future__ import annotations

import logging
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config.settings import settings
from src.alerts.dedup import is_alert_already_sent, mark_alert_sent

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
FOOTER_TEXT = (
    "\n\n--------------------------------------------------\n"
    "Timestamp: {timestamp}\n"
    "Paper Trading Only — No Real Money\n"
    "--------------------------------------------------"
)
FOOTER_HTML = (
    '<hr style="border:0;border-top:1px solid #e0e0e0;margin:20px 0;">'
    '<p style="color:#757575;font-size:12px;margin:4px 0;">'
    '<strong>Timestamp:</strong> {timestamp}<br>'
    '<em>Paper Trading Only — No Real Money</em>'
    '</p>'
)

# Queue holding failed emails for retry: [{subject, body_text, body_html, retry_at, attempts}]
_retry_queue: List[Dict[str, Any]] = []
_retry_lock = threading.Lock()
RETRY_DELAY_SEC: float = 60.0


def _get_timestamp_str() -> str:
    """Returns human-readable current Eastern Time timestamp."""
    return datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _is_email_configured() -> bool:
    """Checks if email alerts are enabled and required Gmail SMTP credentials are present in settings."""
    if not getattr(settings, "email_alerts_enabled", False):
        return False
    sender = getattr(settings, "alert_email_sender", "") or ""
    password = getattr(settings, "alert_email_password", "") or ""
    recipient = getattr(settings, "alert_email_recipient", "") or ""
    return bool(sender.strip() and password.strip() and recipient.strip())


def _send_smtp(
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    """
    Low-level SMTP sender using Gmail STARTTLS.
    Returns True if sent successfully, raises Exception on failure.
    """
    sender = settings.alert_email_sender.strip()
    password = settings.alert_email_password.strip()
    recipient = settings.alert_email_recipient.strip()
    host = getattr(settings, "alert_email_smtp_host", "smtp.gmail.com")
    port = int(getattr(settings, "alert_email_smtp_port", 587))

    ts_str = _get_timestamp_str()
    full_text = body_text.strip() + FOOTER_TEXT.format(timestamp=ts_str)

    if body_html:
        full_html = (
            f"<!DOCTYPE html><html><body style='font-family: Arial, sans-serif; line-height: 1.5; color: #212121;'>"
            f"{body_html}"
            f"{FOOTER_HTML.format(timestamp=ts_str)}"
            f"</body></html>"
        )
    else:
        escaped_body = full_text.replace("\n", "<br>")
        full_html = (
            f"<!DOCTYPE html><html><body style='font-family: Arial, sans-serif; line-height: 1.5; color: #212121;'>"
            f"<p>{escaped_body}</p>"
            f"</body></html>"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(full_text, "plain", "utf-8"))
    msg.attach(MIMEText(full_html, "html", "utf-8"))

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.send_message(msg)

    logger.info("Email alert sent successfully: '%s' to %s", subject, recipient)
    return True


def _retry_worker(item: Dict[str, Any], delay: float) -> None:
    """Background worker that waits and retries a failed email once."""
    if delay > 0:
        time.sleep(delay)
    try:
        _send_smtp(item["subject"], item["body_text"], item["body_html"])
        logger.info("Retried email alert sent successfully: '%s'", item["subject"])
    except Exception as exc:
        logger.error("Retry email alert failed for '%s': %s", item["subject"], exc)
    finally:
        with _retry_lock:
            if item in _retry_queue:
                _retry_queue.remove(item)


def send_alert(
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    retry_delay: Optional[float] = None,
    alert_key: Optional[str] = None,
) -> bool:
    """
    Sends an email alert notification via Gmail SMTP.

    Guarantees:
      - Never crashes the caller. Wrapped in try/except always.
      - If credentials missing or email_alerts_enabled is False: logs warning/info and skips silently.
      - Failed sends are queued and retried once after 60 seconds (in a daemon thread).

    Args:
        subject: Email subject line.
        body_text: Plain text body content.
        body_html: Optional formatted HTML body content.
        retry_delay: Delay in seconds before background retry (defaults to RETRY_DELAY_SEC).
        alert_key: Optional unique alert deduplication key.

    Returns:
        True if sent immediately, False if skipped or queued for retry.
    """
    try:
        if alert_key and is_alert_already_sent(alert_key):
            logger.info("DUPLICATE_ALERT_SKIPPED: %s", alert_key)
            return False

        if not _is_email_configured():
            logger.warning(
                "Email alert skipped (email_alerts_enabled is False or credentials not set): '%s'",
                subject,
            )
            return False

        _send_smtp(subject, body_text, body_html)
        if alert_key:
            mark_alert_sent(alert_key)
            logger.info("ALERT_SENT: %s", alert_key)
        return True

    except Exception as exc:
        logger.warning("Failed to send email alert '%s': %s. Queuing for 60s retry.", subject, exc)
        item = {
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html,
            "attempts": 1,
            "timestamp": time.time(),
        }
        with _retry_lock:
            _retry_queue.append(item)

        delay = RETRY_DELAY_SEC if retry_delay is None else retry_delay
        t = threading.Thread(target=_retry_worker, args=(item, delay), daemon=True)
        t.start()
        return False


# ── Specific Alert Trigger Builders ──────────────────────────────────────────

def send_trade_executed_alert(
    action: str,
    ticker: str,
    price: float,
    shares: float,
    pnl: Optional[float] = None,
    model_confidence: Optional[float] = None,
    exit_reason: Optional[str] = None,
    portfolio_value: Optional[float] = None,
) -> bool:
    """
    Trigger 1: Trade executed (buy or sell).
    Subject: '📈 BOUGHT AAPL @ $182.50' or '📉 SOLD AAPL @ $195.00 (+$84.32)'
    """
    if not _is_email_configured():
        return False

    act_upper = action.upper()
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"{act_upper}_{ticker.upper()}_{date_str}"
    if act_upper == "BUY":
        subject = f"📈 BOUGHT {ticker.upper()} @ ${price:.2f}"
    else:
        pnl_str = f" ({pnl:+.2f})" if pnl is not None else ""
        subject = f"📉 SOLD {ticker.upper()} @ ${price:.2f}{pnl_str}"

    text_lines = [
        f"Trade Execution Notification:",
        f"• Action: {act_upper}",
        f"• Ticker: {ticker.upper()}",
        f"• Shares: {shares:.4f}",
        f"• Fill Price: ${price:.2f}",
    ]
    if act_upper == "SELL" and pnl is not None:
        text_lines.append(f"• Realized P&L: ${pnl:+.2f}")
    if model_confidence is not None:
        text_lines.append(f"• Model Confidence Score: {model_confidence:.4f}")
    if act_upper == "SELL" and exit_reason:
        text_lines.append(f"• Exit Reason: {exit_reason}")
    if portfolio_value is not None:
        text_lines.append(f"• Portfolio Value After Trade: ${portfolio_value:.2f}")

    body_text = "\n".join(text_lines)

    pnl_html = f"<li><strong>Realized P&L:</strong> ${pnl:+.2f}</li>" if (act_upper == "SELL" and pnl is not None) else ""
    conf_html = f"<li><strong>Model Confidence Score:</strong> {model_confidence:.4f}</li>" if model_confidence is not None else ""
    exit_html = f"<li><strong>Exit Reason:</strong> {exit_reason}</li>" if (act_upper == "SELL" and exit_reason) else ""
    port_html = f"<li><strong>Portfolio Value After Trade:</strong> ${portfolio_value:.2f}</li>" if portfolio_value is not None else ""

    body_html = f"""
    <h3>Trade Execution Notification</h3>
    <ul>
      <li><strong>Action:</strong> {act_upper}</li>
      <li><strong>Ticker:</strong> {ticker.upper()}</li>
      <li><strong>Shares:</strong> {shares:.4f}</li>
      <li><strong>Fill Price:</strong> ${price:.2f}</li>
      {pnl_html}
      {conf_html}
      {exit_html}
      {port_html}
    </ul>
    """
    return send_alert(subject, body_text, body_html, alert_key=alert_key)


def send_stop_loss_alert(
    ticker: str,
    entry_price: float,
    exit_price: float,
    loss_amount: float,
    loss_pct: Optional[float] = None,
    days_held: Optional[int] = None,
    portfolio_value: Optional[float] = None,
) -> bool:
    """
    Trigger 2: Stop-loss triggered.
    Subject: '🔴 STOP-LOSS: MSFT sold at -7.8%'
    """
    if not _is_email_configured():
        return False

    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"STOP_LOSS_{ticker.upper()}_{date_str}"
    pct = loss_pct if loss_pct is not None else (
        ((exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
    )
    subject = f"🔴 STOP-LOSS: {ticker.upper()} sold at {pct:+.1f}%"

    text_lines = [
        f"Stop-Loss Execution Alert:",
        f"• Ticker: {ticker.upper()}",
        f"• Entry Price: ${entry_price:.2f}",
        f"• Exit Price: ${exit_price:.2f}",
        f"• Loss Amount: ${abs(loss_amount):.2f} ({pct:+.2f}%)",
    ]
    if days_held is not None:
        text_lines.append(f"• Days Held: {days_held} trading days")
    if portfolio_value is not None:
        text_lines.append(f"• Current Portfolio Value: ${portfolio_value:.2f}")

    body_text = "\n".join(text_lines)

    days_html = f"<li><strong>Days Held:</strong> {days_held} trading days</li>" if days_held is not None else ""
    port_html = f"<li><strong>Current Portfolio Value:</strong> ${portfolio_value:.2f}</li>" if portfolio_value is not None else ""

    body_html = f"""
    <h3 style="color:#d32f2f;">Stop-Loss Execution Alert</h3>
    <ul>
      <li><strong>Ticker:</strong> {ticker.upper()}</li>
      <li><strong>Entry Price:</strong> ${entry_price:.2f}</li>
      <li><strong>Exit Price:</strong> ${exit_price:.2f}</li>
      <li><strong>Loss Amount:</strong> -${abs(loss_amount):.2f} ({pct:+.2f}%)</li>
      {days_html}
      {port_html}
    </ul>
    """
    return send_alert(subject, body_text, body_html, alert_key=alert_key)


def send_circuit_breaker_alert(
    spy_drop_pct: float,
    lookback_window: int,
    pause_days: int,
    portfolio_value: Optional[float] = None,
    cash_pct: Optional[float] = None,
) -> bool:
    """
    Trigger 3: Circuit breaker activated.
    Subject: '⚠️ CIRCUIT BREAKER ACTIVE — Buys paused'
    """
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"CIRCUIT_BREAKER_{date_str}"
    subject = "⚠️ CIRCUIT BREAKER ACTIVE — Buys paused"

    text_lines = [
        f"Market Circuit Breaker Activated:",
        f"• SPY Drop: {spy_drop_pct:.2f}%",
        f"• Lookback Window: {lookback_window} trading days",
        f"• Buys Paused: {pause_days} trading days",
    ]
    if portfolio_value is not None:
        text_lines.append(f"• Current Portfolio Value: ${portfolio_value:.2f}")
    if cash_pct is not None:
        text_lines.append(f"• Cash Reserve: {cash_pct:.1f}%")

    body_text = "\n".join(text_lines)

    port_html = f"<li><strong>Current Portfolio Value:</strong> ${portfolio_value:.2f}</li>" if portfolio_value is not None else ""
    cash_html = f"<li><strong>Cash Reserve:</strong> {cash_pct:.1f}%</li>" if cash_pct is not None else ""

    body_html = f"""
    <h3 style="color:#e65100;">⚠️ Market Circuit Breaker Activated</h3>
    <ul>
      <li><strong>SPY Drop:</strong> {spy_drop_pct:.2f}%</li>
      <li><strong>Lookback Window:</strong> {lookback_window} trading days</li>
      <li><strong>Buys Paused For:</strong> {pause_days} trading days</li>
      {port_html}
      {cash_html}
    </ul>
    <p>All new BUY orders are strictly vetoed until market volatility stabilizes.</p>
    """
    return send_alert(subject, body_text, body_html, alert_key=alert_key)


def send_pipeline_failure_alert(
    error_message: str,
    run_date: Optional[str] = None,
    timestamp: Optional[str] = None,
    last_portfolio_value: Optional[float] = None,
) -> bool:
    """
    Trigger 4: Pipeline failure.
    Subject: '🚨 PIPELINE FAILED — Action needed'
    """
    subject = "🚨 PIPELINE FAILED — Action needed"
    date_str = run_date or datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"PIPELINE_FAILURE_{date_str}"
    ts_str = timestamp or _get_timestamp_str()

    text_lines = [
        f"Autonomous Daily Pipeline Failure:",
        f"• Error: {error_message}",
        f"• Date: {date_str}",
        f"• Time: {ts_str}",
    ]
    if last_portfolio_value is not None:
        text_lines.append(f"• Last Known Portfolio Value: ${last_portfolio_value:.2f}")

    body_text = "\n".join(text_lines)

    port_html = f"<li><strong>Last Known Portfolio Value:</strong> ${last_portfolio_value:.2f}</li>" if last_portfolio_value is not None else ""

    body_html = f"""
    <h3 style="color:#b71c1c;">🚨 Daily Pipeline Execution Failed</h3>
    <ul>
      <li><strong>Error:</strong> <code>{error_message}</code></li>
      <li><strong>Date:</strong> {date_str}</li>
      <li><strong>Time:</strong> {ts_str}</li>
      {port_html}
    </ul>
    <p>Please inspect system event logs in the operator dashboard for full error stack trace.</p>
    """
    return send_alert(subject, body_text, body_html, alert_key=alert_key)


def send_psi_drift_alert(
    psi_value: float,
    details: Optional[str] = None,
) -> bool:
    """
    Trigger 5: PSI drift alarm (red level only: PSI >= 0.25).
    Subject: '🔴 MODEL DRIFT ALERT — PSI={value}'
    """
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"PSI_DRIFT_{date_str}"
    subject = f"🔴 MODEL DRIFT ALERT — PSI={psi_value:.4f}"

    text_lines = [
        f"Population Stability Index (PSI) Drift Alert:",
        f"• PSI Value: {psi_value:.4f} (Threshold: >= 0.25 RED LEVEL)",
        f"• Diagnosis: Significant shift detected between live inference and baseline training distributions.",
        f"• Action: New candidate buys are paused until human operator reviews calibration and model stability.",
    ]
    if details:
        text_lines.append(f"• Additional Details: {details}")

    body_text = "\n".join(text_lines)

    details_html = f"<p><strong>Additional Details:</strong> {details}</p>" if details else ""

    body_html = f"""
    <h3 style="color:#b71c1c;">🔴 Model Prediction Drift Alarm (Red Alert)</h3>
    <ul>
      <li><strong>PSI Value:</strong> <code>{psi_value:.4f}</code> (Red Level: &ge; 0.25)</li>
      <li><strong>Meaning:</strong> Model predictions have statistically diverged from baseline distribution.</li>
      <li><strong>Safety Rule:</strong> New purchases are vetoed pending operator review.</li>
    </ul>
    {details_html}
    """
    return send_alert(subject, body_text, body_html, alert_key=alert_key)


def send_failover_alert(
    ticker: str,
    primary_source: str = "yfinance",
    backup_source: str = "Alpha Vantage",
) -> bool:
    """Send failover alert when primary data source fails and backup kicks in."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"FAILOVER_{ticker.upper()}_{date_str}"
    subject = f"⚠️ DATA SOURCE FAILOVER: {ticker.upper()}"
    body_text = (
        "⚠️ DATA SOURCE FAILOVER\n"
        f"Primary ({primary_source}) failed for: {ticker.upper()}\n"
        f"Switched to: {backup_source} backup"
    )
    body_html = (
        f"<h2 style='color:#e65100;'>⚠️ DATA SOURCE FAILOVER</h2>"
        f"<p>Primary (<strong>{primary_source}</strong>) failed for: <strong>{ticker.upper()}</strong></p>"
        f"<p>Switched to: <strong>{backup_source} backup</strong></p>"
    )
    return send_alert(subject, body_text, body_html, alert_key=alert_key)

