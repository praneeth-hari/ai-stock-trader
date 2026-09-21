"""
src/alerts — Autonomous email and Telegram alerting (Section 7 Items 1 & 2).
"""

from src.alerts.dedup import (
    is_alert_already_sent,
    mark_alert_sent,
)
from src.alerts.email_alerts import (
    send_alert,
    send_circuit_breaker_alert,
    send_failover_alert,
    send_pipeline_failure_alert,
    send_psi_drift_alert,
    send_stop_loss_alert,
    send_trade_executed_alert,
)
from src.alerts.telegram_alerts import (
    notify,
    notify_circuit_breaker,
    notify_daily_summary,
    notify_failover,
    notify_pipeline_failure,
    notify_psi_drift,
    notify_stop_loss,
    notify_trade_bought,
    notify_trade_sold,
)

__all__ = [
    # Deduplication
    "is_alert_already_sent",
    "mark_alert_sent",
    # Email
    "send_alert",
    "send_trade_executed_alert",
    "send_stop_loss_alert",
    "send_circuit_breaker_alert",
    "send_pipeline_failure_alert",
    "send_psi_drift_alert",
    "send_failover_alert",
    # Telegram
    "notify",
    "notify_trade_bought",
    "notify_trade_sold",
    "notify_stop_loss",
    "notify_circuit_breaker",
    "notify_pipeline_failure",
    "notify_psi_drift",
    "notify_daily_summary",
    "notify_failover",
]
