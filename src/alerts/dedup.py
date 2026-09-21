"""
src/alerts/dedup.py — Alert Deduplication Helper Functions.

Tracks and checks sent alert keys in event_log table to prevent duplicate notifications.
"""

from __future__ import annotations

import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)


def is_alert_already_sent(alert_key: str) -> bool:
    """
    Check if an alert with the given alert_key has already been logged in event_log.
    If settings.suppress_duplicate_alerts is False, always returns False.
    """
    if not getattr(settings, "suppress_duplicate_alerts", True):
        return False

    try:
        from src.db.repository import EventLog, Session, get_engine, select
        target_msg = f"ALERT_SENT: {alert_key}"
        with Session(get_engine()) as session:
            row = session.execute(
                select(EventLog).where(EventLog.message == target_msg)
            ).first()
            return row is not None
    except Exception as exc:
        logger.warning("Error checking alert deduplication for key %s: %s", alert_key, exc)
        return False


def mark_alert_sent(alert_key: str) -> None:
    """
    Log an ALERT_SENT event in event_log table for deduplication tracking.
    """
    try:
        from src.db.repository import log_event
        log_event(
            level="INFO",
            component="alerts",
            message=f"ALERT_SENT: {alert_key}",
            details={"alert_key": alert_key},
        )
    except Exception as exc:
        logger.warning("Error logging mark_alert_sent for key %s: %s", alert_key, exc)
