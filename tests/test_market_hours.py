"""
tests/test_market_hours.py — Test Timezone-Aware Market Hours Check (Section 1).
"""

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.pipeline.scheduler import execute_scheduled_job, is_after_market_close


def test_is_after_market_close_before_cutoff():
    """Any time prior to 16:30 America/New_York must return False."""
    tz = ZoneInfo("America/New_York")
    # 2:15 PM NY time
    test_dt = datetime(2024, 6, 12, 14, 15, 0, tzinfo=tz)
    is_closed, reason = is_after_market_close(test_dt)
    assert is_closed is False
    assert "before" in reason

    # 4:29:59 PM NY time
    test_dt_edge = datetime(2024, 6, 12, 16, 29, 59, tzinfo=tz)
    is_closed_edge, _ = is_after_market_close(test_dt_edge)
    assert is_closed_edge is False


def test_is_after_market_close_at_and_after_cutoff():
    """At 16:30:00 or later America/New_York must return True."""
    tz = ZoneInfo("America/New_York")
    # Exactly 4:30 PM NY time
    test_dt_exact = datetime(2024, 6, 12, 16, 30, 0, tzinfo=tz)
    is_closed, reason = is_after_market_close(test_dt_exact)
    assert is_closed is True
    assert "after" in reason

    # 5:00 PM NY time
    test_dt_after = datetime(2024, 6, 12, 17, 0, 0, tzinfo=tz)
    is_closed_after, _ = is_after_market_close(test_dt_after)
    assert is_closed_after is True


def test_daylight_saving_transitions():
    """
    Verify zoneinfo dynamically switches between EDT (UTC-4) and EST (UTC-5).
    A fixed UTC-5/EST offset would be wrong in July (EDT) by exactly 1 hour.
    """
    # Summer (EDT is UTC-4): 20:30 UTC is 16:30 EDT (market closed + buffer)
    summer_utc = datetime(2024, 7, 15, 20, 30, 0, tzinfo=timezone.utc)
    is_closed_summer, _ = is_after_market_close(summer_utc)
    assert is_closed_summer is True

    # In summer at 20:29 UTC, it is 16:29 EDT (before cutoff)
    summer_utc_early = datetime(2024, 7, 15, 20, 29, 0, tzinfo=timezone.utc)
    is_closed_summer_early, _ = is_after_market_close(summer_utc_early)
    assert is_closed_summer_early is False

    # Winter (EST is UTC-5): 21:30 UTC is 16:30 EST (market closed + buffer)
    winter_utc = datetime(2024, 1, 15, 21, 30, 0, tzinfo=timezone.utc)
    is_closed_winter, _ = is_after_market_close(winter_utc)
    assert is_closed_winter is True

    # In winter at 21:29 UTC, it is 16:29 EST (before cutoff)
    winter_utc_early = datetime(2024, 1, 15, 21, 29, 0, tzinfo=timezone.utc)
    is_closed_winter_early, _ = is_after_market_close(winter_utc_early)
    assert is_closed_winter_early is False


def test_scheduler_skips_when_market_open(monkeypatch, tmp_path):
    """Scheduler must skip the pipeline run if triggered before 16:30 and force=False."""
    # Ensure safe isolated test db
    test_db = f"sqlite:///{tmp_path}/test_sched.db"
    monkeypatch.setattr(settings, "db_url", test_db)

    with patch("src.pipeline.scheduler.is_market_day", return_value=(True, "Trading day")), \
         patch("src.pipeline.scheduler.is_after_market_close", return_value=(False, "Market is still open")):
        
        # Target date is today (or future)
        today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        result = execute_scheduled_job(run_date=today_str, force=False)
        assert result is None


def test_scheduler_force_bypasses_market_hours(monkeypatch, tmp_path):
    """When force=True, scheduler ignores market hours check."""
    test_db = f"sqlite:///{tmp_path}/test_sched.db"
    monkeypatch.setattr(settings, "db_url", test_db)

    with patch("src.pipeline.scheduler.run_daily_pipeline") as mock_pipeline:
        mock_pipeline.return_value = None
        # With force=True, it should proceed to run_daily_pipeline even if market hours is false
        execute_scheduled_job(run_date="2024-06-12", force=True)
        assert mock_pipeline.called
