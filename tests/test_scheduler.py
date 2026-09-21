"""
tests/test_scheduler.py — Phase 13 Automation Scheduler Tests.

Verifies:
  1. Scheduler setup: CronTrigger configured for Mon-Fri 09:00 AM America/New_York with 3600s grace.
  2. Market day / weekend / holiday detection skips non-trading days cleanly.
  3. Successful scheduled execution triggers run_daily_pipeline and records audit event in DB.
  4. Extended Loud Failure Resiliency Test:
     - Day 1: Simulated pipeline crash is caught, logs CRITICAL to DB and logger, returns None.
     - Day 2: Next scheduled cycle executes successfully — confirming one bad day does NOT
       leave the scheduler daemon in a broken or stuck state.
  5. CLI interface --once flag runs single cycle cleanly.
"""

from __future__ import annotations

import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.db import repository
from src.pipeline.daily_pipeline import DailyPipelineResult
from src.pipeline.scheduler import (
    create_scheduler,
    execute_scheduled_job,
    is_market_day,
    is_nyse_holiday,
)


def _make_mock_result(run_date: str) -> DailyPipelineResult:
    return DailyPipelineResult(
        run_date=run_date,
        tickers_fetched=1,
        tickers_valid=1,
        tickers_skipped=0,
        regime="RISK-ON",
        orders_generated=0,
        fills=[],
        pending_orders=[],
        cash=50.0,
        total_equity=50.0,
        errors=[],
    )


@pytest.fixture()
def clean_db(tmp_path):
    """Provides a fresh isolated SQLite database for scheduler tests."""
    db_file = tmp_path / "test_sched.db"
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


# ── Test 1: Scheduler Configuration ───────────────────────────────────────────

def test_1_scheduler_job_configuration():
    """Verifies CronTrigger targets Mon-Fri 09:00 America/New_York with 1hr misfire grace."""
    scheduler = create_scheduler()
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1, "Expected exactly 1 scheduled job"

    job = jobs[0]
    assert job.id == "daily_paper_trading_pipeline"
    assert job.misfire_grace_time == 3600
    assert isinstance(job.trigger, CronTrigger)

    # Check fields of CronTrigger
    fields = {str(f.name): str(f) for f in job.trigger.fields}
    assert fields["day_of_week"] == "mon-fri"
    assert fields["hour"] == str(settings.schedule_hour)
    assert fields["minute"] == str(settings.schedule_minute)
    assert str(job.trigger.timezone) == settings.schedule_timezone


# ── Test 2: Market Day & Holiday Filtering ────────────────────────────────────

def test_2_market_day_and_holiday_filtering(clean_db):
    """Weekend and recognized NYSE holidays should be skipped cleanly."""
    # Weekend
    sat = date(2024, 1, 6)
    is_open, reason = is_market_day(sat)
    assert not is_open
    assert "Saturday" in reason

    sun = date(2024, 1, 7)
    is_open, reason = is_market_day(sun)
    assert not is_open
    assert "Sunday" in reason

    # NYSE Holiday: New Year's Day 2024-01-01
    nyd = date(2024, 1, 1)
    assert is_nyse_holiday(nyd) is True
    is_open, reason = is_market_day(nyd)
    assert not is_open
    assert "holiday" in reason.lower()

    # Normal trading Wednesday 2024-01-03
    wed = date(2024, 1, 3)
    is_open, _ = is_market_day(wed)
    assert is_open is True

    # When scheduled job is called on weekend without force, it skips execution
    res = execute_scheduled_job(run_date="2024-01-06", force=False)
    assert res is None

    events = repository.get_events(component="scheduler")
    assert len(events) >= 1
    assert "SKIPPED_MARKET_CLOSED" in str(events[0]["details"])


# ── Test 3: Successful Scheduled Execution ────────────────────────────────────

def test_3_successful_scheduled_execution(clean_db):
    """Successful pipeline run should record INFO event and return result."""
    mock_result = _make_mock_result("2024-01-03")

    with patch("src.pipeline.scheduler.run_daily_pipeline", return_value=mock_result) as mock_run:
        res = execute_scheduled_job(run_date="2024-01-03", force=True)
        assert res is not None
        assert res.run_date == "2024-01-03"
        mock_run.assert_called_once_with(run_date="2024-01-03")

    events = repository.get_events(component="scheduler")
    success_events = [e for e in events if e.get("details", {}).get("status") == "SUCCESS"]
    assert len(success_events) == 1
    assert "completed" in success_events[0]["message"]


# ── Test 4: Extended Loud Failure Resiliency Test ─────────────────────────────

def test_4_loud_failure_handling_and_multi_day_resiliency(clean_db):
    """
    1. Day 1: Simulated pipeline crash is caught, logged as CRITICAL to DB and logger,
       and does NOT raise an unhandled exception.
    2. Day 2: Next scheduled execution runs normally and succeeds — proving the daemon
       is not stuck or corrupted after a failure.
    """
    # ── Day 1: Crash Simulation ────────────────────────────────────────────────
    day1_date = "2024-01-03"
    with patch("src.pipeline.scheduler.run_daily_pipeline", side_effect=RuntimeError("Data feed corrupted")):
        # Should not raise exception
        res_day1 = execute_scheduled_job(run_date=day1_date, force=True)
        assert res_day1 is None

    # Assert loud failure recorded in SQLite
    events_crit = repository.get_events(level="CRITICAL", component="scheduler")
    assert len(events_crit) == 1, "Expected exactly 1 CRITICAL event logged"
    crit_evt = events_crit[0]
    assert "Data feed corrupted" in crit_evt["message"]
    assert crit_evt["details"]["status"] == "FAILED"
    assert "RuntimeError" in crit_evt["details"]["traceback"]

    # ── Day 2: Recovery on Next Day ────────────────────────────────────────────
    day2_date = "2024-01-04"
    mock_day2_result = _make_mock_result(day2_date)

    with patch("src.pipeline.scheduler.run_daily_pipeline", return_value=mock_day2_result) as mock_run_day2:
        res_day2 = execute_scheduled_job(run_date=day2_date, force=True)
        assert res_day2 is not None
        assert res_day2.run_date == day2_date
        mock_run_day2.assert_called_once_with(run_date=day2_date)

    # Verify second day recorded a SUCCESS event
    events_day2 = repository.get_events(level="INFO", component="scheduler")
    success_day2 = [e for e in events_day2 if e.get("details", {}).get("run_date") == day2_date]
    assert len(success_day2) == 1
    assert success_day2[0]["details"]["status"] == "SUCCESS"


# ── Test 5: CLI Entrypoint (--once) ───────────────────────────────────────────

def test_5_cli_once_argument(clean_db):
    """Verifies main() handles --once and --date flags correctly."""
    mock_result = _make_mock_result("2024-01-05")

    with patch("src.pipeline.scheduler.execute_scheduled_job", return_value=mock_result) as mock_exec, \
         patch("sys.argv", ["scheduler.py", "--once", "--date", "2024-01-05", "--force"]), \
         pytest.raises(SystemExit) as exc_info:
        from src.pipeline.scheduler import main
        main()

    assert exc_info.value.code == 0
    mock_exec.assert_called_once_with(run_date="2024-01-05", force=True)
