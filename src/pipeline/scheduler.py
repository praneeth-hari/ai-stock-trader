"""
src/pipeline/scheduler.py — Local Daily Pipeline Automation Scheduler (Phase 13).

================================================================================
CRITICAL ARCHITECTURAL SAFETY GUARANTEE (CLAUDE.md Rule 10 & §1.9):
--------------------------------------------------------------------------------
THIS SCHEDULER EXCLUSIVELY CALLS run_daily_pipeline(), WHICH ONLY TOUCHES THE
PAPER BROKER (PaperBroker) AND SQLITE DATABASE (src/db/repository.py).
IT CONTAINS NO REAL-MONEY OR LIVE-BROKER-API CODE PATH ANYWHERE.
V1 IS STRICTLY PAPER-TRADING ONLY. RUNS 100% FREE ON YOUR LOCAL LAPTOP.
================================================================================

Schedule:
  Runs Monday through Friday at 09:00 AM US Eastern Time (America/New_York),
  exactly 30 minutes before the NYSE open (09:30 AM ET). This enforces the
  Lag Rule (§1.5): finalized T-1 close data is validated, features and model
  predictions are generated, Risk Engine rules and vetoes are applied, and
  orders are queued for execution at today's open.

Loud Failure Logging:
  Any exception during execution is logged as CRITICAL to stderr, written to
  the SQLite event log repository (repository.log_event), and captured safely
  so the scheduler daemon continues running for future cycles.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

import pytz
from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.db import repository
from src.pipeline.daily_pipeline import DailyPipelineResult, run_daily_pipeline

logger = logging.getLogger("src.pipeline.scheduler")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# ── US Market Holiday Helper ──────────────────────────────────────────────────

def is_nyse_holiday(d: date) -> bool:
    """
    Returns True if the given date is a standard NYSE market closure holiday.
    (New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day,
     Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas Day).
    """
    year = d.year
    month = d.month
    day = d.day

    # Fixed date holidays (with standard observed weekend shifts)
    fixed_holidays = [
        (1, 1),    # New Year's Day
        (6, 19),   # Juneteenth
        (7, 4),    # Independence Day
        (12, 25),  # Christmas
    ]

    for h_month, h_day in fixed_holidays:
        h_date = date(year, h_month, h_day)
        # If Saturday, observed Friday; if Sunday, observed Monday
        if h_date.weekday() == 5:
            observed = date(year, h_month, max(1, h_day - 1))
        elif h_date.weekday() == 6:
            observed = date(year, h_month, min(28, h_day + 1))
        else:
            observed = h_date
        if d == observed:
            return True

    # Floating holidays:
    # MLK Day: 3rd Monday of January
    if month == 1 and d.weekday() == 0 and 15 <= day <= 21:
        return True
    # Washington's Birthday / Presidents' Day: 3rd Monday of February
    if month == 2 and d.weekday() == 0 and 15 <= day <= 21:
        return True
    # Memorial Day: Last Monday of May
    if month == 5 and d.weekday() == 0 and day >= 25:
        return True
    # Labor Day: 1st Monday of September
    if month == 9 and d.weekday() == 0 and day <= 7:
        return True
    # Thanksgiving: 4th Thursday of November
    if month == 11 and d.weekday() == 3 and 22 <= day <= 28:
        return True

    return False


def is_market_day(target_date: Optional[date] = None) -> Tuple[bool, str]:
    """
    Determines if the given date is an active US trading day.
    Returns (is_trading_day, reason_str).
    """
    d = target_date or date.today()
    # Weekend check
    if d.weekday() == 5:
        return False, f"{d.strftime('%Y-%m-%d')} is Saturday (Weekend)"
    if d.weekday() == 6:
        return False, f"{d.strftime('%Y-%m-%d')} is Sunday (Weekend)"
    # Holiday check
    if is_nyse_holiday(d):
        return False, f"{d.strftime('%Y-%m-%d')} is a recognized NYSE market holiday"

    return True, "Active US trading day"


def get_prior_trading_day(target_date: Optional[date] = None) -> date:
    """
    Returns the most recent active US trading day strictly before target_date.
    e.g., If target_date is Monday, returns the preceding Friday (or Thursday if Friday was a holiday).
    """
    d = target_date or date.today()
    curr = d - timedelta(days=1)
    while True:
        is_open, _ = is_market_day(curr)
        if is_open:
            return curr
        curr -= timedelta(days=1)


def is_after_market_close(now_dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Checks if current time in America/New_York is at or after market close + settlement buffer (default 4:30 PM / 16:30).
    Uses zoneinfo.ZoneInfo("America/New_York") to accurately handle EST/EDT transitions without fixed offsets.

    Returns (is_after, reason_str).
    """
    tz = ZoneInfo(settings.market_hours_timezone)
    if now_dt is None:
        current_dt = datetime.now(tz)
    elif now_dt.tzinfo is None:
        current_dt = now_dt.replace(tzinfo=tz)
    else:
        current_dt = now_dt.astimezone(tz)

    cutoff_time = current_dt.replace(
        hour=settings.market_close_hour,
        minute=settings.market_close_minute,
        second=0,
        microsecond=0,
    )
    if current_dt < cutoff_time:
        return (
            False,
            f"Current time {current_dt.strftime('%H:%M:%S %Z')} is before {cutoff_time.strftime('%H:%M %Z')} (market close + settlement buffer)",
        )
    return (
        True,
        f"Current time {current_dt.strftime('%H:%M:%S %Z')} is after market close buffer ({cutoff_time.strftime('%H:%M %Z')})",
    )


# ── Scheduled Job Execution ───────────────────────────────────────────────────

def execute_scheduled_job(
    run_date: Optional[str] = None,
    force: bool = False,
) -> Optional[DailyPipelineResult]:
    """
    Executes a scheduled daily paper trading cycle with comprehensive error handling.

    Args:
        run_date: 'YYYY-MM-DD' date string. Defaults to today's date.
        force: If True, bypasses weekend/holiday checks (useful for backfill/testing).

    Returns:
        PipelineRunResult on success, or None if skipped / failed.
    """
    today_str = run_date or date.today().strftime("%Y-%m-%d")
    target_d = datetime.strptime(today_str, "%Y-%m-%d").date()

    # Ensure tables exist (idempotent)
    repository.create_all_tables()

    logger.info("Scheduler triggered for date: %s", today_str)

    # 1. Market Calendar & Market Hours Check
    if not force:
        is_open, reason = is_market_day(target_d)
        if not is_open:
            logger.info("Market Closed: %s. Skipping paper pipeline cycle.", reason)
            repository.log_event(
                level="INFO",
                component="scheduler",
                message=f"Scheduled cycle skipped: {reason}",
                details={"run_date": today_str, "status": "SKIPPED_MARKET_CLOSED"},
            )
            return None

        # Intraday hours check: ensure run is after 4:30 PM America/New_York for today's run
        tz = ZoneInfo(settings.market_hours_timezone)
        current_ny_date = datetime.now(tz).date()
        if target_d >= current_ny_date:
            is_after, hours_reason = is_after_market_close()
            if not is_after:
                logger.warning("Market Hours Warning: %s. Skipping paper pipeline cycle.", hours_reason)
                repository.log_event(
                    level="WARNING",
                    component="scheduler",
                    message=f"Scheduled cycle skipped: {hours_reason}",
                    details={"run_date": today_str, "status": "SKIPPED_MARKET_HOURS"},
                )
                return None

    # 2. Pipeline Execution with Loud Failure Logging
    try:
        start_time = datetime.now()
        result = run_daily_pipeline(run_date=today_str)
        elapsed = (datetime.now() - start_time).total_seconds()

        if result.errors:
            logger.error(
                "Scheduled pipeline finished with errors for %s in %.2fs. Errors: %s",
                today_str, elapsed, result.errors,
            )
            repository.log_event(
                level="ERROR",
                component="scheduler",
                message=f"Scheduled pipeline run failed for {today_str}: {result.errors[0]}",
                details={
                    "run_date": today_str,
                    "errors": result.errors,
                    "status": "FAILED",
                },
            )
        else:
            logger.info(
                "Scheduled pipeline completed successfully for %s in %.2fs. Fills=%d, Orders=%d, Equity=$%.2f",
                today_str,
                elapsed,
                len(result.fills),
                result.orders_generated,
                result.total_equity,
            )
            repository.log_event(
                level="INFO",
                component="scheduler",
                message=f"Scheduled pipeline run completed for {today_str} ({elapsed:.1f}s)",
                details={
                    "run_date": today_str,
                    "fills": len(result.fills),
                    "orders": result.orders_generated,
                    "total_equity": result.total_equity,
                    "status": "SUCCESS",
                },
            )
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        # Loud failure logging to logger and DB
        logger.critical(
            "CRITICAL: Scheduled daily pipeline execution failed for %s!\nError: %s\n%s",
            today_str, exc, tb,
        )
        repository.log_event(
            level="CRITICAL",
            component="scheduler",
            message=f"CRITICAL FAILURE in daily pipeline: {str(exc)}",
            details={
                "run_date": today_str,
                "error": str(exc),
                "traceback": tb,
                "status": "FAILED",
            },
        )
        # Note: Do not raise here; returning None ensures the scheduler daemon stays alive for future cycles.
        return None


# ── Scheduler Setup ───────────────────────────────────────────────────────────

def create_scheduler() -> BlockingScheduler:
    """
    Instantiates and configures APScheduler with NYSE schedule settings.
    """
    tz = pytz.timezone(settings.schedule_timezone)
    scheduler = BlockingScheduler(timezone=tz)

    trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=settings.schedule_hour,
        minute=settings.schedule_minute,
        timezone=tz,
    )

    scheduler.add_job(
        execute_scheduled_job,
        trigger=trigger,
        id="daily_paper_trading_pipeline",
        name="Daily Paper Trading Pipeline",
        misfire_grace_time=3600,  # 1 hour grace time in case laptop was temporarily asleep
        replace_existing=True,
    )

    return scheduler


# ── CLI Runner ────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint for the scheduler module."""
    parser = argparse.ArgumentParser(description="AI Stock Trader — Local Daily Automation Scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pipeline cycle immediately and exit.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for --once run in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if date is a weekend or NYSE market holiday.",
    )

    args = parser.parse_args()

    if args.once or args.force:
        target_date = args.date or date.today().strftime("%Y-%m-%d")
        print(f"Executing scheduled pipeline cycle for {target_date} (force={args.force})...")
        res = execute_scheduled_job(run_date=target_date, force=args.force)
        if res is not None:
            print("\n" + res.to_markdown())
            if res.errors and res.tickers_valid == 0 and res.regime != "SKIPPED_MARKET_CLOSED":
                print("PIPELINE FAILED: " + "; ".join(res.errors))
                sys.exit(1)
            sys.exit(0)
        else:
            print("Cycle did not execute (market closed or failure logged).")
            if args.force:
                print("PIPELINE FAILED: Execution encountered critical failure.")
                sys.exit(1)
            sys.exit(0)

    # Continuous daemon mode
    print("=" * 80)
    print("AI STOCK TRADER — LOCAL AUTOMATION SCHEDULER")
    print(f"Schedule: Mon-Fri @ {settings.schedule_hour:02d}:{settings.schedule_minute:02d} {settings.schedule_timezone}")
    print(f"Active Model: logistic_regression_baseline_v1 | DB: {settings.db_url}")
    print("Running locally on laptop. Press Ctrl+C to terminate.")
    print("=" * 80)

    scheduler = create_scheduler()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler shut down cleanly.")


if __name__ == "__main__":
    main()
