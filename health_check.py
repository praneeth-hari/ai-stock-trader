"""
health_check.py — Morning & On-Demand System Health Monitor.

Part 2 / Part 3 Health Monitoring:
  - Verifies whether the daily pipeline executed on schedule.
  - Inspects the latest log file in logs/ for "PIPELINE FAILED" or absence.
  - Queries database event log and portfolio snapshots for health metrics.
  - Prints clear human-readable status alerts so you know the state
    of the system before US markets open (9:00 AM Monday-Friday)
    or immediately upon system boot/startup.

Exit Codes:
  0: Healthy or informational status
  1: Critical failure or missing expected run
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Self-locating base path
PROJECT_DIR = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_DIR / "logs"

# Ensure root is in sys.path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def find_latest_daily_log() -> Optional[Path]:
    """Finds the most recently modified daily run log file in logs/."""
    if not LOGS_DIR.exists():
        return None
    logs = list(LOGS_DIR.glob("daily_run_*.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def inspect_log_file(log_path: Path) -> Tuple[bool, List[str]]:
    """
    Reads a log file and returns:
      (is_failed, list_of_error_lines)
    """
    has_failed = False
    errors: List[str] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if "PIPELINE FAILED" in line_str:
                    has_failed = True
                    errors.append(line_str)
                elif "CRITICAL" in line_str:
                    errors.append(line_str)
                elif "MISSED RUN DETECTED" in line_str:
                    errors.append(line_str)
    except Exception as exc:
        return True, [f"Could not read log file: {exc}"]

    return has_failed, errors


def get_db_health() -> Dict[str, Any]:
    """Queries repository database for latest portfolio state and recent critical events."""
    result: Dict[str, Any] = {
        "db_connected": False,
        "latest_snapshot_date": None,
        "total_equity": None,
        "cash": None,
        "critical_events": [],
    }
    try:
        from src.db import repository
        repository.create_all_tables()
        snap = repository.get_latest_portfolio_snapshot()
        result["db_connected"] = True
        if snap:
            result["latest_snapshot_date"] = snap.get("run_date")
            result["total_equity"] = float(snap.get("total_value", 0.0))
            result["cash"] = float(snap.get("cash", 0.0))
            result["total_slippage_cost"] = float(snap.get("total_slippage_cost", 0.0) or 0.0)

        crit_events = repository.get_events(level="CRITICAL", limit=5)
        result["critical_events"] = crit_events
    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_health_check() -> int:
    """
    Executes full health check inspection and prints clear status summary.
    Returns exit code (0 = success, 1 = failure).
    """
    now = datetime.now()
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    # Determine last expected run date (yesterday, or Friday if today is Monday/weekend)
    if today.weekday() == 0:  # Monday
        expected_prev_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    elif today.weekday() == 6:  # Sunday
        expected_prev_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    elif today.weekday() == 5:  # Saturday
        expected_prev_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        expected_prev_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    LOGS_DIR.mkdir(exist_ok=True)
    today_log_path = LOGS_DIR / f"daily_run_{today_str}.log"
    latest_log = find_latest_daily_log()

    db_info = get_db_health()

    print("=" * 78)
    print(" AI STOCK TRADER - SYSTEM HEALTH & CRASH RECOVERY MONITOR")
    print(f" Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')} | Target: Pre-Market 9:00 AM / Startup")
    print("=" * 78)

    status = "HEALTHY"
    alerts: List[str] = []

    # Check 1: Log file inspection
    if latest_log is None:
        status = "WARNING"
        alerts.append("No daily pipeline logs found in logs/ directory.")
        print(f"\n[!] LOG STATUS: No daily run logs found.")
    else:
        log_date = latest_log.stem.replace("daily_run_", "")
        failed, err_lines = inspect_log_file(latest_log)
        print(f"\n[*] LATEST LOG FILE: {latest_log.name} ({log_date})")

        if failed:
            status = "CRITICAL"
            alerts.append(f"Latest run ({log_date}) failed with 'PIPELINE FAILED'!")
            print(f"    Status: FAILED")
            for err in err_lines[:5]:
                print(f"    - {err}")
        else:
            print(f"    Status: SUCCESS / CLEAN EXECUTION")

        if log_date < expected_prev_date and today.weekday() < 5:
            if status != "CRITICAL":
                status = "WARNING"
            alerts.append(f"Last pipeline run was {log_date}, but expected {expected_prev_date}. Computer may have been off.")
            print(f"    Notice: Expected run from {expected_prev_date} not found. (Last was {log_date})")

    # Check 2: Database health
    print("\n[*] DATABASE STATE:")
    if db_info["db_connected"]:
        eq_str = f"${db_info['total_equity']:,.2f}" if db_info["total_equity"] is not None else "N/A"
        cash_str = f"${db_info['cash']:,.2f}" if db_info["cash"] is not None else "N/A"
        slip_str = f"${db_info.get('total_slippage_cost', 0.0):,.4f}"
        snap_date = db_info["latest_snapshot_date"] or "None"
        print(f"    Connected: YES (SQLAlchemy / SQLite-PostgreSQL)")
        print(f"    Latest Snapshot Date: {snap_date}")
        print(f"    Portfolio Total Equity: {eq_str}")
        print(f"    Cash Cushion: {cash_str}")
        print(f"    Cumulative Slippage: {slip_str}")

        if db_info["critical_events"]:
            recent_crit = db_info["critical_events"][0]
            event_ts = recent_crit.get("timestamp", "")
            msg = recent_crit.get("message", "")
            print(f"    [!] Recent Critical Event ({event_ts}): {msg}")
    else:
        status = "CRITICAL"
        err_msg = db_info.get("error", "Unknown database error")
        alerts.append(f"Database connection failed: {err_msg}")
        print(f"    Connected: NO ({err_msg})")

    # Final Evaluation & Verdict
    print("\n" + "=" * 78)
    if status == "HEALTHY":
        print(" [OK] HEALTH CHECK PASSED: System is operating normally.")
        print("      Last scheduled cycle ran cleanly. Portfolio state is sound.")
        exit_code = 0
    elif status == "WARNING":
        print(" [!] HEALTH CHECK WARNING: Non-critical anomalies detected:")
        for a in alerts:
            print(f"     * {a}")
        print("     Action: Verify scheduled task execution in Windows Task Scheduler.")
        exit_code = 0  # Warning does not block startup
    else:
        print(" [X] HEALTH CHECK CRITICAL: Action required before trading opens:")
        for a in alerts:
            print(f"     * {a}")
        print(f"     Action: Review details in {latest_log} and run:")
        print("             cmd /c run_trader.bat")
        exit_code = 1

    print("=" * 78)

    # Save to health check log
    hc_log = LOGS_DIR / f"health_check_{today_str}.log"
    with open(hc_log, "a", encoding="utf-8") as f:
        f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status} | ExitCode: {exit_code} | Alerts: {alerts}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_health_check())
