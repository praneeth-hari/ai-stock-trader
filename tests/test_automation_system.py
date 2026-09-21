"""
Unit tests for the hands-off automation system:
- health_check.py
- daily_status.py
- batch runners and log management
"""

import sys
from pathlib import Path
from datetime import datetime, date
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from health_check import run_health_check
from daily_status import generate_status_report, main as run_daily_status_main


def test_health_check_execution():
    """Verify health_check runs without errors and produces exit code (0=clean, 1=warning)."""
    exit_code = run_health_check()
    assert exit_code in (0, 1)


def test_daily_status_report_generation():
    """Verify daily_status prints report and creates status log."""
    report_text, summary = generate_status_report()
    assert report_text is not None
    assert "DAILY EVENING STATUS REPORT" in report_text
    assert "PORTFOLIO VALUATION" in report_text
    assert "Total Portfolio Value" in report_text
    assert "HOLDINGS & OPEN POSITIONS" in report_text
    assert "Uninvested Cash" in report_text
    assert isinstance(summary, dict)

    # Run main to verify file writing
    run_daily_status_main()

    today_str = datetime.now().strftime("%Y-%m-%d")
    status_log = PROJECT_ROOT / "logs" / f"status_{today_str}.log"
    assert status_log.exists()
    content = status_log.read_text(encoding="utf-8")
    assert "DAILY EVENING STATUS REPORT" in content
