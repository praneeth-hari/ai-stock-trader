"""
tests/pipeline/test_workflow_summary.py — Unit tests for GitHub Actions Workflow Summary report generator.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.workflow_summary import (
    check_database_verification,
    extract_audit_block,
    extract_error_lines,
    format_workflow_summary,
    parse_audit_fields,
    sanitize_text,
)


SAMPLE_SUCCESS_LOG = """
2026-10-01 11:15:00 [INFO] scheduler: Executing scheduled pipeline cycle for 2026-10-01 (force=True)...
2026-10-01 11:15:01 [INFO] repository: Database engine created: postgresql+psycopg://postgres.abc:SecretPass123@aws-0.pooler.supabase.com:5432/postgres
2026-10-01 11:15:02 [INFO] repository: save_market_data: inserted 4 rows (skipped existing).
2026-10-01 11:15:03 [INFO] repository: save_portfolio_snapshot: saved snapshot for 2026-10-01

## Daily Pipeline Audit — 2026-10-01

- Status: **SUCCESS**

### Data Fetch & Validation
- Tickers fetched: **26**
- Passed validation: **26**
- Skipped (bad data): **0**
- Market Regime: **BULL**

### Orders & Fills
- Orders generated: **1**
- Fills executed: **1** (1 BUY, 0 SELL)
- Pending orders for next open: **0**

| Date | Action | Ticker | Shares | Fill Price | Gross | Fee | Cash Impact | Net PnL | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | BUY | AAPL | 5.0000 | $150.00 | $750.00 | $1.5000 | -$751.5000 | +0.0000 | Signal entry |

### Portfolio State (End of Day)
- Cash: **$9248.5000** (92.5% of equity)
- Total Equity: **$10000.0000**
- Realized PnL Today: **+0.0000**

PIPELINE STATUS: SUCCESS (exit code 0)
"""

SAMPLE_FAILURE_LOG = """
2026-10-01 11:15:00 [INFO] scheduler: Executing scheduled pipeline cycle for 2026-10-01 (force=True)...
2026-10-01 11:15:01 [ERROR] repository: Connection failed for postgresql+psycopg://user:P@ssw0rd!@db.host.com:5432/postgres
Traceback (most recent call last):
  File "src/pipeline/daily_pipeline.py", line 250, in run_daily_pipeline
    repository.create_all_tables()
  File "src/db/repository.py", line 180, in create_all_tables
psycopg.OperationalError: connection to server at "db.host.com" failed: Connection timed out
"""

SAMPLE_SKIPPED_LOG = """
2026-10-10 11:15:00 [INFO] scheduler: Executing scheduled pipeline cycle for 2026-10-10 (force=False)...
2026-10-10 11:15:00 [INFO] scheduler: Market Closed (US): Weekend (Saturday). Skipping daily paper pipeline cycle.

## Daily Pipeline Audit — 2026-10-10

- Status: **SKIPPED**

### Data Fetch & Validation
- Tickers fetched: **0**
- Passed validation: **0**
- Skipped (bad data): **26**
- Market Regime: **SKIPPED_MARKET_CLOSED**

### Orders & Fills
- Orders generated: **0**
- Fills executed: **0** (0 BUY, 0 SELL)
- Pending orders for next open: **0**

### Portfolio State (End of Day)
- Cash: **$0.0000** (100.0% of equity)
- Total Equity: **$0.0000**
- Realized PnL Today: **+0.0000**

### ⚠ Errors
- Market Closed (US): Weekend (Saturday). Skipping daily paper pipeline cycle.

PIPELINE STATUS: SKIPPED (exit code 0)
"""

SAMPLE_SKIPPED_NO_AUDIT_LOG = """
2026-10-10 11:15:00 [INFO] scheduler: Scheduler triggered for date: 2026-10-10
2026-10-10 11:15:00 [INFO] scheduler: Market Closed: Weekend (Saturday). Skipping paper pipeline cycle.
Cycle did not execute (market closed or failure logged).
PIPELINE STATUS: SKIPPED (exit code 0)
"""


def test_sanitize_text_masks_credentials_and_tokens(monkeypatch):
    """Verify secrets, database passwords, tokens, and sensitive env variables are redacted."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "987654321:SecretTelegramToken_XYZ1234567890123")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySecretGeminiKey12345")

    raw = (
        "Connecting to postgresql://postgres.user:VerySecretDbPass@aws-0.supabase.com:5432/postgres\n"
        "Connecting with sqlite URL: sqlite:///test.db\n"
        "Telegram alert sent via 987654321:SecretTelegramToken_XYZ1234567890123\n"
        "Gemini request with AIzaSySecretGeminiKey12345\n"
        "Alpha vantage call: https://alphavantage.co/query?apikey=MySecretAVKey&symbol=IBM"
    )

    clean = sanitize_text(raw)

    assert "VerySecretDbPass" not in clean
    assert "postgresql://postgres.user:***@aws-0.supabase.com:5432/postgres" in clean
    assert "987654321:SecretTelegramToken_XYZ1234567890123" not in clean
    assert "AIzaSySecretGeminiKey12345" not in clean
    assert "MySecretAVKey" not in clean
    assert "***REDACTED***" in clean


def test_extract_audit_block_and_parse_fields():
    """Verify audit block and structured metrics are cleanly parsed from log text."""
    audit_text, status = extract_audit_block(SAMPLE_SUCCESS_LOG)
    assert audit_text is not None
    assert "## Daily Pipeline Audit — 2026-10-01" in audit_text
    assert status == "SUCCESS"

    fields = parse_audit_fields(audit_text)
    assert fields["run_date"] == "2026-10-01"
    assert fields["status"] == "SUCCESS"
    assert fields["tickers_fetched"] == "26"
    assert fields["tickers_valid"] == "26"
    assert fields["tickers_skipped"] == "0"
    assert fields["regime"] == "BULL"
    assert fields["orders_generated"] == "1"
    assert fields["fills_executed"] == "1 (1 BUY, 0 SELL)"
    assert fields["pending_orders"] == "0"
    assert fields["cash"] == "$9248.5000 (92.5% of equity)"
    assert fields["total_equity"] == "$10000.0000"
    assert fields["realized_pnl"] == "+0.0000"


def test_format_workflow_summary_successful_run(monkeypatch):
    """Verify successful summary contains run metadata, metrics, fills table, and persistence distinction."""
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "101")
    monkeypatch.setenv("GITHUB_RUN_ID", "55555")
    monkeypatch.setenv("GITHUB_REPOSITORY", "praneeth-hari/ai-stock-trader")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")

    summary = format_workflow_summary(SAMPLE_SUCCESS_LOG, outcome="success", verify_db=False)

    assert "# ✅ Daily Trading Pipeline — Run #101" in summary
    assert "[#101](https://github.com/praneeth-hari/ai-stock-trader/actions/runs/55555)" in summary
    assert "[`abcdef1`](https://github.com/praneeth-hari/ai-stock-trader/commit/abcdef1234567890)" in summary
    assert "**Pipeline Status** | **SUCCESS**" in summary
    assert "**Market Regime** | BULL" in summary
    assert "**Fills Executed:** 1 (1 BUY, 0 SELL)" in summary
    assert "| 2026-10-01 | BUY | AAPL | 5.0000 |" in summary
    assert "Database Persistence Status" in summary
    assert "Pipeline Log Indicator" in summary


def test_format_workflow_summary_failed_run_without_audit(monkeypatch):
    """Verify failed run produces concise failure report with sanitized error lines and no invented metrics."""
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "102")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")

    summary = format_workflow_summary(SAMPLE_FAILURE_LOG, outcome="failure", verify_db=False)

    assert "# ❌ Daily Trading Pipeline — Run #102" in summary
    assert "Pipeline Execution Failure Report" in summary
    assert "The trading pipeline terminated before emitting the final Daily Pipeline Audit." in summary
    # Traceback should be included and sanitized
    assert "psycopg.OperationalError: connection to server" in summary
    assert "P@ssw0rd!" not in summary
    assert "user:***@" in summary
    # Should not invent metrics
    assert "**Tickers Fetched:**" not in summary
    assert "Database Persistence Status" in summary


def test_database_verification_distinguishes_pipeline_and_db_results():
    """Verify independent database verification checks distinction."""
    # When no run_date established
    indicator, verif = check_database_verification(None)
    assert "Not determined" in indicator
    assert "Unverified" in verif

    # When cycle is skipped
    indicator_skip, verif_skip = check_database_verification("2026-10-10", is_skipped=True)
    assert "Not applicable — market closed" in indicator_skip
    assert "Not applicable — no portfolio snapshot expected for a skipped cycle" in verif_skip

    # Mock snapshot exists
    with patch("src.db.repository.get_portfolio_snapshot", return_value={"total_value": 10000.0}):
        indicator, verif = check_database_verification("2026-10-01")
        assert "Write activity logged" in indicator
        assert "Confirmed — portfolio snapshot for 2026-10-01 confirmed present" in verif

    # Mock snapshot missing
    with patch("src.db.repository.get_portfolio_snapshot", return_value=None):
        indicator, verif = check_database_verification("2026-10-01")
        assert "Unconfirmed — portfolio snapshot for 2026-10-01 was not found" in verif


def test_format_workflow_summary_skipped_market_closed(monkeypatch):
    """Verify skipped run summary displays skip icon, separates workflow outcome, and avoids $0 / false DB claims."""
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "201")
    monkeypatch.setenv("GITHUB_SHA", "fedcba9876543210")

    summary = format_workflow_summary(SAMPLE_SKIPPED_LOG, outcome="success", verify_db=True)

    # 1. Title heading has skip icon, not red failure heading
    assert "# ⏭️ Daily Trading Pipeline — Run #201" in summary
    assert "# ❌" not in summary

    # 2. Workflow outcome is separated from trading pipeline status
    assert "| **Workflow Outcome** | ✅ Success |" in summary
    assert "| **Pipeline Status** | ⏭️ SKIPPED — US market closed |" in summary
    assert "| **Market Regime** | SKIPPED_MARKET_CLOSED |" in summary

    # 3. Tickers, orders, and fills are shown as 0 / appropriate, and bad data is N/A
    assert "- **Tickers Fetched:** 0" in summary
    assert "- **Passed Validation:** 0" in summary
    assert "- **Skipped (Bad Data):** N/A — no data validation performed" in summary
    assert "Skipped (Bad Data): 26" not in summary
    assert "- **Orders Generated:** 0" in summary
    assert "- **Fills Executed:** 0 (0 BUY, 0 SELL)" in summary

    # 4. Cash, total equity, and realized PnL are N/A, not $0.0000
    assert "- **Cash:** N/A — no trading cycle executed" in summary
    assert "- **Total Equity:** N/A — no trading cycle executed" in summary
    assert "- **Realized PnL Today:** N/A — no trading cycle executed" in summary
    assert "$0.0000" not in summary

    # 5. Database verification is 'Not applicable' and does not claim writes
    assert "- **Independent Database Verification:** Not applicable — no portfolio snapshot expected for a skipped cycle" in summary
    assert "Write activity logged" not in summary

    # 6. Does not report as an unhandled failure
    assert "Pipeline Execution Failure Report" not in summary


def test_format_workflow_summary_skipped_without_audit_block(monkeypatch):
    """Verify skipped cycle without audit markdown block still renders cleanly as skipped."""
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "202")

    summary = format_workflow_summary(SAMPLE_SKIPPED_NO_AUDIT_LOG, outcome="success", verify_db=True)

    assert "# ⏭️ Daily Trading Pipeline — Run #202" in summary
    assert "# ❌" not in summary
    assert "| **Workflow Outcome** | ✅ Success |" in summary
    assert "| **Pipeline Status** | ⏭️ SKIPPED — US market closed |" in summary
    assert "- **Tickers Fetched:** 0" in summary
    assert "- **Passed Validation:** 0" in summary
    assert "- **Skipped (Bad Data):** N/A — no data validation performed" in summary
    assert "- **Orders Generated:** 0" in summary
    assert "- **Cash:** N/A — no trading cycle executed" in summary
    assert "- **Independent Database Verification:** Not applicable — no portfolio snapshot expected for a skipped cycle" in summary
    assert "Pipeline Execution Failure Report" not in summary


def test_skipped_report_does_not_display_misleading_bad_data_count():
    """Verify that market-closed cycles never report tickers as bad data."""
    log_with_skipped_tickers = """
## Daily Pipeline Audit — 2026-10-10
- Status: **SKIPPED**
### Data Fetch & Validation
- Tickers fetched: **0**
- Passed validation: **0**
- Skipped (bad data): **26**
- Market Regime: **SKIPPED_MARKET_CLOSED**
### Orders & Fills
- Orders generated: **0**
- Fills executed: **0**
- Pending orders for next open: **0**
### Portfolio State (End of Day)
- Cash: **$0.0000**
- Total Equity: **$0.0000**
- Realized PnL Today: **+0.0000**
PIPELINE STATUS: SKIPPED (exit code 0)
"""
    summary = format_workflow_summary(log_with_skipped_tickers, outcome="success", verify_db=False)
    assert "- **Skipped (Bad Data):** N/A — no data validation performed" in summary
    assert "Skipped (Bad Data): 26" not in summary


def test_format_workflow_summary_explicit_pipeline_failure(monkeypatch):
    """Verify actual pipeline failure with status FAILED remains clearly marked as a failure with red icon."""
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "203")

    failing_audit_log = """
## Daily Pipeline Audit — 2026-10-01

- Status: **FAILED**

### Data Fetch & Validation
- Tickers fetched: **10**
- Passed validation: **0**
- Skipped (bad data): **10**
- Market Regime: **CRASH**

### ⚠ Errors
- All tickers failed validation!

PIPELINE STATUS: FAILED (exit code 1)
"""
    summary = format_workflow_summary(failing_audit_log, outcome="failure", verify_db=False)

    assert "# ❌ Daily Trading Pipeline — Run #203" in summary
    assert "| **Workflow Outcome** | ❌ Failed |" in summary
    assert "| **Pipeline Status** | **FAILED** |" in summary
    assert "⏭️ SKIPPED" not in summary


def test_main_cli_writes_to_step_summary(tmp_path, monkeypatch):
    """Verify main() CLI reads log file and appends markdown to GITHUB_STEP_SUMMARY."""
    log_file = tmp_path / "pipeline.log"
    log_file.write_text(SAMPLE_SUCCESS_LOG, encoding="utf-8")

    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setenv("PIPELINE_OUTCOME", "success")

    from src.pipeline.workflow_summary import main
    with patch("sys.argv", ["workflow_summary.py", str(log_file)]), \
         patch("src.pipeline.workflow_summary.check_database_verification", return_value=("Logged", "Confirmed")):
        main()

    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "Daily Trading Pipeline" in content
    assert "2026-10-01" in content
