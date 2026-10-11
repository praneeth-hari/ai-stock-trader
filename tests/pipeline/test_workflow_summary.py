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

    # Mock snapshot exists
    with patch("src.db.repository.get_portfolio_snapshot", return_value={"total_value": 10000.0}):
        indicator, verif = check_database_verification("2026-10-01")
        assert "Write activity logged" in indicator
        assert "Confirmed — portfolio snapshot for 2026-10-01 confirmed present" in verif

    # Mock snapshot missing
    with patch("src.db.repository.get_portfolio_snapshot", return_value=None):
        indicator, verif = check_database_verification("2026-10-01")
        assert "Unconfirmed — portfolio snapshot for 2026-10-01 was not found" in verif


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
