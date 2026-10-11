"""
src/pipeline/workflow_summary.py — GitHub Actions Workflow Summary Generator.

Parses captured pipeline output from daily_trading.yml, extracts the
Daily Pipeline Audit, independently checks database persistence if configured,
sanitizes secrets, and writes a rich formatted Markdown report to
$GITHUB_STEP_SUMMARY.

Can be run standalone:
    python -m src.pipeline.workflow_summary /tmp/pipeline_run.log
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def sanitize_text(text: str) -> str:
    """
    Remove secrets, passwords, tokens, and database credentials from text.
    Ensures no sensitive data is leaked into the GitHub Actions step summary.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Mask known sensitive environment variable values if set
    sensitive_keys = [
        "DATABASE_URL",
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "FRED_API_KEY",
    ]
    for key in sensitive_keys:
        val = os.environ.get(key, "").strip()
        if val and len(val) >= 5:
            sanitized = sanitized.replace(val, "***REDACTED***")

    # 2. Mask database connection strings containing credentials:
    #    e.g. postgresql://user:password@host:port/db -> postgresql://user:***@host:port/db
    sanitized = re.sub(
        r"(postgres(?:ql)?(?:\+[a-zA-Z0-9_]+)?://)([^:\s/@]+):([^@\s/]+)@",
        r"\1\2:***@",
        sanitized,
    )

    # 3. Mask Telegram bot tokens: e.g. 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
    sanitized = re.sub(
        r"\b\d{8,11}:[a-zA-Z0-9_-]{35}\b",
        "***REDACTED_TELEGRAM_TOKEN***",
        sanitized,
    )

    # 4. Mask query param / key-value secrets (apikey=..., token=..., password=...)
    sanitized = re.sub(
        r"((?:api[_-]?key|token|secret|password)=)[^\s&\"']+",
        r"\1***REDACTED***",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


def extract_audit_block(log_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the markdown Daily Pipeline Audit section and pipeline status from log text.

    Returns:
        (audit_markdown, pipeline_status)
    """
    if not log_text:
        return None, None

    status_match = re.search(r"PIPELINE STATUS:\s*([A-Za-z0-9_]+)", log_text)
    pipeline_status = status_match.group(1) if status_match else None

    header_match = re.search(r"(## Daily Pipeline Audit — [^\r\n]+)", log_text)
    if not header_match:
        return None, pipeline_status

    start_idx = header_match.start()
    rest = log_text[start_idx:]

    # The audit ends either before 'PIPELINE STATUS:' or end of text
    end_marker = re.search(r"(\r?\nPIPELINE STATUS:)", rest)
    if end_marker:
        audit_text = rest[: end_marker.start()].strip()
    else:
        audit_text = rest.strip()

    if not pipeline_status:
        inner_status = re.search(r"- Status:\s*\*\*([^\*]+)\*\*", audit_text)
        if inner_status:
            pipeline_status = inner_status.group(1)

    return audit_text, pipeline_status


def parse_audit_fields(audit_text: str) -> Dict[str, str]:
    """Parse structured fields from the markdown Daily Pipeline Audit block."""
    fields: Dict[str, str] = {}
    if not audit_text:
        return fields

    patterns = {
        "run_date": r"^## Daily Pipeline Audit — (\S+)",
        "status": r"^- Status:\s*(.+)",
        "tickers_fetched": r"^- Tickers fetched:\s*(.+)",
        "tickers_valid": r"^- Passed validation:\s*(.+)",
        "tickers_skipped": r"^- Skipped \(bad data\):\s*(.+)",
        "regime": r"^- Market Regime:\s*(.+)",
        "orders_generated": r"^- Orders generated:\s*(.+)",
        "fills_executed": r"^- Fills executed:\s*(.+)",
        "pending_orders": r"^- Pending orders for next open:\s*(.+)",
        "cash": r"^- Cash:\s*(.+)",
        "total_equity": r"^- Total Equity:\s*(.+)",
        "realized_pnl": r"^- Realized PnL Today:\s*(.+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, audit_text, re.MULTILINE)
        if m:
            raw_val = m.group(1).strip()
            clean_val = re.sub(r"\*\*([^\*]+)\*\*", r"\1", raw_val).strip()
            fields[key] = clean_val

    # Extract errors section if present
    err_section = re.search(r"### ⚠ Errors\s*\n((?:- .*\n?)+)", audit_text)
    if err_section:
        fields["errors"] = err_section.group(1).strip()

    return fields


def extract_error_lines(log_text: str, max_lines: int = 25) -> str:
    """Extract relevant error lines or traceback from failure logs."""
    if not log_text:
        return "No log output captured."

    lines = log_text.splitlines()
    error_lines: List[str] = []

    # Check for Python traceback
    tb_indices = [idx for idx, line in enumerate(lines) if "Traceback (most recent call last):" in line]
    if tb_indices:
        last_tb = tb_indices[-1]
        start_tb = max(0, last_tb - 2)
        error_lines = lines[start_tb : last_tb + max_lines]
    else:
        # Collect lines with ERROR, CRITICAL, or FAILED
        for line in lines:
            if any(term in line for term in ("CRITICAL", "ERROR", "PIPELINE FAILED", "FAILED:", "Exception:", "Error:")):
                error_lines.append(line)
        if not error_lines:
            # Fallback to the tail of the log
            error_lines = lines[-max_lines:]

    snippet = "\n".join(error_lines[-max_lines:])
    return sanitize_text(snippet)


def check_database_verification(
    run_date: Optional[str],
    is_skipped: bool = False,
) -> Tuple[str, str]:
    """
    Perform an independent, safe read-only check to verify whether today's
    portfolio snapshot actually exists in the database.

    Returns:
        (pipeline_log_indicator, independent_verification_result)
    """
    if is_skipped:
        return (
            "Not applicable — market closed (no database writes scheduled)",
            "Not applicable — no portfolio snapshot expected for a skipped cycle",
        )

    if not run_date:
        return (
            "Not determined (pipeline did not establish run date)",
            "⚠️ Unverified — run date was not established.",
        )

    try:
        from src.db import repository

        if not getattr(repository, "_db_available", True):
            return (
                "Stateless mode (database unavailable during run)",
                "⚠️ Not verified — database connection is unavailable.",
            )

        snapshot = repository.get_portfolio_snapshot(run_date)
        if snapshot:
            total_val = snapshot.get("total_value", 0.0)
            return (
                "Write activity logged by pipeline",
                f"✅ Confirmed — portfolio snapshot for {run_date} confirmed present in database (total_value=${float(total_val):,.2f}).",
            )
        else:
            return (
                "Write activity logged by pipeline",
                f"⚠️ Unconfirmed — portfolio snapshot for {run_date} was not found in database.",
            )
    except Exception as exc:
        err_msg = sanitize_text(str(exc))
        return (
            "Unable to determine from logs",
            f"⚠️ Independent check skipped or failed: {err_msg}",
        )


def format_workflow_summary(
    log_text: str,
    outcome: Optional[str] = None,
    verify_db: bool = True,
) -> str:
    """
    Build the complete GitHub Actions Step Summary markdown report.
    """
    run_num = os.environ.get("GITHUB_RUN_NUMBER", "N/A")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")

    run_url = f"{server_url}/{repo}/actions/runs/{run_id}" if (server_url and repo and run_id) else None
    commit_url = f"{server_url}/{repo}/commit/{sha}" if (server_url and repo and sha) else None

    # Step outcome
    if not outcome:
        outcome = os.environ.get("PIPELINE_OUTCOME", "unknown").lower()
    is_success = outcome == "success"

    # Extract audit and fields
    audit_text, pipeline_status = extract_audit_block(log_text)
    fields = parse_audit_fields(audit_text) if audit_text else {}

    raw_status = pipeline_status or fields.get("status") or ("SUCCESS" if is_success else "FAILED")
    status = raw_status.strip()

    # Determine if this was a skipped cycle (e.g. market closed)
    is_skipped = (
        status in ("SKIPPED", "SKIPPED_MARKET_CLOSED")
        or fields.get("status") in ("SKIPPED", "SKIPPED_MARKET_CLOSED")
        or fields.get("regime") == "SKIPPED_MARKET_CLOSED"
        or "SKIPPED_MARKET_CLOSED" in log_text
        or (is_success and "Market Closed" in log_text)
    ) and status not in ("FAILED", "HALTED")

    run_date = fields.get("run_date")
    if not run_date and is_skipped:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", log_text)
        if date_match:
            run_date = date_match.group(1)

    # DB persistence check
    log_indicator, db_verification = (
        check_database_verification(run_date, is_skipped=is_skipped)
        if verify_db
        else (
            (
                "Not applicable — market closed (no database writes scheduled)"
                if is_skipped
                else "Pipeline logs not verified"
            ),
            (
                "Not applicable — no portfolio snapshot expected for a skipped cycle"
                if is_skipped
                else "Independent database check skipped."
            ),
        )
    )

    # Check if pipeline explicitly reported database unavailability
    if not is_skipped and log_text and "DATABASE UNAVAILABLE" in log_text:
        log_indicator = "❌ Database was UNAVAILABLE during run (stateless mode — no DB writes)"

    # Build markdown
    lines: List[str] = []

    # Title
    if is_skipped:
        status_icon = "⏭️"
    elif is_success and status == "SUCCESS":
        status_icon = "✅"
    elif status == "DEGRADED":
        status_icon = "⚠️"
    else:
        status_icon = "❌"
    lines.append(f"# {status_icon} Daily Trading Pipeline — Run #{run_num}")
    lines.append("")

    # Overview table
    lines.append("| Property | Value |")
    lines.append("|:---|:---|")
    if run_url:
        lines.append(f"| **Workflow Run** | [#{run_num}]({run_url}) |")
    else:
        lines.append(f"| **Workflow Run** | #{run_num} |")

    if commit_url and sha:
        lines.append(f"| **Commit** | [`{sha[:7]}`]({commit_url}) |")
    elif sha:
        lines.append(f"| **Commit** | `{sha[:7]}` |")

    lines.append(f"| **Workflow Outcome** | {'✅ Success' if is_success else '❌ Failed'} |")
    pipeline_status_display = "⏭️ SKIPPED — US market closed" if is_skipped else f"**{status}**"
    lines.append(f"| **Pipeline Status** | {pipeline_status_display} |")
    lines.append(f"| **Run Date** | {run_date or '*Not established*'} |")
    regime_display = fields.get("regime", "SKIPPED_MARKET_CLOSED" if is_skipped else "*Not established*")
    lines.append(f"| **Market Regime** | {regime_display} |")
    lines.append("")

    # Audit Metrics or Failure Report
    if audit_text and fields:
        lines.append("## 📊 Trading Execution & Portfolio Metrics")
        lines.append("")
        lines.append("### Data Ingestion & Validation")
        lines.append(f"- **Tickers Fetched:** {fields.get('tickers_fetched', 'N/A')}")
        lines.append(f"- **Passed Validation:** {fields.get('tickers_valid', 'N/A')}")
        skipped_bad_data = (
            "N/A — no data validation performed"
            if is_skipped
            else fields.get("tickers_skipped", "N/A")
        )
        lines.append(f"- **Skipped (Bad Data):** {skipped_bad_data}")
        lines.append("")

        lines.append("### Orders & Fills")
        lines.append(f"- **Orders Generated:** {fields.get('orders_generated', 'N/A')}")
        lines.append(f"- **Fills Executed:** {fields.get('fills_executed', 'N/A')}")
        lines.append(f"- **Pending Orders for Next Open:** {fields.get('pending_orders', 'N/A')}")
        lines.append("")

        lines.append("### Portfolio State (End of Day)")
        if is_skipped:
            lines.append("- **Cash:** N/A — no trading cycle executed")
            lines.append("- **Total Equity:** N/A — no trading cycle executed")
            lines.append("- **Realized PnL Today:** N/A — no trading cycle executed")
        else:
            lines.append(f"- **Cash:** {fields.get('cash', 'N/A')}")
            lines.append(f"- **Total Equity:** {fields.get('total_equity', 'N/A')}")
            lines.append(f"- **Realized PnL Today:** {fields.get('realized_pnl', 'N/A')}")
        lines.append("")

        if "errors" in fields:
            lines.append("### ℹ️ Execution Notes / Skip Reason" if is_skipped else "### ⚠ Errors Reported by Pipeline")
            lines.append(fields["errors"])
            lines.append("")

        # Include detailed fills table if present in audit
        if "| Date | Action | Ticker |" in audit_text:
            table_match = re.search(r"(\| Date \| Action \| Ticker \|[\s\S]+?)(?=\n###|\Z)", audit_text)
            if table_match:
                lines.append("### Fills Executed Today")
                lines.append(table_match.group(1).strip())
                lines.append("")
    elif is_skipped:
        # Skipped run without a full audit block
        lines.append("## 📊 Trading Execution & Portfolio Metrics")
        lines.append("")
        lines.append("### Data Ingestion & Validation")
        lines.append("- **Tickers Fetched:** 0")
        lines.append("- **Passed Validation:** 0")
        lines.append("- **Skipped (Bad Data):** N/A — no data validation performed")
        lines.append("")

        lines.append("### Orders & Fills")
        lines.append("- **Orders Generated:** 0")
        lines.append("- **Fills Executed:** 0")
        lines.append("- **Pending Orders for Next Open:** 0")
        lines.append("")

        lines.append("### Portfolio State (End of Day)")
        lines.append("- **Cash:** N/A — no trading cycle executed")
        lines.append("- **Total Equity:** N/A — no trading cycle executed")
        lines.append("- **Realized PnL Today:** N/A — no trading cycle executed")
        lines.append("")

        skip_match = re.search(r"Market Closed[^\r\n]+", log_text)
        if skip_match:
            lines.append("### ℹ️ Execution Notes / Skip Reason")
            lines.append(f"- {sanitize_text(skip_match.group(0).strip())}")
            lines.append("")
    else:
        # Incomplete / failed run
        lines.append("## ⚠️ Pipeline Execution Failure Report")
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append("> The trading pipeline terminated before emitting the final Daily Pipeline Audit.")
        lines.append("")
        lines.append("### Relevant Sanitized Error Log")
        lines.append("```text")
        lines.append(extract_error_lines(log_text))
        lines.append("```")
        lines.append("")

    # Database Verification Section (Requirement 6 & 7)
    lines.append("## 🗄️ Database Persistence Status")
    lines.append("")
    lines.append(f"- **Pipeline Log Indicator:** {log_indicator}")
    lines.append(f"- **Independent Database Verification:** {db_verification}")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Pipeline results above reflect simulated paper trading execution. Real money is disabled in V1.")

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for publishing summary report to $GITHUB_STEP_SUMMARY."""
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pipeline_run.log"
    log_text = ""

    if Path(log_path).exists():
        try:
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log_text = f"Failed to read log file {log_path}: {exc}"
    else:
        log_text = f"Log file not found at: {log_path}"

    try:
        report_md = format_workflow_summary(log_text)
    except Exception as exc:
        report_md = f"# ❌ Workflow Summary Generation Error\n\nFailed to generate report: {sanitize_text(str(exc))}"

    # Write to $GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(report_md + "\n")
            print(f"Workflow summary written to {summary_file}")
        except Exception as exc:
            print(f"Warning: Failed to write to GITHUB_STEP_SUMMARY: {exc}", file=sys.stderr)

    # Always echo summary to stdout
    print("\n" + report_md)


if __name__ == "__main__":
    main()
