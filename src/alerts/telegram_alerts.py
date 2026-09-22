"""
src/alerts/telegram_alerts.py — Telegram Notifications (Section 7 Item 2).

Delivers real-time Telegram messages for:
  1. Trade executed — BUY  (✅ BOUGHT ...)
  2. Trade executed — SELL (📤 SOLD ...)
  3. Stop-loss triggered   (🔴 STOP-LOSS TRIGGERED)
  4. Circuit breaker       (⚠️ CIRCUIT BREAKER ACTIVE)
  5. Pipeline failure      (🚨 PIPELINE FAILED)
  6. PSI drift alarm       (🔴 MODEL DRIFT ALERT)
  7. Daily summary         (📊 DAILY SUMMARY)

SAFETY GUARANTEES:
  - Never crashes or blocks pipeline execution. All ops wrapped in try/except.
  - If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing: logs warning, skips.
  - Async Bot.send_message() is executed via a private event loop run — no
    external event loop dependency, safe to call from any thread or sync code.
  - Uses parse_mode=None (plain text with emoji) — no Markdown escaping needed.

USAGE:
  from src.alerts.telegram_alerts import notify
  notify("🤖 Hello from the pipeline!")

CLI TEST:
  python -m src.alerts.telegram_alerts --test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from config.settings import settings

from src.alerts.dedup import is_alert_already_sent, mark_alert_sent

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")


# ── Credential check ──────────────────────────────────────────────────────────

def _is_telegram_configured() -> bool:
    """Return True only when both BOT_TOKEN and CHAT_ID are non-empty."""
    token = getattr(settings, "telegram_bot_token", "") or ""
    chat_id = getattr(settings, "telegram_chat_id", "") or ""
    return bool(token.strip() and chat_id.strip())


# ── Low-level async sender ────────────────────────────────────────────────────

async def _async_send(token: str, chat_id: str, text: str) -> None:
    """Send a single Telegram message asynchronously."""
    from telegram import Bot  # local import — keeps startup cost at zero when Telegram unused
    async with Bot(token) as bot:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            # No parse_mode — emoji + plain text, zero escaping headaches
        )


# ── Public sync interface ─────────────────────────────────────────────────────

def notify(message: str, alert_key: Optional[str] = None) -> bool:
    """
    Send a Telegram notification message with optional deduplication.

    Synchronous wrapper over the async Bot API. Handles the event loop
    internally — safe to call from any sync context (pipeline, scheduler, etc).

    Returns:
        True  — message sent successfully.
        False — skipped (credentials missing or duplicate key) or send failed (logged).
    """
    try:
        if alert_key and is_alert_already_sent(alert_key):
            logger.info("DUPLICATE_ALERT_SKIPPED: %s", alert_key)
            return False

        if not _is_telegram_configured():
            logger.warning(
                "Telegram notification skipped "
                "(TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set): %r",
                message[:60],
            )
            return False

        token = settings.telegram_bot_token.strip()
        chat_id = settings.telegram_chat_id.strip()

        # Run async sender in a fresh event loop — avoids colliding with any
        # existing loop (e.g., Streamlit's or Jupyter's).
        asyncio.run(_async_send(token, chat_id, message))
        if alert_key:
            mark_alert_sent(alert_key)
            logger.info("ALERT_SENT: %s", alert_key)
        logger.info("Telegram notification sent: %r", message[:60])
        return True

    except Exception as exc:
        logger.warning("Telegram notification failed: %s", exc)
        return False


# ── Trigger message builders ──────────────────────────────────────────────────

def notify_trade_bought(
    ticker: str,
    price: float,
    shares: float,
    model_confidence: Optional[float] = None,
    portfolio_value: Optional[float] = None,
) -> bool:
    """Trigger 1: BUY trade executed."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"BUY_{ticker.upper()}_{date_str}"
    conf_str = f"{model_confidence * 100:.0f}% (High)" if (model_confidence is not None and model_confidence <= 1.0) else (f"{model_confidence:.2f}" if model_confidence is not None else "High")
    lines = [
        "✅ BOUGHT A STOCK",
        f"📌 Stock: {ticker.upper()}",
        f"💰 Bought at: ${price:,.2f} per share",
        f"📦 Shares: {shares:.4f} shares",
        f"🎯 AI Confidence: {conf_str}",
    ]
    if portfolio_value is not None:
        lines.append(f"💼 Portfolio now: ${portfolio_value:,.2f}")
    return notify("\n".join(lines), alert_key=alert_key)


def notify_trade_sold(
    ticker: str,
    exit_price: float,
    shares: float,
    pnl: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    exit_reason: Optional[str] = None,
    portfolio_value: Optional[float] = None,
) -> bool:
    """Trigger 2: SELL trade executed."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"SELL_{ticker.upper()}_{date_str}"
    lines = [
        "📤 SOLD A STOCK",
        f"📌 Stock: {ticker.upper()}",
        f"💰 Exit Price: ${exit_price:,.2f}",
        f"📦 Shares: {shares:.4f} shares",
    ]
    if pnl is not None:
        pnl_pct_str = f" ({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
        icon = "📈" if pnl >= 0 else "📉"
        lines.append(f"{icon} Profit/Loss: ${pnl:+,.2f}{pnl_pct_str}")
    if exit_reason:
        lines.append(f"🚪 Reason: {exit_reason}")
    if portfolio_value is not None:
        lines.append(f"💼 Portfolio now: ${portfolio_value:,.2f}")
    return notify("\n".join(lines), alert_key=alert_key)


def notify_stop_loss(
    ticker: str,
    loss_amount: float,
    loss_pct: float,
    days_held: Optional[int] = None,
    portfolio_value: Optional[float] = None,
) -> bool:
    """Trigger 3: Stop-loss triggered."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"STOP_LOSS_{ticker.upper()}_{date_str}"
    lines = [
        "🔴 SOLD TO PREVENT BIGGER LOSS",
        f"📌 Stock: {ticker.upper()}",
        f"📉 We lost: -${abs(loss_amount):,.2f} ({loss_pct:.1f}%)",
        "💡 Why: Stock fell too much so we sold automatically to protect the rest of your money",
    ]
    if days_held is not None:
        lines.append(f"📅 Days Held: {days_held}")
    if portfolio_value is not None:
        lines.append(f"💼 Portfolio now: ${portfolio_value:,.2f}")
    return notify("\n".join(lines), alert_key=alert_key)


def notify_circuit_breaker(
    spy_drop_pct: float,
    lookback_days: int,
    pause_days: int,
    cash_pct: Optional[float] = None,
) -> bool:
    """Trigger 4: Circuit breaker activated."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"CIRCUIT_BREAKER_{date_str}"
    lines = [
        "⚠️ MARKET CRASH PROTECTION ON",
        f"📉 The overall market dropped {spy_drop_pct:.1f}% in the last {lookback_days} days",
        f"🛑 We stopped buying new stocks until the market calms down",
        "💼 Your money is safe in cash",
    ]
    if cash_pct is not None:
        lines.append(f"💼 Cash Reserve: {cash_pct:.1f}%")
    return notify("\n".join(lines), alert_key=alert_key)


def notify_pipeline_failure(
    error_message: str,
    run_date: Optional[str] = None,
) -> bool:
    """Trigger 5: Pipeline failure."""
    date_str = run_date or datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"PIPELINE_FAILURE_{date_str}"
    lines = [
        "🚨 PIPELINE FAILED",
        f"📅 Date: {date_str}",
        f"❌ Error: {error_message}",
        "⚡ Action needed — check logs",
    ]
    return notify("\n".join(lines), alert_key=alert_key)


def notify_psi_drift(psi_value: float) -> bool:
    """Trigger 6: PSI drift red alarm (>= 0.25)."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"PSI_DRIFT_{date_str}"
    lines = [
        "⚠️ AI MODEL NEEDS UPDATING",
        "🤖 Our AI predictions have become less reliable recently",
        f"📊 Model Health (PSI): {psi_value:.2f}",
        "🛑 Stopped buying new stocks for now",
        "👤 The AI will be retrained soon",
        "💼 Existing stocks held safely",
    ]
    return notify("\n".join(lines), alert_key=alert_key)


def _get_top_ai_pick(run_date: str) -> Optional[str]:
    """Retrieve top AI pick formatted string for run_date from repository predictions."""
    try:
        from src.db import repository
        best_ticker = None
        best_prob = -1.0

        tickers = getattr(settings, "ticker_list", [])
        for t in tickers:
            try:
                df = repository.get_predictions(t, start_date=run_date, end_date=run_date)
                if not df.empty and "probability" in df.columns:
                    prob = float(df.iloc[-1]["probability"])
                    if prob > best_prob:
                        best_prob = prob
                        best_ticker = str(df.iloc[-1]["ticker"]).upper()
            except Exception:
                continue

        if best_ticker and best_prob >= 0:
            return f"{best_ticker} ({round(best_prob * 100)}%)"

        if getattr(repository, "_db_available", False):
            from sqlalchemy import select
            from sqlalchemy.orm import Session
            from src.db.repository import get_engine
            from src.db.models import PredictionRow
            with Session(get_engine()) as session:
                row = session.execute(
                    select(PredictionRow)
                    .where(PredictionRow.date == run_date)
                    .order_by(PredictionRow.probability.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if row:
                    return f"{row.ticker.upper()} ({round(row.probability * 100)}%)"
    except Exception as exc:
        logger.debug("Failed to fetch top AI pick: %s", exc)
    return None


def notify_daily_summary(
    run_date: str,
    portfolio_value: float,
    daily_pnl: Optional[float] = None,
    daily_pnl_pct: Optional[float] = None,
    cash: Optional[float] = None,
    cash_pct: Optional[float] = None,
    positions: Optional[Any] = None,
    max_positions: Optional[int] = None,
    status: str = "SUCCESS",
    top_pick: Optional[str] = None,
    alerts: Optional[str] = None,
) -> bool:
    """Trigger 7: Daily summary on every successful pipeline run (Freqtrade style format)."""
    alert_key = f"DAILY_SUMMARY_{run_date}"
    try:
        snapshot = None
        try:
            from src.db import repository
            snapshot = repository.get_latest_portfolio_snapshot()
        except Exception as exc:
            logger.debug("Failed to fetch latest portfolio snapshot for daily summary: %s", exc)

        if snapshot:
            if not portfolio_value:
                portfolio_value = float(snapshot.get("total_value", 0.0) or 0.0)
            if cash is None:
                cash = float(snapshot.get("cash", 0.0) or 0.0)

        if cash is not None and portfolio_value and portfolio_value > 0 and cash_pct is None:
            cash_pct = (cash / portfolio_value) * 100.0

        # Portfolio Today line
        if daily_pnl is not None:
            pnl_sign = "+" if daily_pnl >= 0 else "-"
            pct_val = daily_pnl_pct if daily_pnl_pct is not None else 0.0
            pct_sign = "+" if pct_val >= 0 else "-"
            icon = "📈" if daily_pnl >= 0 else "📉"
            today_str = f"{icon} Today: {pnl_sign}${abs(daily_pnl):,.2f} ({pct_sign}{abs(pct_val):.2f}%)"
        else:
            today_str = "📈 Today: +$0.00 (+0.00%)"

        # Cash line
        cash_val = cash if cash is not None else 0.0
        cash_p = cash_pct if cash_pct is not None else 0.0
        cash_str = f"🏦 Cash: ${cash_val:,.2f} ({cash_p:.1f}%)"

        # Extract positions dictionary (explicit argument takes precedence over snapshot)
        pos_dict = {}
        if isinstance(positions, dict) and positions:
            pos_dict = positions
        elif isinstance(positions, list) and positions:
            for item in positions:
                if isinstance(item, dict) and "ticker" in item:
                    pos_dict[str(item["ticker"]).upper()] = item
                elif isinstance(item, str):
                    pos_dict[item.upper()] = {"ticker": item.upper()}
        elif snapshot and isinstance(snapshot.get("positions"), dict) and snapshot["positions"]:
            pos_dict = snapshot["positions"]


        max_pos = max_positions if max_positions is not None else getattr(settings, "max_positions", 3)
        num_pos = len(pos_dict) if pos_dict else (len(positions) if isinstance(positions, list) else 0)

        pos_lines = []
        if pos_dict:
            for ticker, info in pos_dict.items():
                t_upper = str(ticker).upper()
                if isinstance(info, dict):
                    qty = float(info.get("quantity") or info.get("shares") or 0.0)
                    entry_p = float(info.get("entry_price") or 0.0)
                    curr_p = float(info.get("current_price") or entry_p)
                    pnl_val = info.get("unrealized_pnl")
                    if pnl_val is None:
                        pnl_val = (curr_p - entry_p) * qty if qty > 0 and entry_p > 0 else 0.0
                    else:
                        pnl_val = float(pnl_val)

                    pnl_p = info.get("unrealized_pnl_pct")
                    if pnl_p is None:
                        pnl_p = ((curr_p - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0
                    else:
                        pnl_p = float(pnl_p)

                    pnl_s = "+" if pnl_val >= 0 else "-"
                    pct_s = "+" if pnl_p >= 0 else "-"
                    pos_icon = "📈" if pnl_val >= 0 else "📉"

                    pos_lines.append(
                        f"- {t_upper}: {qty:.2f} shares\n"
                        f"  Entry: ${entry_p:,.2f} → Now: ${curr_p:,.2f}\n"
                        f"  P&L: {pnl_s}${abs(pnl_val):,.2f} ({pct_s}{abs(pnl_p):.2f}%) {pos_icon}"
                    )
                else:
                    pos_lines.append(f"- {t_upper}: No data available")
        elif isinstance(positions, list) and positions:
            for t in positions:
                t_upper = str(t).upper()
                pos_lines.append(f"- {t_upper}: No data available")
        else:
            pos_lines.append("No open positions held")

        positions_block = "\n\n".join(pos_lines)

        # AI Top Pick
        ai_pick_str = top_pick
        if not ai_pick_str:
            ai_pick_str = _get_top_ai_pick(run_date)
        if not ai_pick_str:
            ai_pick_str = "No data available"

        alerts_str = alerts or "None"
        status_icon = "✅" if status.upper() == "SUCCESS" else "⚠️"

        lines = [
            "📊 DAILY SUMMARY",
            f"📅 Date: {run_date}",
            "━━━━━━━━━━━━━━━━━━━━",
            f"💼 Portfolio: ${portfolio_value:,.2f}",
            today_str,
            cash_str,
            "━━━━━━━━━━━━━━━━━━━━",
            f"📌 POSITIONS ({num_pos}/{max_pos}):",
            positions_block,
            "━━━━━━━━━━━━━━━━━━━━",
            f"🤖 AI Top Pick Today: {ai_pick_str}",
            f"⚠️ Alerts: {alerts_str}",
            f"{status_icon} Pipeline: {status.upper()}",
            "🎓 Paper trading only!",
        ]

        msg_text = "\n".join(lines)
        return notify(msg_text, alert_key=alert_key)
    except Exception as exc:
        logger.warning("Failed to build daily summary notification: %s", exc)
        fallback_msg = (
            f"📊 DAILY SUMMARY\n"
            f"📅 Date: {run_date}\n"
            f"💼 Portfolio: ${portfolio_value:,.2f}\n"
            f"✅ Pipeline: {status.upper()}"
        )
        return notify(fallback_msg, alert_key=alert_key)



def notify_failover(
    ticker: str,
    primary_source: str = "yfinance",
    backup_source: str = "Alpha Vantage",
) -> bool:
    """Trigger: Data source failover."""
    date_str = datetime.now(NY_TZ).strftime("%Y-%m-%d")
    alert_key = f"FAILOVER_{ticker.upper()}_{date_str}"
    lines = [
        "⚠️ DATA SOURCE FAILOVER",
        f"Primary ({primary_source}) failed for: {ticker.upper()}",
        f"Switched to: {backup_source} backup",
    ]
    return notify("\n".join(lines), alert_key=alert_key)



# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="AI Stock Trader — Telegram Alerts Test")
    parser.add_argument("--test", action="store_true", help="Send a test notification")
    args = parser.parse_args()

    if args.test:
        if not _is_telegram_configured():
            print(
                "❌ Telegram credentials not configured.\n"
                "   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.\n"
                "   Get token from @BotFather on Telegram.\n"
                "   Get chat ID from @userinfobot on Telegram."
            )
            sys.exit(1)

        print("Sending test Telegram notification …")
        success = notify(
            "🤖 AI Stock Trader connected!\n"
            "Notifications are working. ✅"
        )
        if success:
            print("✅ Test notification sent successfully.")
        else:
            print("❌ Test notification failed. Check credentials and logs.")
            sys.exit(1)
    else:
        parser.print_help()
