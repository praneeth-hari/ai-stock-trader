"""
src/reports/weekly_summary.py — Weekly Portfolio Summary Digest via Telegram (Section 7 Item 3).

Generates and delivers a weekly summary digest of portfolio performance, trades,
holdings, alerts, and upcoming earnings watch every Sunday morning at 9:00 AM.
Delivered via Telegram only (email is disabled).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import settings
from src.alerts.telegram_alerts import _is_telegram_configured, notify
from src.db import repository

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


def _get_week_date_range(target_date: Optional[date] = None) -> Tuple[date, date]:
    """
    Returns (monday_date, friday_date) for the trading week ending on or immediately preceding target_date.
    If target_date is Sunday (e.g. 2026-09-20), the trading week is Mon 2026-09-14 to Fri 2026-09-18.
    If target_date is Saturday, week is Mon to Fri of that week.
    If target_date is a weekday, week is Mon to Fri of that week.
    """
    if target_date is None:
        target_date = date.today()

    # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    w = target_date.weekday()
    if w == 6:  # Sunday
        friday = target_date - timedelta(days=2)
    elif w == 5:  # Saturday
        friday = target_date - timedelta(days=1)
    else:
        friday = target_date + timedelta(days=(4 - w))

    monday = friday - timedelta(days=4)
    return monday, friday


def generate_weekly_summary_data(target_date: Optional[date] = None, market: str = "US") -> Optional[Dict[str, Any]]:
    """
    Gathers data from DB for the weekly summary digest.
    Returns a dict with formatted values, or None if no portfolio data exists for the week.
    """
    monday_date, friday_date = _get_week_date_range(target_date)
    mon_str = monday_date.strftime("%Y-%m-%d")
    fri_str = friday_date.strftime("%Y-%m-%d")

    # Fetch snapshots
    all_snapshots = repository.get_portfolio_snapshots(limit=500, market=market) if hasattr(repository, "get_portfolio_snapshots") else []
    
    if not all_snapshots:
        # Fallback check if single snapshot exists
        latest_snap = repository.get_latest_portfolio_snapshot(market=market)
        if latest_snap:
            all_snapshots = [latest_snap]

    if not all_snapshots:
        logger.warning("No portfolio snapshots found in database. Weekly summary skipped.")
        return None

    # Filter snapshots for the target week or earlier
    week_snaps = [s for s in all_snapshots if mon_str <= str(s.get("run_date", "")) <= fri_str]
    prior_snaps = [s for s in all_snapshots if str(s.get("run_date", "")) <= mon_str]

    if week_snaps:
        start_snap = prior_snaps[-1] if prior_snaps else week_snaps[0]
        end_snap = week_snaps[-1]
    elif prior_snaps:
        start_snap = prior_snaps[0]
        end_snap = prior_snaps[-1]
    else:
        start_snap = all_snapshots[0]
        end_snap = all_snapshots[-1]

    starting_value = float(start_snap.get("total_value", settings.initial_capital))
    ending_value = float(end_snap.get("total_value", settings.initial_capital))
    weekly_gain_dollar = ending_value - starting_value
    weekly_gain_pct = (weekly_gain_dollar / starting_value * 100.0) if starting_value > 0 else 0.0

    # Benchmark comparison
    bm_symbol = settings.get_benchmark(market)
    try:
        spy_df = repository.get_market_data(bm_symbol, start_date=mon_str, end_date=fri_str)
    except Exception:
        spy_df = pd.DataFrame()

    if not spy_df.empty and "close" in spy_df.columns and len(spy_df) >= 1:
        spy_start = float(spy_df["close"].iloc[0])
        spy_end = float(spy_df["close"].iloc[-1])
        spy_return_pct = ((spy_end - spy_start) / spy_start * 100.0) if spy_start > 0 else 0.0
    else:
        spy_return_pct = 0.0

    spy_diff = weekly_gain_pct - spy_return_pct
    if spy_diff >= 0:
        spy_comp_str = f"+{spy_diff:.1f} pts ahead ✅"
    else:
        spy_comp_str = f"{spy_diff:.1f} pts behind 🔴"

    # Trades this week
    try:
        from dashboard.data_loader import get_recent_trades_df
        trades_df = get_recent_trades_df(limit=200, market=market)
    except Exception:
        trades_df = pd.DataFrame()

    if not trades_df.empty:
        date_col = "Date Sold" if "Date Sold" in trades_df.columns else ("date" if "date" in trades_df.columns else "run_date")
        if date_col in trades_df.columns:
            week_trades = trades_df[
                (trades_df[date_col].astype(str) >= mon_str) & (trades_df[date_col].astype(str) <= fri_str)
            ]
        else:
            week_trades = trades_df
    else:
        week_trades = pd.DataFrame()

    total_trades = len(week_trades)
    wins = 0
    losses = 0
    best_trade_str = "None"
    worst_trade_str = "None"

    if total_trades > 0:
        pnl_col = "Profit / Loss ($)" if "Profit / Loss ($)" in week_trades.columns else ("net_pnl" if "net_pnl" in week_trades.columns else "pnl")
        ticker_col = "Ticker" if "Ticker" in week_trades.columns else "ticker"
        pct_col = "Profit / Loss (%)" if "Profit / Loss (%)" in week_trades.columns else "pnl_pct"

        if pnl_col in week_trades.columns:
            def parse_pnl_num(val):
                try:
                    return float(str(val).replace("$", "").replace(",", "").replace("+", ""))
                except Exception:
                    return 0.0

            week_trades = week_trades.copy()
            week_trades["_num_pnl"] = week_trades[pnl_col].apply(parse_pnl_num)

            for val in week_trades["_num_pnl"]:
                if val > 0:
                    wins += 1
                elif val < 0:
                    losses += 1

            best_idx = week_trades["_num_pnl"].idxmax()
            worst_idx = week_trades["_num_pnl"].idxmin()

            best_tr = week_trades.loc[best_idx]
            worst_tr = week_trades.loc[worst_idx]

            best_t = str(best_tr.get(ticker_col, "N/A"))
            worst_t = str(worst_tr.get(ticker_col, "N/A"))

            bp = float(best_tr.get("_num_pnl", 0.0))
            wp = float(worst_tr.get("_num_pnl", 0.0))

            bp_pct = str(best_tr.get(pct_col, ""))
            wp_pct = str(worst_tr.get(pct_col, ""))

            best_trade_str = f"{best_t} +${bp:,.2f}" + (f" ({bp_pct})" if bp_pct else "")
            worst_trade_str = f"{worst_t} -${abs(wp):,.2f}" + (f" ({wp_pct})" if wp_pct else "")

        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    else:
        win_rate = 0.0

    # Current holdings from latest snapshot
    positions = end_snap.get("positions", {}) or {}
    current_cash = float(end_snap.get("cash", settings.initial_capital))
    total_eq = float(end_snap.get("total_value", settings.initial_capital))
    cash_pct = (current_cash / total_eq * 100.0) if total_eq > 0 else 100.0

    holdings_lines = []
    if isinstance(positions, dict):
        for t, pos in positions.items():
            if isinstance(pos, dict):
                sh = float(pos.get("shares", 0.0))
                pnl_p = float(pos.get("unrealized_pnl_pct", pos.get("unrealized_pnl", 0.0)))
                icon = "🟢" if pnl_p >= 0 else "🔴"
                holdings_lines.append(f"- {t}: {sh:.2f} shares ({pnl_p:+.2f}%) {icon}")
            else:
                holdings_lines.append(f"- {t}")
    elif isinstance(positions, list):
        for pos in positions:
            t = pos.get("ticker", "N/A")
            sh = float(pos.get("shares", 0.0))
            pnl_p = float(pos.get("unrealized_pnl_pct", pos.get("unrealized_pnl", 0.0)))
            icon = "🟢" if pnl_p >= 0 else "🔴"
            holdings_lines.append(f"- {t}: {sh:.2f} shares ({pnl_p:+.2f}%) {icon}")

    if not holdings_lines:
        holdings_formatted = "- None (100% Cash)"
    else:
        holdings_formatted = "\n".join(holdings_lines)

    # Alerts this week (inspect events log)
    try:
        events = repository.get_events(limit=100)
    except Exception:
        events = []

    cb_alert = "No circuit breaker events ✅"
    ai_alert = "AI model healthy ✅"
    data_alert = "All data feeds normal ✅"

    if events:
        for e in events:
            msg = str(e.get("message", "")).upper()
            if "CIRCUIT_BREAKER" in msg or "CIRCUIT BREAKER" in msg:
                cb_alert = "Circuit breaker activated ⚠️"
            if "DRIFT" in msg or "PSI" in msg:
                ai_alert = "AI model drift alert triggered ⚠️"
            if "FAILOVER" in msg or "DATA FETCH ERROR" in msg:
                data_alert = "Data feed failover occurred ⚠️"

    # Next week watch (earnings & regime)
    earnings_watch = "No earnings blackouts next week ✅"
    try:
        if hasattr(repository, "get_latest_earnings_calendar"):
            earn_cal = repository.get_latest_earnings_calendar()
            if earn_cal and isinstance(earn_cal, dict):
                upcoming = [
                    (t, d.get("days_until_earnings"))
                    for t, d in earn_cal.items()
                    if isinstance(d, dict) and d.get("days_until_earnings") is not None and 1 <= d["days_until_earnings"] <= 7
                ]
                if upcoming:
                    upcoming.sort(key=lambda x: x[1])
                    top_t, top_d = upcoming[0]
                    earnings_watch = f"{top_t} earnings in {top_d} days ⚠️"
    except Exception:
        pass

    regime_str = "Risk-ON ✅"
    try:
        if hasattr(repository, "get_market_regime_and_predictions"):
            regime_info, _ = repository.get_market_regime_and_predictions()
            if isinstance(regime_info, dict) and not regime_info.get("is_risk_on", True):
                regime_str = "Risk-OFF 🛑"
    except Exception:
        pass

    m_fmt = monday_date.strftime("%d %b")
    f_fmt = friday_date.strftime("%d %b %Y")
    week_label = f"{m_fmt} — {f_fmt}"

    return {
        "benchmark": bm_symbol,
        "monday_date": mon_str,
        "friday_date": fri_str,
        "week_label": week_label,
        "starting_value": starting_value,
        "ending_value": ending_value,
        "weekly_gain_dollar": weekly_gain_dollar,
        "weekly_gain_pct": weekly_gain_pct,
        "spy_comp_str": spy_comp_str,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "best_trade_str": best_trade_str,
        "worst_trade_str": worst_trade_str,
        "holdings_formatted": holdings_formatted,
        "cash": current_cash,
        "cash_pct": cash_pct,
        "circuit_breaker_alert_str": cb_alert,
        "ai_model_alert_str": ai_alert,
        "data_feed_alert_str": data_alert,
        "earnings_watch_str": earnings_watch,
        "regime_str": regime_str,
    }


def format_weekly_summary_message(data: Dict[str, Any]) -> str:
    """Formats data dict into the exact beginner-friendly Telegram weekly summary message."""
    gain_sign = "+" if data["weekly_gain_dollar"] >= 0 else ""
    bm_symbol = data.get("benchmark", "SPY")
    return (
        f"📊 WEEKLY SUMMARY\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Week: {data['week_label']}\n\n"
        f"💼 PORTFOLIO\n"
        f"Starting Value:  ${data['starting_value']:,.2f}\n"
        f"Ending Value:    ${data['ending_value']:,.2f}\n"
        f"Weekly Gain:     {gain_sign}${data['weekly_gain_dollar']:,.2f} ({data['weekly_gain_pct']:+.2f}%)\n"
        f"vs Market ({bm_symbol}): {data['spy_comp_str']}\n\n"
        f"📈 THIS WEEK'S TRADES\n"
        f"Total Trades:    {data['total_trades']}\n"
        f"Wins:            {data['wins']} ✅\n"
        f"Losses:          {data['losses']} ❌\n"
        f"Win Rate:        {data['win_rate']:.1f}%\n\n"
        f"Best Trade:  {data['best_trade_str']}\n"
        f"Worst Trade: {data['worst_trade_str']}\n\n"
        f"💰 CURRENT HOLDINGS\n"
        f"{data['holdings_formatted']}\n"
        f"- Cash: ${data['cash']:,.2f} ({data['cash_pct']:.1f}%)\n\n"
        f"⚠️ ALERTS THIS WEEK\n"
        f"- {data['circuit_breaker_alert_str']}\n"
        f"- {data['ai_model_alert_str']}\n"
        f"- {data['data_feed_alert_str']}\n\n"
        f"📅 NEXT WEEK WATCH\n"
        f"- {data['earnings_watch_str']}\n"
        f"- Market regime: {data['regime_str']}\n\n"
        f"🎓 Remember: Paper trading only!\n"
        f"   Learning to trade risk-free 🚀\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


def send_weekly_summary(target_date: Optional[date] = None, market: str = "US") -> bool:
    """
    Generates and delivers weekly summary digest via Telegram only.
    Skipped gracefully if Telegram credentials are not set or if no data exists.
    """
    if not _is_telegram_configured():
        logger.warning("Telegram notification skipped (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set)")
        return False

    try:
        data = generate_weekly_summary_data(target_date, market=market)
        if not data:
            logger.warning("No weekly summary data available for target date %s. Skipping.", target_date)
            return False

        message_text = format_weekly_summary_message(data)
        alert_key = f"WEEKLY_SUMMARY_{data['friday_date']}"
        success = notify(message_text, alert_key=alert_key)
        if success:
            logger.info("Weekly summary sent successfully via Telegram for week ending %s", data['friday_date'])
        return success
    except Exception as exc:
        logger.error("Failed to generate or send weekly summary: %s", exc, exc_info=True)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Executing weekly summary digest runner...")
    send_weekly_summary()


if __name__ == "__main__":
    main()
