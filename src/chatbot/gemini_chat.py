"""
src/chatbot/gemini_chat.py — Section 10 Option C: AI Chatbot Engine.

Provides live system context gathering and integrates with the Google Gemini API
to answer user questions about portfolio performance, risk rules, and trading decisions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from config.settings import settings
from dashboard.data_loader import (
    get_market_regime_and_predictions,
    get_portfolio_summary,
    get_recent_trades_df,
)
from src.db import repository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a friendly AI assistant for an automated paper stock trading system.
You help the user understand their portfolio, trades, and system decisions in plain English.
Be friendly, casual and concise.
Always explain things in simple terms as if talking to someone who has never invested before. Never use financial jargon without immediately explaining what it means in brackets.

Examples of good responses:
- Instead of 'RSI is overbought at 74' say 'The stock has been rising very fast lately (score: 74/100)'
- Instead of 'MACD crossover detected' say 'The trend direction just changed which could mean the stock is about to move'
- Instead of 'portfolio drawdown -8%' say 'at its worst point your portfolio was down $800 from your starting $10,000'

Always end responses with:
Remember: This is paper trading with fake money — great for learning with zero risk! 🎓

Current Portfolio Data:
{portfolio_data}

Recent Trades:
{recent_trades}

Active Alerts:
{active_alerts}

Model Rankings Today:
{model_rankings}
"""

FRIENDLY_ERROR_MSG = "Sorry I am having trouble connecting. Please try again in a moment."


def build_live_context_strings() -> Dict[str, str]:
    """
    Gathers live context from the portfolio, trades, risk engine, and ML model.
    """
    # 1. Portfolio Data
    try:
        port = get_portfolio_summary()
        pos_str_list = []
        for p in port.get("positions", []):
            pos_str_list.append(
                f"- {p['ticker']}: {p['shares']:.2f} shares @ ${p['entry_price']:.2f} "
                f"(Current: ${p['current_price']:.2f}, PnL: ${p['unrealized_pnl']:+.2f})"
            )
        pos_str = "\n".join(pos_str_list) if pos_str_list else "No open positions (100% Cash)."

        portfolio_data = (
            f"Total Equity: ${port.get('total_equity', 10000.0):,.2f}\n"
            f"Cash: ${port.get('cash', 10000.0):,.2f} ({port.get('cash_reserve_pct', 100.0):.1f}%)\n"
            f"Open Positions Count: {port.get('open_positions_count', 0)} / {port.get('max_positions', 3)}\n"
            f"Open Positions:\n{pos_str}"
        )
    except Exception as e:
        logger.warning("Error gathering portfolio context: %s", e)
        portfolio_data = "Portfolio Equity: $10,000.00, Cash: $10,000.00"

    # 2. Recent Trades
    try:
        trades_df = get_recent_trades_df(limit=10)
        if not trades_df.empty:
            tr_lines = []
            for _, r in trades_df.iterrows():
                tr_lines.append(
                    f"- {r.get('date')}: {r.get('action')} {r.get('shares')} {r.get('ticker')} @ "
                    f"${r.get('price'):.2f} (Net PnL: ${r.get('net_pnl'):+.2f})"
                )
            recent_trades = "\n".join(tr_lines)
        else:
            recent_trades = "No recent trades recorded."
    except Exception as e:
        logger.warning("Error gathering trades context: %s", e)
        recent_trades = "No recent trades available."

    # 3. Active Alerts
    try:
        events = repository.get_events(limit=20)
        alert_lines = []
        for ev in events:
            if ev.get("level") in ("WARNING", "ERROR", "CRITICAL"):
                alert_lines.append(f"- [{ev.get('level')}] {ev.get('component')}: {ev.get('message')}")
        active_alerts = "\n".join(alert_lines) if alert_lines else "All systems operating normally. No active risk alerts."
    except Exception as e:
        logger.warning("Error gathering alerts context: %s", e)
        active_alerts = "No active risk alerts."

    # 4. Model Rankings Today
    try:
        regime, preds_df = get_market_regime_and_predictions()
        regime_status = "Risk-ON" if regime.get("is_risk_on") else "Risk-OFF (SPY < 200d MA)"
        if not preds_df.empty:
            rank_lines = [f"Regime: {regime_status}"]
            for rank_idx, (_, r) in enumerate(preds_df.head(5).iterrows()):
                prob_val = r.get("probability", 0.5)
                prob_str = f"{prob_val:.1%}" if isinstance(prob_val, (int, float)) else str(prob_val)
                rank_lines.append(
                    f"- #{rank_idx + 1} {r.get('ticker', '')} ({r.get('sector', '')}): Prob={prob_str}, "
                    f"Action={r.get('action_signal', 'HOLD')}, Sentiment={r.get('sentiment', '—')}"
                )
            model_rankings = "\n".join(rank_lines)
        else:
            model_rankings = f"Regime: {regime_status} — No candidate predictions available."
    except Exception as e:
        logger.warning("Error gathering model predictions context: %s", e)
        model_rankings = "Model: HistGradientBoostingActive, Regime: Normal."

    return {
        "portfolio_data": portfolio_data,
        "recent_trades": recent_trades,
        "active_alerts": active_alerts,
        "model_rankings": model_rankings,
    }


def generate_system_prompt() -> str:
    """
    Builds the complete System Prompt with live data injected.
    """
    ctx = build_live_context_strings()
    return SYSTEM_PROMPT_TEMPLATE.format(
        portfolio_data=ctx["portfolio_data"],
        recent_trades=ctx["recent_trades"],
        active_alerts=ctx["active_alerts"],
        model_rankings=ctx["model_rankings"],
    )


def ask_gemini_chatbot(
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Sends the user message along with injected live context to the Google Gemini API.
    Returns the chatbot response text or a friendly error message on failure.
    """
    key_to_use = (
        api_key
        or getattr(settings, "gemini_api_key", "")
        or os.getenv("GEMINI_API_KEY", "")
    )
    if key_to_use:
        key_to_use = str(key_to_use).strip()

    if not key_to_use:
        logger.warning("GEMINI_API_KEY is not set.")
        return FRIENDLY_ERROR_MSG

    effective_model = (
        model_name
        or getattr(settings, "gemini_model", "gemini-2.5-flash")
        or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    if effective_model:
        effective_model = str(effective_model).strip()

    try:
        from google import genai

        client = genai.Client(api_key=key_to_use)
        system_instruction = generate_system_prompt()

        # Format conversation history if provided
        history_lines = []
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 3 turns for turn memory
                role = "user" if msg.get("role") == "user" else "model"
                history_lines.append(f"{role.upper()}: {msg.get('content', '')}")

        history_text = "\n".join(history_lines) if history_lines else "None"
        full_prompt = (
            f"{system_instruction}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"USER: {user_message}"
        )

        logger.debug("Sending prompt to Gemini API model %s...", effective_model)
        response = client.models.generate_content(
            model=effective_model,
            contents=full_prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return FRIENDLY_ERROR_MSG

    except Exception as exc:
        logger.error("Gemini API call error: %s", exc, exc_info=True)
        return FRIENDLY_ERROR_MSG
