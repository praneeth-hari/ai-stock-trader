"""
SPY >= 200-day SMA shadow benchmark (V1-safe, measurement only).

A separate $-for-$ paper portfolio that holds SPY while SPY's close is at or above its 200-day SMA
and cash otherwise. It starts from V1's day-0 snapshot (same date, same capital) and follows V1's
conventions exactly, so the two equity curves are comparable:

- Lag rule: a decision made on the close of the latest completed bar before run_date is executed by
  the NEXT run at the open V1 uses for its own fills (run_date's bar if present, else the latest bar).
- Costs: V1's slippage model (compute_slippage, ADV-tiered) plus the 0.2% fee on the slipped gross.
- Point-in-time: only bars dated <= run_date are read; the regime uses bars before run_date only.

It has its own table (benchmark_snapshots), never touches the V1 portfolio, and is updated after the
V1 day is committed, so it cannot affect any V1 decision.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import settings
from src.db import repository
from src.trading.paper_broker import compute_slippage

logger = logging.getLogger(__name__)

BENCHMARK_NAME = "SPY_200D"


def _floor4(x: float) -> float:
    return math.floor(x * 10_000) / 10_000


def _buy_all(cash: float, open_px: float, adv: Optional[float]) -> Dict[str, float]:
    fee_rate = settings.simulated_cost_per_trade
    rate = settings.slippage_tier_low_pct
    shares = _floor4(cash / (open_px * (1 + rate) * (1 + fee_rate)))
    while shares > 0:
        eff, _, slip = compute_slippage(shares, open_px, "BUY", adv=adv)
        gross = round(shares * eff, 4)
        fee = round(gross * fee_rate, 4)
        if gross + fee <= cash:
            return {"shares": shares, "cash": round(cash - gross - fee, 4), "price": eff, "fee": fee, "slippage": slip}
        shares = round(shares - 0.0001, 4)
    return {"shares": 0.0, "cash": cash, "price": open_px, "fee": 0.0, "slippage": 0.0}


def _sell_all(shares: float, cash: float, open_px: float, adv: Optional[float]) -> Dict[str, float]:
    eff, _, slip = compute_slippage(shares, open_px, "SELL", adv=adv)
    gross = round(shares * eff, 4)
    fee = round(gross * settings.simulated_cost_per_trade, 4)
    return {"shares": 0.0, "cash": round(cash + gross - fee, 4), "price": eff, "fee": fee, "slippage": slip}


def update_spy_200d_benchmark(run_date: str, spy_df: pd.DataFrame, market: str = "US") -> Optional[Dict[str, Any]]:
    """Advance the shadow benchmark to run_date and persist the day. Returns the saved row."""
    history = repository.get_benchmark_snapshots(BENCHMARK_NAME)
    if not history:
        day0 = repository.get_earliest_portfolio_snapshot(market)
        if day0 is None:
            return None
        capital = float(day0["total_value"])
        seed = {"decision_bar_date": None, "cash": capital, "spy_shares": 0.0, "spy_close": None, "sma_200": None,
                "regime_risk_on": None, "decision": "START", "pending_action": None, "equity": capital,
                "daily_return": 0.0, "cumulative_return": 0.0, "peak_equity": capital, "drawdown": 0.0,
                "details": {"start_date": day0["run_date"], "start_equity": capital, "source": "V1 day-0 snapshot"}}
        repository.save_benchmark_snapshot(BENCHMARK_NAME, day0["run_date"], seed)
        history = [{"run_date": day0["run_date"], **seed}]
    last = history[-1]
    if run_date <= last["run_date"]:
        return last                                      # already recorded (or before day 0)
    start = history[0]
    start_equity = float(start["details"]["start_equity"])

    bars = spy_df[spy_df["date"].astype(str) <= run_date].sort_values("date").reset_index(drop=True)
    decision_bars = bars[bars["date"].astype(str) < run_date]
    if bars.empty or decision_bars.empty:
        raise ValueError(f"No SPY bars on or before {run_date}")

    fill_rows = bars[bars["date"].astype(str) == run_date]
    fill_bar = fill_rows.iloc[0] if not fill_rows.empty else bars.iloc[-1]
    open_px = float(fill_bar["open"])
    adv = float(bars["volume"].tail(20).mean()) if "volume" in bars.columns else None

    cash, shares = float(last["cash"]), float(last["spy_shares"])
    executed = None
    if last["pending_action"] == "BUY" and shares == 0:
        f = _buy_all(cash, open_px, adv)
        cash, shares, executed = f["cash"], f["shares"], {"action": "BUY", **f, "fill_bar": str(fill_bar["date"])}
    elif last["pending_action"] == "SELL" and shares > 0:
        f = _sell_all(shares, cash, open_px, adv)
        cash, shares, executed = f["cash"], f["shares"], {"action": "SELL", **f, "fill_bar": str(fill_bar["date"])}

    window = int(settings.regime_ma_window)
    closes = decision_bars["close"].astype(float)
    decision_close = float(closes.iloc[-1])
    if len(closes) >= window:
        sma = float(closes.tail(window).mean())
        risk_on: Optional[bool] = decision_close >= sma
    else:
        sma, risk_on = None, None
    if risk_on is None:
        pending, decision = None, "INSUFFICIENT_HISTORY_HOLD"
    elif risk_on and shares == 0:
        pending, decision = "BUY", "BUY_NEXT_OPEN"
    elif not risk_on and shares > 0:
        pending, decision = "SELL", "SELL_NEXT_OPEN"
    else:
        pending, decision = None, "HOLD_SPY" if shares > 0 else "HOLD_CASH"

    mark = float(bars["close"].astype(float).iloc[-1])
    equity = round(cash + shares * mark, 4)
    peak = max(float(last["peak_equity"]), equity)
    row = {
        "decision_bar_date": str(decision_bars["date"].iloc[-1]),
        "cash": cash, "spy_shares": shares, "spy_close": mark,
        "sma_200": round(sma, 4) if sma is not None else None,
        "regime_risk_on": risk_on, "decision": decision, "pending_action": pending, "equity": equity,
        "daily_return": round(equity / float(last["equity"]) - 1.0, 6) if float(last["equity"]) > 0 else 0.0,
        "cumulative_return": round(equity / start_equity - 1.0, 6),
        "peak_equity": round(peak, 4), "drawdown": round(equity / peak - 1.0, 6),
        "details": {"start_date": start["details"]["start_date"], "start_equity": start_equity,
                    "decision_close": decision_close, "executed": executed},
    }
    repository.save_benchmark_snapshot(BENCHMARK_NAME, run_date, row)
    return {"run_date": run_date, **row}
