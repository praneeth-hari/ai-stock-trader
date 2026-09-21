"""
src/trading/leaderboard.py — Section 10 Item 1: Paper Trading Leaderboard.

PURPOSE
-------
Simulate and track multiple strategy variants simultaneously on paper money.
Ranks strategies by total portfolio return and displays a comparative leaderboard.

PRESET STRATEGIES
-----------------
1. "Conservative": buy_bar=0.67, trailing_stop=0.08, max_positions=2, sizing="confidence"
2. "Balanced":     buy_bar=0.60, trailing_stop=0.08, max_positions=3, sizing="confidence"
3. "Aggressive":   buy_bar=0.60, trailing_stop=0.05, max_positions=3, sizing="full"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.db import repository

logger = logging.getLogger(__name__)

DEFAULT_STARTING_CAPITAL: float = 10000.0

DEFAULT_STRATEGIES: List[Dict[str, Any]] = [
    {
        "name": "Conservative",
        "settings": {
            "buy_bar": 0.75,
            "trailing_stop_pct": 0.08,
            "max_positions": 2,
            "position_sizing": "confidence_based",
            "take_profit_pct": 0.15,
        },
    },
    {
        "name": "Balanced",
        "settings": {
            "buy_bar": 0.60,
            "trailing_stop_pct": 0.08,
            "max_positions": 3,
            "position_sizing": "confidence_based",
            "take_profit_pct": 0.15,
        },
    },
    {
        "name": "Aggressive",
        "settings": {
            "buy_bar": 0.60,
            "trailing_stop_pct": 0.05,
            "max_positions": 3,
            "position_sizing": "full_always",
            "take_profit_pct": 0.15,
        },
    },
]


def init_default_strategies(created_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Seeds the default 3 strategy variants into the DB if none exist yet.
    Returns list of strategy variant dicts.
    """
    repository.create_all_tables()
    existing = repository.get_strategy_variants(only_active=False)
    if existing:
        return existing

    today_str = created_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    created = []
    for strat in DEFAULT_STRATEGIES:
        sid = repository.save_strategy_variant(
            name=strat["name"],
            settings_dict=strat["settings"],
            starting_capital=DEFAULT_STARTING_CAPITAL,
            created_date=today_str,
        )
        # Seed initial snapshot ($10,000 cash, 0 positions)
        repository.save_strategy_snapshot(
            strategy_id=sid,
            date_str=today_str,
            portfolio_value=DEFAULT_STARTING_CAPITAL,
            cash=DEFAULT_STARTING_CAPITAL,
            positions={},
            daily_return=0.0,
        )
        created.append({
            "id": sid,
            "name": strat["name"],
            "settings": strat["settings"],
            "starting_capital": DEFAULT_STARTING_CAPITAL,
            "created_date": today_str,
            "is_active": True,
        })
    logger.info("Initialized %d default strategy variants in DB.", len(created))
    return repository.get_strategy_variants(only_active=True)


def run_leaderboard_cycle(
    run_date: Optional[str] = None,
    prices: Optional[Dict[str, float]] = None,
    predictions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates all active strategy variants against current predictions and prices,
    executes virtual fills, records trades, and updates daily portfolio snapshots.
    """
    repository.create_all_tables()
    today_str = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    variants = init_default_strategies(created_date=today_str)

    # Load latest market prices if not supplied
    if prices is None:
        prices = {}
        for ticker in settings.ticker_list:
            df_m = repository.get_market_data(ticker, end_date=today_str)
            if not df_m.empty and "close" in df_m.columns:
                prices[ticker] = float(df_m["close"].iloc[-1])

    # Load latest model predictions if not supplied
    if predictions is None:
        predictions = {}
        for ticker in settings.ticker_list:
            df_p = repository.get_predictions(ticker, today_str, today_str)
            if not df_p.empty and "probability" in df_p.columns:
                predictions[ticker] = float(df_p["probability"].iloc[-1])

    cycle_results = {}

    for v in variants:
        sid = v["id"]
        name = v["name"]
        st_config = v["settings"]
        start_cap = float(v.get("starting_capital", DEFAULT_STARTING_CAPITAL))

        # Get latest snapshot for this strategy
        snaps = repository.get_strategy_snapshots(strategy_id=sid)
        if snaps:
            last_snap = snaps[-1]
            cash = float(last_snap["cash"])
            positions = dict(last_snap["positions"])
            prev_val = float(last_snap["portfolio_value"])
        else:
            cash = start_cap
            positions = {}
            prev_val = start_cap

        buy_bar = float(st_config.get("buy_bar", 0.60))
        trailing_stop_pct = float(st_config.get("trailing_stop_pct", 0.08))
        take_profit_pct = float(st_config.get("take_profit_pct", 0.15))
        max_pos = int(st_config.get("max_positions", 3))
        sizing_mode = st_config.get("position_sizing", "confidence")

        # 1. Update mark-to-market positions & check exits
        new_positions = {}
        realized_pnl = 0.0

        for ticker, pos in positions.items():
            curr_price = prices.get(ticker, pos["entry_price"])
            shares = float(pos["shares"])
            entry_p = float(pos["entry_price"])
            peak_p = max(float(pos.get("peak_price", entry_p)), curr_price)
            prob = predictions.get(ticker, 0.50)

            ret_from_entry = (curr_price - entry_p) / entry_p if entry_p > 0 else 0.0
            ret_from_peak = (curr_price - peak_p) / peak_p if peak_p > 0 else 0.0

            should_exit = False
            exit_reason = ""

            # Stop loss (-8%)
            if ret_from_entry <= -0.08:
                should_exit = True
                exit_reason = "STOP_LOSS"
            # Trailing stop
            elif ret_from_peak <= -trailing_stop_pct:
                should_exit = True
                exit_reason = "TRAILING_STOP"
            # Take profit (+15%)
            elif ret_from_entry >= take_profit_pct:
                should_exit = True
                exit_reason = "TAKE_PROFIT"
            # Signal exit (< 0.45)
            elif prob < 0.45:
                should_exit = True
                exit_reason = "SIGNAL_EXIT"

            if should_exit:
                trade_pnl = (curr_price - entry_p) * shares
                cash += shares * curr_price
                realized_pnl += trade_pnl
                repository.save_strategy_trade(
                    strategy_id=sid,
                    date_str=today_str,
                    ticker=ticker,
                    action="SELL",
                    price=curr_price,
                    shares=shares,
                    pnl=trade_pnl,
                )
                logger.info("Strategy '%s' SELL %s (%s, pnl=+$%.2f)", name, ticker, exit_reason, trade_pnl)
            else:
                new_positions[ticker] = {
                    "shares": shares,
                    "entry_price": entry_p,
                    "peak_price": peak_p,
                    "current_price": curr_price,
                }

        # 2. Check BUY opportunities
        open_count = len(new_positions)
        if open_count < max_pos and predictions:
            # Sort candidates by probability descending
            cand_pairs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            for ticker, prob in cand_pairs:
                if open_count >= max_pos:
                    break
                if ticker in new_positions:
                    continue
                if prob < buy_bar:
                    continue

                curr_price = prices.get(ticker)
                if not curr_price or curr_price <= 0:
                    continue

                # Sizing math
                if sizing_mode == "full":
                    alloc_mult = 1.0
                else:
                    # Confidence scaling
                    alloc_mult = 0.50 if prob < 0.65 else (0.75 if prob < 0.75 else 1.0)

                target_alloc = (start_cap / max_pos) * alloc_mult
                avail_cash = cash * 0.85  # keep 15% cash buffer
                buy_dollars = min(target_alloc, avail_cash)

                if buy_dollars >= settings.min_trade_size:
                    shares = buy_dollars / curr_price
                    cash -= buy_dollars
                    new_positions[ticker] = {
                        "shares": shares,
                        "entry_price": curr_price,
                        "peak_price": curr_price,
                        "current_price": curr_price,
                    }
                    open_count += 1
                    repository.save_strategy_trade(
                        strategy_id=sid,
                        date_str=today_str,
                        ticker=ticker,
                        action="BUY",
                        price=curr_price,
                        shares=shares,
                        pnl=0.0,
                    )
                    logger.info("Strategy '%s' BUY %s (prob=%.3f, shares=%.2f)", name, ticker, prob, shares)

        # 3. Mark to market portfolio value
        positions_val = sum(
            pos["shares"] * prices.get(tk, pos["entry_price"])
            for tk, pos in new_positions.items()
        )
        total_val = cash + positions_val
        daily_ret = (total_val - prev_val) / prev_val if prev_val > 0 else 0.0

        repository.save_strategy_snapshot(
            strategy_id=sid,
            date_str=today_str,
            portfolio_value=total_val,
            cash=cash,
            positions=new_positions,
            daily_return=daily_ret,
        )

        cycle_results[name] = {
            "strategy_id": sid,
            "portfolio_value": round(total_val, 2),
            "cash": round(cash, 2),
            "positions_count": len(new_positions),
            "daily_return": round(daily_ret, 4),
        }

    return cycle_results


def get_leaderboard_rankings() -> List[Dict[str, Any]]:
    """
    Computes leaderboard summary rankings for all active strategy variants.
    Ranks by Return % descending. Adds 'trophy': '🏆' to #1.
    """
    repository.create_all_tables()
    variants = init_default_strategies()
    rankings = []

    for v in variants:
        sid = v["id"]
        name = v["name"]
        start_cap = float(v.get("starting_capital", DEFAULT_STARTING_CAPITAL))

        snaps = repository.get_strategy_snapshots(strategy_id=sid)
        trades = repository.get_strategy_trades(strategy_id=sid)

        curr_val = float(snaps[-1]["portfolio_value"]) if snaps else start_cap
        ret_pct = ((curr_val - start_cap) / start_cap) * 100.0

        closed_trades = [t for t in trades if t["action"] == "SELL"]
        n_trades = len(trades)
        n_wins = sum(1 for t in closed_trades if t["pnl"] > 0)
        win_rate = (n_wins / len(closed_trades) * 100.0) if closed_trades else 0.0

        rankings.append({
            "strategy_id": sid,
            "name": name,
            "starting_capital": start_cap,
            "portfolio_value": round(curr_val, 2),
            "return_pct": round(ret_pct, 2),
            "win_rate_pct": round(win_rate, 1),
            "trades_count": n_trades,
            "settings": v["settings"],
        })

    # Sort descending by return_pct
    rankings.sort(key=lambda x: x["return_pct"], reverse=True)

    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx
        item["is_winning"] = (idx == 1)
        item["trophy"] = "🏆" if idx == 1 else ""

    return rankings
