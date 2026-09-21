"""
src/portfolio/intelligence.py — Portfolio Intelligence, Sector Analysis, and Structural What-If Simulator (V2.2 Wave 1).

PURPOSE:
--------
1. Sector Analysis: Deconstructs current portfolio holdings, candidate trading universe,
   and screener universe across 8 distinct economic sectors.
2. Diversification Score: Transparent, rule-based concentration metric using
   Sector/Position Herfindahl-Hirschman Index (HHI) and a 3-pillar explainable 0-100 rubric.
3. Structural What-If Simulator: In-memory structural portfolio composition analysis
   ("what happens to sector exposure and diversification if stock X is added/removed?").

CRITICAL BOUNDARIES:
--------------------
1. ZERO CHANGES TO RISK ENGINE OR TRADING EXECUTION: Purely informational/analytical.
2. NO RETURN/P&L PREDICTIONS: Structural composition math only (weights, sectors, HHI).
3. SINGLE SOURCE OF TRUTH: Reuses sector metadata from src.screening.universe.
4. EXACT ALLOCATION SIZING PARITY: Default simulated buy allocation uses the exact
   formula from settings: (total_equity * (1 - cash_reserve)) / max_positions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import settings
from src.screening.universe import (
    SCREENER_UNIVERSE,
    ScreenerCompany,
    get_company_meta,
)

logger = logging.getLogger(__name__)

SECTOR_CONCENTRATION_LIMIT_PCT: float = 40.0
STRUCTURAL_DISCLAIMER: str = (
    "STRUCTURAL COMPOSITION ONLY: Analyzes portfolio weight and sector diversification impact. "
    "Does not predict returns, forecast P&L, or execute trades."
)


def get_ticker_sector(ticker: str) -> str:
    """Returns the sector for a given ticker from screener universe metadata."""
    meta = get_company_meta(ticker)
    return meta.sector


def compute_default_allocation_amount(
    total_equity: float,
    current_cash: Optional[float] = None,
) -> float:
    """
    Computes default position sizing matching src/risk/risk_engine.py and settings.

    Formula:
      target_slot_size = (total_equity * (1.0 - settings.cash_reserve)) / float(settings.max_positions)
      spendable_cash = max(0.0, current_cash - (total_equity * settings.cash_reserve))
      allocated_amount = min(target_slot_size, spendable_cash)
    """
    target_slot_size = (total_equity * (1.0 - float(settings.cash_reserve))) / float(settings.max_positions)
    if current_cash is not None:
        cash_floor = total_equity * float(settings.cash_reserve)
        spendable = max(0.0, current_cash - cash_floor)
        return round(float(min(target_slot_size, spendable)), 2)
    return round(float(target_slot_size), 2)


def compute_sector_breakdown(
    positions: Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]],
    cash: float,
    total_equity: float,
) -> Dict[str, Any]:
    """
    Computes sector exposure breakdown for active portfolio holdings.
    Also returns cross-sectional sector breakdown for trading and screener universes.
    """
    pos_list: List[Dict[str, Any]] = []
    if isinstance(positions, dict):
        for tkr, pdata in positions.items():
            qty = float(pdata.get("quantity", pdata.get("shares", 0.0)))
            if qty > 0:
                price = float(pdata.get("current_price", pdata.get("price", pdata.get("avg_cost", 0.0))))
                mv = float(pdata.get("market_value", qty * price))
                pos_list.append({
                    "ticker": tkr.upper(),
                    "shares": qty,
                    "price": price,
                    "market_value": mv,
                })
    else:
        for p in positions:
            mv = float(p.get("market_value", float(p.get("shares", 0.0)) * float(p.get("current_price", 0.0))))
            pos_list.append({
                "ticker": str(p.get("ticker", "")).upper(),
                "shares": float(p.get("shares", 0.0)),
                "price": float(p.get("current_price", p.get("price", 0.0))),
                "market_value": mv,
            })

    total_eq = float(total_equity) if total_equity > 0 else 10000.0
    invested_equity = sum(p["market_value"] for p in pos_list)
    cash_val = float(cash)

    # 1. Holdings Sector Breakdown
    sector_mv: Dict[str, float] = {}
    sector_tickers: Dict[str, List[str]] = {}

    for p in pos_list:
        sec = get_ticker_sector(p["ticker"])
        sector_mv[sec] = sector_mv.get(sec, 0.0) + p["market_value"]
        if sec not in sector_tickers:
            sector_tickers[sec] = []
        sector_tickers[sec].append(p["ticker"])

    holdings_sectors: List[Dict[str, Any]] = []
    concentration_warning = False
    top_sector_name = None
    top_sector_pct = 0.0

    for sec, mv in sorted(sector_mv.items(), key=lambda x: x[1], reverse=True):
        pct_portfolio = round((mv / total_eq) * 100.0, 2)
        pct_invested = round((mv / invested_equity) * 100.0, 2) if invested_equity > 0 else 0.0
        if pct_portfolio > top_sector_pct:
            top_sector_pct = pct_portfolio
            top_sector_name = sec
        if pct_portfolio > SECTOR_CONCENTRATION_LIMIT_PCT:
            concentration_warning = True

        holdings_sectors.append({
            "sector": sec,
            "market_value": round(mv, 2),
            "weight_portfolio_pct": pct_portfolio,
            "weight_invested_pct": pct_invested,
            "tickers": sector_tickers[sec],
            "position_count": len(sector_tickers[sec]),
        })

    cash_reserve_pct = round((cash_val / total_eq) * 100.0, 2)

    # 2. Trading Universe Sector Distribution (10 tickers)
    trading_universe_sectors: Dict[str, int] = {}
    for t in settings.ticker_list:
        sec = get_ticker_sector(t)
        trading_universe_sectors[sec] = trading_universe_sectors.get(sec, 0) + 1

    # 3. Screener Universe Sector Distribution (45 tickers)
    screener_universe_sectors: Dict[str, int] = {}
    for sc in SCREENER_UNIVERSE:
        screener_universe_sectors[sc.sector] = screener_universe_sectors.get(sc.sector, 0) + 1

    return {
        "total_equity": round(total_eq, 2),
        "invested_equity": round(invested_equity, 2),
        "cash": round(cash_val, 2),
        "cash_reserve_pct": cash_reserve_pct,
        "active_positions_count": len(pos_list),
        "holdings_sectors": holdings_sectors,
        "top_sector": top_sector_name,
        "top_sector_pct": top_sector_pct,
        "concentration_warning": concentration_warning,
        "concentration_limit_pct": SECTOR_CONCENTRATION_LIMIT_PCT,
        "trading_universe_sectors": trading_universe_sectors,
        "screener_universe_sectors": screener_universe_sectors,
    }


def compute_diversification_score(
    positions: Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]],
    cash: float,
    total_equity: float,
) -> Dict[str, Any]:
    """
    Computes rule-based Diversification Score (0-100) and Herfindahl-Hirschman Index (HHI).

    "Show Your Work" 3-Pillar Rubric:
      - Pillar 1: Sector Breadth (Max 40 pts) -> Distinct Sectors / Positions
      - Pillar 2: Sector Concentration HHI (Max 40 pts) -> Normalized HHI
      - Pillar 3: Position Balance HHI (Max 20 pts) -> Equal Weighting Parity
    """
    sector_data = compute_sector_breakdown(positions, cash, total_equity)
    num_positions = sector_data["active_positions_count"]
    holdings_sectors = sector_data["holdings_sectors"]
    invested_eq = sector_data["invested_equity"]

    # Special Case: 100% Cash Portfolio (Zero Active Equity Concentration Risk)
    if num_positions == 0 or invested_eq <= 0:
        return {
            "score": 100.0,
            "rating": "FULL_CASH_PRESERVATION",
            "rating_label": "100% Cash Reserve (Zero Equity Concentration)",
            "is_cash_reserve": True,
            "sector_hhi": 0.0,
            "position_hhi": 0.0,
            "effective_sectors": 0.0,
            "effective_positions": 0.0,
            "pillar_breakdown": {
                "sector_breadth_pts": 40.0,
                "sector_hhi_pts": 40.0,
                "position_balance_pts": 20.0,
            },
            "formula_explanation": (
                "Portfolio is held 100% in cash reserve. Zero equity concentration risk; "
                "full capital preservation mandate honored (§1.1)."
            ),
        }

    # 1. Sector HHI (on invested equity)
    sector_weights = [(s["market_value"] / invested_eq) for s in holdings_sectors]
    sector_hhi = sum(w ** 2 for w in sector_weights)
    effective_sectors = round(1.0 / sector_hhi, 2) if sector_hhi > 0 else 1.0

    # 2. Position HHI (on invested equity)
    pos_mvs: List[float] = []
    if isinstance(positions, dict):
        for p in positions.values():
            qty = float(p.get("quantity", p.get("shares", 0.0)))
            price = float(p.get("current_price", p.get("price", p.get("avg_cost", 0.0))))
            pos_mvs.append(qty * price)
    else:
        for p in positions:
            mv = float(p.get("market_value", float(p.get("shares", 0.0)) * float(p.get("current_price", 0.0))))
            pos_mvs.append(mv)

    pos_weights = [(mv / invested_eq) for mv in pos_mvs if mv > 0]
    position_hhi = sum(w ** 2 for w in pos_weights) if pos_weights else 1.0
    effective_positions = round(1.0 / position_hhi, 2) if position_hhi > 0 else 1.0

    # 3. Transparent 3-Pillar Scoring
    # Pillar 1: Sector Breadth (Max 40 pts)
    distinct_sectors = len(holdings_sectors)
    breadth_ratio = distinct_sectors / float(num_positions)
    pillar1_pts = round(40.0 * min(1.0, breadth_ratio), 1)

    # Pillar 2: Sector Concentration HHI (Max 40 pts)
    # Ideal HHI for K positions (up to 3 max slots) is 1/K. Worst is 1.0.
    min_possible_hhi = 1.0 / float(min(settings.max_positions, num_positions))
    if min_possible_hhi >= 1.0:
        pillar2_pts = 20.0  # Single position held
    else:
        hhi_score_ratio = (1.0 - sector_hhi) / (1.0 - min_possible_hhi)
        pillar2_pts = round(40.0 * max(0.0, min(1.0, hhi_score_ratio)), 1)

    # Pillar 3: Position Balance HHI (Max 20 pts)
    if min_possible_hhi >= 1.0:
        pillar3_pts = 10.0
    else:
        pos_score_ratio = (1.0 - position_hhi) / (1.0 - min_possible_hhi)
        pillar3_pts = round(20.0 * max(0.0, min(1.0, pos_score_ratio)), 1)

    total_score = round(pillar1_pts + pillar2_pts + pillar3_pts, 1)

    if total_score >= 80.0:
        rating = "HIGHLY_DIVERSIFIED"
        rating_label = "Optimal Cross-Sector Diversification"
    elif total_score >= 60.0:
        rating = "MODERATELY_DIVERSIFIED"
        rating_label = "Acceptable Diversification (Some Sector Clustering)"
    else:
        rating = "CONCENTRATED"
        rating_label = "Elevated Sector / Single-Stock Concentration"

    return {
        "score": total_score,
        "rating": rating,
        "rating_label": rating_label,
        "is_cash_reserve": False,
        "sector_hhi": round(sector_hhi, 4),
        "position_hhi": round(position_hhi, 4),
        "effective_sectors": effective_sectors,
        "effective_positions": effective_positions,
        "pillar_breakdown": {
            "sector_breadth_pts": pillar1_pts,
            "sector_hhi_pts": pillar2_pts,
            "position_balance_pts": pillar3_pts,
        },
        "formula_explanation": (
            f"Score = Breadth ({pillar1_pts}/40 pts) + Sector HHI ({pillar2_pts}/40 pts) + "
            f"Position Balance ({pillar3_pts}/20 pts) = {total_score}/100 pts. "
            f"HHI_sector: {sector_hhi:.3f} ({effective_sectors} effective sectors)."
        ),
    }


def simulate_what_if(
    positions: Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]],
    cash: float,
    total_equity: float,
    action: str,  # 'ADD' or 'REMOVE'
    ticker: str,
    simulated_amount: Optional[float] = None,
    simulated_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    STRUCTURAL Portfolio Composition What-If Simulator.

    Calculates in-memory structural delta:
    - Before vs. After sector weights
    - Before vs. After position count & cash %
    - Before vs. After Diversification Score & HHI

    STRICT GUARANTEES:
    - NEVER projects hypothetical returns or P&L.
    - Zero execution or mutation of real database state.
    - Uses exact settings sizing formula for ADD: (total_equity * (1 - cash_reserve)) / max_positions.
    """
    act = action.strip().upper()
    if act in ("ADD_POSITION", "ADD"):
        act = "ADD"
    elif act in ("REMOVE_POSITION", "REMOVE"):
        act = "REMOVE"

    t = ticker.strip().upper()
    total_eq = float(total_equity) if total_equity > 0 else 10000.0

    # 1. Deep copy current positions list
    current_pos: List[Dict[str, Any]] = []
    if isinstance(positions, dict):
        for tkr, p in positions.items():
            qty = float(p.get("quantity", p.get("shares", 0.0)))
            if qty > 0:
                price = float(p.get("current_price", p.get("price", p.get("avg_cost", 100.0))))
                mv = float(p.get("market_value", qty * price))
                current_pos.append({
                    "ticker": tkr.upper(),
                    "shares": qty,
                    "current_price": price,
                    "market_value": mv,
                })
    else:
        for p in positions:
            mv = float(p.get("market_value", float(p.get("shares", 0.0)) * float(p.get("current_price", 100.0))))
            current_pos.append({
                "ticker": str(p.get("ticker", "")).upper(),
                "shares": float(p.get("shares", 0.0)),
                "current_price": float(p.get("current_price", p.get("price", 100.0))),
                "market_value": mv,
            })

    current_cash = float(cash)

    # Calculate BEFORE baseline
    before_breakdown = compute_sector_breakdown(current_pos, current_cash, total_eq)
    before_diversification = compute_diversification_score(current_pos, current_cash, total_eq)

    simulated_pos = [dict(p) for p in current_pos]
    simulated_cash = current_cash
    warnings: List[str] = []

    if act == "ADD":
        # Check if already held
        existing_idx = next((i for i, p in enumerate(simulated_pos) if p["ticker"] == t), None)
        if existing_idx is not None:
            warnings.append(f"Ticker {t} is already held in portfolio. Sizing addition on top of existing slot.")

        # Exact settings default allocation formula
        alloc_val = (
            float(simulated_amount)
            if simulated_amount is not None and simulated_amount > 0
            else compute_default_allocation_amount(total_eq, current_cash)
        )

        # Check cash reserve ceiling
        cash_floor = total_eq * float(settings.cash_reserve)
        max_spendable = max(0.0, current_cash - cash_floor)
        if alloc_val > max_spendable:
            warnings.append(
                f"Simulated allocation (${alloc_val:.2f}) exceeds spendable cash reserve (${max_spendable:.2f}). "
                f"Capped to maintain statutory {settings.cash_reserve * 100:.0f}% cash floor."
            )
            alloc_val = max_spendable

        if alloc_val <= 0:
            warnings.append("Insufficient cash to open new simulated position without violating cash reserve floor.")

        price_val = float(simulated_price) if simulated_price is not None and simulated_price > 0 else 100.0
        shares_val = round(alloc_val / price_val, 4) if price_val > 0 else 0.0

        if existing_idx is not None:
            simulated_pos[existing_idx]["shares"] += shares_val
            simulated_pos[existing_idx]["market_value"] += alloc_val
        else:
            simulated_pos.append({
                "ticker": t,
                "shares": shares_val,
                "current_price": price_val,
                "market_value": alloc_val,
            })

        simulated_cash = max(0.0, round(simulated_cash - alloc_val, 2))

        # Position cap warning
        if len(simulated_pos) > settings.max_positions:
            warnings.append(
                f"Adding {t} brings position count to {len(simulated_pos)}, exceeding "
                f"the system's {settings.max_positions}-position maximum cap (§1.4)."
            )

    elif act == "REMOVE":
        existing = next((p for p in simulated_pos if p["ticker"] == t), None)
        if existing is None:
            warnings.append(f"Cannot remove {t}: ticker is not currently held in portfolio.")
        else:
            simulated_cash = round(simulated_cash + existing["market_value"], 2)
            simulated_pos = [p for p in simulated_pos if p["ticker"] != t]
    else:
        warnings.append(f"Invalid simulation action '{act}'. Use 'ADD' or 'REMOVE'.")

    # Calculate AFTER metrics
    after_breakdown = compute_sector_breakdown(simulated_pos, simulated_cash, total_eq)
    after_diversification = compute_diversification_score(simulated_pos, simulated_cash, total_eq)

    # Sector Delta comparison
    all_sectors = sorted(list(set(
        [s["sector"] for s in before_breakdown["holdings_sectors"]] +
        [s["sector"] for s in after_breakdown["holdings_sectors"]]
    )))

    before_sec_map = {s["sector"]: s["weight_portfolio_pct"] for s in before_breakdown["holdings_sectors"]}
    after_sec_map = {s["sector"]: s["weight_portfolio_pct"] for s in after_breakdown["holdings_sectors"]}

    sector_deltas = []
    for s_name in all_sectors:
        w_before = before_sec_map.get(s_name, 0.0)
        w_after = after_sec_map.get(s_name, 0.0)
        diff = round(w_after - w_before, 2)
        sector_deltas.append({
            "sector": s_name,
            "weight_before_pct": w_before,
            "weight_after_pct": w_after,
            "delta_pct": diff,
        })

    score_delta = round(after_diversification["score"] - before_diversification["score"], 1)

    return {
        "action": act,
        "ticker": t,
        "simulated_sector": get_ticker_sector(t),
        "disclaimer": STRUCTURAL_DISCLAIMER,
        "before": {
            "positions_count": before_breakdown["active_positions_count"],
            "cash": before_breakdown["cash"],
            "cash_reserve_pct": before_breakdown["cash_reserve_pct"],
            "diversification_score": before_diversification["score"],
            "sector_hhi": before_diversification["sector_hhi"],
            "effective_sectors": before_diversification["effective_sectors"],
            "holdings_sectors": before_breakdown["holdings_sectors"],
        },
        "after": {
            "positions_count": after_breakdown["active_positions_count"],
            "cash": after_breakdown["cash"],
            "cash_reserve_pct": after_breakdown["cash_reserve_pct"],
            "diversification_score": after_diversification["score"],
            "sector_hhi": after_diversification["sector_hhi"],
            "effective_sectors": after_diversification["effective_sectors"],
            "holdings_sectors": after_breakdown["holdings_sectors"],
        },
        "deltas": {
            "score_delta": score_delta,
            "positions_count_delta": after_breakdown["active_positions_count"] - before_breakdown["active_positions_count"],
            "cash_delta": round(after_breakdown["cash"] - before_breakdown["cash"], 2),
            "cash_reserve_delta_pct": round(after_breakdown["cash_reserve_pct"] - before_breakdown["cash_reserve_pct"], 2),
            "sector_deltas": sector_deltas,
        },
        "warnings": warnings,
    }
