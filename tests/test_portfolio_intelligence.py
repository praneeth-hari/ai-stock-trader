"""
tests/test_portfolio_intelligence.py — Unit, Sizing Parity, and Isolation Tests for Portfolio Intelligence.

Verifies:
1. Sector Coverage Check: ALL 10 trading-universe tickers have valid sector metadata.
2. Sizing Formula Parity Test: Simulator default sizing exactly matches settings and risk_engine.py formula.
3. Diversification Score: HHI, effective sectors, and 3-pillar explainable rubric.
4. What-If Structural Simulator: Exact Before vs. After delta math with zero P&L / return forecasting.
5. Strict Isolation: Zero database mutation and zero broker imports.
"""

import ast
import hashlib
from pathlib import Path
from typing import Dict
import numpy as np
import pytest

from config.settings import settings
from src.portfolio.intelligence import (
    SECTOR_CONCENTRATION_LIMIT_PCT,
    STRUCTURAL_DISCLAIMER,
    compute_default_allocation_amount,
    compute_diversification_score,
    compute_sector_breakdown,
    get_ticker_sector,
    simulate_what_if,
)
from src.screening.universe import get_company_meta


# ── Test 1: Sector Coverage Check (Mandatory User Fix 1) ───────────────────────

def test_sector_coverage_all_trading_universe_tickers():
    """
    CRITICAL CHECK: ALL 10 trading-universe tickers must have valid sector metadata.
    Specifically checks META has valid non-'Other' sector mapping.
    """
    required_10 = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM", "V", "UNH"]
    for t in required_10:
        assert t in settings.ticker_list

    for tkr in required_10:
        meta = get_company_meta(tkr)
        assert meta.ticker == tkr
        assert meta.sector != "Other", f"Ticker {tkr} returned generic 'Other' sector fallback!"
        assert len(meta.sector) > 2

    # Specific check for META
    meta_info = get_company_meta("META")
    assert meta_info.sector in ("Communications", "Communication Services", "Technology")
    assert meta_info.name == "Meta Platforms Inc."


# ── Test 2: Sizing Formula Parity Test (Mandatory User Fix 2) ──────────────────

def test_what_if_default_sizing_formula_exact_parity():
    """
    CRITICAL CHECK: Simulator default allocation sizing must exactly match
    the real formula from settings and src/risk/risk_engine.py:
      target_slot_size = (total_equity * (1.0 - settings.cash_reserve)) / float(settings.max_positions)
    """
    total_equity = 10000.0
    cash = 10000.0

    # 1. Expected theoretical sizing from settings
    expected_target = (total_equity * (1.0 - settings.cash_reserve)) / float(settings.max_positions)
    assert expected_target == (10000.0 * 0.85) / 3.0  # Exactly $2,833.3333...

    # 2. Simulator's sizing calculation
    sim_size = compute_default_allocation_amount(total_equity=total_equity, current_cash=cash)
    assert round(sim_size, 2) == round(expected_target, 2)
    assert sim_size == 2833.33

    # 3. Spendable cash constraint check
    tight_cash = 2000.0
    cash_floor = total_equity * settings.cash_reserve  # $1,500
    spendable = tight_cash - cash_floor                # $500
    constrained_size = compute_default_allocation_amount(total_equity=total_equity, current_cash=tight_cash)
    assert constrained_size == round(spendable, 2)
    assert constrained_size == 500.0


# ── Test 3: Sector Breakdown & Diversification Scoring ─────────────────────────

def test_diversification_score_100_percent_cash():
    """Empty portfolio (100% cash) must be cleanly scored as capital preservation."""
    res = compute_diversification_score(positions=[], cash=10000.0, total_equity=10000.0)
    assert res["is_cash_reserve"] is True
    assert res["score"] == 100.0
    assert res["rating"] == "FULL_CASH_PRESERVATION"
    assert res["sector_hhi"] == 0.0


def test_diversification_score_optimal_3_sectors():
    """3 equal positions in 3 different sectors should score near optimal (~100 pts)."""
    positions = [
        {"ticker": "AAPL", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},  # Tech
        {"ticker": "JPM", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},   # Financials
        {"ticker": "UNH", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},   # Healthcare
    ]
    res = compute_diversification_score(positions=positions, cash=7000.0, total_equity=10000.0)
    assert res["is_cash_reserve"] is False
    assert res["score"] >= 95.0
    assert res["rating"] == "HIGHLY_DIVERSIFIED"
    assert abs(res["sector_hhi"] - 0.3333) < 0.01
    assert abs(res["effective_sectors"] - 3.0) < 0.1


def test_diversification_score_concentrated_all_same_sector():
    """3 positions all in Technology should score low (HHI = 1.0)."""
    positions = [
        {"ticker": "AAPL", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},   # Tech
        {"ticker": "MSFT", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},   # Tech
        {"ticker": "NVDA", "shares": 10.0, "current_price": 100.0, "market_value": 1000.0},   # Tech
    ]
    res = compute_diversification_score(positions=positions, cash=7000.0, total_equity=10000.0)
    assert res["sector_hhi"] == 1.0
    assert res["effective_sectors"] == 1.0
    assert res["score"] < 50.0
    assert res["rating"] == "CONCENTRATED"


# ── Test 4: Structural What-If Simulator Math & Bounds ─────────────────────────

def test_what_if_simulator_add_position():
    """Adding a position shifts cash and sector weights accurately with zero return forecasts."""
    current_positions = [
        {"ticker": "JPM", "shares": 10.0, "current_price": 200.0, "market_value": 2000.0},  # Financials
    ]
    sim = simulate_what_if(
        positions=current_positions,
        cash=8000.0,
        total_equity=10000.0,
        action="ADD",
        ticker="AAPL",  # Technology
    )
    assert sim["action"] == "ADD"
    assert sim["ticker"] == "AAPL"
    assert sim["simulated_sector"] == "Technology"
    assert STRUCTURAL_DISCLAIMER in sim["disclaimer"]

    # Positions increased from 1 to 2
    assert sim["before"]["positions_count"] == 1
    assert sim["after"]["positions_count"] == 2
    assert sim["deltas"]["positions_count_delta"] == 1

    # Cash decreased by default allocation amount ($2,833.33)
    assert sim["after"]["cash"] == round(8000.0 - 2833.33, 2)

    # Sector deltas present
    sec_names = [d["sector"] for d in sim["deltas"]["sector_deltas"]]
    assert "Financials" in sec_names
    assert "Technology" in sec_names


def test_what_if_simulator_remove_position():
    """Removing a position returns proceeds to cash and eliminates the sector."""
    current_positions = [
        {"ticker": "AAPL", "shares": 10.0, "current_price": 150.0, "market_value": 1500.0},
        {"ticker": "JPM", "shares": 10.0, "current_price": 150.0, "market_value": 1500.0},
    ]
    sim = simulate_what_if(
        positions=current_positions,
        cash=7000.0,
        total_equity=10000.0,
        action="REMOVE",
        ticker="AAPL",
    )
    assert sim["before"]["positions_count"] == 2
    assert sim["after"]["positions_count"] == 1
    assert sim["after"]["cash"] == 8500.0  # 7000 + 1500


def test_what_if_exceeding_max_positions_warns():
    """Simulating adding a 4th position must flag max positions warning (§1.4)."""
    current_positions = [
        {"ticker": "AAPL", "shares": 5.0, "current_price": 100.0, "market_value": 500.0},
        {"ticker": "JPM", "shares": 5.0, "current_price": 100.0, "market_value": 500.0},
        {"ticker": "UNH", "shares": 5.0, "current_price": 100.0, "market_value": 500.0},
    ]
    sim = simulate_what_if(
        positions=current_positions,
        cash=8500.0,
        total_equity=10000.0,
        action="ADD",
        ticker="AMZN",
    )
    assert any("exceeding" in w.lower() for w in sim["warnings"])


# ── Test 5: Strict Architectural Isolation ─────────────────────────────────────

def test_intelligence_never_imports_broker_or_mutates_db():
    """
    CRITICAL ARCHITECTURAL ISOLATION TEST:
    Verifies that src/portfolio/intelligence.py contains zero imports of
    paper_broker or trade execution, and executing simulations never alters SQLite.
    """
    code_path = Path("src/portfolio/intelligence.py")
    assert code_path.exists()

    with open(code_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(code_path))

    forbidden_imports = ("paper_broker", "execute_order", "place_order")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forb in forbidden_imports:
                    assert forb not in alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forb in forbidden_imports:
                    assert forb not in node.module

    # Verify SQLite database hash remains identical before and after simulations
    db_file = Path("data/processed/trader.db")
    if db_file.exists():
        h_before = hashlib.sha256(db_file.read_bytes()).hexdigest()
        for tkr in ("AAPL", "MSFT", "NVDA", "JPM"):
            _ = simulate_what_if([], 10000.0, 10000.0, "ADD", tkr)
            _ = simulate_what_if([], 10000.0, 10000.0, "REMOVE", tkr)
        h_after = hashlib.sha256(db_file.read_bytes()).hexdigest()
        assert h_before == h_after, "CRITICAL: Database was modified by what-if simulations!"

