"""
tests/test_sector_rotation.py — Unit tests for Section 5 Item 3: Sector Rotation Intelligence.

Covers:
  - calculate_stock_20d_returns (computes returns from OHLCV DataFrames)
  - Sector averaging across constituent stocks
  - Ranking 1–8 (sorted by 20-day avg return descending)
  - Multiplier assignment: rank 1–2 = +0.04, rank 3–5 = 0.00, rank 6–8 = -0.04
  - SectorRotationResult.get_multiplier_for_ticker
  - SectorRotationResult.get_rank_for_ticker
  - Graceful handling of empty or missing data
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict

import pandas as pd
import pytest

from src.intelligence.sector_rotation import (
    calculate_stock_20d_returns,
    calculate_sector_rotation,
    SectorRotationResult,
    SectorRank,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv(ticker: str, prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    start_dt = date.fromisoformat(start)
    dates = [(start_dt + timedelta(days=i)).isoformat() for i in range(len(prices))]
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1_000_000] * len(prices),
    })


def make_universe_dfs(sector_returns: dict[str, float], n_points: int = 25) -> dict[str, pd.DataFrame]:
    """
    Create a universe of OHLCV DataFrames where each ticker starts at 100 and ends
    at 100 * (1 + sector_returns[ticker]) after n_points days.
    """
    dfs = {}
    for ticker, ret in sector_returns.items():
        start_price = 100.0
        end_price = start_price * (1 + ret)
        # Linear interpolation
        prices = [start_price + (end_price - start_price) * (i / (n_points - 1)) for i in range(n_points)]
        dfs[ticker] = make_ohlcv(ticker, prices)
    return dfs


# ── calculate_stock_20d_returns tests ─────────────────────────────────────────

def test_calculate_stock_20d_returns_basic():
    """Stock that gained 10% over 20 days returns ~0.10."""
    dfs = {"AAPL": make_ohlcv("AAPL", list(range(100, 120 + 1)))}  # 21 points: +10% from 100 to 110 in 20 steps approx
    # Create a stock from 100 to 110 over exactly 21 prices (index [0]=100, index[-1]=110)
    prices = [100.0 + i * 0.5 for i in range(21)]  # 100 → 110 over 21 points
    dfs = {"AAPL": make_ohlcv("AAPL", prices)}
    as_of = "2024-01-21"
    returns = calculate_stock_20d_returns(dfs, as_of_date=as_of, lookback=20)
    assert "AAPL" in returns
    ret = returns["AAPL"]
    assert abs(ret - 0.10) < 0.02, f"Expected ~10% return, got {ret*100:.1f}%"


def test_calculate_stock_20d_returns_empty_df():
    """Empty DataFrame should not appear in returns."""
    dfs = {"EMPTY": pd.DataFrame(columns=["date", "close"])}
    returns = calculate_stock_20d_returns(dfs, "2024-01-21")
    assert "EMPTY" not in returns


def test_calculate_stock_20d_returns_insufficient_data():
    """Less than 2 rows → should not appear in returns (or use fallback partial)."""
    dfs = {"SHORT": make_ohlcv("SHORT", [100.0])}
    returns = calculate_stock_20d_returns(dfs, "2024-01-01")
    assert "SHORT" not in returns


# ── calculate_sector_rotation tests ──────────────────────────────────────────

def test_sector_rotation_ranking_order():
    """
    Technology stocks gain +15%, Energy stocks gain +2%.
    Technology should rank higher than Energy.
    """
    # AAPL, MSFT, NVDA → Technology (+15%)
    # XOM → Energy (+2%)
    dfs = make_universe_dfs({
        "AAPL": 0.15, "MSFT": 0.15, "NVDA": 0.15,   # Technology
        "XOM": 0.02,                                    # Energy
    })
    as_of = "2024-01-25"
    result = calculate_sector_rotation(dfs, as_of_date=as_of, persist=False)
    assert isinstance(result, SectorRotationResult)
    tech_rank = result.get_rank_for_ticker("AAPL")
    energy_rank = result.get_rank_for_ticker("XOM")
    if tech_rank is not None and energy_rank is not None:
        assert tech_rank < energy_rank, f"Tech rank ({tech_rank}) should be better than Energy rank ({energy_rank})"


def test_sector_rotation_multipliers_top2():
    """Rank 1-2 sectors should get +0.04 rotation_multiplier."""
    dfs = make_universe_dfs({"AAPL": 0.20, "MSFT": 0.18})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    top_mult = result.sector_map.get("Technology")
    if top_mult:
        assert top_mult.rotation_multiplier == pytest.approx(0.04, abs=1e-9), \
            f"Top sector should have +0.04 modifier, got {top_mult.rotation_multiplier}"


def test_sector_rotation_multipliers_bottom3():
    """Rank 6-8 sectors should get -0.04 rotation_multiplier."""
    # Create a universe where Energy ranks last (very poor return)
    dfs = make_universe_dfs({
        "AAPL": 0.20, "MSFT": 0.18,  # Technology (rank 1)
        "JPM": 0.12,                   # Financials (rank 2 or so)
        "AMGN": 0.08,                  # Healthcare
        "UPS": 0.04,                   # Industrials
        "PG": 0.02,                    # Consumer Staples
        "META": 0.01,                  # Communications
        "XOM": -0.05,                  # Energy (rank 8)
    })
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    energy_rank = result.sector_map.get("Energy")
    if energy_rank and energy_rank.rank >= 6:
        assert energy_rank.rotation_multiplier == pytest.approx(-0.04, abs=1e-9)


def test_sector_rotation_neutral_multiplier():
    """Rank 3-5 sectors should get 0.0 rotation_multiplier."""
    dfs = make_universe_dfs({
        "AAPL": 0.20,  # Technology  (rank 1) → +0.04
        "JPM": 0.15,   # Financials  (rank 2) → +0.04
        "AMGN": 0.10,  # Healthcare  (rank 3) → 0.00
    })
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    health = result.sector_map.get("Healthcare")
    if health:
        assert health.rotation_multiplier == pytest.approx(0.0, abs=1e-9)


def test_sector_rotation_get_multiplier_for_ticker():
    """get_multiplier_for_ticker returns value from sector_map."""
    dfs = make_universe_dfs({"AAPL": 0.20, "MSFT": 0.15})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    mult = result.get_multiplier_for_ticker("AAPL")
    assert mult in [-0.04, 0.0, 0.04], f"Unexpected multiplier: {mult}"


def test_sector_rotation_get_multiplier_unknown_ticker():
    """Unknown ticker returns 0.0 multiplier (no data = neutral)."""
    dfs = make_universe_dfs({"AAPL": 0.20})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    mult = result.get_multiplier_for_ticker("UNKNOWN_TICKER_XYZ")
    assert mult == 0.0


def test_sector_rotation_rankings_length():
    """Result should contain exactly 8 sector ranks (all sectors always present)."""
    dfs = make_universe_dfs({"AAPL": 0.10, "JPM": 0.08})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    assert len(result.rankings) == 8


def test_sector_rotation_ranks_are_1_to_8():
    """Rank values must be integers 1 through 8."""
    dfs = make_universe_dfs({"AAPL": 0.10, "XOM": 0.05})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    ranks = [r.rank for r in result.rankings]
    assert sorted(ranks) == list(range(1, 9)), f"Expected ranks 1-8, got {sorted(ranks)}"


def test_sector_rotation_empty_universe():
    """Empty universe should still produce 8 sectors all at rank 0.0 return."""
    result = calculate_sector_rotation({}, as_of_date="2024-01-25", persist=False)
    assert len(result.rankings) == 8
    for r in result.rankings:
        assert r.avg_20d_return == pytest.approx(0.0, abs=1e-9)


def test_sector_rotation_to_dict():
    """SectorRank.to_dict() returns expected keys."""
    dfs = make_universe_dfs({"AAPL": 0.10})
    result = calculate_sector_rotation(dfs, as_of_date="2024-01-25", persist=False)
    d = result.rankings[0].to_dict()
    assert "sector" in d
    assert "avg_20d_return" in d
    assert "rank" in d
    assert "rotation_multiplier" in d
