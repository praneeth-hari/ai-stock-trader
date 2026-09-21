"""
tests/test_leaderboard.py — Unit tests for Section 10 Item 1: Paper Trading Leaderboard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import settings
from src.db import repository
from src.trading.leaderboard import (
    DEFAULT_STRATEGIES,
    get_leaderboard_rankings,
    init_default_strategies,
    run_leaderboard_cycle,
)
from dashboard.data_loader import (
    get_leaderboard_equity_curves,
    get_leaderboard_summary_data,
)
from dashboard.charts import build_leaderboard_equity_chart


@pytest.fixture
def tmp_db(tmp_path):
    """Fixture providing an isolated SQLite database for testing repository functions."""
    db_file = tmp_path / "test_trader.db"
    db_url = f"sqlite:///{db_file}"

    orig_url = repository.settings.db_url
    repository.settings.__dict__["db_url"] = db_url
    repository._engine = None
    repository.create_all_tables()

    yield db_url

    repository._engine = None
    repository.settings.__dict__["db_url"] = orig_url


def test_settings_leaderboard_defaults():
    assert settings.leaderboard_enabled is True
    assert settings.leaderboard_strategies == 3


def test_init_default_strategies(tmp_db):
    variants = init_default_strategies(created_date="2026-09-19")
    assert len(variants) == 3
    names = [v["name"] for v in variants]
    assert "Conservative" in names
    assert "Balanced" in names
    assert "Aggressive" in names

    # Re-running init should be idempotent
    variants_again = init_default_strategies(created_date="2026-09-19")
    assert len(variants_again) == 3


def test_repository_strategy_crud(tmp_db):
    sid = repository.save_strategy_variant(
        name="Custom_Test_Strat",
        settings_dict={"buy_bar": 0.65},
        starting_capital=10000.0,
        created_date="2026-09-19",
    )
    assert sid > 0

    variants = repository.get_strategy_variants(only_active=True)
    assert any(v["name"] == "Custom_Test_Strat" for v in variants)

    # Save snapshot
    snap_id = repository.save_strategy_snapshot(
        strategy_id=sid,
        date_str="2026-09-19",
        portfolio_value=10500.0,
        cash=5000.0,
        positions={"AAPL": {"shares": 25, "entry_price": 200.0}},
        daily_return=0.05,
    )
    assert snap_id > 0

    snaps = repository.get_strategy_snapshots(strategy_id=sid)
    assert len(snaps) == 1
    assert snaps[0]["portfolio_value"] == 10500.0

    # Record trade
    trade_id = repository.save_strategy_trade(
        strategy_id=sid,
        date_str="2026-09-19",
        ticker="AAPL",
        action="BUY",
        price=200.0,
        shares=25.0,
        pnl=0.0,
    )
    assert trade_id > 0

    trades = repository.get_strategy_trades(strategy_id=sid)
    assert len(trades) == 1
    assert trades[0]["ticker"] == "AAPL"


def test_run_leaderboard_cycle(tmp_db):
    date_str = "2026-09-19"
    prices = {"AAPL": 220.0, "MSFT": 420.0, "NVDA": 130.0}
    predictions = {"AAPL": 0.78, "MSFT": 0.62, "NVDA": 0.55}

    cycle_res = run_leaderboard_cycle(
        run_date=date_str,
        prices=prices,
        predictions=predictions,
    )

    assert isinstance(cycle_res, dict)
    assert "Conservative" in cycle_res
    assert "Balanced" in cycle_res
    assert "Aggressive" in cycle_res

    # Conservative buy_bar is 0.75 -> should buy AAPL (0.78)
    cons_res = cycle_res["Conservative"]
    assert cons_res["portfolio_value"] > 0

    # Verify snapshots were saved in DB
    snaps = repository.get_strategy_snapshots()
    assert len(snaps) >= 3


def test_get_leaderboard_rankings(tmp_db):
    date_str = "2026-09-19"
    prices = {"AAPL": 220.0, "MSFT": 420.0}
    predictions = {"AAPL": 0.68, "MSFT": 0.62}

    run_leaderboard_cycle(run_date=date_str, prices=prices, predictions=predictions)

    rankings = get_leaderboard_rankings()
    assert len(rankings) == 3

    top_rank = rankings[0]
    assert top_rank["rank"] == 1
    assert top_rank["is_winning"] is True
    assert top_rank["trophy"] == "🏆"

    # Must be sorted descending by return_pct
    returns = [r["return_pct"] for r in rankings]
    assert returns == sorted(returns, reverse=True)


def test_dashboard_data_loaders_and_chart_builders(tmp_db):
    date_str = "2026-09-19"
    prices = {"AAPL": 220.0, "MSFT": 420.0}
    predictions = {"AAPL": 0.68, "MSFT": 0.62}
    run_leaderboard_cycle(run_date=date_str, prices=prices, predictions=predictions)

    rankings, winning = get_leaderboard_summary_data()
    assert len(rankings) == 3
    assert winning is not None
    assert winning["rank"] == 1

    curves_df = get_leaderboard_equity_curves()
    assert not curves_df.empty
    assert "Strategy" in curves_df.columns
    assert "portfolio_value" in curves_df.columns

    chart = build_leaderboard_equity_chart(curves_df)
    assert chart is not None
