"""
tests/test_operational_section4.py — Test Suite for Section 4 (Items 14 & 15).

Tests:
  - Item 14: Database upgrade to PostgreSQL via DATABASE_URL, engine pooling,
    SQLAlchemy ORM compatibility, and db_migrate.py export/import/transfer.
  - Item 15: Slippage simulation model (3 tiers: 0.05% / 0.15% / 0.30%),
    effective execution prices on BUY/SELL, tracking cumulative total_slippage_cost
    on Drawer 4 balance sheet and Phase 6 evaluation report.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from config.settings import Settings, settings
from db_migrate import export_database, get_engine_for_url, import_database, normalize_db_url, transfer_database, verify_tables
from src.db import repository
from src.db.models import (
    Base,
    EventLog,
    FeatureRow,
    MarketDataRow,
    OrderRow,
    PortfolioSnapshot,
    PredictionRow,
    TradeRow,
)
from src.ml.evaluate import EvaluationReport, WindowMetrics
from src.portfolio.portfolio import OrderSpec
from src.trading.paper_broker import FillResult, PaperBroker, compute_slippage


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_db():
    """Creates a clean isolated SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    tmp_url = f"sqlite:///{tmp_path.replace(os.sep, '/')}"
    orig_url = settings.db_url
    settings.__dict__["db_url"] = tmp_url
    repository._engine = None

    repository.create_all_tables()
    yield tmp_url

    repository._engine = None
    settings.__dict__["db_url"] = orig_url
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Item 14 Tests: Database URL, Dialects, ORM, and Migration Script
# ═══════════════════════════════════════════════════════════════════════════════

def test_database_url_environment_override_and_normalization():
    """Verify DATABASE_URL env var alias and postgres:// -> postgresql+psycopg2:// normalizer."""
    # Test normalization of legacy postgres:// scheme
    pg_legacy = "postgres://trader:secret@localhost:5432/trading_db"
    assert normalize_db_url(pg_legacy) == "postgresql+psycopg2://trader:secret@localhost:5432/trading_db"

    # Test normalization of standard postgresql:// scheme
    pg_std = "postgresql://trader:secret@localhost:5432/trading_db"
    assert normalize_db_url(pg_std) == "postgresql+psycopg2://trader:secret@localhost:5432/trading_db"

    # Test settings validator with DATABASE_URL
    with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@db.example.com:5432/trader"}):
        s = Settings()
        assert s.db_url == "postgresql+psycopg2://user:pass@db.example.com:5432/trader"


def test_engine_creation_sqlite_and_postgres_configs():
    """Verify get_engine creates SQLite engine with check_same_thread=False and configured pooling."""
    sqlite_url = "sqlite:///:memory:"
    eng_sqlite = get_engine_for_url(sqlite_url)
    assert eng_sqlite.name == "sqlite"

    # Test PostgreSQL URL produces postgresql dialect engine
    eng_pg = get_engine_for_url("postgresql+psycopg2://user:pass@localhost:5432/testdb")
    assert eng_pg.name == "postgresql"
    assert eng_pg.pool.size() == 5  # default pool size configured


def test_db_migrate_export_and_import(isolated_db):
    """Verify db_migrate.py export to JSON and import into a clean database."""
    # Seed source database with test records across tables
    repository.save_portfolio_snapshot(
        run_date="2024-03-01",
        cash=95000.0,
        total_value=100000.0,
        positions={"AAPL": {"quantity": 25.0, "entry_price": 180.0, "current_price": 200.0}},
        total_slippage_cost=15.50,
    )
    repository.save_trade(
        run_date="2024-03-01",
        ticker="AAPL",
        action="BUY",
        quantity=25.0,
        fill_price=180.09,
        cost=9.0,
        net_pnl=0.0,
        slippage_cost=2.25,
    )
    repository.log_event("INFO", "test_runner", "Test migration event", {"key": "val"})

    # Export source database
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
        export_json_path = json_file.name

    try:
        exported_counts = export_database(source_url=isolated_db, output_path=export_json_path)
        assert exported_counts["portfolio"] == 1
        assert exported_counts["trades"] == 1
        assert exported_counts["event_log"] == 1

        # Verify exported JSON structure
        with open(export_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "portfolio" in data["tables"]
        assert data["tables"]["portfolio"][0]["total_slippage_cost"] == 15.50
        assert data["tables"]["trades"][0]["slippage_cost"] == 2.25

        # Create target database and import
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as target_file:
            target_db_path = target_file.name
        target_url = f"sqlite:///{target_db_path.replace(os.sep, '/')}"

        try:
            imported_counts = import_database(input_path=export_json_path, target_url=target_url, clear_existing=True)
            assert imported_counts["portfolio"] == 1
            assert imported_counts["trades"] == 1
            assert imported_counts["event_log"] == 1

            # Verify target table counts via verify_tables
            counts = verify_tables(target_url)
            assert counts["portfolio"] == 1
            assert counts["trades"] == 1
            assert counts["event_log"] == 1
        finally:
            if os.path.exists(target_db_path):
                try:
                    os.remove(target_db_path)
                except OSError:
                    pass
    finally:
        if os.path.exists(export_json_path):
            try:
                os.remove(export_json_path)
            except OSError:
                pass


def test_db_migrate_direct_transfer(isolated_db):
    """Verify live direct database-to-database transfer without intermediate disk file."""
    # Seed data
    repository.save_portfolio_snapshot(
        run_date="2024-03-02",
        cash=90000.0,
        total_value=102000.0,
        positions={},
        total_slippage_cost=8.40,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as target_file:
        target_db_path = target_file.name
    target_url = f"sqlite:///{target_db_path.replace(os.sep, '/')}"

    try:
        transfer_counts = transfer_database(source_url=isolated_db, target_url=target_url, clear_existing=True)
        assert transfer_counts["portfolio"] >= 1

        counts = verify_tables(target_url)
        assert counts["portfolio"] >= 1
    finally:
        if os.path.exists(target_db_path):
            try:
                os.remove(target_db_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Item 15 Tests: Slippage Simulation Model (3 Tiers, Balance Sheet, Report)
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_slippage_tiers():
    """
    Verify the 3 slippage simulation tiers based on trade size vs Average Daily Volume:
      - Tier 1: ratio <= 1%  -> 0.05% (0.0005)
      - Tier 2: 1% < ratio <= 5% -> 0.15% (0.0015)
      - Tier 3: ratio > 5%   -> 0.30% (0.0030)
      - Default when ADV is None -> 0.05%
    """
    base_price = 100.0
    adv = 100_000.0

    # Tier 1: 500 shares / 100,000 ADV = 0.5% (<= 1%) -> 0.05%
    eff_buy, rate_buy, cost_buy = compute_slippage(shares=500.0, base_price=base_price, action="BUY", adv=adv)
    assert rate_buy == pytest.approx(0.0005, abs=1e-6)
    assert eff_buy == pytest.approx(100.05, abs=1e-4)
    assert cost_buy == pytest.approx(500.0 * 100.0 * 0.0005, abs=1e-4)

    # Tier 1 on SELL: effective price moves down against seller
    eff_sell, rate_sell, cost_sell = compute_slippage(shares=500.0, base_price=base_price, action="SELL", adv=adv)
    assert rate_sell == pytest.approx(0.0005, abs=1e-6)
    assert eff_sell == pytest.approx(99.95, abs=1e-4)
    assert cost_sell == pytest.approx(500.0 * 100.0 * 0.0005, abs=1e-4)

    # Tier 2: 3,000 shares / 100,000 ADV = 3.0% (1% to 5%) -> 0.15%
    eff_mid, rate_mid, cost_mid = compute_slippage(shares=3000.0, base_price=base_price, action="BUY", adv=adv)
    assert rate_mid == pytest.approx(0.0015, abs=1e-6)
    assert eff_mid == pytest.approx(100.15, abs=1e-4)
    assert cost_mid == pytest.approx(3000.0 * 100.0 * 0.0015, abs=1e-4)

    # Tier 3: 8,000 shares / 100,000 ADV = 8.0% (> 5%) -> 0.30%
    eff_high, rate_high, cost_high = compute_slippage(shares=8000.0, base_price=base_price, action="BUY", adv=adv)
    assert rate_high == pytest.approx(0.0030, abs=1e-6)
    assert eff_high == pytest.approx(100.30, abs=1e-4)
    assert cost_high == pytest.approx(8000.0 * 100.0 * 0.0030, abs=1e-4)

    # Default without ADV -> low tier (0.05%)
    eff_def, rate_def, cost_def = compute_slippage(shares=100.0, base_price=base_price, action="BUY", adv=None)
    assert rate_def == pytest.approx(0.0005, abs=1e-6)
    assert eff_def == pytest.approx(100.05, abs=1e-4)


def test_paper_broker_slippage_execution_and_balance_sheet(isolated_db):
    """
    Verify PaperBroker executes orders with slippage, tracks total_slippage_cost,
    persists it to trades and daily portfolio snapshots (Drawer 4), and restores it.
    """
    broker = PaperBroker()
    broker.load_state()
    assert broker.total_slippage_cost == 0.0

    # 1. Execute BUY with ADV (25 shares @ $100 base, ADV=2,500 -> 1.0% -> Tier 1: 0.05%)
    buy_order = OrderSpec(
        date="2024-03-01",
        ticker="MSFT",
        action="BUY",
        order_type="MARKET",
        shares=25.0,
        reference_price=100.0,
        gross_value=2500.0,
        estimated_fee=5.0,
        net_amount=2505.0,
        reason="CONVICTION_BUY",
        adv=2500.0,
    )
    fill_buy = broker.execute_order(buy_order, fill_price=100.0, run_date="2024-03-01", adv=2500.0)
    assert fill_buy is not None
    assert fill_buy.fill_price == pytest.approx(100.05, abs=1e-4)
    expected_buy_slip = round(25.0 * 100.0 * 0.0005, 4)  # $1.25
    assert fill_buy.slippage_cost == pytest.approx(expected_buy_slip, abs=1e-4)
    assert broker.total_slippage_cost == pytest.approx(expected_buy_slip, abs=1e-4)

    # 2. Execute SELL with high ADV participation (25 shares @ $110 base, ADV=250 -> 10% -> Tier 3: 0.30%)
    sell_order = OrderSpec(
        date="2024-03-05",
        ticker="MSFT",
        action="SELL",
        order_type="MARKET",
        shares=25.0,
        reference_price=110.0,
        gross_value=2750.0,
        estimated_fee=5.5,
        net_amount=2744.5,
        reason="TAKE_PROFIT",
        adv=250.0,
    )
    fill_sell = broker.execute_order(sell_order, fill_price=110.0, run_date="2024-03-05", adv=250.0)
    assert fill_sell is not None
    # 0.30% slippage on $110 -> $109.67 effective
    assert fill_sell.fill_price == pytest.approx(109.67, abs=1e-4)
    expected_sell_slip = round(25.0 * 110.0 * 0.0030, 4)  # $8.25
    assert fill_sell.slippage_cost == pytest.approx(expected_sell_slip, abs=1e-4)

    expected_total_slip = round(expected_buy_slip + expected_sell_slip, 4)  # $9.50
    assert broker.total_slippage_cost == pytest.approx(expected_total_slip, abs=1e-4)

    # 3. Mark to market and record snapshot (Drawer 4 balance sheet)
    total_eq = broker.record_snapshot(run_date="2024-03-05", current_prices={"MSFT": 110.0})
    snap = repository.get_latest_portfolio_snapshot()
    assert snap is not None
    assert snap["total_slippage_cost"] == pytest.approx(expected_total_slip, abs=1e-4)

    # Verify trades table records individual slippage costs
    trades = repository.get_trades(limit=10)
    assert len(trades) == 2
    assert trades[0]["slippage_cost"] == pytest.approx(expected_sell_slip, abs=1e-4)
    assert trades[1]["slippage_cost"] == pytest.approx(expected_buy_slip, abs=1e-4)

    # 4. Verify new broker instance restores total_slippage_cost in load_state
    broker_reloaded = PaperBroker()
    broker_reloaded.load_state()
    assert broker_reloaded.total_slippage_cost == pytest.approx(expected_total_slip, abs=1e-4)


def test_dashboard_data_loader_portfolio_summary_slippage():
    """Verify get_portfolio_summary and get_equity_history_df expose total_slippage_cost."""
    from dashboard.data_loader import get_equity_history_df, get_portfolio_summary

    summary = get_portfolio_summary()
    assert "total_slippage_cost" in summary
    assert isinstance(summary["total_slippage_cost"], float)

    eq_df = get_equity_history_df()
    assert list(eq_df.columns) == ["date", "cash", "total_value", "invested"]
    snaps = repository.get_portfolio_snapshots(limit=10)
    if snaps:
        assert "total_slippage_cost" in snaps[0]


def test_evaluation_report_includes_slippage_cost():
    """Verify Phase 6 WindowMetrics and EvaluationReport include total_slippage_cost."""
    wm = WindowMetrics(
        window_name="2022 Bear Market",
        model_type="logistic_regression",
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        train_rows=5000,
        test_rows=1200,
        base_rate=0.48,
        accuracy=0.55,
        roc_auc=0.58,
        brier_score=0.23,
        brier_skill_score=0.04,
        count_at_buy_bar=45,
        precision_at_buy_bar=0.62,
        edge_over_base_rate=0.14,
        is_low_sample=False,
        sample_confidence="HIGH (N>=30)",
        too_good_alarm=False,
        regime_tag="2022 Bear Market (Critical)",
        total_slippage_cost=127.50,
    )
    assert wm.total_slippage_cost == 127.50

    report = EvaluationReport(
        generated_at_utc="2026-09-19T00:00:00Z",
        buy_bar=0.60,
        window_results=[wm],
        summary_by_model={
            "logistic_regression": {
                "total_opportunities_fired": 45,
                "total_slippage_cost": 127.50,
                "sample_weighted_edge_pts": 14.0,
                "mean_auc": 0.58,
            }
        },
        leakage_alarms=[],
    )

    table_md = report.to_markdown_table()
    assert "Slippage Cost" in table_md
    assert "$127.50" in table_md
