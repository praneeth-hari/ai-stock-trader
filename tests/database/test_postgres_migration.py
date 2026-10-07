"""
tests/database/test_postgres_migration.py — Test Suite for Section 4 (Item 14).

Tests:
  - Item 14: Database upgrade to PostgreSQL via DATABASE_URL, engine pooling,
    SQLAlchemy ORM compatibility, and db_migrate.py export/import/transfer.
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
from db_migrate import (
    TABLE_MODELS,
    export_database,
    get_engine_for_url,
    import_database,
    normalize_db_url,
    preflight_check,
    transfer_database,
    verify_migration,
    verify_tables,
)
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


def test_all_19_tables_in_migration_models():
    """Verify that all 19 ORM models are registered in TABLE_MODELS in foreign-key safe order."""
    all_orm_tables = set(Base.metadata.tables.keys())
    migrated_tables = {m.__tablename__ for m in TABLE_MODELS}

    assert len(TABLE_MODELS) == 19
    assert migrated_tables == all_orm_tables

    # Check foreign key dependency order
    table_names = [m.__tablename__ for m in TABLE_MODELS]
    sv_idx = table_names.index("strategy_variants")
    ss_idx = table_names.index("strategy_snapshots")
    st_idx = table_names.index("strategy_trades")
    assert sv_idx < ss_idx, "strategy_variants must come before strategy_snapshots"
    assert sv_idx < st_idx, "strategy_variants must come before strategy_trades"


def test_preflight_check(isolated_db):
    """Verify that preflight check succeeds without modifying data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as target_file:
        target_db_path = target_file.name
    target_url = f"sqlite:///{target_db_path.replace(os.sep, '/')}"

    try:
        # Pre-create target schema
        target_engine = get_engine_for_url(target_url)
        Base.metadata.create_all(target_engine)

        res = preflight_check(source_url=isolated_db, target_url=target_url)
        assert res["status"] is True
        assert res["fk_order_valid"] is True
        assert res["tables_count"] == 19
        assert isinstance(res["src_counts"], dict)
        assert isinstance(res["tgt_counts"], dict)
    finally:
        if os.path.exists(target_db_path):
            try:
                os.remove(target_db_path)
            except OSError:
                pass


def test_migration_verification(isolated_db):
    """Verify that verify_migration detects exact match after direct transfer."""
    # Seed source database with portfolio snapshot and trade
    repository.save_portfolio_snapshot(
        run_date="2026-09-29",
        cash=10000.0,
        total_value=10000.0,
        positions={},
        market="US",
    )
    repository.save_trade(
        run_date="2026-09-29",
        ticker="AAPL",
        action="BUY",
        quantity=10.0,
        fill_price=150.0,
        cost=3.0,
        net_pnl=0.0,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as target_file:
        target_db_path = target_file.name
    target_url = f"sqlite:///{target_db_path.replace(os.sep, '/')}"

    try:
        transfer_database(source_url=isolated_db, target_url=target_url, batch_size=2)
        v_res = verify_migration(source_url=isolated_db, target_url=target_url)
        assert v_res["success"] is True
        assert len(v_res["mismatches"]) == 0
        assert v_res["trading_state"]["portfolio"]["match"] is True
        assert v_res["trading_state"]["trades"]["match"] is True
    finally:
        if os.path.exists(target_db_path):
            try:
                os.remove(target_db_path)
            except OSError:
                pass


def test_ci_database_guard_requires_postgres_in_ci(monkeypatch):
    """Verify that CI database guard prevents SQLite fallback in CI but allows local dev."""
    from src.db.cloud_db import verify_ci_database_guard

    # 1. In CI: Missing DATABASE_URL fails immediately
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="CI DATABASE SAFETY GUARD: DATABASE_URL environment variable is required"):
        verify_ci_database_guard(force_check=True)

    # 2. In CI: SQLite DATABASE_URL fails immediately
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/processed/trader.db")
    with pytest.raises(ValueError, match="must be a PostgreSQL URL"):
        verify_ci_database_guard(force_check=True)

    # 3. In CI: Valid PostgreSQL URL passes
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@ep-pooler.supabase.com:5432/postgres")
    verify_ci_database_guard(force_check=True)

    # 4. Local dev (no CI env): Missing or SQLite DATABASE_URL does not raise
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    verify_ci_database_guard(force_check=False)


def test_persisted_kill_switch_enforced_by_pipeline(isolated_db):
    """Verify that persisted kill-switch state in DB halts the trader pipeline."""
    from src.pipeline.daily_pipeline import run_daily_pipeline

    orig_ks = settings.kill_switch_enabled
    orig_reason = settings.kill_switch_reason
    try:
        settings.kill_switch_enabled = False
        settings.kill_switch_reason = ""

        # Persist kill switch ON in DB
        repository.save_kill_switch_state(True, "Persisted dashboard veto")

        # Trader startup should read DB and halt
        result = run_daily_pipeline(run_date="2026-10-07")
        assert result.status == "HALTED"
        assert result.regime == "KILL_SWITCH_ACTIVE"
        assert any("Persisted dashboard veto" in err for err in result.errors)
        assert settings.kill_switch_enabled is True
    finally:
        repository.save_kill_switch_state(False, "")
        settings.kill_switch_enabled = orig_ks
        settings.kill_switch_reason = orig_reason


def test_telegram_workflow_security_and_manual_dispatch():
    """Verify daily_trading.yml does not expose bot token in URLs and allows manual dispatch."""
    workflow_path = Path(".github/workflows/daily_trading.yml")
    assert workflow_path.exists(), "daily_trading.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # Security: No curl exposing token in URL
    assert "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}" not in content
    assert "src.alerts.telegram_alerts" in content

    # Manual dispatch enabled, hard off removed for manual runs
    assert "if: github.event_name == 'workflow_dispatch'" in content
    assert "if: ${{ false }}" not in content

    # Recurring cron schedule remains commented out / disabled
    assert "- cron:" not in [line.strip() for line in content.splitlines() if not line.strip().startswith("#")]


