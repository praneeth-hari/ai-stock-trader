"""Old-schema compatibility (portfolio.pending_orders + new tables) and no stateless trading."""

import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from config.settings import settings
from src.db import repository
from src.features.engineer import FEATURE_COLUMNS
import src.pipeline.daily_pipeline as dp
import src.pipeline.scheduler as sch

OLD_PORTFOLIO_DDL = """
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date VARCHAR(10) NOT NULL,
    market VARCHAR(10) DEFAULT 'US',
    cash FLOAT NOT NULL,
    total_value FLOAT NOT NULL,
    total_slippage_cost FLOAT DEFAULT 0.0,
    highest_price_since_entry FLOAT,
    trailing_stop_price FLOAT,
    positions JSON,
    created_at DATETIME,
    CONSTRAINT uq_portfolio_run_date_market UNIQUE (run_date, market)
)"""
DAY0 = ("2026-09-25", "US", 10000.0, 10000.0, 0.0, None, None, "{}", "2026-09-26 14:00:00")


@pytest.fixture
def old_db(tmp_path):
    """A database exactly as V1 had it before commit ec32b8a: no pending_orders, no new tables."""
    path = tmp_path / "old_schema.db"
    con = sqlite3.connect(path)
    con.execute(OLD_PORTFOLIO_DDL)
    con.execute("INSERT INTO portfolio (run_date, market, cash, total_value, total_slippage_cost, "
                "highest_price_since_entry, trailing_stop_price, positions, created_at) VALUES (?,?,?,?,?,?,?,?,?)", DAY0)
    con.commit()
    con.close()
    settings.db_url = f"sqlite:///{path.as_posix()}"
    repository._engine, repository._db_available = None, True
    yield path
    repository._db_available = True


def _columns(path, table):
    return [r[1] for r in sqlite3.connect(path).execute(f"pragma table_info('{table}')")]


def _tables(path):
    return sorted(r[0] for r in sqlite3.connect(path).execute("select name from sqlite_master where type='table'"))


def _fingerprint(path):
    con = sqlite3.connect(path)
    out = {}
    for t in _tables(path):
        rows = con.execute(f"select * from '{t}' order by rowid").fetchall()
        out[t] = (_columns(path, t), hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest())
    return out


def test_migration_fixes_reads_and_preserves_the_existing_day0_row(old_db):
    with pytest.raises(OperationalError, match="no such column: portfolio.pending_orders"):
        repository.get_portfolio_snapshots(market="US")        # the dashboard error, reproduced

    repository._engine = None
    repository.create_all_tables()
    assert repository._db_available is True
    assert _columns(old_db, "portfolio")[-1] == "pending_orders"
    assert {"decision_log", "benchmark_snapshots"} <= set(_tables(old_db))
    old_cols = ["run_date", "market", "cash", "total_value", "total_slippage_cost", "highest_price_since_entry",
                "trailing_stop_price", "positions", "created_at"]
    row = sqlite3.connect(old_db).execute(f"select {', '.join(old_cols)}, pending_orders from portfolio").fetchall()
    assert row == [DAY0 + (None,)], "day-0 row unchanged; the new column is empty"
    snaps = repository.get_portfolio_snapshots(market="US")
    assert [(s["run_date"], s["cash"], s["total_value"], s["positions"]) for s in snaps] == [("2026-09-25", 10000.0, 10000.0, {})]
    assert repository.get_latest_portfolio_snapshot("US")["pending_orders"] is None


def test_migration_is_idempotent(old_db):
    repository.create_all_tables()
    first = _fingerprint(old_db)
    repository._engine = None
    repository.create_all_tables()
    assert _fingerprint(old_db) == first


def test_dashboard_chart_data_works_against_an_unmigrated_database(old_db):
    from dashboard.data_loader import get_portfolio_vs_spy_chart_data

    df = get_portfolio_vs_spy_chart_data(market="US")          # no explicit migration first
    assert list(df["My Portfolio"]) == [10000.0]
    assert "pending_orders" in _columns(old_db, "portfolio")


def test_weekly_summary_works_against_an_unmigrated_database(old_db):
    from src.reports.weekly_summary import generate_weekly_summary_data

    generate_weekly_summary_data(target_date=date(2026, 9, 25), market="US")   # must not raise
    assert "pending_orders" in _columns(old_db, "portfolio")


def _synthetic(run_date):
    days = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(340)]
    days = [d.strftime("%Y-%m-%d") for d in days if d.weekday() < 5 and d.strftime("%Y-%m-%d") <= run_date]
    n = len(days)

    def frame(t, c):
        return pd.DataFrame({"date": days, "open": [c * 0.995] * n, "high": [c * 1.01] * n, "low": [c * 0.99] * n,
                             "close": [c] * n, "volume": [500_000] * n, "ticker": [t] * n})
    return frame("SPY", 500.0), {"AAPL": frame("AAPL", 100.0)}


def test_a_failed_migration_stops_the_pipeline_with_no_stateless_trading(old_db):
    run_date = "2024-01-04"
    spy, uni = _synthetic(run_date)
    before = _fingerprint(old_db)
    model = MagicMock()
    alerts = [patch.object(dp, n, MagicMock()) for n in dir(dp) if n.startswith(("send_", "notify_"))]
    broker_cls = MagicMock(wraps=dp.PaperBroker)
    try:
        for a in alerts:
            a.start()
        with patch.object(repository.Base.metadata, "create_all",
                          side_effect=OperationalError("CREATE TABLE", {}, Exception("database is locked"))), \
             patch.object(dp, "load_active_model", return_value=model), \
             patch.object(dp, "PaperBroker", broker_cls):
            res = dp.run_daily_pipeline(run_date=run_date, tickers=["AAPL"], _spy_df=spy, _universe_dfs=uni)
            assert repository._db_available is False, "the real create_all_tables() fell back to stateless mode"
        sent = [a.new.called for a in alerts if a.attribute == "send_pipeline_failure_alert"]
    finally:
        for a in alerts:
            a.stop()
        repository._db_available = True

    assert res.status == "FAILED" and res.regime == "DATABASE_UNAVAILABLE"
    assert res.orders_generated == 0 and res.fills == [] and res.pending_orders == []
    assert sch.EXIT_CODES[res.status] == 1
    assert not broker_cls.called, "no (stateless) portfolio was created"
    assert not model.predict_proba.called, "no model inference, no trading decisions"
    assert sent == [True], "a failure alert was sent"
    assert _fingerprint(old_db) == before, "database untouched"
