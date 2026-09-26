"""Suite-wide test isolation: tests must never write production state or tracked reports."""

import sqlite3
from pathlib import Path

import pytest

PRODUCTION_DB = Path(__file__).resolve().parents[1] / "data" / "processed" / "trader.db"


@pytest.fixture(scope="session", autouse=True)
def _session_test_database(tmp_path_factory):
    """The whole session runs on a private copy of trader.db: tests can read real market data,
    but no test can write the production database."""
    from config.settings import settings
    from src.db import repository

    copy = tmp_path_factory.mktemp("db") / "trader_test_copy.db"
    if PRODUCTION_DB.exists():
        src = sqlite3.connect(f"file:{PRODUCTION_DB}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        src.backup(dst)
        dst.close()
        src.close()
    url = f"sqlite:///{copy.as_posix()}"
    original = settings.db_url
    settings.db_url = url
    repository._engine = None
    yield url
    settings.db_url = original
    repository._engine = None


@pytest.fixture(autouse=True)
def _reset_database_between_tests(_session_test_database):
    """Start every test on the isolated copy with a fresh engine, so a test that pointed db_url
    elsewhere (or cached an engine) cannot leak into the next one."""
    from config.settings import settings
    from src.db import repository

    settings.db_url = _session_test_database
    repository._engine = None
    yield
    settings.db_url = _session_test_database
    repository._engine = None


@pytest.fixture(autouse=True)
def _isolate_pending_orders_dir(tmp_path, monkeypatch):
    """Every test writes PaperBroker pending orders to its own temp dir, never to production data/."""
    import src.trading.paper_broker as paper_broker

    monkeypatch.setattr(paper_broker, "PENDING_ORDERS_DIR", tmp_path / "pending_orders")


@pytest.fixture(autouse=True)
def _isolate_tax_report_dir(tmp_path, monkeypatch):
    """Dashboard renders auto-generate the tax report; default its output to a temp dir, not tracked reports/."""
    from src.reports import tax_report

    real_generate = tax_report.generate_tax_report

    def generate_into_tmp(year, country="India", reports_dir=None):
        return real_generate(year=year, country=country, reports_dir=reports_dir or str(tmp_path / "reports"))

    monkeypatch.setattr(tax_report, "generate_tax_report", generate_into_tmp)
