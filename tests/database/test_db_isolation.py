"""Tests can never write the production trader.db (regression for test events leaking into the audit trail)."""

import pytest

from config.settings import settings
from src.db import repository


def test_suite_runs_on_an_isolated_database_copy():
    assert "data/processed/trader.db" not in settings.db_url.replace("\\", "/")
    engine_url = str(repository.get_engine().url).replace("\\", "/")
    assert "data/processed/trader.db" not in engine_url


def test_write_guard_checks_the_cached_engine_not_only_the_configured_url(tmp_path, monkeypatch):
    # Reproduce the leak: db_url patched to a temp file, but a stale engine still targets "production".
    fake_prod = tmp_path / "data" / "processed" / "trader.db"
    fake_prod.parent.mkdir(parents=True)
    repository._engine = repository.create_engine(f"sqlite:///{fake_prod.as_posix()}")
    monkeypatch.setattr(settings, "db_url", f"sqlite:///{(tmp_path / 'isolated.db').as_posix()}")

    with pytest.raises(RuntimeError, match="PRODUCTION DATABASE WRITE BLOCKED"):
        repository.log_event("CRITICAL", "scheduler", "must never reach production")
    with pytest.raises(RuntimeError, match="PRODUCTION DATABASE WRITE BLOCKED"):
        repository.save_kill_switch_state(False, "")
    assert not fake_prod.exists() or fake_prod.stat().st_size == 0
