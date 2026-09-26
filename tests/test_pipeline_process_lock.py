"""Machine-wide trading-cycle lock, tested with independent OS processes."""

import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from src.db import repository
import src.pipeline.daily_pipeline as dp
import src.pipeline.scheduler as sch
from src.pipeline.run_lock import PROJECT_ROOT, PipelineLockBusy, lock_dir, lock_paths, pipeline_run_lock
from src.portfolio.portfolio import OrderSpec
from src.trading.paper_broker import PaperBroker, _pending_orders_cache

CHILD = Path(__file__).with_name("_pipeline_lock_child.py")
_DAYS = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(340)]
DAYS = [d.strftime("%Y-%m-%d") for d in _DAYS if d.weekday() < 5][:224]
D, PREV = DAYS[-2], DAYS[-3]


def _spawn(*args):
    return subprocess.Popen([sys.executable, str(CHILD), *map(str, args)], cwd=PROJECT_ROOT,
                            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _wait_for(path: Path, proc=None, timeout=120.0):
    end = time.time() + timeout
    while not path.exists():
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"child exited early: {proc.stderr.read().decode()[-2000:]}")
        assert time.time() < end, f"timed out waiting for {path}"
        time.sleep(0.05)


def _finish(proc, timeout=180):
    out, err = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, err.decode()[-3000:]


@pytest.fixture
def world(tmp_path):
    """Shared temp DB + lock dir + pending dir for this test and its child processes."""
    locks = tmp_path / "locks"
    db = tmp_path / "shared.db"
    settings.pipeline_lock_dir = locks
    settings.db_url = f"sqlite:///{db.as_posix()}"
    repository._engine, repository._db_available = None, True
    repository.create_all_tables()
    _pending_orders_cache.clear()
    return {"locks": locks, "db_url": settings.db_url, "pending": tmp_path / "pending", "tmp": tmp_path}


def _commit_pending_buy():
    b = PaperBroker(market="US")
    b.load_state()
    b.sync_pending_orders([OrderSpec(date=PREV, ticker="AAPL", action="BUY", order_type="MARKET", shares=10.0,
                                     reference_price=100.0, gross_value=1000.0, estimated_fee=2.0,
                                     net_amount=1002.0, reason="QUALIFIED_CONVICTION_BUY")])
    b.record_snapshot(run_date=PREV, current_prices={})


def _run_child(world, name, pause=None):
    result = world["tmp"] / f"{name}.json"
    args = ["run", world["locks"], world["db_url"], world["pending"], D, result]
    return _spawn(*(args + ([pause] if pause else []))), result


def test_second_process_is_refused_while_another_process_holds_the_lock(world):
    ready = world["tmp"] / "holder.ready"
    holder = _spawn("hold", world["locks"], "US", ready)
    try:
        _wait_for(ready, holder)
        runner, result = _run_child(world, "second")
        _finish(runner)
        r = json.loads(result.read_text())
        assert r["status"] == "HALTED" and r["regime"] == "CONCURRENT_RUN_REJECTED"
        assert r["orders"] == 0 and r["fills"] == []
        assert repository.get_portfolio_snapshot(D, "US") is None
        assert repository.get_trades(limit=10, market="US") == []
        blocked = [e for e in repository.get_events(limit=20) if "CONCURRENT RUN BLOCKED" in (e.get("message") or "")]
        assert blocked and blocked[0]["details"]["lock_holder"]["pid"] == holder.pid
    finally:
        (world["tmp"] / "holder.ready.release").write_text("")
        holder.communicate(timeout=60)


def test_two_simultaneous_pipeline_processes_only_one_executes_the_cycle(world):
    _commit_pending_buy()
    pause = world["tmp"] / "a.paused"
    a, a_result = _run_child(world, "a", pause=pause)
    try:
        _wait_for(pause, a)                       # A is mid-cycle: morning fill done, inside the risk step
        b, b_result = _run_child(world, "b")
        _finish(b)
    finally:
        Path(str(pause) + ".go").write_text("")
    _finish(a)
    ra, rb = json.loads(a_result.read_text()), json.loads(b_result.read_text())
    assert rb["status"] == "HALTED" and rb["orders"] == 0 and rb["fills"] == []
    assert ra["status"] in ("SUCCESS", "DEGRADED") and ra["fills"] == [["AAPL", "BUY"]]
    trades = repository.get_trades(limit=10, market="US")
    assert [(t["ticker"], t["action"]) for t in trades] == [("AAPL", "BUY")], "exactly one execution"
    assert list(repository.get_portfolio_snapshot(D, "US")["positions"]) == ["AAPL"]

    # The same-day duplicate guard still applies once the lock is free again.
    c, c_result = _run_child(world, "c")
    _finish(c)
    assert json.loads(c_result.read_text())["status"] == "SKIPPED"
    assert len(repository.get_trades(limit=10, market="US")) == 1


def test_a_crashed_lock_holder_never_leaves_a_stale_lock(world):
    ready = world["tmp"] / "doomed.ready"
    holder = _spawn("hold", world["locks"], "US", ready)
    _wait_for(ready, holder)
    with pytest.raises(PipelineLockBusy):
        with pipeline_run_lock("US"):
            pass
    holder.kill()                                  # hard kill: no cleanup code runs in the child
    holder.communicate(timeout=60)
    assert lock_paths("US")[0].exists(), "the lock file stays on disk after the crash"
    deadline = time.time() + 10
    while True:                                    # the OS releases the lock when the process dies
        try:
            with pipeline_run_lock("US"):
                break
        except PipelineLockBusy:
            assert time.time() < deadline, "lock still held after its process was killed"
            time.sleep(0.1)
    runner, result = _run_child(world, "after_crash")
    _finish(runner)
    assert json.loads(result.read_text())["status"] != "HALTED"


def test_lock_is_released_after_a_normal_run_and_after_an_exception(world):
    with patch.object(dp, "_run_daily_pipeline_internal", return_value=dp.DailyPipelineResult(
            D, 0, 0, 0, "RISK-ON", 0, [], [], 0.0, 0.0, [])):
        dp.run_daily_pipeline(run_date=D, tickers=["AAPL"])
    with pipeline_run_lock("US"):
        pass
    with patch.object(dp, "_run_daily_pipeline_internal", side_effect=RuntimeError("crash inside the cycle")):
        with pytest.raises(RuntimeError):
            dp.run_daily_pipeline(run_date=D, tickers=["AAPL"])
    with pipeline_run_lock("US"):
        pass


def test_scheduler_and_dashboard_are_refused_by_the_same_machine_wide_lock(world):
    from dashboard.data_loader import run_daily_paper_cycle_trigger

    ready = world["tmp"] / "holder.ready"
    holder = _spawn("hold", world["locks"], "US", ready)
    try:
        _wait_for(ready, holder)
        res = sch.execute_scheduled_job(run_date=D, force=True)
        assert res.status == "HALTED" and res.orders_generated == 0 and not res.fills
        assert sch.EXIT_CODES[res.status] == 3
        ev = [e for e in repository.get_events(limit=20) if e.get("component") == "scheduler"][0]
        assert "Status=HALTED" in ev["message"]
        dash = run_daily_paper_cycle_trigger(run_date=D, market="US")
        assert dash["orders_count"] == 0 and dash["fills_count"] == 0 and "HALTED" in dash["audit_markdown"]
    finally:
        (world["tmp"] / "holder.ready.release").write_text("")
        holder.communicate(timeout=60)


def test_threads_in_one_process_are_also_serialised(world):
    with pipeline_run_lock("US"):
        res = dp.run_daily_pipeline(run_date=D, tickers=["AAPL"])   # same process, second handle
    assert res.status == "HALTED" and res.orders_generated == 0


def test_lock_location_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "pipeline_lock_dir", Path("data/locks"))
    here = lock_dir()
    monkeypatch.chdir(tmp_path)
    assert lock_dir() == here == PROJECT_ROOT / "data" / "locks"
