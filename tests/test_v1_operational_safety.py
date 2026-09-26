"""V1 operational safety: crash-consistent state (G3), exit codes (G4), intelligence status (G7),
DST-safe timing (G8) and the explicit model-failure rule (G9). None of these change a decision."""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.db import repository
from src.features.engineer import FEATURE_COLUMNS
from src.intelligence.macro import MacroEnvironmentResult
import src.pipeline.daily_pipeline as dp
import src.pipeline.scheduler as sch
from src.portfolio.portfolio import OrderSpec
from src.trading.paper_broker import PaperBroker, _pending_orders_cache

_DAYS = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(340)]
DAYS = [d.strftime("%Y-%m-%d") for d in _DAYS if d.weekday() < 5][:224]
D, D2, PREV = DAYS[-2], DAYS[-1], DAYS[-3]


def _frame(ticker, close, upto):
    d = [x for x in DAYS if x <= upto]
    n = len(d)
    return pd.DataFrame({"date": d, "open": [close * 0.995] * n, "high": [close * 1.01] * n,
                         "low": [close * 0.99] * n, "close": [close] * n, "volume": [500_000] * n,
                         "ticker": [ticker] * n})


def _model(p=0.5):
    m = MagicMock()
    f = lambda X: np.array([[1 - p, p]] * (len(X) if hasattr(X, "__len__") else 1))
    m.predict_proba = MagicMock(side_effect=f)
    m.model = MagicMock()
    m.model.predict_proba = MagicMock(side_effect=f)
    m.model.feature_names_in_ = FEATURE_COLUMNS
    m.metadata = {"model_type": "baseline", "feature_columns": FEATURE_COLUMNS}
    return m


def _buy(ticker, on, shares=10.0, price=100.0):
    gross = shares * price
    return OrderSpec(date=on, ticker=ticker, action="BUY", order_type="MARKET", shares=shares,
                     reference_price=price, gross_value=gross, estimated_fee=gross * 0.002,
                     net_amount=gross * 1.002, reason="QUALIFIED_CONVICTION_BUY")


@pytest.fixture
def fresh_db(tmp_path):
    settings.db_url = f"sqlite:///{(tmp_path / 'v1_ops.db').as_posix()}"
    repository._engine, repository._db_available = None, True
    repository.create_all_tables()
    _pending_orders_cache.clear()
    yield
    _pending_orders_cache.clear()


@pytest.fixture
def quiet():
    """No Telegram/email and no live intelligence calls (offline -> neutral fallback, recorded as FAILED)."""
    offline = RuntimeError("offline in test")
    patches = [patch.object(dp, n, MagicMock()) for n in dir(dp) if n.startswith(("send_", "notify_"))]
    patches += [patch.object(dp, n, side_effect=offline) for n in
                ("analyze_universe_sentiment", "get_universe_earnings_calendar",
                 "calculate_sector_rotation", "get_macro_environment")]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def _commit_day0(pending, positions_prices=None):
    b = PaperBroker(market="US")
    b.load_state()
    b.sync_pending_orders(pending)
    b.record_snapshot(run_date=PREV, current_prices=positions_prices or {})
    return b


def _run(run_date, upto=None, model=None, tickers=("AAPL",)):
    upto = upto or run_date
    universe = {t: _frame(t, 100.0, upto) for t in tickers}
    with patch.object(dp, "load_active_model", return_value=model or _model()):
        return dp.run_daily_pipeline(run_date=run_date, tickers=list(tickers),
                                     _spy_df=_frame("SPY", 500.0, upto), _universe_dfs=universe)


def _us_trades():
    return [t for t in repository.get_trades(limit=100, market="US")]


# ── G3: crash-consistent state ──────────────────────────────────────────────────

def test_snapshot_commits_pending_orders_in_the_same_row(fresh_db):
    _commit_day0([_buy("AAPL", PREV)])
    snap = repository.get_latest_portfolio_snapshot("US")
    assert [o["ticker"] for o in snap["pending_orders"]] == ["AAPL"]


def test_committed_pending_orders_survive_a_cleared_mirror(fresh_db):
    b = _commit_day0([_buy("AAPL", PREV)])
    b.clear_pending_orders()          # what a crashed run does to the file/cache mirror before filling
    fresh = PaperBroker(market="US")
    fresh.load_state()
    assert [o.ticker for o in fresh.pending_orders] == ["AAPL"]


@pytest.mark.parametrize("recover_on", ["same_day", "next_day"])
def test_crash_after_fills_is_rolled_back_and_reexecuted_exactly_once(fresh_db, quiet, recover_on):
    _commit_day0([_buy("AAPL", PREV)])
    with patch.object(dp, "evaluate_portfolio_risk", side_effect=RuntimeError("simulated crash after fills")):
        crashed = _run(D)
    assert crashed.status == "FAILED"
    assert len(_us_trades()) == 1, "the crashed run did record its fill"
    assert repository.get_portfolio_snapshot(D, "US") is None, "but never committed the day"

    rerun_date = D if recover_on == "same_day" else D2
    rec = _run(rerun_date)
    trades = _us_trades()
    snap = repository.get_latest_portfolio_snapshot("US")
    assert len(trades) == 1 and trades[0]["ticker"] == "AAPL" and trades[0]["action"] == "BUY"
    assert trades[0]["run_date"] == rerun_date
    assert snap["run_date"] == rerun_date and list(snap["positions"]) == ["AAPL"]
    assert snap["pending_orders"] == []
    assert snap["cash"] == pytest.approx(10000.0 - float(trades[0]["quantity"]) * float(trades[0]["fill_price"])
                                         - float(trades[0]["cost"]), abs=0.01)
    assert rec.status == "DEGRADED" and any(e.startswith("RECOVERY") for e in rec.errors)
    assert any(e.get("component") == "recovery_us" for e in repository.get_events(limit=50))


def test_crash_in_the_middle_of_fills_leaves_no_duplicate_or_missing_execution(fresh_db, quiet):
    _commit_day0([_buy("AAPL", PREV), _buy("MSFT", PREV)])
    real_save_trade = repository.save_trade
    calls = {"n": 0}

    def dies_on_second(**kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("process killed mid-fill")
        return real_save_trade(**kw)

    with patch.object(repository, "save_trade", side_effect=dies_on_second):
        with pytest.raises(RuntimeError):
            _run(D, tickers=("AAPL", "MSFT"))
    _run(D, tickers=("AAPL", "MSFT"))
    trades = _us_trades()
    assert sorted(t["ticker"] for t in trades) == ["AAPL", "MSFT"]
    assert sorted(repository.get_latest_portfolio_snapshot("US")["positions"]) == ["AAPL", "MSFT"]


def test_an_order_is_never_filled_in_the_run_that_created_it(fresh_db, quiet):
    _commit_day0([_buy("AAPL", D)])      # dated D: can only come from an interrupted run of D
    res = _run(D)
    assert _us_trades() == [] and res.fills == []
    assert any("LAG RULE" in (e.get("message") or "") for e in repository.get_events(limit=50))


# ── G4: status and exit codes ───────────────────────────────────────────────────

def _result(status=None, regime="RISK-ON", errors=()):
    return dp.DailyPipelineResult(D, 2, 2, 0, regime, 0, [], [], 1e4, 1e4, list(errors), status=status)


@pytest.mark.parametrize("status,code", [
    ("SUCCESS", 0), ("SKIPPED", 0), ("DEGRADED", 2), ("FAILED", 1), ("HALTED", 3), (None, 1),
])
def test_process_exit_code_reflects_pipeline_status(status, code):
    result = _result(status) if status else None
    with patch.object(sch, "execute_scheduled_job", return_value=result), \
         patch.object(sys, "argv", ["scheduler", "--force"]):
        with pytest.raises(SystemExit) as exc:
            sch.main()
    assert exc.value.code == code


def test_status_is_derived_for_early_returns():
    assert _result(regime="ALREADY_RUN").status == "SKIPPED"
    assert _result(regime="SKIPPED_MARKET_CLOSED", errors=["Market Closed"]).status == "SKIPPED"
    assert _result(regime="KILL_SWITCH_ACTIVE", errors=["kill"]).status == "HALTED"
    assert _result(regime="UNKNOWN", errors=["CRITICAL: SPY fetch failed"]).status == "FAILED"


def test_scheduler_event_and_dashboard_show_degraded(fresh_db):
    from dashboard.data_loader import get_last_pipeline_status

    with patch.object(sch, "run_daily_pipeline", return_value=_result("DEGRADED", errors=["SKIP MSFT: bad data"])):
        sch.execute_scheduled_job(run_date=D, force=True)
    ev = [e for e in repository.get_events(limit=10) if e.get("component") == "scheduler"][0]
    assert "Status=DEGRADED" in ev["message"]
    assert get_last_pipeline_status()["status"] == "DEGRADED"


# ── G7: intelligence status ─────────────────────────────────────────────────────

def test_failed_intelligence_is_recorded_and_marks_the_run_degraded(fresh_db, quiet):
    _commit_day0([])
    res = _run(D)
    assert {k: v["status"] for k, v in res.intelligence.items()} == \
        {"sentiment": "FAILED", "earnings": "FAILED", "sector": "FAILED", "macro": "FAILED"}
    assert res.status == "DEGRADED"
    msgs = [e.get("message") or "" for e in repository.get_events(limit=50)]
    assert sum(m.startswith("INTELLIGENCE_FAILED") for m in msgs) == 4


def _intel_stubs(unavailable=(), macro_source="2026-09-01", unknown_earnings=()):
    sent = SimpleNamespace(scores={
        "AAPL": SimpleNamespace(sentiment_label="UNAVAILABLE" if "AAPL" in unavailable else "NEUTRAL",
                                modifier=0.0, composite_score=0.0, is_veto=False)})
    info = SimpleNamespace(status="UNKNOWN" if "AAPL" in unknown_earnings else "OK", earnings_date=None,
                           days_until_earnings=None, badge_text="", is_blocked=False)
    earn = SimpleNamespace(calendar={"AAPL": info}, get_info=lambda t: info, has_hold_protection=lambda t: False)
    sector = SimpleNamespace(rankings=[SimpleNamespace(sector="Technology", stock_count=1)],
                             get_multiplier_for_ticker=lambda t: 0.0, get_rank_for_ticker=lambda t: 3)
    macro = MacroEnvironmentResult(date=D, fed_funds_rate=2.0, cpi_yoy=2.0, unemployment_rate=4.0, treasury_10y=3.0,
                                   macro_regime="FAVORABLE", position_size_multiplier=1.0, buy_bar_shift=0.0,
                                   is_cached=False, source_date=macro_source)
    return [patch.object(dp, "analyze_universe_sentiment", return_value=sent),
            patch.object(dp, "get_universe_earnings_calendar", return_value=earn),
            patch.object(dp, "calculate_sector_rotation", return_value=sector),
            patch.object(dp, "get_macro_environment", return_value=macro)]


@pytest.mark.parametrize("kwargs,expected,overall", [
    ({}, {"sentiment": "SUCCESS", "earnings": "SUCCESS", "sector": "SUCCESS", "macro": "SUCCESS"}, "SUCCESS"),
    ({"unavailable": ("AAPL",), "macro_source": "BASELINE_DEFAULT"},
     {"sentiment": "DEGRADED", "earnings": "SUCCESS", "sector": "SUCCESS", "macro": "DEGRADED"}, "DEGRADED"),
    ({"unknown_earnings": ("AAPL",)},
     {"sentiment": "SUCCESS", "earnings": "DEGRADED", "sector": "SUCCESS", "macro": "SUCCESS"}, "DEGRADED"),
])
def test_neutral_fallbacks_are_distinguishable_from_real_intelligence(fresh_db, quiet, kwargs, expected, overall):
    _commit_day0([])
    stubs = _intel_stubs(**kwargs)
    for s in stubs:
        s.start()
    try:
        res = _run(D)
    finally:
        for s in stubs:
            s.stop()
    assert {k: v["status"] for k, v in res.intelligence.items()} == expected
    assert res.status == overall


# ── G8: DST-safe timing ─────────────────────────────────────────────────────────

def test_scheduled_run_time_is_after_the_close_in_both_us_dst_regimes():
    ist = ZoneInfo("Asia/Kolkata")
    for d in ("2026-07-01", "2026-12-01"):     # EDT, EST
        assert sch.is_after_market_close(datetime.fromisoformat(f"{d}T03:30:00").replace(tzinfo=ist))[0]
    # The old 2:00 AM IST start was before the close once the US left daylight time.
    assert not sch.is_after_market_close(datetime.fromisoformat("2026-12-01T02:00:00").replace(tzinfo=ist))[0]
    setup = Path(__file__).resolve().parents[1] / "setup_task_scheduler.ps1"
    text = setup.read_text(encoding="utf-8")
    assert "weekly /d MON,TUE,WED,THU,FRI /st 03:30" in text and "StartWhenAvailable = $true" in text


def test_unfinished_bar_is_refused():
    ny_today = datetime.now(ZoneInfo(settings.market_hours_timezone)).strftime("%Y-%m-%d")
    today_df = pd.DataFrame({"date": [ny_today]})
    with patch.object(sch, "is_after_market_close", return_value=(False, "before 16:30")):
        with pytest.raises(ValueError, match="INCOMPLETE_BAR"):
            dp._assert_latest_bar_complete(today_df, "SPY")
        dp._assert_latest_bar_complete(pd.DataFrame({"date": ["2000-01-03"]}), "SPY")   # finished bar: ok
    with patch.object(sch, "is_after_market_close", return_value=(True, "after 16:30")):
        dp._assert_latest_bar_complete(today_df, "SPY")


def test_live_run_on_an_unfinished_bar_fails_with_zero_orders(fresh_db, quiet):
    _commit_day0([_buy("AAPL", PREV)])
    ny_today = datetime.now(ZoneInfo(settings.market_hours_timezone)).strftime("%Y-%m-%d")
    spy = _frame("SPY", 500.0, D)
    spy.loc[spy.index[-1], "date"] = ny_today
    with patch.object(dp, "fetch_ticker_data", return_value=spy), \
         patch.object(sch, "is_after_market_close", return_value=(False, "before 16:30")):
        res = dp.run_daily_pipeline(run_date=D2, tickers=["AAPL"])
    assert res.status == "FAILED" and res.orders_generated == 0 and not res.fills
    assert any("INCOMPLETE_BAR" in e for e in res.errors)
    assert _us_trades() == []


# ── G9: model failure while positions are held ──────────────────────────────────

def test_model_failure_halts_all_and_leaves_held_positions_and_pending_orders_untouched(fresh_db, quiet, tmp_path):
    held = PaperBroker(market="US")
    held.load_state()
    held.execute_order(order=_buy("AAPL", DAYS[-5]), fill_price=100.0, run_date=DAYS[-4])
    held.sync_pending_orders([_buy("MSFT", PREV)])
    held.record_snapshot(run_date=PREV, current_prices={"AAPL": 100.0})
    before = repository.get_latest_portfolio_snapshot("US")
    n_trades = len(_us_trades())

    bad_models = tmp_path / "models"
    bad_models.mkdir()
    (bad_models / "active_model.joblib").write_bytes(b"not the frozen model")
    with patch.object(settings, "data_models_dir", bad_models), patch.object(settings, "model_freeze_enabled", True):
        res = dp.run_daily_pipeline(run_date=D, tickers=["AAPL", "MSFT"], _spy_df=_frame("SPY", 500.0, D),
                                    _universe_dfs={t: _frame(t, 70.0, D) for t in ("AAPL", "MSFT")})

    assert res.status == "FAILED" and not res.fills and res.orders_generated == 0
    after = repository.get_latest_portfolio_snapshot("US")
    assert after == before, "no snapshot written; positions and pending orders exactly as committed"
    assert len(_us_trades()) == n_trades, "no exit (AAPL is down 30%, past the stop-loss) and no pending fill"
    crit = [e for e in repository.get_events(limit=50) if e.get("level") == "CRITICAL"][0]
    assert crit["details"]["policy"] == dp.MODEL_FAILURE_POLICY == "HALT_ALL"
    assert crit["details"]["held_positions"] == ["AAPL"] and crit["details"]["pending_orders_untouched"] == 1
