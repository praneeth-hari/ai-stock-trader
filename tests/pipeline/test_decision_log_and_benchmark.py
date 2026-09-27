"""V1-safe measurement: per-ticker decision log and the SPY >= 200-day SMA shadow benchmark."""

import hashlib
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.db import repository
from src.features.engineer import FEATURE_COLUMNS
import src.pipeline.daily_pipeline as dp
from src.portfolio.portfolio import OrderSpec
from src.ranking.ranking import TIER_BUY, TIER_NEUTRAL
from src.reports.decision_log import build_decision_records
from src.trading import shadow_benchmark as sb
from src.trading.paper_broker import PaperBroker, _pending_orders_cache

_DAYS = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(340)]
DAYS = [d.strftime("%Y-%m-%d") for d in _DAYS if d.weekday() < 5][:224]
PREV, D, D2, D3 = DAYS[-4], DAYS[-3], DAYS[-2], DAYS[-1]
REQUIRED = ("model_sha256", "raw_probability", "sector", "sector_rank", "sector_modifier", "macro_state",
            "regime_state", "correlation", "final_probability", "buy_threshold", "size_tier", "decision", "reason")


def _frame(ticker, closes, upto=None):
    d = [x for x in DAYS if upto is None or x <= upto]
    c = list(closes)[: len(d)] if not callable(closes) else [closes(i) for i in range(len(d))]
    n = len(d)
    return pd.DataFrame({"date": d, "open": [x * 0.995 for x in c], "high": [x * 1.01 for x in c],
                         "low": [x * 0.99 for x in c], "close": c, "volume": [2_000_000] * n, "ticker": [ticker] * n})


def _rising(i):
    return 400.0 * (1 + 0.001 * i)


def _wiggle(base):
    """Varying prices: perfectly flat series leave RSI/volatility features undefined, so nothing ranks."""
    return lambda i: base * (1 + 0.02 * np.sin(i / 4.0)) + 0.01 * i


def _model(probs_in_row_order):
    def f(X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.array([[1 - p, p] for p in (list(probs_in_row_order) * n)[:n]])
    m = MagicMock()
    m.predict_proba = MagicMock(side_effect=f)
    m.model = MagicMock()
    m.model.predict_proba = MagicMock(side_effect=f)
    m.model.feature_names_in_ = FEATURE_COLUMNS
    m.metadata = {"model_type": "baseline", "feature_columns": FEATURE_COLUMNS}
    return m


@pytest.fixture
def fresh_db(tmp_path):
    settings.db_url = f"sqlite:///{(tmp_path / 'obs.db').as_posix()}"
    repository._engine, repository._db_available = None, True
    repository.create_all_tables()
    _pending_orders_cache.clear()
    yield
    _pending_orders_cache.clear()


@pytest.fixture
def quiet():
    offline = RuntimeError("offline in test")
    ps = [patch.object(dp, n, MagicMock()) for n in dir(dp) if n.startswith(("send_", "notify_"))]
    ps += [patch.object(dp, n, side_effect=offline) for n in ("analyze_universe_sentiment",
           "get_universe_earnings_calendar", "calculate_sector_rotation", "get_macro_environment")]
    for p in ps:
        p.start()
    yield
    for p in ps:
        p.stop()


def _day0(pending=(), held=None):
    b = PaperBroker(market="US")
    b.load_state()
    if held:
        t, price = held
        b.execute_order(order=OrderSpec(date=DAYS[-7], ticker=t, action="BUY", order_type="MARKET", shares=10.0,
                                        reference_price=price, gross_value=10 * price, estimated_fee=0.0,
                                        net_amount=10 * price, reason="QUALIFIED_CONVICTION_BUY"),
                        fill_price=price, run_date=DAYS[-6])
    b.sync_pending_orders(list(pending))
    b.record_snapshot(run_date=PREV, current_prices={held[0]: held[1]} if held else {})


def _run(run_date, universe, spy, model):
    with patch.object(dp, "load_active_model", return_value=model):
        return dp.run_daily_pipeline(run_date=run_date, tickers=list(universe), _spy_df=spy, _universe_dfs=universe)


# ── Decision log ─────────────────────────────────────────────────────────────────

def test_every_ranked_ticker_gets_a_decision_row_with_all_required_fields(fresh_db, quiet):
    _day0()
    uni = {t: _frame(t, _wiggle(100.0), D) for t in ("AAPL", "MSFT", "JPM")}
    res = _run(D, uni, _frame("SPY", _rising, D), _model([0.80, 0.30, 0.30]))
    rows = {r["ticker"]: r for r in repository.get_decision_log(D)}
    assert set(rows) == {"AAPL", "MSFT", "JPM"}
    frozen = hashlib.sha256((Path(settings.data_models_dir) / "active_model.joblib").read_bytes()).hexdigest()
    for r in rows.values():
        assert all(k in r for k in REQUIRED)
        assert r["model_sha256"] == frozen
        assert r["decision"] in {"BUY", "HOLD", "SELL", "REJECT"} and r["reason"]
        assert {"regime", "size_multiplier", "buy_bar_shift"} <= set(r["macro_state"])
        assert {"spy_above_200d", "volatility_regime", "circuit_breaker_active"} <= set(r["regime_state"])
    buy = rows["AAPL"]
    assert buy["decision"] == "BUY" and buy["size_tier"] == "FULL" and "Won slot 1 of 3" in buy["order_reason"]
    assert [o.ticker for o in res.pending_orders] == ["AAPL"]
    assert rows["MSFT"]["decision"] == "REJECT" and rows["MSFT"]["reason"].startswith("BELOW_BUY_THRESHOLD")


def test_held_stop_loss_position_is_logged_as_sell_with_the_exit_rule(fresh_db, quiet):
    _day0(held=("AAPL", 100.0))
    uni = {"AAPL": _frame("AAPL", lambda i: 88.0 * (1 + 0.005 * np.sin(i / 3.0)), D)}   # about -12%: stop-loss
    _run(D, uni, _frame("SPY", _rising, D), _model([0.55]))
    row = repository.get_decision_log(D)[0]
    assert row["decision"] == "SELL" and row["reason"].startswith("STOP_LOSS") and "STOP_LOSS" in row["order_reason"]
    assert row["raw_probability"] is not None


def test_held_position_the_model_could_not_rank_is_still_logged(fresh_db, quiet):
    _day0(held=("AAPL", 100.0))
    uni = {"AAPL": _frame("AAPL", lambda i: 88.0, D)}          # flat: incomplete features, not ranked
    _run(D, uni, _frame("SPY", _rising, D), _model([0.55]))
    row = repository.get_decision_log(D)[0]
    assert row["ticker"] == "AAPL" and row["raw_probability"] is None and row["final_probability"] is None
    assert row["decision"] == "SELL" and row["reason"].startswith("UNRANKED (incomplete features); STOP_LOSS")


def _opp(t, p, tier=TIER_BUY, eligible=True, rank=1, **kw):
    return SimpleNamespace(ticker=t, probability=p, base_probability=p - 0.01, sector="Technology", sector_rank=2,
                           sector_modifier=0.01, conviction_tier=tier, is_buy_eligible=eligible, rank=rank,
                           rel_strength_21=0.05, price_to_ma50=0.02, sentiment_modifier=0.0,
                           sentiment_label=kw.get("sentiment", "NEUTRAL"), earnings_status=kw.get("earn", "OK"))


def _decision(t, action="VETO", veto=None, reason=None, details="", tier=None, alloc=0.0):
    return SimpleNamespace(ticker=t, action=action, veto_reason=veto, reason=reason, details=details,
                           confidence_tier=tier, allocated_amount=alloc)


def test_every_decision_branch_has_an_explicit_reason():
    ranking = SimpleNamespace(effective_buy_bar=0.60, regime_risk_on=True, ranked_opportunities=[
        _opp("WIN", 0.80), _opp("FULLUP", 0.78, rank=2), _opp("CORR", 0.77, rank=3), _opp("ADAPT", 0.61, rank=4),
        _opp("SENT", 0.70, eligible=False, rank=5, sentiment="VERY_NEGATIVE"), _opp("LOW", 0.40, TIER_NEUTRAL, False, 6),
        _opp("KEEP", 0.55, TIER_NEUTRAL, False, 7), _opp("EXIT", 0.30, TIER_NEUTRAL, False, 8)])
    risk = SimpleNamespace(
        exit_orders=[_decision("EXIT", "SELL", reason="SIGNAL_EXIT", details="P below exit")],
        held_unchanged=[_decision("KEEP", "HOLD", reason=None)],
        buy_orders=[_decision("WIN", "BUY", reason="QUALIFIED_CONVICTION_BUY", tier="FULL", alloc=2833.33)],
        vetoed_orders=[_decision("CORR", veto=SimpleNamespace(value="CORRELATION_VETO"), details="r=0.93 with KEEP"),
                       _decision("FULLUP", veto=SimpleNamespace(value="MAX_POSITIONS_REACHED"), details="no slots")],
        evaluated_candidates=["WIN", "FULLUP", "CORR", "ADAPT"],
        correlation_checks={"WIN": {"max_correlation": 0.41, "max_correlated_ticker": "KEEP", "warning": False,
                                    "allowed": True, "result": "CORRELATION_OK"}},
        active_buy_bar=0.63, volatility_regime="NORMAL_VOLATILITY", circuit_breaker_active=False, positions_after=2)
    orders = [SimpleNamespace(ticker="WIN", action="BUY"), SimpleNamespace(ticker="EXIT", action="SELL")]
    rows = {r["ticker"]: r for r in build_decision_records(ranking, risk, orders, ["KEEP", "EXIT"], None, {}, "abc")}
    assert rows["WIN"]["decision"] == "BUY" and "Won slot 1 of 2" in rows["WIN"]["order_reason"]
    assert "Lost out for lack of slots: ['FULLUP']" in rows["WIN"]["order_reason"]
    assert rows["FULLUP"]["reason"].startswith("MAX_POSITIONS_REACHED")
    assert rows["CORR"]["reason"].startswith("CORRELATION_VETO")
    assert rows["ADAPT"]["reason"].startswith("BELOW_ADAPTIVE_BUY_BAR")
    assert rows["SENT"]["reason"].startswith("SENTIMENT_VETO")
    assert rows["LOW"]["reason"].startswith("BELOW_BUY_THRESHOLD")
    assert rows["KEEP"]["decision"] == "HOLD" and rows["EXIT"]["decision"] == "SELL"
    assert all(r["buy_threshold"] == 0.63 for r in rows.values()), "records the binding bar: max(ranking, adaptive)"


# ── SPY >= 200-day SMA shadow benchmark ─────────────────────────────────────────

def _v1_state_by_day(benchmark_ok: bool, tmp_path):
    settings.db_url = f"sqlite:///{(tmp_path / f'iso_{benchmark_ok}.db').as_posix()}"
    repository._engine = None
    repository.create_all_tables()
    _pending_orders_cache.clear()
    _day0()
    probs = [[0.80, 0.30], [0.30, 0.80], [0.40, 0.40]]
    out = []
    ctx = patch.object(sb, "update_spy_200d_benchmark", side_effect=RuntimeError("benchmark exploded")) \
        if not benchmark_ok else patch.object(sb, "BENCHMARK_NAME", sb.BENCHMARK_NAME)
    with ctx:
        for day, p in zip((D, D2, D3), probs):
            _pending_orders_cache.clear()
            uni = {t: _frame(t, _wiggle(100.0), day) for t in ("AAPL", "MSFT")}
            r = _run(day, uni, _frame("SPY", _rising, day), _model(p))
            snap = repository.get_portfolio_snapshot(day, "US")
            out.append((r.status, sorted((o.ticker, o.action, o.shares) for o in r.pending_orders),
                        sorted((f.ticker, f.action, f.shares, f.fill_price) for f in r.fills),
                        snap["cash"], sorted(snap["positions"]), snap["pending_orders"]))
    return out, repository.get_benchmark_snapshots(sb.BENCHMARK_NAME)


def test_shadow_benchmark_cannot_alter_v1_orders_or_state(quiet, tmp_path):
    with_bench, bench_rows = _v1_state_by_day(True, tmp_path)
    without_bench, broken_rows = _v1_state_by_day(False, tmp_path)
    assert with_bench == without_bench, "V1 orders, fills, cash, positions and status identical"
    assert len(bench_rows) == 4 and broken_rows == []           # day 0 + 3 runs, and nothing when it fails
    assert any(o[1] for o in with_bench), "scenario actually produced V1 orders"


def test_benchmark_uses_only_point_in_time_data(fresh_db):
    _day0()
    base = _frame("SPY", _rising, D)
    future = _frame("SPY", lambda i: 50.0, None)
    future = future[future["date"] > D]                          # absurd future prices must be ignored
    row_clean = sb.update_spy_200d_benchmark(D, base)
    repository._engine = None
    settings.db_url = settings.db_url.replace("obs.db", "obs2.db")
    repository.create_all_tables()
    _day0()
    row_with_future = sb.update_spy_200d_benchmark(D, pd.concat([base, future], ignore_index=True))
    assert {k: v for k, v in row_clean.items() if k != "details"} == \
           {k: v for k, v in row_with_future.items() if k != "details"}
    prior = base[base["date"] < D]
    assert row_clean["decision_bar_date"] == prior["date"].iloc[-1]
    assert row_clean["sma_200"] == pytest.approx(prior["close"].tail(200).mean(), abs=1e-4)


def test_regime_is_decided_on_the_prior_close_not_the_run_date_bar(fresh_db):
    _day0()
    spy = _frame("SPY", _rising, D)
    spy.loc[spy.index[-1], ["open", "close"]] = [100.0, 100.0]   # run_date's own bar crashes below the SMA
    row = sb.update_spy_200d_benchmark(D, spy)
    assert row["regime_risk_on"] is True and row["decision"] == "BUY_NEXT_OPEN"


def test_both_portfolios_start_from_the_same_baseline(fresh_db, quiet):
    _day0()
    uni = {"AAPL": _frame("AAPL", _wiggle(100.0), D)}
    _run(D, uni, _frame("SPY", _rising, D), _model([0.3]))
    v1_day0 = repository.get_earliest_portfolio_snapshot("US")
    bench = repository.get_benchmark_snapshots(sb.BENCHMARK_NAME)
    assert bench[0]["run_date"] == v1_day0["run_date"] == PREV
    assert bench[0]["equity"] == v1_day0["total_value"] == settings.initial_capital
    assert bench[0]["decision"] == "START" and bench[1]["run_date"] == D


def test_benchmark_follows_v1_lag_rule_and_cost_convention(fresh_db):
    _day0()
    spy = _frame("SPY", _rising, D3)
    r1 = sb.update_spy_200d_benchmark(D, spy)                    # decides on the prior close: risk-on
    assert r1["decision"] == "BUY_NEXT_OPEN" and r1["spy_shares"] == 0 and r1["equity"] == 10000.0
    r2 = sb.update_spy_200d_benchmark(D2, spy)                   # fills at the run_date open, like V1
    ex = r2["details"]["executed"]
    open_px = float(spy.loc[spy["date"] == D2, "open"].iloc[0])
    assert ex["action"] == "BUY" and ex["price"] == pytest.approx(open_px * (1 + settings.slippage_tier_low_pct), abs=1e-4)
    assert ex["fee"] == pytest.approx(r2["spy_shares"] * ex["price"] * settings.simulated_cost_per_trade, abs=0.01)
    assert r2["cash"] >= 0 and r2["decision"] == "HOLD_SPY"
    assert r2["equity"] == pytest.approx(r2["cash"] + r2["spy_shares"] * r2["spy_close"], abs=1e-3)
    assert r2["daily_return"] == pytest.approx(r2["equity"] / 10000.0 - 1, abs=1e-6)
    crash = spy.copy()
    crash.loc[crash["date"] == D2, "close"] = 100.0              # prior close falls below the SMA
    r3 = sb.update_spy_200d_benchmark(D3, crash)
    assert r3["regime_risk_on"] is False and r3["decision"] == "SELL_NEXT_OPEN" and r3["pending_action"] == "SELL"
    assert r3["drawdown"] == pytest.approx(r3["equity"] / r3["peak_equity"] - 1.0, abs=1e-6) and r3["drawdown"] <= 0
    again = sb.update_spy_200d_benchmark(D3, crash)              # a re-run of a recorded day changes nothing
    assert again["equity"] == r3["equity"] and len(repository.get_benchmark_snapshots(sb.BENCHMARK_NAME)) == 4
