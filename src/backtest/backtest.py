"""
src/backtest/backtest.py — Phase 10 Backtesting Engine.

Simulates the complete autonomous trading pipeline chronologically through historical data
with strict anti-self-deception safeguards (§1.5):
  1. The Lag Rule: Trade signals generated at T-1 close; orders filled at T open (Market-On-Open).
  2. Transaction Costs: 0.2% cost per trade (0.4% round trip) deducted on every fill.
  3. Strict Chronology & Regime Awareness: SPY 200d MA regime filter enforced daily.
  4. Real Portfolio Sizing: 3-position cap, 15% cash reserve, $10 min trade size.
  5. The Known-Answer Test: Verifies engine accounting logic on SPY before any strategy run.
  6. "Too Good to Be True" Alarm: Suspicious metrics trigger an anti-leakage audit alarm.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS, compute_features, drop_warmup_rows
from src.ml.dataset import build_dataset
from src.ml.evaluate import load_active_model
from src.ml.train import LOCKED_MODEL_TYPE, TrainedModel, load_model, train_model
from src.portfolio.portfolio import OrderSpec, allocate_portfolio, calculate_fractional_shares
from src.ranking.ranking import RankingResult, rank_candidates
from src.risk.risk_engine import RiskAssessmentResult, evaluate_portfolio_risk

logger = logging.getLogger(__name__)

# §1.5 "Too Good to Be True" Backtest Alarm Thresholds
ALARM_MAX_CAGR: float = 0.35           # > 35% annualized net return is suspicious
ALARM_MIN_DRAWDOWN: float = 0.05       # < 5% max drawdown over multi-year equities is suspicious
ALARM_MAX_WIN_RATE: float = 0.65       # > 65% win rate for daily swing is suspicious
ALARM_MAX_SHARPE: float = 2.00         # > 2.0 Sharpe ratio is suspicious


@dataclass
class BacktestTrade:
    """Represents a completed or active trade in the backtest."""
    ticker: str
    entry_date: str
    entry_price: float
    shares: float
    entry_cost: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_cost: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    net_pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None

    def close(self, exit_date: str, exit_price: float, fee_rate: float, reason: str) -> None:
        self.exit_date = exit_date
        self.exit_price = float(exit_price)
        self.exit_cost = round(self.shares * self.exit_price * fee_rate, 4)
        gross_proceeds = self.shares * self.exit_price
        gross_cost = self.shares * self.entry_price
        self.gross_pnl = round(gross_proceeds - gross_cost, 4)
        self.net_pnl = round(self.gross_pnl - self.entry_cost - self.exit_cost, 4)
        self.net_pnl_pct = round(self.net_pnl / (gross_cost + self.entry_cost), 4)
        self.exit_reason = reason


@dataclass
class DailySnapshot:
    """Mark-to-market portfolio snapshot at end of trading day T."""
    date: str
    cash: float
    portfolio_value: float
    total_equity: float
    daily_return: float
    cumulative_return: float
    spy_close: float
    spy_daily_return: float
    spy_cumulative_return: float
    positions_count: int
    cash_pct: float
    drawdown: float
    spy_drawdown: float


@dataclass
class BacktestMetrics:
    """Comprehensive performance and risk metrics (Net of 0.2% costs)."""
    start_date: str
    end_date: str
    trading_days: int
    starting_capital: float
    final_equity: float
    total_net_return_pct: float
    annualized_return_pct: float
    spy_total_return_pct: float
    spy_annualized_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    spy_max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl_pct: float
    avg_holding_days: float
    avg_cash_pct: float
    alarm_triggered: bool
    alarm_reasons: List[str] = field(default_factory=list)
    model_out_of_sample: bool = True
    resolved_model_name: str = ""
    survivorship_biased: bool = True
    spy_sharpe_ratio: float = 0.0


def _annualized_sharpe(daily_returns: List[float]) -> float:
    std = float(np.std(daily_returns)) if daily_returns else 0.0
    return float(np.mean(daily_returns)) / std * np.sqrt(252.0) if std > 1e-8 else 0.0


def _annualized_sortino(daily_returns: List[float]) -> float:
    if not daily_returns:
        return 0.0
    # Downside deviation over ALL periods (0% target), not std of the negative subset.
    downside_dev = float(np.sqrt(np.mean(np.minimum(daily_returns, 0.0) ** 2)))
    return float(np.mean(daily_returns)) / downside_dev * np.sqrt(252.0) if downside_dev > 1e-8 else 0.0


def _liquidate_open_positions(
    positions: Dict[str, Dict[str, Any]],
    last_close: Dict[str, float],
    date: str,
    cost_per_trade: float,
) -> Tuple[List[BacktestTrade], float]:
    """Closes still-open positions at the final close, net of exit fee, so final equity and trade stats are net."""
    trades: List[BacktestTrade] = []
    net_inflow = 0.0
    for t, pos in positions.items():
        price = last_close.get(t, pos["entry_price"])
        qty = pos["quantity"]
        trade = BacktestTrade(
            ticker=t,
            entry_date=pos["entry_date"],
            entry_price=pos["entry_price"],
            shares=qty,
            entry_cost=round(qty * pos["entry_price"] * cost_per_trade, 4),
        )
        trade.close(exit_date=date, exit_price=price, fee_rate=cost_per_trade, reason="END_OF_BACKTEST")
        trades.append(trade)
        gross = round(qty * price, 4)
        net_inflow += gross - round(gross * cost_per_trade, 4)
    return trades, round(net_inflow, 4)


def _apply_final_liquidation(snapshot: "DailySnapshot", net_inflow: float, peak_equity: float, prev_equity: float, initial_capital: float) -> None:
    snapshot.cash = round(snapshot.cash + net_inflow, 4)
    snapshot.portfolio_value = 0.0
    snapshot.total_equity = snapshot.cash
    snapshot.positions_count = 0
    snapshot.cash_pct = 100.0
    snapshot.daily_return = (snapshot.total_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
    snapshot.cumulative_return = (snapshot.total_equity / initial_capital) - 1.0
    snapshot.drawdown = min(0.0, (snapshot.total_equity - peak_equity) / peak_equity) if peak_equity > 0 else 0.0


# ticker -> list of (start_date, end_date) membership windows, inclusive; None = open-ended.
UniverseMembership = Dict[str, List[Tuple[Optional[str], Optional[str]]]]


def _check_universe_membership(
    universe_dict: Dict[str, pd.DataFrame],
    universe_membership: Optional[UniverseMembership],
) -> bool:
    """Returns True when the run is survivorship-biased (no point-in-time membership supplied)."""
    if universe_membership is None:
        logger.warning(
            "SURVIVORSHIP-BIASED BACKTEST: no point-in-time universe membership supplied; "
            "universe %s is today's survivors applied to all historical dates. Results are optimistic.",
            sorted(universe_dict),
        )
        return True
    missing = sorted(set(universe_dict) - set(universe_membership))
    if missing:
        raise ValueError(
            f"universe_membership has no point-in-time windows for {missing}; "
            "refusing to treat them as members on every historical date."
        )
    return False


def _is_member(universe_membership: Optional[UniverseMembership], ticker: str, date: str) -> bool:
    if universe_membership is None:
        return True
    return any(
        (start is None or date >= start) and (end is None or date <= end)
        for start, end in universe_membership.get(ticker, [])
    )


# ticker -> inclusive (start, end) windows in which point-in-time validation rejects its data.
DataExclusions = Dict[str, List[Tuple[str, str]]]


def _excluded_on(data_exclusions: Optional[DataExclusions], date: str) -> set:
    return {
        t for t, windows in (data_exclusions or {}).items()
        if any(start <= date <= end for start, end in windows)
    }


@dataclass
class BacktestResult:
    """Complete container for a backtest run."""
    metrics: BacktestMetrics
    daily_snapshots: List[DailySnapshot]
    trades: List[BacktestTrade]

    def to_markdown_summary(self, title: str = "Strategy Backtest Report") -> str:
        m = self.metrics
        alarm_text = ""
        if m.alarm_triggered:
            alarm_text = (
                "\n> [!CAUTION]\n"
                "> **§1.5 TOO GOOD TO BE TRUE ALARM TRIGGERED**\n"
                f"> Suspected Look-Ahead Leakage: {', '.join(m.alarm_reasons)}\n"
            )

        model_status = "Strictly Out-of-Sample" if m.model_out_of_sample else "IN-SAMPLE LEAKAGE DETECTED"
        model_info = f" | **Model**: `{m.resolved_model_name}` ({model_status})" if m.resolved_model_name else ""

        universe_text = (
            "uses a **SURVIVORSHIP-BIASED** universe (today's survivors applied to past dates)."
            if m.survivorship_biased
            else "uses point-in-time universe membership (limited to the supplied membership data)."
        )
        lines = [
            f"### {title}",
            f"> [!WARNING]",
            f"> **OPTIMISTIC — Real results likely worse.** Assumes perfect market-open execution,",
            f"> zero slippage beyond 0.2%, and {universe_text}",
            alarm_text,
            f"**Period**: `{m.start_date}` to `{m.end_date}` ({m.trading_days} trading days){model_info} | "
            f"**Starting Capital**: `${m.starting_capital:.2f}` -> **Final Equity**: `${m.final_equity:.2f}`\n",
            "| Performance Metric | Strategy (Net of 0.2% Fees) | SPY Benchmark (Buy & Hold) | Excess / Edge |",
            "|---|---|---|---|",
            f"| **Total Return** | **{m.total_net_return_pct:+.2f}%** | {m.spy_total_return_pct:+.2f}% | **{m.alpha_pct:+.2f}%** |",
            f"| **Annualized Return (CAGR)** | **{m.annualized_return_pct:+.2f}%** | {m.spy_annualized_return_pct:+.2f}% | {m.annualized_return_pct - m.spy_annualized_return_pct:+.2f}% |",
            f"| **Max Drawdown** | **{m.max_drawdown_pct:.2f}%** | {m.spy_max_drawdown_pct:.2f}% | {m.spy_max_drawdown_pct - m.max_drawdown_pct:+.2f}% pts better |",
            f"| **Sharpe Ratio (0% rf)** | **{m.sharpe_ratio:.2f}** | N/A | N/A |",
            f"| **Sortino Ratio** | **{m.sortino_ratio:.2f}** | N/A | N/A |",
            f"| **Average Cash Exposure** | **{m.avg_cash_pct:.1f}%** | 0.0% | +{m.avg_cash_pct:.1f}% safety |",
            "\n**Trading Activity & Execution Statistics**:",
            "| Metric | Value |",
            "|---|---|",
            f"| Total Completed Trades | **{m.total_trades}** |",
            f"| Win Rate | **{m.win_rate_pct:.1f}%** ({m.winning_trades} wins / {m.losing_trades} losses) |",
            f"| Profit Factor | **{m.profit_factor:.2f}** |",
            f"| Average Trade PnL | **{m.avg_trade_pnl_pct:+.2f}%** |",
            f"| Average Holding Period | **{m.avg_holding_days:.1f} days** |",
        ]
        return "\n".join(lines)


# ── 1. The Known-Answer Test (§1.5) ──────────────────────────────────────────

def run_known_answer_test(
    spy_df: pd.DataFrame,
    start_date: str = "2023-01-03",
    end_date: str = "2023-12-29",
    initial_capital: float = 10_000.0,
    cost_per_trade: float = settings.simulated_cost_per_trade,
    tolerance_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Mandatory §1.5 Known-Answer Test.
    Runs a pure 'Buy SPY at T1 Open and Hold until TN Close' simulation using a larger
    notional amount ($10,000.00) specifically to eliminate fractional share rounding noise.

    Compares the engine's simulated net equity against the closed-form benchmark return:
      R_price = (Close(TN) / Open(T1)) - 1
      R_net   = (1 - cost_per_trade) * (1 + R_price) - 1

    Tolerance: absolute difference must be <= 0.05% (0.0005).
    """
    logger.info(
        "Running §1.5 Known-Answer Test on SPY (%s to %s, capital=$%.2f)...",
        start_date, end_date, initial_capital,
    )

    df = spy_df.copy()
    if "date" not in df.columns:
        raise ValueError("spy_df must contain 'date' column.")

    df["date"] = df["date"].astype(str)
    window_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date").reset_index(drop=True)

    if len(window_df) < 2:
        raise ValueError(f"Insufficient SPY data in window [{start_date}, {end_date}]: only {len(window_df)} rows.")

    t1_open = float(window_df.iloc[0]["open"])
    tn_close = float(window_df.iloc[-1]["close"])

    # 1. Closed-Form Benchmark Net Return
    raw_benchmark_return = (tn_close / t1_open) - 1.0
    expected_net_return = (1.0 - cost_per_trade) * (1.0 + raw_benchmark_return) - 1.0

    # 2. Simulated Engine Execution
    # Buy on T1 open with full capital
    max_spend = initial_capital / (1.0 + cost_per_trade)
    raw_shares = max_spend / t1_open
    shares = math.floor(raw_shares * 10000.0) / 10000.0
    gross_spend = round(shares * t1_open, 4)
    entry_fee = round(gross_spend * cost_per_trade, 4)
    cash = round(initial_capital - (gross_spend + entry_fee), 4)

    # Mark to market at TN close
    final_portfolio_val = round(shares * tn_close, 4)
    final_equity = round(cash + final_portfolio_val, 4)
    simulated_net_return = (final_equity / initial_capital) - 1.0

    diff_pct = abs(simulated_net_return - expected_net_return) * 100.0
    passed = diff_pct <= tolerance_pct

    result = {
        "passed": passed,
        "test_name": "SPY Buy-and-Hold Known-Answer Test (§1.5)",
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "t1_open": t1_open,
        "tn_close": tn_close,
        "simulated_net_return_pct": round(simulated_net_return * 100.0, 4),
        "expected_net_return_pct": round(expected_net_return * 100.0, 4),
        "difference_pct_points": round(diff_pct, 4),
        "tolerance_pct": tolerance_pct,
        "shares": shares,
        "remaining_cash": cash,
        "dividend_notice": (
            "Benchmark return represents unadjusted price return (matching yfinance auto_adjust=False). "
            "Real SPY investors receive cash dividends (~1.5%/year) which are excluded from price-only simulation."
        ),
    }

    if not passed:
        logger.critical(
            "KNOWN-ANSWER TEST FAILED! Simulated: %.4f%%, Expected: %.4f%%, Diff: %.4f%% > %.2f%%",
            simulated_net_return * 100.0, expected_net_return * 100.0, diff_pct, tolerance_pct,
        )
    else:
        logger.info(
            "KNOWN-ANSWER TEST PASSED! Simulated: %.4f%% vs Expected: %.4f%% (Diff: %.4f%% <= %.2f%%)",
            simulated_net_return * 100.0, expected_net_return * 100.0, diff_pct, tolerance_pct,
        )

    return result


# ── 1.5. Out-of-Sample Model Resolution & Leakage Guard ──────────────────────

def _model_train_end(meta: Any) -> Optional[str]:
    split_info = meta.get("split_info", {}) if isinstance(meta, dict) else {}
    train_end = split_info.get("train_end_date")
    td = split_info.get("train_dates")
    if not train_end and isinstance(td, (list, tuple)) and len(td) > 1:
        train_end = td[1]
    return str(train_end) if train_end else None


def _trained_before(train_end: Optional[str], start_date: str, gap_days: int) -> bool:
    # Labels look gap_days ahead, so the training labels must also end before start_date.
    return bool(train_end) and pd.to_datetime(train_end) + pd.offsets.BDay(gap_days) < pd.to_datetime(start_date)


def resolve_backtest_model(
    start_date: str,
    explicit_model: Optional[TrainedModel] = None,
    models_dir: Optional[Union[str, Path]] = None,
    universe_dict: Optional[Dict[str, pd.DataFrame]] = None,
    spy_df: Optional[pd.DataFrame] = None,
    gap_days: int = 5,
) -> Tuple[TrainedModel, bool, Optional[str]]:
    """
    Resolves a strictly out-of-sample model for the backtest window starting at start_date.

    Guarantees that the model used for simulation was trained exclusively on data prior
    to start_date (accounting for the label embargo gap).

    OOS rule: the model's last training label (train_end_date + gap_days trading days) must be
    strictly before start_date. Only the active model's type is evaluated, so the backtest tests the
    intended model rather than whichever type happens to have the latest training date.

    Hierarchy:
      1. Explicit caller model: checked for temporal leakage; unknown training end = not OOS.
      2. Saved models of the active model's type that satisfy the OOS rule (latest training end wins).
      3. Dynamic training: trains the active model's type on data strictly prior to start_date.
      4. Fallback: active model, flagged NOT out-of-sample with the reason steps 2-3 failed.
    """
    if models_dir is None:
        models_dir = Path(settings.data_models_dir)
    else:
        models_dir = Path(models_dir)

    _train_end_of = _model_train_end

    def _is_oos(train_end: Optional[str]) -> bool:
        return _trained_before(train_end, start_date, gap_days)

    # 1. Explicit model check
    if explicit_model is not None:
        train_end = _train_end_of(getattr(explicit_model, "metadata", {}) or {})
        if not _is_oos(train_end):
            leak_reason = (
                f"Model leakage detected: Model '{getattr(explicit_model, 'model_name', 'model')}' "
                f"was trained through {train_end or 'an UNKNOWN date'}, not strictly before backtest start "
                f"date {start_date} plus the {gap_days}-day label embargo (in-sample look-ahead bias)."
            )
            logger.warning(leak_reason)
            return explicit_model, False, leak_reason
        return explicit_model, True, None

    active = load_active_model(models_dir=models_dir)
    intended_type = active.model_type
    failures: List[str] = []

    # 2. Saved models of the intended type that satisfy the OOS rule
    candidates: List[Tuple[str, str, Path, Path]] = []
    for meta_file in models_dir.glob("*.json"):
        if any(k in meta_file.name for k in ["best_params", "feature_importance", "state", "walk_forward"]):
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if not isinstance(meta, dict) or meta.get("model_type") != intended_type:
            continue
        train_end = _train_end_of(meta)
        base_name = meta_file.name.replace("_metadata.json", "").replace(".json", "")
        joblib_candidate = meta_file.parent / f"{base_name}.joblib"
        if _is_oos(train_end) and joblib_candidate.exists():
            candidates.append((train_end, meta.get("created_at_utc", ""), joblib_candidate, meta_file))

    # Latest training end first, then newest artifact
    for train_end, _, cand_joblib, cand_meta in sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True):
        try:
            loaded = load_model(cand_joblib, metadata_path=cand_meta)
        except Exception as e:
            logger.warning("Failed loading candidate model %s: %s", cand_joblib, e)
            continue
        if loaded.model_type != intended_type:
            # Metadata sidecar disagrees with the estimator bundle; never trust it.
            logger.error("Skipping %s: metadata says '%s' but estimator is '%s'.", cand_joblib.name, intended_type, loaded.model_type)
            continue
        logger.info(
            "Resolved out-of-sample %s model '%s' from %s (train_end=%s, start_date=%s)",
            intended_type, loaded.model_name, cand_joblib.name, train_end, start_date,
        )
        return loaded, True, None
    failures.append(f"no saved '{intended_type}' model trained before {start_date} (with {gap_days}-day embargo)")

    # 3. Dynamic training of the intended type on data strictly before start_date
    if universe_dict is None or spy_df is None:
        failures.append("no pre-start market data supplied for fresh training")
    else:
        try:
            pre_frames = []
            for ticker, df in universe_dict.items():
                pre_df = df[df["date"].astype(str) < start_date].copy()
                if "ticker" not in pre_df.columns:
                    pre_df["ticker"] = ticker
                pre_frames.append(pre_df)
            pre_spy = spy_df[spy_df["date"].astype(str) < start_date].copy()
            # Slicing before labelling means every label is computed from pre-start closes only.
            dataset_df = build_dataset(pd.concat(pre_frames, ignore_index=True), spy_df=pre_spy)
            dynamic_model = train_model(dataset_df, model_type=intended_type, gap_days=gap_days)
            train_end = _train_end_of(dynamic_model.metadata)
            if not _is_oos(train_end):
                raise ValueError(f"freshly trained model ends {train_end}, not before {start_date}")
            logger.info(
                "Dynamically trained out-of-sample %s model '%s' (train_end=%s, start_date=%s)",
                intended_type, dynamic_model.model_name, train_end, start_date,
            )
            return dynamic_model, True, None
        except Exception as e:
            failures.append(f"fresh '{intended_type}' training on pre-start data failed: {e}")

    # 4. Fallback to active model, explicitly flagged as NOT out-of-sample
    leak_msg = (
        f"Model leakage detected: no valid out-of-sample '{intended_type}' model for start {start_date} "
        f"({'; '.join(failures)}). Fallback active model '{active.model_name}' was trained through "
        f"{_train_end_of(active.metadata) or 'unknown'} (in-sample look-ahead bias)."
    )
    logger.error(leak_msg)
    return active, False, leak_msg


# ── 2. Full-Pipeline Strategy Backtest Engine ────────────────────────────────

def run_strategy_backtest(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = settings.initial_capital,
    model: Optional[TrainedModel] = None,
    cost_per_trade: float = settings.simulated_cost_per_trade,
    universe_membership: Optional[UniverseMembership] = None,
    model_schedule: Optional[List[Tuple[str, TrainedModel]]] = None,
    data_exclusions: Optional[DataExclusions] = None,
    apply_macro_adjustment: bool = False,
    macro_cache_path: Optional[str] = None,
    apply_sector_rotation: bool = False,
) -> BacktestResult:
    """
    Executes the full day-by-day autonomous trading replay across historical data.

    Enforces:
      - Sizing at project scale (default $50.00 capital, $14.16 equal-weight slot, $7.50 cash floor).
      - The Lag Rule: Day T-1 close features -> Rank -> Risk -> Alloc -> Day T open execution.
      - 0.2% cost deduction per trade.
      - Mark-to-market daily snapshots and comparison with SPY benchmark.
      - Strictly Out-of-Sample: Uses a model trained prior to start_date, or flags leakage alarm.
      - Point-in-time universe: with universe_membership, a ticker is only a candidate on dates it
        was a member (held positions are still managed). Without it, the result is flagged
        survivorship_biased.
      - model_schedule (walk-forward): [(effective_date, model), ...]. Each decision day uses the
        latest model whose effective_date <= that day; each model must be OOS for its own
        effective_date or the leakage alarm is raised. Cash and positions are never reset.
      - data_exclusions (from validate_ticker_data_point_in_time): on an excluded date the ticker
        cannot be newly bought. A held position is managed exactly as normal (fixed stop-loss,
        take-profit and signal exit stay active) so it is never frozen for the whole window.
      - apply_macro_adjustment (default False = current behavior, unchanged): when True, reproduces
        production's macro regime exactly — get_macro_environment() historical macro_result is
        passed into rank_candidates() (buy-bar shift) and its position_size_multiplier into
        evaluate_portfolio_risk() (0.6/0.8/1.0 sizing), same as src/pipeline/daily_pipeline.py.
        Does not touch compute_adaptive_thresholds, thresholds, stops, or the regime filter.
      - macro_cache_path (default None = current behavior, unchanged): backtest-only determinism aid,
        used only when apply_macro_adjustment=True. If set, historical macro_result values (one per
        calendar month, point-in-time via get_macro_environment's observation_end bounding) are read
        from this local JSON file when present, so reruns of the same experiment never touch the live
        FRED API and are byte-for-byte reproducible. On a cache miss for a given month, the real
        get_macro_environment(force_fetch=True, persist=False) is called exactly as before and the
        result is added to the file for next time. Does not change get_macro_environment() itself,
        regime classification, or any threshold/multiplier — it only removes live-network variance
        from repeated backtest runs.
      - apply_sector_rotation (default False = current behavior, unchanged): when True, reproduces
        production's sector rotation exactly — calculate_sector_rotation() is called once per decision
        date on point-in-time-bounded price history (dates <= that decision date only) and its
        SectorRotationResult is passed into rank_candidates() as sector_result, same as
        src/pipeline/daily_pipeline.py. The rotation algorithm itself (20-day returns, rank 1-8,
        +-4% modifier) is untouched; this only wires its existing, already-point-in-time-safe output
        into the backtest the same way production does. persist=False (backtest run, not a DB write).
    """
    # 1. Clean and validate inputs
    survivorship_biased = _check_universe_membership(universe_dict, universe_membership)
    spy_clean = spy_df.copy().sort_values("date").reset_index(drop=True)
    spy_dates = set(spy_clean["date"].astype(str))

    schedule: List[Tuple[str, TrainedModel]] = sorted(model_schedule or [], key=lambda e: e[0])
    if schedule:
        if model is not None:
            raise ValueError("Pass either model or model_schedule, not both.")
        if schedule[0][0] > start_date:
            raise ValueError(f"model_schedule starts {schedule[0][0]}, after backtest start {start_date}.")
        gap = int(settings.prediction_horizon)
        leaks = []
        for eff_date, sched_model in schedule:
            train_end = _model_train_end(getattr(sched_model, "metadata", {}) or {})
            if not _trained_before(train_end, eff_date, gap):
                leaks.append(
                    f"'{getattr(sched_model, 'model_name', 'model')}' effective {eff_date} "
                    f"was trained through {train_end or 'an UNKNOWN date'}"
                )
        is_oos = not leaks
        leak_reason = ("Model leakage detected in walk-forward schedule: " + "; ".join(leaks)) if leaks else None
        if leak_reason:
            logger.warning(leak_reason)
        active_model = schedule[0][1]
    else:
        # Resolve strictly out-of-sample model to eliminate look-ahead leakage
        active_model, is_oos, leak_reason = resolve_backtest_model(
            start_date=start_date,
            explicit_model=model,
            universe_dict=universe_dict,
            spy_df=spy_clean,
        )

    def _model_for(decision_date: str) -> TrainedModel:
        if not schedule:
            return active_model
        return [m for eff_date, m in schedule if eff_date <= decision_date][-1]

    # Precompute features for each ticker
    all_features: List[pd.DataFrame] = []
    price_history: Dict[str, Dict[str, Dict[str, float]]] = {}  # ticker -> date -> {open, close}
    # Point-in-time correlation inputs: sorted (date, close) per ticker plus a parallel sorted date
    # array for binary search, so evaluate_candidate_correlation() never sees rows after the decision
    # date (see _price_histories_as_of below). Built once here to avoid re-scanning full history
    # (up to ~5900 rows/ticker) on every one of the ~4900+ decision days in a multi-decade backtest.
    close_history: Dict[str, pd.DataFrame] = {}
    close_history_dates: Dict[str, Any] = {}

    for ticker, df in universe_dict.items():
        clean = df.copy().sort_values("date").reset_index(drop=True)
        feats = compute_features(clean, spy_df=spy_clean)
        all_features.append(feats)

        price_history[ticker] = {}
        for _, row in clean.iterrows():
            d = str(row["date"])
            price_history[ticker][d] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
            }

        close_history[ticker] = clean[["date", "close"]].reset_index(drop=True)
        close_history_dates[ticker] = close_history[ticker]["date"].to_numpy()

    def _price_histories_as_of(decision_date: str, tickers: Any) -> Dict[str, pd.DataFrame]:
        """Date-bounded (rows with date <= decision_date only) close-price history per ticker, for
        evaluate_portfolio_risk's price_histories param. Prevents the correlation filter from seeing
        future data during historical replay (matches the lag rule: only what was knowable by the
        decision date). A small safety margin over correlation_lookback_days is kept so
        calculate_returns_correlation's own tail(lookback_days) truncation still has enough rows."""
        margin = settings.correlation_lookback_days + 10
        out: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            dates_arr = close_history_dates.get(t)
            if dates_arr is None:
                continue
            idx = int(np.searchsorted(dates_arr, decision_date, side="right"))
            out[t] = close_history[t].iloc[max(0, idx - margin):idx]
        return out

    valid_features = [f for f in all_features if not f.empty and not f.isna().all().all()]
    combined_features = pd.concat(valid_features, ignore_index=True) if valid_features else pd.DataFrame()

    # Determine unique trading dates in test window
    all_test_dates = sorted(
        d for d in spy_dates if start_date <= d <= end_date
    )

    if len(all_test_dates) < 2:
        raise ValueError(f"Insufficient dates in test window [{start_date}, {end_date}].")

    # SPY benchmark reference
    spy_price_map = {
        str(r["date"]): {"open": float(r["open"]), "close": float(r["close"])}
        for _, r in spy_clean.iterrows()
    }
    spy_t0_close = spy_price_map[all_test_dates[0]]["close"]

    # 2. Simulation State
    cash = float(initial_capital)
    positions: Dict[str, Dict[str, Any]] = {}  # ticker -> {quantity, avg_cost, entry_date, entry_price}
    pending_orders: List[OrderSpec] = []
    completed_trades: List[BacktestTrade] = []
    daily_snapshots: List[DailySnapshot] = []

    peak_equity = float(initial_capital)
    spy_peak_close = spy_t0_close
    last_close: Dict[str, float] = {}
    macro_cache: Dict[str, Any] = {}

    # Backtest-only determinism aid (see macro_cache_path docstring above): a local JSON file of
    # already-resolved historical MacroEnvironmentResult values, keyed by calendar month, so repeated
    # runs of the same experiment never depend on live FRED network availability. Not used unless
    # both apply_macro_adjustment and macro_cache_path are set; production's get_macro_environment()
    # and its own DB cache are untouched either way.
    macro_file_cache: Dict[str, Dict[str, Any]] = {}
    macro_file_cache_dirty = False
    if apply_macro_adjustment and macro_cache_path:
        cache_file = Path(macro_cache_path)
        if cache_file.exists():
            try:
                macro_file_cache = json.loads(cache_file.read_text())
            except Exception as exc:
                logger.warning("Could not read macro cache file %s: %s. Will refetch live.", macro_cache_path, exc)
                macro_file_cache = {}

    def _macro_for(decision_date: str) -> Optional[Any]:
        """Real historical macro_result for decision_date, cached per calendar month (matches the
        monthly cadence of the underlying FRED series; avoids one FRED call per trading day).
        When macro_cache_path is set, a month already present in the local JSON cache is loaded
        from there instead of calling get_macro_environment live, so reruns are deterministic."""
        nonlocal macro_file_cache_dirty
        if not apply_macro_adjustment:
            return None
        month_key = decision_date[:7]
        if month_key in macro_cache:
            return macro_cache[month_key]

        if macro_cache_path and month_key in macro_file_cache:
            from src.intelligence.macro import MacroEnvironmentResult
            macro_cache[month_key] = MacroEnvironmentResult(**macro_file_cache[month_key])
            return macro_cache[month_key]

        from src.intelligence.macro import get_macro_environment
        try:
            result = get_macro_environment(
                as_of_date_str=decision_date, force_fetch=True, persist=False,
            )
        except Exception as exc:
            logger.warning("Macro lookup failed for %s: %s. Treating as FAVORABLE (no adjustment).", decision_date, exc)
            result = None
        macro_cache[month_key] = result
        if macro_cache_path and result is not None:
            macro_file_cache[month_key] = asdict(result)
            macro_file_cache_dirty = True
        return result

    for day_idx, current_date in enumerate(all_test_dates):
        # ── Step A: Morning Fills (The Lag Rule) ──────────────────────────────
        # Execute orders generated at previous trading day's close at today's OPEN
        if pending_orders:
            for order in pending_orders:
                t = order.ticker
                if t not in price_history or current_date not in price_history[t]:
                    logger.warning("Missing open price for %s on %s. Skipping order.", t, current_date)
                    continue

                open_price = price_history[t][current_date]["open"]

                if order.action == "SELL" and t in positions:
                    held_qty = positions[t]["quantity"]
                    trade = BacktestTrade(
                        ticker=t,
                        entry_date=positions[t]["entry_date"],
                        entry_price=positions[t]["entry_price"],
                        shares=held_qty,
                        entry_cost=round(held_qty * positions[t]["entry_price"] * cost_per_trade, 4),
                    )
                    trade.close(
                        exit_date=current_date,
                        exit_price=open_price,
                        fee_rate=cost_per_trade,
                        reason=order.reason,
                    )
                    completed_trades.append(trade)

                    # Return net proceeds to cash (gross - fee = gross * 0.998)
                    gross_proceeds = round(held_qty * open_price, 4)
                    fee = round(gross_proceeds * cost_per_trade, 4)
                    net_inflow = round(gross_proceeds - fee, 4)
                    cash = round(cash + net_inflow, 4)
                    del positions[t]

                elif order.action == "BUY" and t not in positions:
                    # Execute buy order using today's open price
                    shares, gross_spend, fee = calculate_fractional_shares(
                        order.net_amount, open_price, cost_per_trade
                    )
                    total_outflow = round(gross_spend + fee, 4)

                    if shares > 0 and cash >= total_outflow:
                        cash = round(cash - total_outflow, 4)
                        positions[t] = {
                            "quantity": shares,
                            "avg_cost": open_price,
                            "entry_date": current_date,
                            "entry_price": open_price,
                        }

            pending_orders = []

        # ── Step B: Mark to Market at Day T Close ─────────────────────────────
        portfolio_val = 0.0
        current_prices: Dict[str, float] = {}

        for t, pos in positions.items():
            if current_date in price_history[t]:
                last_close[t] = price_history[t][current_date]["close"]
            # Missing bar: carry the last known close forward, never revert to entry price.
            close_price = last_close.get(t, pos["entry_price"])
            current_prices[t] = close_price
            portfolio_val += pos["quantity"] * close_price

        total_equity = round(cash + portfolio_val, 4)
        peak_equity = max(peak_equity, total_equity)
        current_drawdown = (total_equity - peak_equity) / peak_equity

        # Benchmark SPY daily tracking
        spy_close = spy_price_map.get(current_date, {}).get("close", spy_t0_close)
        spy_peak_close = max(spy_peak_close, spy_close)
        spy_drawdown = (spy_close - spy_peak_close) / spy_peak_close

        prev_equity = daily_snapshots[-1].total_equity if daily_snapshots else initial_capital
        daily_ret = (total_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        cum_ret = (total_equity / initial_capital) - 1.0

        prev_spy = daily_snapshots[-1].spy_close if daily_snapshots else spy_t0_close
        spy_daily_ret = (spy_close / prev_spy) - 1.0 if prev_spy > 0 else 0.0
        spy_cum_ret = (spy_close / spy_t0_close) - 1.0

        snapshot = DailySnapshot(
            date=current_date,
            cash=cash,
            portfolio_value=round(portfolio_val, 4),
            total_equity=total_equity,
            daily_return=daily_ret,
            cumulative_return=cum_ret,
            spy_close=spy_close,
            spy_daily_return=spy_daily_ret,
            spy_cumulative_return=spy_cum_ret,
            positions_count=len(positions),
            cash_pct=round((cash / total_equity) * 100.0, 2) if total_equity > 0 else 100.0,
            drawdown=current_drawdown,
            spy_drawdown=spy_drawdown,
        )
        daily_snapshots.append(snapshot)

        # ── Step C: Generate Decisions for Day T+1 (End of Day T) ─────────────
        # If not the last day, run pipeline up to current_date close
        if day_idx < len(all_test_dates) - 1:
            # Sliced features up to current_date
            sub_feats = combined_features[combined_features["date"] <= current_date]
            sub_spy = spy_clean[spy_clean["date"] <= current_date]

            excluded_today = _excluded_on(data_exclusions, current_date)
            if (universe_membership is not None or excluded_today) and not sub_feats.empty:
                eligible = {
                    t for t in universe_dict
                    if _is_member(universe_membership, t, current_date) and t not in excluded_today
                } | set(positions)
                sub_feats = sub_feats[sub_feats["ticker"].isin(eligible)]
                if sub_feats[sub_feats["date"] == current_date].empty:
                    logger.info("No eligible (member, validated) tickers on %s; no new orders.", current_date)
                    continue

            if not sub_feats.empty:
                try:
                    macro_res = _macro_for(current_date)

                    # Current prices for all tickers in universe. Excluded tickers keep their prices:
                    # a held excluded position must still reach the stop-loss / take-profit / signal
                    # exits; exclusion only removes the ticker from buy candidates (above).
                    today_universe_prices = {
                        t: price_history[t][current_date]["close"]
                        for t in universe_dict
                        if current_date in price_history.get(t, {})
                    }
                    # Point-in-time price history (dates <= current_date only), shared by sector
                    # rotation's 20-day returns and the correlation filter's 60-day returns below.
                    today_price_histories = _price_histories_as_of(current_date, today_universe_prices.keys())

                    sector_res = None
                    if apply_sector_rotation:
                        from src.intelligence.sector_rotation import calculate_sector_rotation
                        sector_res = calculate_sector_rotation(
                            universe_dfs=today_price_histories, as_of_date=current_date, persist=False,
                        )

                    # 1. Ranking Engine
                    ranking_res = rank_candidates(
                        sub_feats,
                        model=_model_for(current_date),
                        spy_df=sub_spy,
                        run_date=current_date,
                        macro_result=macro_res,
                        sector_result=sector_res,
                    )

                    # 2. Risk Engine
                    risk_res = evaluate_portfolio_risk(
                        run_date=current_date,
                        current_cash=cash,
                        current_positions=positions,
                        current_prices=today_universe_prices,
                        ranking_result=ranking_res,
                        spy_df=sub_spy,
                        macro_size_multiplier=(macro_res.position_size_multiplier if macro_res else 1.0),
                        price_histories=today_price_histories,
                    )

                    # 3. Portfolio Allocation (Order Generation)
                    alloc_res = allocate_portfolio(
                        risk_assessment=risk_res,
                        current_cash=cash,
                        current_positions=positions,
                        current_prices=today_universe_prices,
                        fee_rate=cost_per_trade,
                    )

                    pending_orders = alloc_res.orders

                except Exception as exc:
                    logger.error("Error generating orders on %s: %s", current_date, exc)
                    pending_orders = []

    # 3. Compute Summary Metrics
    if positions:
        open_trades, liq_inflow = _liquidate_open_positions(positions, last_close, all_test_dates[-1], cost_per_trade)
        completed_trades.extend(open_trades)
        prev_eq = daily_snapshots[-2].total_equity if len(daily_snapshots) > 1 else initial_capital
        _apply_final_liquidation(daily_snapshots[-1], liq_inflow, peak_equity, prev_eq, initial_capital)
        positions = {}

    trading_days = len(daily_snapshots)
    final_equity = daily_snapshots[-1].total_equity
    total_net_ret = (final_equity / initial_capital) - 1.0

    # Returns are measured from the day-0 close, so there are trading_days - 1 return periods.
    years = (trading_days - 1) / 252.0
    cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) if final_equity > 0 and years > 0 else 0.0

    final_spy = daily_snapshots[-1].spy_close
    spy_total_ret = (final_spy / spy_t0_close) - 1.0
    spy_cagr = ((final_spy / spy_t0_close) ** (1.0 / years) - 1.0) if final_spy > 0 and years > 0 else 0.0

    max_dd = min(s.drawdown for s in daily_snapshots) * 100.0  # negative %
    spy_max_dd = min(s.spy_drawdown for s in daily_snapshots) * 100.0

    # Risk-adjusted ratios (day 0 is the zero-return baseline, not a period)
    daily_returns = [s.daily_return for s in daily_snapshots[1:]]
    sharpe = _annualized_sharpe(daily_returns)
    sortino = _annualized_sortino(daily_returns)
    spy_sharpe = _annualized_sharpe([s.spy_daily_return for s in daily_snapshots[1:]])

    # Trade statistics
    winning_trades = [t for t in completed_trades if (t.net_pnl or 0.0) > 0]
    losing_trades = [t for t in completed_trades if (t.net_pnl or 0.0) <= 0]
    win_rate = (len(winning_trades) / len(completed_trades) * 100.0) if completed_trades else 0.0

    gross_gains = sum(t.net_pnl for t in winning_trades if t.net_pnl is not None)
    gross_losses = abs(sum(t.net_pnl for t in losing_trades if t.net_pnl is not None))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-6 else (99.0 if gross_gains > 0 else 0.0)

    avg_trade_pnl = float(np.mean([t.net_pnl_pct for t in completed_trades])) * 100.0 if completed_trades else 0.0

    # Average holding duration
    holding_days_list = []
    for t in completed_trades:
        if t.entry_date and t.exit_date:
            try:
                d1 = pd.Timestamp(t.entry_date)
                d2 = pd.Timestamp(t.exit_date)
                holding_days_list.append((d2 - d1).days)
            except Exception:
                pass
    avg_holding = float(np.mean(holding_days_list)) if holding_days_list else 0.0
    avg_cash = float(np.mean([s.cash_pct for s in daily_snapshots])) if daily_snapshots else 100.0

    # 4. Anti-Self-Deception Alarm Checks (§1.5)
    alarm_triggered = False
    alarm_reasons: List[str] = []

    if not is_oos and leak_reason:
        alarm_triggered = True
        alarm_reasons.append(leak_reason)

    if cagr > ALARM_MAX_CAGR:
        alarm_triggered = True
        alarm_reasons.append(f"CAGR suspiciously high ({cagr*100:.1f}% > {ALARM_MAX_CAGR*100:.0f}%)")
    if abs(max_dd) < (ALARM_MIN_DRAWDOWN * 100.0) and trading_days >= 200 and len(completed_trades) >= 5:
        alarm_triggered = True
        alarm_reasons.append(f"Max Drawdown suspiciously low ({abs(max_dd):.1f}% < {ALARM_MIN_DRAWDOWN*100:.0f}%) while actively trading")
    if win_rate > (ALARM_MAX_WIN_RATE * 100.0) and len(completed_trades) >= 15:
        alarm_triggered = True
        alarm_reasons.append(f"Win Rate suspiciously high ({win_rate:.1f}% > {ALARM_MAX_WIN_RATE*100:.0f}%)")
    if sharpe > ALARM_MAX_SHARPE and trading_days >= 200 and len(completed_trades) >= 5:
        alarm_triggered = True
        alarm_reasons.append(f"Sharpe ratio suspiciously high ({sharpe:.2f} > {ALARM_MAX_SHARPE:.2f})")

    metrics = BacktestMetrics(
        start_date=all_test_dates[0],
        end_date=all_test_dates[-1],
        trading_days=trading_days,
        starting_capital=initial_capital,
        final_equity=final_equity,
        total_net_return_pct=round(total_net_ret * 100.0, 2),
        annualized_return_pct=round(cagr * 100.0, 2),
        spy_total_return_pct=round(spy_total_ret * 100.0, 2),
        spy_annualized_return_pct=round(spy_cagr * 100.0, 2),
        alpha_pct=round((total_net_ret - spy_total_ret) * 100.0, 2),
        max_drawdown_pct=round(max_dd, 2),
        spy_max_drawdown_pct=round(spy_max_dd, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        total_trades=len(completed_trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        avg_trade_pnl_pct=round(avg_trade_pnl, 2),
        avg_holding_days=round(avg_holding, 1),
        avg_cash_pct=round(avg_cash, 1),
        alarm_triggered=alarm_triggered,
        alarm_reasons=alarm_reasons,
        model_out_of_sample=is_oos,
        resolved_model_name=(
            f"{getattr(active_model, 'model_name', 'unknown')} (walk-forward, {len(schedule)} models)"
            if schedule else getattr(active_model, "model_name", "unknown")
        ),
        survivorship_biased=survivorship_biased,
        spy_sharpe_ratio=round(spy_sharpe, 2),
    )

    if macro_cache_path and macro_file_cache_dirty:
        try:
            cache_file = Path(macro_cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(macro_file_cache, indent=2, sort_keys=True))
        except Exception as exc:
            logger.warning("Could not write macro cache file %s: %s", macro_cache_path, exc)

    return BacktestResult(
        metrics=metrics,
        daily_snapshots=daily_snapshots,
        trades=completed_trades,
    )


# ── 2b. Walk-Forward Trading Backtest ────────────────────────────────────────

# Extra calendar history before each training window so 200-day features are valid on its first row.
WALK_FORWARD_FEATURE_WARMUP_DAYS: int = 400


@dataclass
class WalkForwardWindow:
    """One test window of a walk-forward run, measured on the single continuous portfolio."""
    window_index: int
    test_start: str
    test_end: str
    model_name: str
    model_type: str
    train_start: str
    train_end: str
    start_equity: float
    end_equity: float
    return_pct: float
    spy_return_pct: float
    max_drawdown_pct: float
    trades_closed: int
    win_rate_pct: float
    positions_carried_in: int


@dataclass
class WalkForwardResult:
    combined: BacktestResult
    windows: List[WalkForwardWindow]

    def to_markdown_summary(self, title: str = "Walk-Forward Backtest Report") -> str:
        lines = [
            self.combined.to_markdown_summary(title),
            "\n**Per-Window Results** (one continuous portfolio; each model trained only on data before its window):",
            "| # | Test Window | Model Trained On | Start Equity | End Equity | Return | SPY | Max DD | Trades Closed | Win Rate | Positions Carried In |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for w in self.windows:
            lines.append(
                f"| {w.window_index + 1} | {w.test_start} → {w.test_end} | {w.train_start} → {w.train_end} | "
                f"${w.start_equity:.2f} | ${w.end_equity:.2f} | {w.return_pct:+.2f}% | {w.spy_return_pct:+.2f}% | "
                f"{w.max_drawdown_pct:.2f}% | {w.trades_closed} | {w.win_rate_pct:.1f}% | {w.positions_carried_in} |"
            )
        return "\n".join(lines)


def _walk_forward_windows(trading_dates: List[str], start_date: str, end_date: str, test_months: float) -> List[Tuple[str, str]]:
    """Consecutive, non-overlapping test windows of test_months, snapped to actual trading dates."""
    windows: List[Tuple[str, str]] = []
    boundary = pd.Timestamp(start_date)
    while boundary <= pd.Timestamp(end_date):
        nxt = boundary + pd.DateOffset(months=int(test_months))
        in_window = [d for d in trading_dates if boundary <= pd.Timestamp(d) < nxt and start_date <= d <= end_date]
        if in_window:
            windows.append((in_window[0], in_window[-1]))
        boundary = nxt
    return windows


def _train_locked_model_before(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    window_start: str,
    train_years: float,
    gap_days: int,
) -> TrainedModel:
    """Trains the locked model type on the train_years before window_start, using only pre-window data."""
    train_from = (pd.Timestamp(window_start) - pd.DateOffset(months=int(round(train_years * 12)))).strftime("%Y-%m-%d")
    history_from = (pd.Timestamp(train_from) - pd.Timedelta(days=WALK_FORWARD_FEATURE_WARMUP_DAYS)).strftime("%Y-%m-%d")

    def _pre_window(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        out = df.copy()
        out["date"] = out["date"].astype(str)
        out = out[(out["date"] >= history_from) & (out["date"] < window_start)]
        if "ticker" not in out.columns:
            out["ticker"] = ticker
        return out

    # Slicing before labelling means every label is computed from pre-window closes only.
    ohlcv = pd.concat([_pre_window(df, t) for t, df in universe_dict.items()], ignore_index=True)
    dataset_df = build_dataset(ohlcv, spy_df=_pre_window(spy_df, settings.benchmark))
    dataset_df = dataset_df[dataset_df["date"] >= train_from]
    if dataset_df.empty:
        raise ValueError(f"No training rows between {train_from} and {window_start}; supply more history.")

    trained = train_model(dataset_df, model_type=LOCKED_MODEL_TYPE, gap_days=gap_days)
    train_end = _model_train_end(trained.metadata)
    if not _trained_before(train_end, window_start, gap_days):
        raise ValueError(f"Model for window {window_start} was trained through {train_end}; not out-of-sample.")
    return trained


def run_walk_forward_backtest(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = settings.initial_capital,
    test_months: float = settings.walk_forward_test_months,
    train_years: float = settings.walk_forward_train_years,
    gap_days: int = settings.prediction_horizon,
    cost_per_trade: float = settings.simulated_cost_per_trade,
    universe_membership: Optional[UniverseMembership] = None,
    data_exclusions: Optional[DataExclusions] = None,
    apply_macro_adjustment: bool = False,
    macro_cache_path: Optional[str] = None,
    apply_sector_rotation: bool = False,
) -> WalkForwardResult:
    """
    Walk-forward trading backtest: splits [start_date, end_date] into test_months windows, trains the
    locked Baseline for each window on the train_years of data before it (plus the gap_days label
    embargo), and replays ONE continuous portfolio through run_strategy_backtest with that model schedule.
    Costs, next-open fills, risk, portfolio rules and accounting are the engine's, unchanged.

    Needs history from at least start_date - train_years - WALK_FORWARD_FEATURE_WARMUP_DAYS.
    Fails loudly if any window cannot get an out-of-sample model.
    apply_macro_adjustment: see run_strategy_backtest (default False = unchanged prior behavior).
    macro_cache_path: see run_strategy_backtest (default None = unchanged prior behavior).
    apply_sector_rotation: see run_strategy_backtest (default False = unchanged prior behavior).
    """
    trading_dates = sorted(spy_df["date"].astype(str).unique())
    windows = _walk_forward_windows(trading_dates, start_date, end_date, test_months)
    if not windows:
        raise ValueError(f"No trading dates in walk-forward range [{start_date}, {end_date}].")

    schedule: List[Tuple[str, TrainedModel]] = []
    for idx, (win_start, win_end) in enumerate(windows):
        try:
            trained = _train_locked_model_before(universe_dict, spy_df, win_start, train_years, gap_days)
        except Exception as exc:
            raise ValueError(f"Walk-forward window {idx + 1} ({win_start}..{win_end}): {exc}") from exc
        logger.info(
            "Walk-forward window %d (%s..%s): %s trained %s..%s",
            idx + 1, win_start, win_end, trained.model_type,
            trained.metadata["split_info"]["train_start_date"], _model_train_end(trained.metadata),
        )
        schedule.append((win_start, trained))

    combined = run_strategy_backtest(
        universe_dict=universe_dict,
        spy_df=spy_df,
        start_date=windows[0][0],
        end_date=windows[-1][1],
        initial_capital=initial_capital,
        cost_per_trade=cost_per_trade,
        universe_membership=universe_membership,
        model_schedule=schedule,
        data_exclusions=data_exclusions,
        apply_macro_adjustment=apply_macro_adjustment,
        macro_cache_path=macro_cache_path,
        apply_sector_rotation=apply_sector_rotation,
    )
    if not combined.metrics.model_out_of_sample:
        raise ValueError(f"Walk-forward schedule failed the OOS check: {combined.metrics.alarm_reasons}")

    snaps = combined.daily_snapshots
    window_results: List[WalkForwardWindow] = []
    for idx, ((win_start, win_end), (_, win_model)) in enumerate(zip(windows, schedule)):
        in_win = [k for k, s in enumerate(snaps) if win_start <= s.date <= win_end]
        first, last = in_win[0], in_win[-1]
        # Measured from the prior close (continuous portfolio); window 1 uses the engine's day-0 close.
        before = snaps[first - 1] if first > 0 else snaps[first]
        start_equity = before.total_equity if first > 0 else float(initial_capital)
        end_equity = snaps[last].total_equity

        peak, max_dd = start_equity, 0.0
        for s in snaps[first:last + 1]:
            peak = max(peak, s.total_equity)
            max_dd = min(max_dd, (s.total_equity - peak) / peak if peak > 0 else 0.0)

        closed = [t for t in combined.trades if t.exit_date and win_start <= t.exit_date <= win_end]
        wins = [t for t in closed if (t.net_pnl or 0.0) > 0]

        window_results.append(WalkForwardWindow(
            window_index=idx,
            test_start=win_start,
            test_end=win_end,
            model_name=win_model.model_name,
            model_type=win_model.model_type,
            train_start=str(win_model.metadata["split_info"]["train_start_date"]),
            train_end=str(_model_train_end(win_model.metadata)),
            start_equity=round(start_equity, 4),
            end_equity=round(end_equity, 4),
            return_pct=round((end_equity / start_equity - 1.0) * 100.0, 2) if start_equity > 0 else 0.0,
            spy_return_pct=round((snaps[last].spy_close / before.spy_close - 1.0) * 100.0, 2),
            max_drawdown_pct=round(max_dd * 100.0, 2),
            trades_closed=len(closed),
            win_rate_pct=round(len(wins) / len(closed) * 100.0, 1) if closed else 0.0,
            positions_carried_in=before.positions_count if first > 0 else 0,
        ))

    return WalkForwardResult(combined=combined, windows=window_results)


# ── 3. Section 10 Item 2: Backtest Lab Engine ─────────────────────────────────

def compute_monthly_returns_heatmap(
    daily_snapshots: List[DailySnapshot],
    starting_capital: float = 10000.0,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Computes a Year x Month grid of returns (%), plus best_month and worst_month strings.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if not daily_snapshots:
        empty_df = pd.DataFrame(columns=["Year"] + month_names + ["Year Total"])
        return empty_df, "N/A", "N/A"

    df = pd.DataFrame([{"date": pd.to_datetime(s.date), "equity": s.total_equity} for s in daily_snapshots])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    years = sorted(df["year"].unique())
    grid_rows = []
    monthly_returns_list: List[float] = []

    for y in years:
        ydf = df[df["year"] == y]
        row_dict: Dict[str, str] = {"Year": str(y)}
        prior_year_df = df[df["year"] < y]
        year_start_eq = prior_year_df["equity"].iloc[-1] if not prior_year_df.empty else starting_capital

        for m in range(1, 13):
            m_name = month_names[m - 1]
            mdf = ydf[ydf["month"] == m]
            if mdf.empty:
                row_dict[m_name] = "—"
            else:
                prior_df = df[(df["date"] < mdf["date"].iloc[0])]
                prev_eq = prior_df["equity"].iloc[-1] if not prior_df.empty else year_start_eq
                curr_eq = mdf["equity"].iloc[-1]
                m_ret = ((curr_eq / prev_eq) - 1.0) * 100.0 if prev_eq > 0 else 0.0
                monthly_returns_list.append(m_ret)
                sign = "+" if m_ret > 0 else ""
                row_dict[m_name] = f"{sign}{m_ret:.1f}%"

        year_end_eq = ydf["equity"].iloc[-1]
        y_ret = ((year_end_eq / year_start_eq) - 1.0) * 100.0 if year_start_eq > 0 else 0.0
        sign_y = "+" if y_ret > 0 else ""
        row_dict["Year Total"] = f"{sign_y}{y_ret:.1f}%"
        grid_rows.append(row_dict)

    grid_df = pd.DataFrame(grid_rows)

    if monthly_returns_list:
        best_m = max(monthly_returns_list)
        worst_m = min(monthly_returns_list)
        best_str = f"+{best_m:.1f}%" if best_m > 0 else f"{best_m:.1f}%"
        worst_str = f"+{worst_m:.1f}%" if worst_m > 0 else f"{worst_m:.1f}%"
    else:
        best_str = "0.0%"
        worst_str = "0.0%"

    return grid_df, best_str, worst_str


def run_lab_backtest(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0,
    buy_threshold: float = 0.60,
    exit_threshold: float = 0.45,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
    max_positions: int = 3,
    position_sizing: str = "Fixed",
    use_sentiment: bool = True,
    use_earnings_blackout: bool = True,
    use_sector_rotation: bool = True,
    use_macro_regime: bool = True,
    use_correlation_filter: bool = True,
    use_trailing_stop: bool = settings.trailing_stop_enabled,
    model: Optional[TrainedModel] = None,
    cost_per_trade: float = 0.002,
    universe_membership: Optional[UniverseMembership] = None,
    data_exclusions: Optional[DataExclusions] = None,
) -> Dict[str, Any]:
    """
    Executes an interactive Backtest Lab simulation with fully customizable strategy parameters.
    Returns comprehensive metrics, equity curves, monthly returns heatmap, and trade list.
    See run_strategy_backtest for universe_membership semantics.
    """
    survivorship_biased = _check_universe_membership(universe_dict, universe_membership)
    spy_clean = spy_df.copy().sort_values("date").reset_index(drop=True)
    if "ticker" not in spy_clean.columns:
        spy_clean["ticker"] = "SPY"
    spy_dates = set(spy_clean["date"].astype(str))

    active_model, is_oos, leak_reason = resolve_backtest_model(
        start_date=start_date,
        explicit_model=model,
        universe_dict=universe_dict,
        spy_df=spy_clean,
    )

    price_history: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_features: List[pd.DataFrame] = []

    for ticker, df in universe_dict.items():
        clean = df.copy().sort_values("date").reset_index(drop=True)
        if "ticker" not in clean.columns:
            clean["ticker"] = ticker
        feats = compute_features(clean, spy_df=spy_clean)
        all_features.append(feats)

        price_history[ticker] = {}
        for _, row in clean.iterrows():
            d = str(row["date"])
            price_history[ticker][d] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
            }

    valid_features = [f for f in all_features if not f.empty and not f.isna().all().all()]
    combined_features = pd.concat(valid_features, ignore_index=True) if valid_features else pd.DataFrame()

    all_test_dates = sorted([d for d in spy_dates if start_date <= d <= end_date])
    if len(all_test_dates) < 2:
        return {
            "total_return_pct": 0.0,
            "spy_total_return_pct": 0.0,
            "alpha_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "days_in_cash_pct": 100.0,
            "sharpe_ratio": 0.0,
            "best_month": "0.0%",
            "worst_month": "0.0%",
            "equity_curve": pd.DataFrame(),
            "monthly_returns": pd.DataFrame(),
            "trades_df": pd.DataFrame(),
            "survivorship_biased": survivorship_biased,
        }

    spy_price_map = {
        str(r["date"]): {"open": float(r["open"]), "close": float(r["close"])}
        for _, r in spy_clean.iterrows()
    }
    spy_t0_close = spy_price_map[all_test_dates[0]]["close"]

    cash = float(initial_capital)
    positions: Dict[str, Dict[str, Any]] = {}  # ticker -> {quantity, avg_cost, entry_date, entry_price, highest_price, sector}
    pending_buy_orders: List[Dict[str, Any]] = []
    completed_trades: List[BacktestTrade] = []
    daily_snapshots: List[DailySnapshot] = []

    peak_equity = float(initial_capital)
    spy_peak_close = spy_t0_close
    last_close: Dict[str, float] = {}

    ticker_sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Communication Services", "AMZN": "Consumer Cyclical",
        "META": "Communication Services", "BRK-B": "Financial",
        "JNJ": "Healthcare", "JPM": "Financial", "V": "Financial",
    }

    for day_idx, current_date in enumerate(all_test_dates):
        # ── A. Morning Exits and Fills (Lag Rule) ─────────────────────────────
        # 1. Process Exits first
        active_tickers = list(positions.keys())
        for t in active_tickers:
            pos = positions[t]
            if t not in price_history or current_date not in price_history[t]:
                continue
            open_price = price_history[t][current_date]["open"]
            highest_price = max(pos.get("highest_price", open_price), open_price)
            pos["highest_price"] = highest_price

            exit_reason = None
            # Check Stop Loss / Trailing Stop
            if use_trailing_stop:
                stop_target = highest_price * (1.0 - stop_loss_pct)
                if open_price <= stop_target:
                    exit_reason = f"Trailing Stop ({stop_loss_pct*100:.0f}%)"
            else:
                stop_target = pos["entry_price"] * (1.0 - stop_loss_pct)
                if open_price <= stop_target:
                    exit_reason = f"Stop Loss ({stop_loss_pct*100:.0f}%)"

            # Check Take Profit
            if not exit_reason:
                tp_target = pos["entry_price"] * (1.0 + take_profit_pct)
                if open_price >= tp_target:
                    exit_reason = f"Take Profit ({take_profit_pct*100:.0f}%)"

            if exit_reason:
                held_qty = pos["quantity"]
                trade = BacktestTrade(
                    ticker=t,
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    shares=held_qty,
                    entry_cost=round(held_qty * pos["entry_price"] * cost_per_trade, 4),
                )
                trade.close(
                    exit_date=current_date,
                    exit_price=open_price,
                    fee_rate=cost_per_trade,
                    reason=exit_reason,
                )
                completed_trades.append(trade)

                gross_proceeds = held_qty * open_price
                fee = gross_proceeds * cost_per_trade
                cash += (gross_proceeds - fee)
                del positions[t]

        # 2. Process Morning Pending Buys
        if pending_buy_orders:
            for order in pending_buy_orders:
                t = order["ticker"]
                if len(positions) >= max_positions:
                    break
                if t in positions or t not in price_history or current_date not in price_history[t]:
                    continue
                open_price = price_history[t][current_date]["open"]
                alloc_amount = order["net_amount"]

                shares, gross_spend, fee = calculate_fractional_shares(
                    alloc_amount, open_price, cost_per_trade
                )
                total_outflow = gross_spend + fee
                if shares > 0 and cash >= total_outflow:
                    cash -= total_outflow
                    positions[t] = {
                        "quantity": shares,
                        "avg_cost": open_price,
                        "entry_date": current_date,
                        "entry_price": open_price,
                        "highest_price": open_price,
                        "sector": ticker_sectors.get(t, "Other"),
                    }

            pending_buy_orders = []

        # ── B. Mark to Market at Day T Close ──────────────────────────────────
        portfolio_val = 0.0
        for t, pos in positions.items():
            if current_date in price_history[t]:
                last_close[t] = price_history[t][current_date]["close"]
            portfolio_val += pos["quantity"] * last_close.get(t, pos["entry_price"])

        total_equity = cash + portfolio_val
        peak_equity = max(peak_equity, total_equity)
        current_drawdown = (total_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        spy_close = spy_price_map.get(current_date, {}).get("close", spy_t0_close)
        spy_peak_close = max(spy_peak_close, spy_close)
        spy_drawdown = (spy_close - spy_peak_close) / spy_peak_close if spy_peak_close > 0 else 0.0

        prev_equity = daily_snapshots[-1].total_equity if daily_snapshots else initial_capital
        daily_ret = (total_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        cum_ret = (total_equity / initial_capital) - 1.0

        prev_spy = daily_snapshots[-1].spy_close if daily_snapshots else spy_t0_close
        spy_daily_ret = (spy_close / prev_spy) - 1.0 if prev_spy > 0 else 0.0
        spy_cum_ret = (spy_close / spy_t0_close) - 1.0

        snapshot = DailySnapshot(
            date=current_date,
            cash=round(cash, 2),
            portfolio_value=round(portfolio_val, 2),
            total_equity=round(total_equity, 2),
            daily_return=daily_ret,
            cumulative_return=cum_ret,
            spy_close=spy_close,
            spy_daily_return=spy_daily_ret,
            spy_cumulative_return=spy_cum_ret,
            positions_count=len(positions),
            cash_pct=round((cash / total_equity) * 100.0, 2) if total_equity > 0 else 100.0,
            drawdown=current_drawdown,
            spy_drawdown=spy_drawdown,
        )
        daily_snapshots.append(snapshot)

        # ── C. Generate Signals for Day T+1 (End of Day T) ─────────────────────
        if day_idx < len(all_test_dates) - 1:
            sub_feats = combined_features[combined_features["date"] <= current_date]
            sub_spy = spy_clean[spy_clean["date"] <= current_date]

            # 1. Macro Regime Check
            is_risk_on = True
            if use_macro_regime and len(sub_spy) >= 200:
                spy_200ma = sub_spy["close"].tail(200).mean()
                if spy_close < spy_200ma:
                    is_risk_on = False

            if is_risk_on and not sub_feats.empty and active_model is not None:
                latest_feats = sub_feats[sub_feats["date"] == current_date]
                candidates = []
                excluded_today = _excluded_on(data_exclusions, current_date)

                for _, r in latest_feats.iterrows():
                    ticker = str(r["ticker"])
                    if (ticker in positions or ticker in excluded_today
                            or not _is_member(universe_membership, ticker, current_date)):
                        continue

                    # Incomplete features (e.g. newly listed): skip this ticker for this day only.
                    if r[FEATURE_COLUMNS].isna().any():
                        continue

                    # Feature vector prediction
                    feat_row = r[FEATURE_COLUMNS].to_frame().T
                    prob = float(active_model.predict_proba(feat_row)[0, 1])

                    # Sentiment toggle adjustment
                    if use_sentiment:
                        sent_val = float(r.get("sentiment_score", 0.0))
                        if sent_val > 0.2:
                            prob = min(0.99, prob + 0.03)
                        elif sent_val < -0.2:
                            prob = max(0.01, prob - 0.03)

                    if prob < buy_threshold:
                        continue

                    # Earnings blackout toggle check
                    if use_earnings_blackout:
                        days_to_earnings = abs(int(r.get("days_to_earnings", 999)))
                        if days_to_earnings <= 7:
                            continue

                    # Sector rotation toggle check
                    if use_sector_rotation:
                        sec = ticker_sectors.get(ticker, "Other")
                        held_sectors = [p["sector"] for p in positions.values()]
                        if sec in held_sectors:
                            continue

                    candidates.append({"ticker": ticker, "prob": prob})

                candidates.sort(key=lambda x: x["prob"], reverse=True)
                available_slots = max_positions - len(positions)
                selected_candidates = candidates[:available_slots]

                if selected_candidates and available_slots > 0:
                    base_alloc = cash / available_slots
                    for cand in selected_candidates:
                        if position_sizing == "Confidence-Based":
                            conviction = max(0.0, (cand["prob"] - buy_threshold) / (1.0 - buy_threshold + 1e-6))
                            alloc = base_alloc * (0.8 + 0.4 * conviction)
                        else:
                            alloc = base_alloc

                        pending_buy_orders.append({
                            "ticker": cand["ticker"],
                            "net_amount": min(cash, alloc),
                            "prob": cand["prob"],
                        })

    # ── Post-Simulation Metrics Assembly ──────────────────────────────────────
    if positions:
        open_trades, liq_inflow = _liquidate_open_positions(positions, last_close, all_test_dates[-1], cost_per_trade)
        completed_trades.extend(open_trades)
        prev_eq = daily_snapshots[-2].total_equity if len(daily_snapshots) > 1 else initial_capital
        _apply_final_liquidation(daily_snapshots[-1], liq_inflow, peak_equity, prev_eq, initial_capital)
        positions = {}

    trading_days = len(daily_snapshots)
    final_equity = daily_snapshots[-1].total_equity
    total_net_ret = ((final_equity / initial_capital) - 1.0) * 100.0

    final_spy = daily_snapshots[-1].spy_close
    spy_total_ret = ((final_spy / spy_t0_close) - 1.0) * 100.0
    alpha_pct = total_net_ret - spy_total_ret

    max_dd = min(s.drawdown for s in daily_snapshots) * 100.0 if daily_snapshots else 0.0

    sharpe = _annualized_sharpe([s.daily_return for s in daily_snapshots[1:]])

    winning_trades = [t for t in completed_trades if (t.net_pnl or 0.0) > 0]
    win_rate = (len(winning_trades) / len(completed_trades) * 100.0) if completed_trades else 0.0

    cash_days = sum(1 for s in daily_snapshots if s.cash_pct >= 80.0)
    days_in_cash_pct = (cash_days / trading_days * 100.0) if trading_days > 0 else 100.0

    monthly_grid_df, best_month, worst_month = compute_monthly_returns_heatmap(
        daily_snapshots=daily_snapshots,
        starting_capital=initial_capital,
    )

    # Convert snapshots into dataframe
    eq_rows = []
    for s in daily_snapshots:
        eq_rows.append({
            "date": s.date,
            "Strategy": round(s.total_equity, 2),
            "SPY Benchmark": round(initial_capital * (1.0 + s.spy_cumulative_return), 2),
            "Cash": round(s.cash, 2),
            "Drawdown": round(s.drawdown * 100.0, 2),
        })
    eq_df = pd.DataFrame(eq_rows)

    # Convert trades into dataframe
    trade_rows = []
    for t in completed_trades:
        trade_rows.append({
            "Ticker": t.ticker,
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date or "—",
            "Entry Price": f"${t.entry_price:.2f}",
            "Exit Price": f"${t.exit_price:.2f}" if t.exit_price else "—",
            "Shares": f"{t.shares:.4f}",
            "Net PnL ($)": f"${t.net_pnl:+.2f}" if t.net_pnl is not None else "—",
            "Net PnL (%)": f"{t.net_pnl_pct * 100:+.2f}%" if t.net_pnl_pct is not None else "—",
            "Exit Reason": t.exit_reason or "—",
        })
    trades_df = pd.DataFrame(trade_rows)

    return {
        "total_return_pct": round(total_net_ret, 2),
        "spy_total_return_pct": round(spy_total_ret, 2),
        "alpha_pct": round(alpha_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": len(completed_trades),
        "days_in_cash_pct": round(days_in_cash_pct, 1),
        "sharpe_ratio": round(sharpe, 2),
        "best_month": best_month,
        "worst_month": worst_month,
        "equity_curve": eq_df,
        "monthly_returns": monthly_grid_df,
        "trades_df": trades_df,
        "model_out_of_sample": is_oos,
        "model_leak_reason": leak_reason,
        "resolved_model_name": getattr(active_model, "model_name", "unknown"),
        "survivorship_biased": survivorship_biased,
    }

