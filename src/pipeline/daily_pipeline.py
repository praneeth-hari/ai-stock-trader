"""
src/pipeline/daily_pipeline.py — Phase 11 Daily Pipeline Orchestrator.

PURPOSE
-------
The single entry point that ties together every proven, tested module into
one end-to-end autonomous daily paper-trading run.

PIPELINE STAGES (enforced in strict order)
------------------------------------------
  1. Market data fetch (src.data.market_data.fetch_ticker_data)
  2. Validation gate  (src.data.validation.validate_ticker_data)
  3. Feature engineering (src.features.engineer.compute_features)
  4. ML inference / ranking (src.ranking.ranking.rank_candidates) — uses active_model.joblib
  5. Portfolio state load from Paper Broker / DB
  6. Risk engine (src.risk.risk_engine.evaluate_portfolio_risk) — exits, sizing, vetoes
  7. Portfolio allocation (src.portfolio.portfolio.allocate_portfolio) — OrderSpecs
  8. Paper broker execution (src.trading.paper_broker.PaperBroker)
  9. End-of-day snapshot + DB persistence
 10. Markdown audit summary emitted to stdout

LAG RULE (§1.5):
  In LIVE mode, the pipeline runs after market CLOSE.
  It fetches data through today's close, generates orders for TOMORROW's market open.
  Orders are NOT executed today — they are returned as pending_orders for the next run
  (or executed manually / by a scheduler at next open).

  In REPLAY mode (date='YYYY-MM-DD' supplied), the pipeline is called once per
  simulated day; the caller controls the date sequence.

ARCHITECTURE RULE:
  This module makes NO trading decisions. It only orchestrates existing modules.
  No new risk logic, no new sizing logic, no new exit logic is written here.

USAGE:
  # Live mode (runs for today's trading data):
  python src/pipeline/daily_pipeline.py

  # Replay / simulation mode:
  python src/pipeline/daily_pipeline.py --date 2024-01-15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.data.market_data import fetch_ticker_data
from src.data.validation import LIVE_FETCH_LOOKBACK_DAYS, validate_ticker_data
from src.db import repository
from src.features.engineer import compute_features
from src.intelligence import (
    analyze_universe_sentiment,
    calculate_sector_rotation,
    get_macro_environment,
    get_universe_earnings_calendar,
)
from src.ml.evaluate import load_active_model, verify_frozen_active_model
from src.portfolio.portfolio import OrderSpec, allocate_portfolio
from src.ranking.ranking import rank_candidates
from src.risk.risk_engine import evaluate_portfolio_risk
from src.trading.paper_broker import FillResult, PaperBroker
from src.alerts import (
    send_circuit_breaker_alert,
    send_pipeline_failure_alert,
    send_psi_drift_alert,
    send_stop_loss_alert,
    send_trade_executed_alert,
    notify_circuit_breaker,
    notify_daily_summary,
    notify_pipeline_failure,
    notify_psi_drift,
    notify_stop_loss,
    notify_trade_bought,
    notify_trade_sold,
)

logger = logging.getLogger(__name__)

# Mutex guard ensuring concurrent pipeline triggers are rejected cleanly
_pipeline_lock = threading.Lock()


# ── Pipeline result dataclass ─────────────────────────────────────────────────

class DailyPipelineResult:
    """Summary of a single daily pipeline execution."""

    def __init__(
        self,
        run_date: str,
        tickers_fetched: int,
        tickers_valid: int,
        tickers_skipped: int,
        regime: str,
        orders_generated: int,
        fills: List[FillResult],
        pending_orders: List[OrderSpec],
        cash: float,
        total_equity: float,
        errors: List[str],
    ) -> None:
        self.run_date = run_date
        self.tickers_fetched = tickers_fetched
        self.tickers_valid = tickers_valid
        self.tickers_skipped = tickers_skipped
        self.regime = regime
        self.orders_generated = orders_generated
        self.fills = fills
        self.pending_orders = pending_orders
        self.cash = cash
        self.total_equity = total_equity
        self.errors = errors

    def to_markdown(self) -> str:
        """Format the daily run result as a human-readable markdown audit report."""
        sells = [f for f in self.fills if f.action == "SELL"]
        buys = [f for f in self.fills if f.action == "BUY"]
        realized_pnl = sum(f.net_pnl for f in sells)
        cash_pct = (self.cash / self.total_equity * 100.0) if self.total_equity > 0 else 100.0

        lines = [
            f"## Daily Pipeline Audit — {self.run_date}",
            "",
            "### Data Fetch & Validation",
            f"- Tickers fetched: **{self.tickers_fetched}**",
            f"- Passed validation: **{self.tickers_valid}**",
            f"- Skipped (bad data): **{self.tickers_skipped}**",
            f"- Market Regime: **{self.regime}**",
            "",
            "### Orders & Fills",
            f"- Orders generated: **{self.orders_generated}**",
            f"- Fills executed: **{len(self.fills)}** ({len(buys)} BUY, {len(sells)} SELL)",
            f"- Pending orders for next open: **{len(self.pending_orders)}**",
        ]

        if self.fills:
            lines.append("")
            lines.append("| Date | Action | Ticker | Shares | Fill Price | Gross | Fee | Cash Impact | Net PnL | Reason |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for f in self.fills:
                lines.append(
                    f"| {f.run_date} | {f.action} | {f.ticker} | {f.shares:.4f} | "
                    f"${f.fill_price:.2f} | ${f.gross_value:.2f} | ${f.fee:.4f} | "
                    f"{'-' if f.action == 'BUY' else '+'}${abs(f.cash_impact):.4f} | "
                    f"{f.net_pnl:+.4f} | {f.reason} |"
                )

        lines += [
            "",
            "### Portfolio State (End of Day)",
            f"- Cash: **${self.cash:.4f}** ({cash_pct:.1f}% of equity)",
            f"- Total Equity: **${self.total_equity:.4f}**",
            f"- Realized PnL Today: **{realized_pnl:+.4f}**",
        ]

        if self.errors:
            lines.append("")
            lines.append("### ⚠ Errors")
            for e in self.errors:
                lines.append(f"- {e}")

        return "\n".join(lines)


# ── Core pipeline function ─────────────────────────────────────────────────────

def run_daily_pipeline(
    run_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    fetch_start: Optional[str] = None,
    broker: Optional[PaperBroker] = None,
    _spy_df: Optional[pd.DataFrame] = None,          # injectable for testing
    _universe_dfs: Optional[Dict[str, pd.DataFrame]] = None,  # injectable for testing
    force: bool = False,
    market_name: Optional[str] = None,
) -> DailyPipelineResult:
    """
    Execute one full daily paper-trading pipeline run.

    Guaranteed Idempotent: If run_daily_pipeline has already executed for run_date,
    it safely skips re-execution unless force=True.
    """
    # ── Emergency Kill Switch ─────────────────────────────────────────────────
    if settings.kill_switch_enabled:
        target_date_str = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        reason = settings.kill_switch_reason or "No reason provided."
        msg = f"EMERGENCY KILL SWITCH ACTIVE — pipeline halted. Reason: {reason}"
        logger.warning(msg)
        repository.log_event(
            "WARNING", "daily_pipeline", msg,
            {"run_date": target_date_str, "status": "KILL_SWITCH_ACTIVE"},
        )
        return DailyPipelineResult(
            run_date=target_date_str,
            tickers_fetched=0,
            tickers_valid=0,
            tickers_skipped=0,
            regime="KILL_SWITCH_ACTIVE",
            orders_generated=0,
            fills=[],
            pending_orders=[],
            cash=0.0,
            total_equity=0.0,
            errors=[msg],
        )

    # Indian market pipeline check (4 PM IST)
    from config.settings import settings as _s
    if getattr(_s, 'india_pipeline_hour_ist', None):
        import pytz
        from datetime import datetime as _dt
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = _dt.now(ist)
        logger.info("India pipeline hour check: current IST hour = %d", now_ist.hour)

    # ── Concurrency Mutex Guard ───────────────────────────────────────────────
    acquired = _pipeline_lock.acquire(blocking=False)
    if not acquired:
        target_date_str = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        msg = f"CONCURRENT RUN BLOCKED: Daily pipeline is already actively executing. Skipping concurrent trigger for {target_date_str}."
        logger.warning(msg)
        repository.log_event(
            "WARNING", "daily_pipeline", msg,
            {"run_date": target_date_str, "status": "CONCURRENT_RUN_REJECTED"}
        )
        return DailyPipelineResult(
            run_date=target_date_str,
            tickers_fetched=0,
            tickers_valid=0,
            tickers_skipped=0,
            regime="CONCURRENT_RUN_REJECTED",
            orders_generated=0,
            fills=[],
            pending_orders=[],
            cash=0.0,
            total_equity=0.0,
            errors=[msg],
        )

    try:
        return _run_daily_pipeline_internal(
            run_date=run_date,
            tickers=tickers,
            fetch_start=fetch_start,
            broker=broker,
            _spy_df=_spy_df,
            _universe_dfs=_universe_dfs,
            force=force,
            market_name=market_name,
        )
    finally:
        _pipeline_lock.release()


def _run_daily_pipeline_internal(
    run_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    fetch_start: Optional[str] = None,
    broker: Optional[PaperBroker] = None,
    _spy_df: Optional[pd.DataFrame] = None,
    _universe_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    force: bool = False,
    market_name: Optional[str] = None,
) -> DailyPipelineResult:
    # ── 0. Setup ──────────────────────────────────────────────────────────────
    repository.create_all_tables()

    if run_date is None:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if market_name:
        market_name = str(market_name).strip().upper()
    elif tickers:
        is_india = any(str(t).endswith(".NS") or str(t).endswith(".BO") for t in tickers)
        market_name = "INDIA" if is_india else "US"
    else:
        market_name = "US"

    if tickers is None:
        tickers = settings.get_universe(market_name)
    market_comp = f"daily_pipeline_{market_name.lower()}"

    # ── Market Calendar Check ─────────────────────────────────────────────────
    if not force:
        from src.pipeline.scheduler import is_market_day
        target_d = datetime.strptime(run_date, "%Y-%m-%d").date()
        is_open, reason = is_market_day(target_d)
        if not is_open:
            msg = f"Market Closed ({market_name}): {reason}. Skipping daily paper pipeline cycle."
            logger.info(msg)
            repository.log_event(
                "INFO", market_comp, msg,
                {"run_date": run_date, "market": market_name, "status": "SKIPPED_MARKET_CLOSED"}
            )
            return DailyPipelineResult(
                run_date=run_date,
                tickers_fetched=0,
                tickers_valid=0,
                tickers_skipped=len(tickers),
                regime="SKIPPED_MARKET_CLOSED",
                orders_generated=0,
                fills=[],
                pending_orders=[],
                cash=0.0,
                total_equity=0.0,
                errors=[msg],
            )

    # ── Idempotency Guard (Market-Specific Double-trigger protection) ───────────
    existing_snap = repository.get_portfolio_snapshot(run_date, market=market_name)
    already_run_event = False
    if not force:
        try:
            recent_events = repository.get_events(limit=100)
            already_run_event = any(
                (e.get("component") in ("daily_pipeline", market_comp))
                and (
                    f"Pipeline complete for {run_date} ({market_name})" in e.get("message", "")
                    or f"Pipeline complete for {run_date}." in e.get("message", "")
                )
                for e in recent_events
            )
        except Exception:
            already_run_event = False

    if (existing_snap is not None or already_run_event) and not force:
        msg = f"IDEMPOTENT GUARD: {market_name} market pipeline already executed for {run_date}. Skipping re-run to prevent duplicate orders or fills."
        logger.info(msg)
        repository.log_event("INFO", market_comp, msg, {"run_date": run_date, "market": market_name, "status": "ALREADY_EXECUTED"})

        existing_orders = repository.get_orders(run_date)
        cash_val = float(existing_snap["cash"]) if existing_snap else 0.0
        eq_val = float(existing_snap["total_value"]) if existing_snap else 0.0
        return DailyPipelineResult(
            run_date=run_date,
            tickers_fetched=0,
            tickers_valid=0,
            tickers_skipped=0,
            regime="ALREADY_RUN",
            orders_generated=len(existing_orders),
            fills=[],
            pending_orders=[],
            cash=cash_val,
            total_equity=eq_val,
            errors=[],
        )

    # Compute fetch_start if not supplied: warmup_bars + 30 calendar day buffer
    if fetch_start is None and _spy_df is None:
        from datetime import timedelta
        rd = datetime.strptime(run_date, "%Y-%m-%d")
        fetch_start = (rd - timedelta(days=LIVE_FETCH_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    errors: List[str] = []
    tickers_fetched = 0
    tickers_skipped = 0

    repository.log_event("INFO", "daily_pipeline", f"Pipeline started for {run_date}",
                         {"run_date": run_date, "tickers": tickers})

    # ── 1. Fetch & Validate SPY (benchmark) ───────────────────────────────────
    if _spy_df is not None:
        spy_clean = _spy_df.copy().sort_values("date").reset_index(drop=True)
        logger.info("Using injected SPY DataFrame (%d rows).", len(spy_clean))
    else:
        try:
            benchmark_symbol = settings.get_benchmark(market_name)
            spy_raw = fetch_ticker_data(benchmark_symbol, start_date=fetch_start, end_date=run_date)
            tickers_fetched += 1
            val_spy = validate_ticker_data(spy_raw, ticker=benchmark_symbol)
            spy_clean = val_spy.cleaned_df.sort_values("date").reset_index(drop=True)
            logger.info("SPY validated: %d rows through %s.", len(spy_clean), run_date)
            try:
                repository.save_market_data(spy_clean)
            except Exception as exc:
                logger.warning("Failed to save SPY market data to repository: %s", exc)
        except Exception as exc:
            err = f"CRITICAL: SPY fetch/validation failed: {exc}"
            logger.critical(err)
            errors.append(err)
            repository.log_event("CRITICAL", "daily_pipeline", err)
            try:
                send_pipeline_failure_alert(
                    error_message=err,
                    run_date=run_date,
                )
            except Exception:
                pass
            try:
                notify_pipeline_failure(
                    error_message=err,
                    run_date=run_date,
                )
            except Exception:
                pass
            # Cannot proceed without SPY (regime filter depends on it)
            return DailyPipelineResult(
                run_date=run_date, tickers_fetched=tickers_fetched,
                tickers_valid=0, tickers_skipped=len(tickers),
                regime="UNKNOWN", orders_generated=0, fills=[],
                pending_orders=[], cash=0.0, total_equity=0.0, errors=errors,
            )

    # ── 2. Fetch, Validate & Feature-Engineer Universe ─────────────────────────
    if _universe_dfs is not None:
        universe_dfs = _universe_dfs
        tickers_valid_list = list(universe_dfs.keys())
        tickers_fetched += len(tickers_valid_list)
        logger.info("Using injected universe DataFrames: %s.", tickers_valid_list)
        for t, u_df in universe_dfs.items():
            try:
                repository.save_market_data(u_df)
            except Exception as exc:
                logger.warning("Failed to save injected %s data: %s", t, exc)
    else:
        universe_dfs: Dict[str, pd.DataFrame] = {}
        tickers_valid_list: List[str] = []

        for ticker in tickers:
            try:
                raw = fetch_ticker_data(ticker, start_date=fetch_start, end_date=run_date)
                tickers_fetched += 1
                val = validate_ticker_data(raw, ticker=ticker)
                if val.is_valid:
                    universe_dfs[ticker] = val.cleaned_df
                    tickers_valid_list.append(ticker)
                    try:
                        repository.save_market_data(val.cleaned_df)
                    except Exception as exc:
                        logger.warning("Failed to save %s market data: %s", ticker, exc)
                else:
                    tickers_skipped += 1
                    err_list = getattr(val, "errors", []) if val else []
                    reason = "; ".join(err_list) if err_list else "unknown"
                    err = f"SKIP {ticker}: validation failed — {reason}"
                    logger.warning(err)
                    errors.append(err)
                    repository.log_event("WARNING", "daily_pipeline", err, {"ticker": ticker})
            except Exception as exc:
                tickers_skipped += 1
                err = f"SKIP {ticker}: fetch/validation error — {exc}"
                logger.error(err)
                errors.append(err)
                repository.log_event("ERROR", "daily_pipeline", err, {"ticker": ticker})

    # ── 3. Compute Features for all valid tickers ──────────────────────────────
    all_feature_dfs: List[pd.DataFrame] = []
    for ticker, df in universe_dfs.items():
        try:
            feats = compute_features(df, spy_df=spy_clean)
            if not feats.empty:
                all_feature_dfs.append(feats)
        except Exception as exc:
            tickers_skipped += 1
            err = f"SKIP {ticker}: feature engineering failed — {exc}"
            logger.error(err)
            errors.append(err)
            repository.log_event("ERROR", "daily_pipeline", err, {"ticker": ticker})

    if not all_feature_dfs:
        err = "No valid features computed for any ticker. Aborting pipeline."
        logger.critical(err)
        errors.append(err)
        repository.log_event("CRITICAL", "daily_pipeline", err)
        return DailyPipelineResult(
            run_date=run_date, tickers_fetched=tickers_fetched,
            tickers_valid=len(tickers_valid_list), tickers_skipped=tickers_skipped,
            regime="UNKNOWN", orders_generated=0, fills=[],
            pending_orders=[], cash=0.0, total_equity=0.0, errors=errors,
        )

    combined_features = pd.concat(all_feature_dfs, ignore_index=True)
    try:
        repository.save_features(combined_features)
    except Exception as exc:
        logger.warning("Failed to save features to repository: %s", exc)

    # ── 3b. Section 5: Gather Market Intelligence Layers ──────────────────────
    # A. News Sentiment Analysis (VADER)
    try:
        sentiment_result = analyze_universe_sentiment(
            tickers=tickers_valid_list,
            date_str=run_date,
            persist=True,
        )
    except Exception as exc:
        logger.warning("SENTIMENT_UNAVAILABLE: Sentiment analysis failed (%s). Continuing normally.", exc)
        sentiment_result = None

    # B. Earnings Calendar Awareness
    try:
        earnings_result = get_universe_earnings_calendar(
            tickers=tickers_valid_list,
            as_of_date_str=run_date,
            persist=True,
        )
    except Exception as exc:
        logger.warning("Earnings calendar retrieval failed (%s). Continuing normally.", exc)
        earnings_result = None

    # C. Sector Rotation Intelligence
    try:
        sector_result = calculate_sector_rotation(
            universe_dfs=universe_dfs,
            as_of_date=run_date,
            persist=True,
        )
    except Exception as exc:
        logger.warning("Sector rotation calculation failed (%s). Continuing normally.", exc)
        sector_result = None

    # D. Macro Economic Indicators
    try:
        macro_result = get_macro_environment(
            as_of_date_str=run_date,
            persist=True,
        )
    except Exception as exc:
        logger.warning("Macro environment inspection failed (%s). Continuing normally.", exc)
        macro_result = None

    # ── 4. Load Active Model & Rank ────────────────────────────────────────────
    try:
        model_file = settings.data_models_dir / "active_model.joblib"
        if settings.model_freeze_enabled:
            verify_frozen_active_model()
        elif not model_file.exists() and not os.path.exists("data/models/active_model.joblib"):
            print("No model found - training new model...")
            logger.info("No active model found at %s - training new model...", model_file)
            from src.ml.train import train_and_promote
            train_and_promote()
            print("Model trained and promoted!")
            logger.info("Model trained and promoted!")

        active_model = load_active_model()
        # Resolve effective feature decision date (latest completed trading day < run_date)
        # Enforce Lag Rule: decisions made at T use information strictly available at T-1 close.
        t_minus_1_dates = combined_features[combined_features["date"] < run_date]["date"]
        if not t_minus_1_dates.empty:
            feature_decision_date = str(t_minus_1_dates.max())
        else:
            # Fallback for single-day test fixtures where no prior date is provided
            valid_feature_dates = combined_features[combined_features["date"] <= run_date]["date"]
            if valid_feature_dates.empty:
                raise ValueError(f"No feature rows found on or before run_date: {run_date}")
            feature_decision_date = str(valid_feature_dates.max())
        logger.info(
            "Ranking candidates for run_date %s using feature close %s (T-1 close)",
            run_date, feature_decision_date,
        )

        ranking_result = rank_candidates(
            features_df=combined_features,
            model=active_model,
            spy_df=spy_clean,
            run_date=feature_decision_date,
            sentiment_result=sentiment_result,
            sector_result=sector_result,
            earnings_result=earnings_result,
            macro_result=macro_result,
        )
        regime = "RISK-ON" if ranking_result.regime_risk_on else "RISK-OFF"

        # Persist predictions to database
        if ranking_result.ranked_opportunities:
            preds_records = [
                {
                    "date": opp.date,
                    "ticker": opp.ticker,
                    "probability": opp.probability,
                    "model_version": getattr(active_model, "metadata", {}).get("model_type", "active_model"),
                }
                for opp in ranking_result.ranked_opportunities
            ]
            try:
                repository.save_predictions(pd.DataFrame(preds_records))
            except Exception as exc:
                logger.warning("Failed to save predictions to database: %s", exc)
    except Exception as exc:
        err = f"CRITICAL: Ranking/model inference failed: {exc}"
        logger.critical(err)
        errors.append(err)
        repository.log_event("CRITICAL", "daily_pipeline", err)
        try:
            send_pipeline_failure_alert(
                error_message=err,
                run_date=run_date,
            )
        except Exception:
            pass
        try:
            notify_pipeline_failure(
                error_message=err,
                run_date=run_date,
            )
        except Exception:
            pass
        return DailyPipelineResult(
            run_date=run_date, tickers_fetched=tickers_fetched,
            tickers_valid=len(tickers_valid_list), tickers_skipped=tickers_skipped,
            regime="UNKNOWN", orders_generated=0, fills=[],
            pending_orders=[], cash=0.0, total_equity=0.0, errors=errors,
        )

    # ── 5. Load Portfolio State from Broker/DB ─────────────────────────────────
    if broker is None:
        broker = PaperBroker(market=market_name)
        broker.load_state()

    # Build open and current (close) prices maps
    open_prices: Dict[str, float] = {}
    current_prices: Dict[str, float] = {}
    for ticker, df in universe_dfs.items():
        df_sorted = df.sort_values("date")
        date_filtered = df_sorted[df_sorted["date"] <= run_date]
        if not date_filtered.empty:
            last_row = date_filtered.iloc[-1]
            close_val = float(last_row["close"])
            current_prices[ticker] = close_val
            # Market open price on run_date (Day T)
            run_date_rows = df_sorted[df_sorted["date"] == run_date]
            if not run_date_rows.empty and "open" in run_date_rows.columns:
                open_val = float(run_date_rows.iloc[0]["open"])
                open_prices[ticker] = open_val if (pd.notna(open_val) and open_val > 0) else close_val
            elif "open" in last_row and pd.notna(last_row["open"]) and float(last_row["open"]) > 0:
                open_prices[ticker] = float(last_row["open"])
            else:
                open_prices[ticker] = close_val

    # ── 5b. Paper Broker: Morning Fills (The Lag Rule) ────────────────────────
    # Orders generated at previous trading day close execute at today's (Day T) market open.
    # Today's newly generated orders will be queued as pending for the next market open.
    fills: List[FillResult] = []
    prior_pending = list(getattr(broker, "pending_orders", []))
    if hasattr(broker, "clear_pending_orders"):
        broker.clear_pending_orders()
    elif hasattr(broker, "pending_orders"):
        broker.pending_orders.clear()

    for order in prior_pending:
        fill_price = open_prices.get(order.ticker)
        if fill_price is None or fill_price <= 0:
            fill_price = current_prices.get(order.ticker)
        if fill_price is None or fill_price <= 0:
            fill_price = order.reference_price

        if fill_price is None or fill_price <= 0:
            err = f"No valid fill price for {order.ticker} on {run_date}. Skipping order."
            logger.warning(err)
            continue

        # Determine 20-day ADV if available in universe_dfs for slippage modeling
        order_adv = getattr(order, "adv", None)
        if order_adv is None:
            t_df = universe_dfs.get(order.ticker)
            if t_df is not None and "volume" in t_df.columns and len(t_df) > 0:
                order_adv = float(t_df["volume"].tail(20).mean())

        result = broker.execute_order(
            order=order,
            fill_price=fill_price,
            run_date=run_date,
            adv=order_adv,
        )
        if result is not None:
            fills.append(result)
            try:
                is_stop_loss = "STOP_LOSS" in str(result.reason).upper()
                if is_stop_loss and result.action == "SELL":
                    pos_entry = broker.positions.get(result.ticker)
                    entry_price = pos_entry.entry_price if pos_entry else result.fill_price
                    loss_pct = ((result.fill_price - entry_price) / entry_price * 100.0
                                if entry_price > 0 else 0.0)
                    send_stop_loss_alert(
                        ticker=result.ticker,
                        entry_price=entry_price,
                        exit_price=result.fill_price,
                        loss_amount=abs(result.net_pnl),
                        loss_pct=loss_pct,
                        portfolio_value=broker.cash,
                    )
                    notify_stop_loss(
                        ticker=result.ticker,
                        loss_amount=abs(result.net_pnl),
                        loss_pct=loss_pct,
                        portfolio_value=broker.cash,
                    )
                elif result.action == "BUY":
                    send_trade_executed_alert(
                        action=result.action,
                        ticker=result.ticker,
                        price=result.fill_price,
                        shares=result.shares,
                        portfolio_value=broker.cash,
                    )
                    notify_trade_bought(
                        ticker=result.ticker,
                        price=result.fill_price,
                        shares=result.shares,
                        portfolio_value=broker.cash,
                    )
                else:  # SELL (non-stop-loss)
                    pnl_pct = (
                        (result.net_pnl / (result.shares * result.fill_price)) * 100.0
                        if result.shares > 0 and result.fill_price > 0 else None
                    )
                    send_trade_executed_alert(
                        action=result.action,
                        ticker=result.ticker,
                        price=result.fill_price,
                        shares=result.shares,
                        pnl=result.net_pnl,
                        exit_reason=result.reason,
                        portfolio_value=broker.cash,
                    )
                    notify_trade_sold(
                        ticker=result.ticker,
                        exit_price=result.fill_price,
                        shares=result.shares,
                        pnl=result.net_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=result.reason,
                        portfolio_value=broker.cash,
                    )
            except Exception:
                pass

    # Reconstruct positions dict compatible with risk engine (reflects morning fills)
    current_positions: Dict[str, Dict[str, Any]] = {
        ticker: {
            "quantity": pos.quantity,
            "avg_cost": pos.entry_price,
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_price,
        }
        for ticker, pos in broker.positions.items()
    }

    # ── 6. Risk Engine ────────────────────────────────────────────────────────
    try:
        macro_mult = macro_result.position_size_multiplier if macro_result else 1.0
        risk_result = evaluate_portfolio_risk(
            run_date=run_date,
            current_cash=broker.cash,
            current_positions=current_positions,
            current_prices=current_prices,
            ranking_result=ranking_result,
            spy_df=spy_clean,
            macro_size_multiplier=macro_mult,
            earnings_result=earnings_result,
        )
    except Exception as exc:
        err = f"CRITICAL: Risk engine failed: {exc}"
        logger.critical(err)
        errors.append(err)
        repository.log_event("CRITICAL", "daily_pipeline", err)
        try:
            send_pipeline_failure_alert(
                error_message=err,
                run_date=run_date,
                last_portfolio_value=broker.cash,
            )
        except Exception:
            pass
        try:
            notify_pipeline_failure(
                error_message=err,
                run_date=run_date,
                last_portfolio_value=broker.cash,
            )
        except Exception:
            pass
        return DailyPipelineResult(
            run_date=run_date, tickers_fetched=tickers_fetched,
            tickers_valid=len(tickers_valid_list), tickers_skipped=tickers_skipped,
            regime=regime, orders_generated=0, fills=[],
            pending_orders=[], cash=broker.cash, total_equity=broker.cash, errors=errors,
        )

    # ── 7. Portfolio Allocation (OrderSpec generation) ─────────────────────────
    try:
        alloc_result = allocate_portfolio(
            risk_assessment=risk_result,
            current_cash=broker.cash,
            current_positions=current_positions,
            current_prices=current_prices,
            fee_rate=settings.simulated_cost_per_trade,
        )
        new_pending_orders: List[OrderSpec] = alloc_result.orders
    except Exception as exc:
        err = f"CRITICAL: Portfolio allocation failed: {exc}"
        logger.critical(err)
        errors.append(err)
        repository.log_event("CRITICAL", "daily_pipeline", err)
        new_pending_orders = []

    orders_generated = len(new_pending_orders)

    # ── 8. Queue Orders as PENDING for Next Market Open (Anti-Leakage Lag Rule) ──
    # Orders generated at Day T close must NEVER execute against Day T close prices.
    # They are queued as pending to execute at the next market open (Day T+1 open).
    if hasattr(broker, "sync_pending_orders"):
        broker.sync_pending_orders(new_pending_orders)
    elif hasattr(broker, "pending_orders"):
        broker.pending_orders = list(new_pending_orders)
    logger.info(
        "Day %s: %d order(s) generated at close — queued as pending for next market open.",
        run_date, orders_generated,
    )

    # ── 8b. Email + Telegram: circuit breaker and PSI drift ──────────────────
    try:
        if risk_result.circuit_breaker_active:
            _cash_pct = (broker.cash / max(risk_result.total_equity, 1.0)) * 100.0
            send_circuit_breaker_alert(
                spy_drop_pct=settings.macro_cb_5d_drop_pct * 100.0,
                lookback_window=5,
                pause_days=settings.macro_cb_5d_halt_days,
                portfolio_value=broker.cash,
                cash_pct=_cash_pct,
            )
            notify_circuit_breaker(
                spy_drop_pct=settings.macro_cb_5d_drop_pct * 100.0,
                lookback_days=5,
                pause_days=settings.macro_cb_5d_halt_days,
                cash_pct=_cash_pct,
            )
    except Exception:
        pass

    try:
        psi_val = getattr(risk_result, "psi_value", None)
        if psi_val is not None and psi_val >= settings.psi_drift_alert_threshold:
            send_psi_drift_alert(
                psi_value=psi_val,
                details=f"Detected on pipeline run for {run_date}.",
            )
            notify_psi_drift(psi_value=psi_val)
    except Exception:
        pass

    # ── 9. End-of-Day Snapshot ────────────────────────────────────────────────
    total_equity = broker.record_snapshot(run_date=run_date, current_prices=current_prices)

    # ── 9c. Section 10 Item 1: Paper Trading Leaderboard ───────────────────────
    try:
        if getattr(settings, "leaderboard_enabled", True):
            from src.trading.leaderboard import run_leaderboard_cycle
            preds_dict = {
                opp.ticker: opp.probability
                for opp in ranking_result.ranked_opportunities
            } if ranking_result and ranking_result.ranked_opportunities else None

            run_leaderboard_cycle(
                run_date=run_date,
                prices=current_prices,
                predictions=preds_dict,
            )
    except Exception as _lb_exc:
        logger.warning("Leaderboard cycle run skipped: %s", _lb_exc)

    status_str = "SUCCESS" if not errors else ("PARTIAL_SUCCESS" if len(tickers_valid_list) > 0 else "FAILED")
    repository.log_event(
        "INFO" if status_str == "SUCCESS" else "WARNING", market_comp,
        f"Pipeline complete for {run_date} ({market_name}). Status={status_str}, Fills={len(fills)}, Equity=${total_equity:.4f}, Slippage=${broker.total_slippage_cost:.4f}",
        {
            "run_date": run_date,
            "market": market_name,
            "status": status_str,
            "fills": len(fills),
            "orders_generated": orders_generated,
            "cash": broker.cash,
            "total_equity": total_equity,
            "total_slippage_cost": broker.total_slippage_cost,
            "errors": errors,
        },
    )

    # ── 9b. Telegram Daily Summary ─────────────────────────────────────────
    try:
        sells_today = [f for f in fills if f.action == "SELL"]
        daily_pnl = sum(f.net_pnl for f in sells_today) if sells_today else None
        daily_pnl_pct = (
            (daily_pnl / (total_equity - daily_pnl) * 100.0)
            if daily_pnl is not None and (total_equity - (daily_pnl or 0.0)) > 0
            else None
        )
        open_positions = list(broker.positions.keys())
        cash_pct = (broker.cash / max(total_equity, 1.0)) * 100.0
        notify_daily_summary(
            run_date=run_date,
            portfolio_value=total_equity,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            cash=broker.cash,
            cash_pct=cash_pct,
            positions=open_positions,
            max_positions=settings.max_positions,
            status=status_str,
        )
    except Exception:
        pass



    return DailyPipelineResult(
        run_date=run_date,
        tickers_fetched=tickers_fetched,
        tickers_valid=len(tickers_valid_list),
        tickers_skipped=tickers_skipped,
        regime=regime,
        orders_generated=orders_generated,
        fills=fills,
        pending_orders=new_pending_orders,
        cash=broker.cash,
        total_equity=total_equity,
        errors=errors,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AI Stock Trader — Daily Paper Trading Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date 'YYYY-MM-DD' (replay mode). Defaults to today (UTC).",
    )
    args = parser.parse_args()

    result = run_daily_pipeline(run_date=args.date)
    print(result.to_markdown())
