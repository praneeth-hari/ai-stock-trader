"""
dashboard/data_loader.py — Data loading, queries, and trigger functions for Streamlit dashboard.

Pulls directly from the DB repository (src.db.repository), settings, active model,
and backtesting engine. Decouples business logic and SQL from UI rendering.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.db import repository

logger = logging.getLogger(__name__)


def get_last_pipeline_status() -> Dict[str, Any]:
    """
    Inspects the database event log to determine the true outcome of the most recent pipeline execution.
    Returns:
        {
            "status": "SUCCESS" | "FAILED" | "SKIPPED" | "NO_RUNS",
            "message": str,
            "timestamp": Optional[str],
            "run_date": Optional[str],
        }
    """
    try:
        events = repository.get_events(limit=30)
    except Exception as exc:
        logger.warning("DB connection error in get_last_pipeline_status: %s", exc)
        return {"status": "NO_RUNS", "message": "Database disconnected", "timestamp": None}

    for e in events:
        comp = str(e.get("component", ""))
        msg = str(e.get("message", ""))
        lvl = str(e.get("level", ""))
        ts = str(e.get("timestamp", ""))

        if comp in ("daily_pipeline", "scheduler"):
            for explicit in ("FAILED", "DEGRADED", "HALTED", "SKIPPED"):
                if f"Status={explicit}" in msg:
                    return {"status": explicit, "message": msg, "timestamp": ts}
            if "Pipeline complete" in msg or "Scheduled pipeline run completed" in msg:
                if "Status=FAILED" in msg or "status: FAILED" in msg or "Status=PARTIAL_SUCCESS" in msg:
                    return {"status": "FAILED" if "Status=FAILED" in msg else "PARTIAL_SUCCESS", "message": msg, "timestamp": ts}
                return {"status": "SUCCESS", "message": msg, "timestamp": ts}
            if "CRITICAL" in lvl or "failed" in msg.lower() or "error" in msg.lower() or "aborting" in msg.lower():
                return {"status": "FAILED", "message": msg, "timestamp": ts}
            if "SKIPPED" in msg or "skipped" in msg.lower():
                return {"status": "SKIPPED", "message": msg, "timestamp": ts}

    return {"status": "NO_RUNS", "message": "No pipeline runs recorded", "timestamp": None}


def get_portfolio_summary(market: str = "US") -> Dict[str, Any]:
    """
    Returns current portfolio snapshot and derived risk metrics.

    Fallback: If database is disconnected or snapshot read fails,
    initializes with settings.initial_capital and zero open positions.
    """
    try:
        snap = repository.get_latest_portfolio_snapshot(market=market)
    except Exception as exc:
        logger.warning("DB connection unavailable in get_portfolio_summary: %s", exc)
        snap = None

    try:
        last_status = get_last_pipeline_status()
    except Exception:
        last_status = {"status": "System Running on Cloud ✅", "message": "Trading pipeline running on GitHub. Check Telegram for live updates! 📱", "timestamp": None}

    if snap is None:
        cash = float(settings.get_initial_capital(market))
        total_equity = float(settings.get_initial_capital(market))
        raw_positions: Dict[str, Any] = {}
        run_date = None
    else:
        cash = float(snap.get("cash", settings.get_initial_capital(market)))
        total_equity = float(snap.get("total_value", settings.get_initial_capital(market)))
        raw_positions = snap.get("positions", {}) or {}
        run_date = snap.get("run_date")

    # Parse positions list
    positions_list: List[Dict[str, Any]] = []
    unrealized_pnl_total = 0.0

    # Fetch latest sentiment & earnings records
    try:
        latest_sent = repository.get_latest_sentiment_scores()
    except Exception:
        latest_sent = {}

    try:
        latest_earn = repository.get_latest_earnings_calendar()
    except Exception:
        latest_earn = {}

    for ticker, pos in raw_positions.items():
        qty = float(pos.get("shares", pos.get("quantity", 0.0)))
        entry_price = float(pos.get("entry_price", pos.get("avg_cost", 0.0)))
        current_price = float(pos.get("current_price", entry_price))
        market_val = float(pos.get("market_value", pos.get("value", round(qty * current_price, 4))))
        unrealized = float(pos.get("unrealized_pnl", round((current_price - entry_price) * qty, 4)))
        unrealized_pct = float(pos.get("unrealized_pnl_pct", round(((current_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 0.0))
        unrealized_pnl_total += unrealized

        # Section 5 Intelligence fields
        s_data = latest_sent.get(ticker.upper(), {})
        s_score = s_data.get("composite_score", 0.0)
        s_str = f"{s_score:+.2f}" if s_data else "N/A"

        e_data = latest_earn.get(ticker.upper(), {})
        e_days = e_data.get("days_until_earnings")
        if e_days is not None:
            if e_days == 0:
                e_badge = "⚠️ Reports Today"
            elif 1 <= e_days <= 5:
                e_badge = f"⚠️ Earnings in {e_days}d"
            else:
                e_badge = f"Reports in {e_days}d"
        else:
            e_badge = "—"

        positions_list.append({
            "ticker": ticker,
            "shares": qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "market_value": market_val,
            "unrealized_pnl": unrealized,
            "unrealized_pnl_pct": unrealized_pct,
            "sentiment": s_str,
            "earnings": e_badge,
            "size_tier": pos.get("confidence_tier", pos.get("size_tier", "FULL")),
        })

    cash_reserve_pct = (cash / total_equity * 100.0) if total_equity > 0 else 100.0

    return {
        "run_date": run_date,
        "cash": round(cash, 4),
        "total_equity": round(total_equity, 4),
        "invested_value": round(max(0.0, total_equity - cash), 4),
        "total_slippage_cost": round(float(snap.get("total_slippage_cost", 0.0) or 0.0), 4) if snap else 0.0,
        "cash_reserve_pct": round(cash_reserve_pct, 2),
        "open_positions_count": len(positions_list),
        "max_positions": settings.max_positions,
        "unrealized_pnl_total": round(unrealized_pnl_total, 4),
        "positions": positions_list,
        "last_run_status": last_status["status"],
        "last_run_message": last_status["message"],
        "last_run_timestamp": last_status["timestamp"],
    }


def get_equity_history_df(market: str = "US") -> pd.DataFrame:
    """
    Returns historical daily portfolio snapshots formatted as a time-series DataFrame.
    """
    try:
        snapshots = repository.get_portfolio_snapshots(limit=500, market=market)
    except Exception as exc:
        logger.warning("DB connection error in get_equity_history_df: %s", exc)
        snapshots = []

    if not snapshots:
        # Default single point with initial capital
        today_str = date.today().strftime("%Y-%m-%d")
        return pd.DataFrame([{
            "date": today_str,
            "cash": float(settings.get_initial_capital(market)),
            "total_value": float(settings.get_initial_capital(market)),
            "invested": 0.0,
        }])

    rows = []
    for s in snapshots:
        c = float(s["cash"])
        tot = float(s["total_value"])
        rows.append({
            "date": s["run_date"],
            "cash": c,
            "total_value": tot,
            "invested": max(0.0, tot - c),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_recent_trades_df(limit: int = 50, market: str = "US") -> pd.DataFrame:
    """
    Returns recent executed trades from DB.
    """
    try:
        trades = repository.get_trades(run_date=None, limit=limit, market=market)
    except Exception as exc:
        logger.warning("DB connection error in get_recent_trades_df: %s", exc)
        trades = []

    if not trades:
        return pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "fee", "net_pnl"])

    rows = []
    for t in trades:
        rows.append({
            "date": t.get("run_date", ""),
            "ticker": t.get("ticker", ""),
            "action": t.get("action", ""),
            "shares": round(float(t.get("quantity", 0.0)), 4),
            "price": round(float(t.get("fill_price", 0.0)), 2),
            "fee": round(float(t.get("cost", 0.0)), 4),
            "net_pnl": round(float(t.get("net_pnl", 0.0)), 4),
        })

    df = pd.DataFrame(rows)
    import datetime
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    if not df.empty and "date" in df.columns:
        df = df[df["date"] >= cutoff]

    return df


def get_recent_orders_df(limit: int = 50, market: str = "US") -> pd.DataFrame:
    """
    Returns recent order decisions from DB.
    """
    try:
        orders = repository.get_orders(run_date=None, limit=limit, market=market)
    except Exception as exc:
        logger.warning("DB connection error in get_recent_orders_df: %s", exc)
        orders = []

    if not orders:
        return pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "reason"])

    rows = []
    for o in orders:
        rows.append({
            "date": o.get("run_date", ""),
            "ticker": o.get("ticker", ""),
            "action": o.get("action", ""),
            "shares": round(float(o.get("quantity", 0.0)), 4),
            "price": round(float(o.get("price", 0.0)), 2),
            "reason": o.get("reason", ""),
        })

    df = pd.DataFrame(rows)
    import datetime
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    if not df.empty and "date" in df.columns:
        df = df[df["date"] >= cutoff]

    return df


get_orders_df = get_recent_orders_df
get_trades_df = get_recent_trades_df


def get_system_events_df(limit: int = 100) -> pd.DataFrame:
    """
    Returns system audit and pipeline events.
    """
    try:
        events = repository.get_events(limit=limit)
    except Exception as exc:
        logger.warning("DB connection error in get_system_events_df: %s", exc)
        events = []

    if not events:
        return pd.DataFrame(columns=["timestamp", "level", "component", "message"])

    rows = []
    for e in events:
        rows.append({
            "timestamp": e.get("timestamp", ""),
            "level": e.get("level", "INFO"),
            "component": e.get("component", "system"),
            "message": e.get("message", ""),
        })

    return pd.DataFrame(rows)


def get_market_regime_and_predictions(market: str = "US") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Pulls regime status (SPY vs 200d MA) and generates/loads latest candidate rankings.
    """
    regime_info: Dict[str, Any] = {
        "status": "UNKNOWN",
        "spy_price": None,
        "spy_200ma": None,
        "is_risk_on": True,
        "explanation": "No SPY data available.",
    }

    # 1. Check regime
    try:
        from src.data.market_data import fetch_ticker_data
        from src.data.validation import validate_ticker_data
        bm_symbol = settings.get_benchmark(market)
        spy_raw = fetch_ticker_data(bm_symbol, period="2y")
        spy_val = validate_ticker_data(spy_raw, ticker=bm_symbol)
        if spy_val.is_valid and len(spy_val.cleaned_df) >= settings.regime_ma_window:
            spy_df = spy_val.cleaned_df.sort_values("date").reset_index(drop=True)
            curr_price = float(spy_df["close"].iloc[-1])
            ma_200 = float(spy_df["close"].rolling(settings.regime_ma_window).mean().iloc[-1])
            is_risk_on = curr_price >= ma_200
            regime_info = {
                "status": "RISK-ON" if is_risk_on else "RISK-OFF",
                "spy_price": round(curr_price, 2),
                "spy_200ma": round(ma_200, 2),
                "is_risk_on": is_risk_on,
                "explanation": (
                    f"SPY (${curr_price:.2f}) >= 200-day MA (${ma_200:.2f}) -> Buys permitted."
                    if is_risk_on else
                    f"SPY (${curr_price:.2f}) < 200-day MA (${ma_200:.2f}) -> Regime filter ACTIVE. Buys vetoed."
                ),
            }
    except Exception as exc:
        logger.warning("Failed to fetch live SPY regime in dashboard: %s", exc)
        regime_info["explanation"] = f"Offline / Cached mode: {exc}"

    # 2. Candidate predictions table
    today_str = date.today().strftime("%Y-%m-%d")
    preds_df = pd.DataFrame(columns=["ticker", "sector", "probability", "sentiment", "earnings", "qualification", "action_signal"])

    # Load latest sentiment and earnings records
    try:
        latest_sent = repository.get_latest_sentiment_scores()
    except Exception:
        latest_sent = {}

    try:
        latest_earn = repository.get_latest_earnings_calendar()
    except Exception:
        latest_earn = {}

    from config.settings import get_ticker_sector

    rows = []
    # If we have universe tickers, check rankings or load from active model
    try:
        from src.ml.evaluate import load_active_model
        model = load_active_model()
        model_name = getattr(model, "metadata", {}).get("model_type", "active_model")
        regime_info["active_model"] = model_name
    except Exception as exc:
        regime_info["active_model"] = f"None ({exc})"

    for ticker in settings.get_universe(market):
        # Check DB prediction for today
        db_p = repository.get_predictions(ticker, today_str, today_str)
        prob = float(db_p["probability"].iloc[-1]) if not db_p.empty else 0.50

        sec = get_ticker_sector(ticker)
        s_data = latest_sent.get(ticker.upper(), {})
        s_score = s_data.get("composite_score", 0.0)
        s_str = f"{s_score:+.2f} ({s_data.get('sentiment_label', 'NEUTRAL')})" if s_data else "—"

        e_data = latest_earn.get(ticker.upper(), {})
        e_days = e_data.get("days_until_earnings")
        if e_days is not None:
            if e_days == 0:
                e_badge = "⚠️ Reports Today"
            elif 1 <= e_days <= 2:
                e_badge = f"⛔ Blackout ({e_days}d)"
            elif 3 <= e_days <= 5:
                e_badge = f"⚠️ Caution ({e_days}d)"
            else:
                e_badge = f"Reports in {e_days}d"
        else:
            e_badge = "—"

        if prob >= settings.buy_bar:
            qual = "BUY (Qualified)" if regime_info["is_risk_on"] else "BLOCKED (Regime Risk-Off)"
            signal = "BUY" if regime_info["is_risk_on"] else "VETO"
        elif prob < settings.signal_exit:
            qual = "EXIT (Below 0.45 threshold)"
            signal = "SELL"
        else:
            qual = "HOLD (Dead-zone 0.45 - 0.60)"
            signal = "HOLD"

        rows.append({
            "ticker": ticker,
            "sector": sec,
            "probability": round(prob, 4),
            "sentiment": s_str,
            "earnings": e_badge,
            "qualification": qual,
            "action_signal": signal,
        })

    if rows:
        preds_df = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)

    return regime_info, preds_df


def run_daily_paper_cycle_trigger(run_date: Optional[str] = None, market: str = "US") -> Dict[str, Any]:
    """
    Executes a daily pipeline paper-trading cycle and returns the audit summary.
    """
    from src.pipeline.daily_pipeline import run_daily_pipeline
    target_date = run_date or date.today().strftime("%Y-%m-%d")
    result = run_daily_pipeline(run_date=target_date, market_name=market)
    return {
        "run_date": target_date,
        "fills_count": len(result.fills),
        "orders_count": result.orders_generated,
        "equity": result.total_equity,
        "audit_markdown": result.to_markdown(),
    }


def _backtest_warmup_start(start_date: str) -> str:
    """Fetch start giving 200-day features valid history on day 1; the test window is unchanged."""
    return (pd.to_datetime(start_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")


def _validate_backtest_universe(
    tickers: List[str], fetch_start: str, fetch_end: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Tuple[str, str]]]]:
    """Point-in-time validation: a later bad row excludes a ticker only from the dates live would have."""
    from src.data.market_data import fetch_ticker_data
    from src.data.validation import validate_ticker_data_point_in_time

    universe_dict: Dict[str, pd.DataFrame] = {}
    data_exclusions: Dict[str, List[Tuple[str, str]]] = {}
    for t in tickers:
        val = validate_ticker_data_point_in_time(
            fetch_ticker_data(t, start_date=fetch_start, end_date=fetch_end), ticker=t,
        )
        if val.is_valid:
            universe_dict[t] = val.cleaned_df
            if val.excluded_windows:
                data_exclusions[t] = val.excluded_windows
    return universe_dict, data_exclusions


def run_backtest_trigger(
    start_date: str = "2021-07-09",
    end_date: str = "2024-08-30",
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes a historical backtest and returns metrics and comparative equity curve.
    """
    from src.data.market_data import fetch_ticker_data
    from src.data.validation import validate_ticker_data
    from src.backtest.backtest import run_strategy_backtest

    t_list = tickers or ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM"]

    # Note on date inclusion: yfinance end_date is exclusive [start, end).
    # Extend fetch window slightly (+2 days) so the requested end_date is fetched.
    try:
        fetch_end = (pd.to_datetime(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    except Exception:
        fetch_end = end_date
    fetch_start = _backtest_warmup_start(start_date)

    spy_raw = fetch_ticker_data("SPY", start_date=fetch_start, end_date=fetch_end)
    spy_df = validate_ticker_data(spy_raw, ticker="SPY").cleaned_df

    universe_dict, data_exclusions = _validate_backtest_universe(t_list, fetch_start, fetch_end)

    result = run_strategy_backtest(
        universe_dict=universe_dict,
        spy_df=spy_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(settings.initial_capital),
        # None -> engine resolves a model trained strictly before start_date, or flags non-OOS.
        model=None,
        data_exclusions=data_exclusions,
    )
    m = result.metrics
    eq_rows = [
        {
            "date": s.date,
            "strategy_equity": round(s.total_equity, 2),
            "benchmark_equity": round(float(settings.initial_capital) * (1.0 + getattr(s, "spy_cumulative_return", 0.0)), 2),
            "cash": round(s.cash, 2),
        }
        for s in result.daily_snapshots
    ]
    eq_df = pd.DataFrame(eq_rows)

    return {
        "cagr": round(m.annualized_return_pct, 2),
        "benchmark_cagr": round(m.spy_annualized_return_pct, 2),
        "max_drawdown": round(m.max_drawdown_pct, 2),
        "benchmark_max_drawdown": round(m.spy_max_drawdown_pct, 2),
        "sharpe_ratio": round(m.sharpe_ratio, 2),
        "benchmark_sharpe_ratio": round(m.spy_sharpe_ratio, 2),
        "win_rate": round(m.win_rate_pct, 2),
        "profit_factor": round(m.profit_factor, 2),
        "total_trades": m.total_trades,
        "net_pnl": round(m.final_equity - m.starting_capital, 2),
        "equity_curve": eq_df,
        "survivorship_biased": m.survivorship_biased,
        "alarm_triggered": m.alarm_triggered,
        "alarm_reasons": list(m.alarm_reasons),
        "model_out_of_sample": m.model_out_of_sample,
        "resolved_model_name": m.resolved_model_name,
        "start_date": m.start_date,
        "end_date": m.end_date,
    }


def run_backtest_lab_trigger(
    start_date: str,
    end_date: str,
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
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Dashboard wrapper for executing an interactive Backtest Lab run.
    """
    from src.backtest.backtest import run_lab_backtest
    from src.data.market_data import fetch_ticker_data, save_raw_snapshot
    from src.data.validation import validate_ticker_data

    t_list = tickers or ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "JNJ", "JPM", "V"]
    try:
        fetch_end = (pd.to_datetime(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    except Exception:
        fetch_end = end_date
    fetch_start = _backtest_warmup_start(start_date)

    spy_raw = fetch_ticker_data("SPY", start_date=fetch_start, end_date=fetch_end)
    spy_df = validate_ticker_data(spy_raw, ticker="SPY").cleaned_df

    universe_dict, data_exclusions = _validate_backtest_universe(t_list, fetch_start, fetch_end)

    return run_lab_backtest(
        universe_dict=universe_dict,
        spy_df=spy_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(settings.initial_capital),
        buy_threshold=buy_threshold,
        exit_threshold=exit_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_positions=max_positions,
        position_sizing=position_sizing,
        use_sentiment=use_sentiment,
        use_earnings_blackout=use_earnings_blackout,
        use_sector_rotation=use_sector_rotation,
        use_macro_regime=use_macro_regime,
        use_correlation_filter=use_correlation_filter,
        use_trailing_stop=use_trailing_stop,
        model=None,
        data_exclusions=data_exclusions,
    )



def get_fundamental_screener_data(fetch_date: Optional[str] = None) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Loads screener results from cache or runs quick screen.
    Returns (DataFrame, List[FundamentalScorecard]).
    """
    from src.screening.screener import run_fundamental_screen, scorecards_to_dataframe
    from src.screening.universe import SCREENER_UNIVERSE

    date_str = fetch_date or date.today().strftime("%Y-%m-%d")
    cache_dir = Path("data/fundamentals") / date_str

    # If cached files exist for at least 5 tickers, load them without re-querying yfinance
    cached_tickers = [c.ticker for c in SCREENER_UNIVERSE if (cache_dir / f"{c.ticker}.json").exists()]
    target_tickers = cached_tickers if len(cached_tickers) >= 5 else [c.ticker for c in SCREENER_UNIVERSE[:10]]

    cards = run_fundamental_screen(tickers=target_tickers, fetch_date=date_str, force_refresh=False)
    df = scorecards_to_dataframe(cards)
    return df, cards


def run_fundamental_screen_trigger(full_universe: bool = False, force: bool = False) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Explicit operator trigger to run fundamental screening across universe.
    """
    from src.screening.screener import run_fundamental_screen, scorecards_to_dataframe
    from src.screening.universe import SCREENER_UNIVERSE

    tickers = [c.ticker for c in SCREENER_UNIVERSE] if full_universe else [c.ticker for c in SCREENER_UNIVERSE[:15]]
    cards = run_fundamental_screen(tickers=tickers, force_refresh=force)
    df = scorecards_to_dataframe(cards)
    return df, cards


# ── V2.1 Wave 1: Explainability & Drift Loaders ───────────────────────────────

def get_prediction_explanation(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Computes SHAP feature attributions for a ticker's latest feature vector.
    """
    from src.db import repository
    from src.ml.explainability import explain_prediction
    from src.ml.evaluate import load_active_model

    df_feat = repository.get_features(ticker)
    if df_feat.empty:
        return None

    latest_row = df_feat.iloc[-1].to_dict()
    try:
        model = load_active_model()
        return explain_prediction(ticker=ticker, feature_data=latest_row, model=model)
    except Exception as exc:
        logger.warning("get_prediction_explanation failed for %s: %s", ticker, exc)
        return None


def get_model_drift_summary() -> Dict[str, Any]:
    """
    Retrieves model drift and population stability metrics against Phase 6 baseline.
    """
    from src.ml.drift import detect_prediction_drift
    try:
        return detect_prediction_drift()
    except Exception as exc:
        logger.warning("get_model_drift_summary failed: %s", exc)
        return {"status": "ERROR", "drift_detected": False, "recommendation": str(exc)}


def get_model_calibration_info() -> Dict[str, Any]:
    """
    Retrieves calibration table and locked thresholds from Phase 6 walk-forward evaluation.
    """
    from src.ml.evaluate import get_calibration_table, MIN_CONVICTION_SAMPLE_SIZE
    return {
        "table": get_calibration_table(),
        "min_sample_size": MIN_CONVICTION_SAMPLE_SIZE,
    }


# ── V2.2 Wave 1: Portfolio Intelligence Loaders ───────────────────────────────

def get_portfolio_sectors_summary(market: str = "US") -> Dict[str, Any]:
    """
    Retrieves sector breakdown and concentration metrics for current portfolio.
    """
    from src.portfolio.intelligence import compute_sector_breakdown
    summary = get_portfolio_summary(market=market)
    return compute_sector_breakdown(
        positions=summary.get("positions", []),
        cash=summary.get("cash", 10000.0),
        total_equity=summary.get("total_equity", 10000.0),
    )


def get_portfolio_diversification_summary(market: str = "US") -> Dict[str, Any]:
    """
    Retrieves Diversification Score (0-100) and HHI concentration analysis.
    """
    from src.portfolio.intelligence import compute_diversification_score
    summary = get_portfolio_summary(market=market)
    return compute_diversification_score(
        positions=summary.get("positions", []),
        cash=summary.get("cash", 10000.0),
        total_equity=summary.get("total_equity", 10000.0),
    )


def simulate_what_if_action(
    action: str,
    ticker: str,
    simulated_amount: Optional[float] = None,
    simulated_price: Optional[float] = None,
    market: str = "US",
) -> Dict[str, Any]:
    """
    Runs in-memory structural what-if simulation on current portfolio.
    """
    from src.portfolio.intelligence import simulate_what_if
    summary = get_portfolio_summary(market=market)
    return simulate_what_if(
        positions=summary.get("positions", []),
        cash=summary.get("cash", 10000.0),
        total_equity=summary.get("total_equity", 10000.0),
        action=action,
        ticker=ticker,
        simulated_amount=simulated_amount,
        simulated_price=simulated_price,
    )


# ── Section 5: Intelligence Layer Dashboard Loaders ───────────────────────────

def get_sector_rotation_summary() -> List[Dict[str, Any]]:
    """
    Returns the latest sector rotation rankings from the database.
    Each dict: {date, sector, avg_20d_return, rank, rotation_multiplier}
    Returns empty list if no data available.
    """
    try:
        today_str = date.today().strftime("%Y-%m-%d")
        rows = repository.get_latest_sector_rankings(as_of_date=today_str)
        return rows
    except Exception as exc:
        logger.warning("get_sector_rotation_summary failed: %s", exc)
        return []


def get_macro_environment_summary() -> Dict[str, Any]:
    """
    Returns the latest macroeconomic environment snapshot from the database.
    Falls back to sensible defaults if no data available.
    """
    defaults = {
        "date": "N/A",
        "fed_funds_rate": None,
        "cpi_yoy": None,
        "unemployment_rate": None,
        "treasury_10y": None,
        "macro_regime": "NEUTRAL",
        "fetched_date": "N/A",
        "is_cached": True,
        "source": "default",
    }
    try:
        today_str = date.today().strftime("%Y-%m-%d")
        row = repository.get_latest_macro_indicators(as_of_date=today_str)
        if row:
            row["source"] = "database"
            return row
        return defaults
    except Exception as exc:
        logger.warning("get_macro_environment_summary failed: %s", exc)
        return defaults


# ── Section 6 Item 1: Performance Charts Loaders ─────────────────────────────

def get_portfolio_vs_spy_chart_data(market: str = "US") -> pd.DataFrame:
    """
    Returns time-series DataFrame comparing Portfolio value to SPY benchmark.
    Data comes strictly from portfolio_snapshots and market_data tables in the database.

    Columns:
      date: str (YYYY-MM-DD)
      My Portfolio: float ($)
      SPY Benchmark: float ($)
      underperforming: bool (My Portfolio < SPY Benchmark)
      deficit: float ($)
    """
    snapshots = repository.get_portfolio_snapshots(limit=1000, market=market)
    cols = ["date", "My Portfolio", "SPY Benchmark", "underperforming", "deficit"]
    if not snapshots:
        today_str = date.today().strftime("%Y-%m-%d")
        c0 = float(settings.initial_capital)
        return pd.DataFrame([{
            "date": today_str,
            "My Portfolio": c0,
            "SPY Benchmark": c0,
            "underperforming": False,
            "deficit": 0.0,
        }])

    # Sort snapshots chronologically
    sorted_snaps = sorted(snapshots, key=lambda s: s["run_date"])
    first_date = sorted_snaps[0]["run_date"]
    initial_equity = (
        float(sorted_snaps[0]["total_value"])
        if float(sorted_snaps[0]["total_value"]) > 0
        else float(settings.initial_capital)
    )

    # Query SPY market data from database
    spy_df = repository.get_market_data(settings.get_benchmark(market))
    spy_map: Dict[str, float] = {}
    spy_base_price: Optional[float] = None

    if not spy_df.empty and "date" in spy_df.columns and "close" in spy_df.columns:
        spy_sorted = spy_df.sort_values("date").reset_index(drop=True)
        spy_sorted["date_str"] = spy_sorted["date"].astype(str).str[:10]
        spy_dict = dict(zip(spy_sorted["date_str"], spy_sorted["close"]))

        # Base price on or before first_date, else first available
        prior_dates = [d for d in spy_sorted["date_str"] if d <= first_date]
        if prior_dates:
            spy_base_price = float(spy_dict[prior_dates[-1]])
        else:
            spy_base_price = float(spy_sorted["close"].iloc[0])

        all_spy_dates = sorted(list(spy_dict.keys()))
        for s in sorted_snaps:
            s_date = s["run_date"]
            if s_date in spy_dict:
                spy_map[s_date] = float(spy_dict[s_date])
            else:
                priors = [d for d in all_spy_dates if d <= s_date]
                if priors:
                    spy_map[s_date] = float(spy_dict[priors[-1]])
                elif all_spy_dates:
                    spy_map[s_date] = float(spy_dict[all_spy_dates[0]])

    rows = []
    for s in sorted_snaps:
        s_date = s["run_date"]
        port_val = float(s["total_value"])
        if spy_base_price and spy_base_price > 0 and s_date in spy_map:
            spy_val = initial_equity * (spy_map[s_date] / spy_base_price)
        else:
            spy_val = initial_equity

        port_val_rnd = round(port_val, 2)
        spy_val_rnd = round(spy_val, 2)
        under = bool(port_val_rnd < spy_val_rnd)
        deficit = round(max(0.0, spy_val_rnd - port_val_rnd), 2)

        rows.append({
            "date": s_date,
            "My Portfolio": port_val_rnd,
            "SPY Benchmark": spy_val_rnd,
            "underperforming": under,
            "deficit": deficit,
        })

    return pd.DataFrame(rows)


def get_daily_returns_histogram_data(num_bins: int = 15, market: str = "US") -> pd.DataFrame:
    """
    Returns binned frequency of daily portfolio % returns.
    Green bars represent positive return days (>= 0%).
    Red bars represent negative return days (< 0%).
    Includes bin_center, bin_label, count, and sign.
    """
    snapshots = repository.get_portfolio_snapshots(limit=1000, market=market)
    cols = ["bin_center", "bin_label", "count", "sign"]
    if not snapshots or len(snapshots) < 2:
        return pd.DataFrame(columns=cols)

    sorted_snaps = sorted(snapshots, key=lambda s: s["run_date"])
    returns = []
    for i in range(1, len(sorted_snaps)):
        prev_val = float(sorted_snaps[i - 1]["total_value"])
        curr_val = float(sorted_snaps[i]["total_value"])
        if prev_val > 0:
            ret_pct = ((curr_val - prev_val) / prev_val) * 100.0
            returns.append(ret_pct)

    if not returns:
        return pd.DataFrame(columns=cols)

    # Check if all returns are zero or near-zero
    if all(abs(r) < 1e-6 for r in returns):
        return pd.DataFrame([{
            "bin_center": 0.0,
            "bin_label": "0.00%",
            "count": len(returns),
            "sign": "Positive",
        }])

    min_ret = min(returns)
    max_ret = max(returns)

    half_bins = max(3, num_bins // 2)
    neg_min = min(min_ret * 1.05, -0.1) if min_ret < 0 else -0.1
    pos_max = max(max_ret * 1.05, 0.1) if max_ret > 0 else 0.1

    neg_edges = np.linspace(neg_min, 0.0, half_bins + 1)
    pos_edges = np.linspace(0.0, pos_max, half_bins + 1)
    bin_edges = np.unique(np.concatenate([neg_edges, pos_edges]))

    cats = pd.cut(returns, bins=bin_edges, include_lowest=True)
    counts = cats.value_counts().sort_index()

    rows = []
    for interval, count in counts.items():
        if count > 0:
            mid = float(interval.mid)
            sign = "Positive" if mid >= 0.0 else "Negative"
            rows.append({
                "bin_center": round(mid, 3),
                "bin_label": f"{interval.left:+.2f}% to {interval.right:+.2f}%",
                "count": int(count),
                "sign": sign,
            })

    return pd.DataFrame(rows)


def get_win_loss_trades_chart_data(limit: int = 200, market: str = "US") -> pd.DataFrame:
    """
    Returns closed trade PnL history for win/loss bar chart.
    Each bar represents one closed trade (action == 'SELL' or net_pnl != 0.0).
    Green bar = profitable trade (net_pnl >= 0).
    Red bar = losing trade (net_pnl < 0).
    Bar height = net profit/loss dollar amount ($).
    """
    trades = repository.get_trades(limit=limit)
    cols = ["trade_id", "date", "ticker", "net_pnl", "sign", "trade_label"]
    if not trades:
        return pd.DataFrame(columns=cols)

    closed_trades = [
        t for t in trades
        if t.get("action") == "SELL" or abs(float(t.get("net_pnl", 0.0))) > 1e-4
    ]

    if not closed_trades:
        return pd.DataFrame(columns=cols)

    sorted_trades = sorted(closed_trades, key=lambda t: (t.get("run_date", ""), t.get("id", 0)))

    rows = []
    for idx, t in enumerate(sorted_trades, start=1):
        pnl = round(float(t.get("net_pnl", 0.0)), 2)
        ticker = str(t.get("ticker", "")).upper()
        d_str = str(t.get("run_date", ""))
        sign = "Win" if pnl >= 0.0 else "Loss"
        label = f"#{idx} {ticker} ({d_str})"
        rows.append({
            "trade_id": idx,
            "date": d_str,
            "ticker": ticker,
            "net_pnl": pnl,
            "sign": sign,
            "trade_label": label,
        })

    return pd.DataFrame(rows)


def get_sector_allocation_donut_data(market: str = "US") -> pd.DataFrame:
    """
    Returns current portfolio split by economic sector, with Cash as its own slice.
    Data comes from the latest portfolio_snapshots and screener universe sector metadata.
    Columns:
      sector: str
      value: float ($)
      pct: float (%)
    """
    from src.portfolio.intelligence import get_ticker_sector

    snap = repository.get_latest_portfolio_snapshot(market=market)
    if snap is None:
        c0 = float(settings.initial_capital)
        return pd.DataFrame([{
            "sector": "Cash Reserve",
            "value": c0,
            "pct": 100.0,
        }])

    cash = float(snap.get("cash", settings.initial_capital))
    total_equity = float(snap.get("total_value", settings.initial_capital))
    positions = snap.get("positions", {}) or {}

    sector_mv: Dict[str, float] = {}
    for ticker, pos in positions.items():
        qty = float(pos.get("shares", pos.get("quantity", 0.0)))
        if qty > 0:
            price = float(pos.get("current_price", pos.get("price", pos.get("avg_cost", 0.0))))
            mv = float(pos.get("market_value", pos.get("value", qty * price)))
            sec = get_ticker_sector(ticker)
            sector_mv[sec] = sector_mv.get(sec, 0.0) + mv

    rows = []
    # 1. Cash slice
    cash_pct = round((cash / total_equity) * 100.0, 2) if total_equity > 0 else 100.0
    rows.append({
        "sector": "Cash Reserve",
        "value": round(cash, 2),
        "pct": cash_pct,
    })

    # 2. Sector slices
    for sec, mv in sorted(sector_mv.items(), key=lambda x: x[1], reverse=True):
        pct = round((mv / total_equity) * 100.0, 2) if total_equity > 0 else 0.0
        rows.append({
            "sector": sec,
            "value": round(mv, 2),
            "pct": pct,
        })

    return pd.DataFrame(rows)


# ── Section 6 Item 2: Trade History Table Loader ─────────────────────────────

def _normalize_exit_reason(reason: Optional[str], pnl: float) -> str:
    """Normalizes raw order reason into one of: Stop-Loss, Take-Profit, Signal, Manual."""
    if not reason:
        return "Take-Profit" if pnl >= 0 else "Stop-Loss"
    r_upper = str(reason).upper()
    if "STOP_LOSS" in r_upper or "STOP" in r_upper:
        return "Stop-Loss"
    if "TAKE_PROFIT" in r_upper or "PROFIT" in r_upper:
        return "Take-Profit"
    if "MANUAL" in r_upper:
        return "Manual"
    return "Signal"


def get_closed_trade_history(market: str = "US") -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Constructs closed trade history by pairing BUY and SELL records chronologically (FIFO).

    Returns:
        (closed_df, summary_metrics, export_df)
    """
    trades = repository.get_trades(limit=10000, market=market)
    orders = repository.get_orders(limit=10000, market=market)

    # Map (date, ticker, 'SELL') to order reason
    order_reason_map: Dict[Tuple[str, str, str], str] = {}
    for o in orders:
        key = (str(o.get("run_date", "")), str(o.get("ticker", "")).upper(), str(o.get("action", "")).upper())
        order_reason_map[key] = str(o.get("reason", ""))

    cols = [
        "Date Bought",
        "Date Sold",
        "Ticker",
        "Entry Price",
        "Exit Price",
        "Shares",
        "Profit / Loss ($)",
        "Profit / Loss (%)",
        "Exit Reason",
        "Model Score at Entry",
    ]

    defaults_summary = {
        "total_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "avg_gain": 0.0,
        "avg_loss": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
    }

    if not trades:
        empty_df = pd.DataFrame(columns=cols)
        return empty_df, defaults_summary, empty_df

    # Sort chronologically by date and id
    sorted_trades = sorted(trades, key=lambda t: (str(t.get("run_date", "")), int(t.get("id", 0))))

    # Group trades by ticker
    ticker_trades: Dict[str, List[Dict[str, Any]]] = {}
    for t in sorted_trades:
        tkr = str(t.get("ticker", "")).upper()
        if tkr not in ticker_trades:
            ticker_trades[tkr] = []
        ticker_trades[tkr].append(t)

    # Cache predictions per ticker
    preds_cache: Dict[str, pd.DataFrame] = {}

    closed_records: List[Dict[str, Any]] = []

    for tkr, t_list in ticker_trades.items():
        buy_lots: List[Dict[str, Any]] = []
        for tr in t_list:
            action = str(tr.get("action", "")).upper()
            qty = float(tr.get("quantity", 0.0))
            price = float(tr.get("fill_price", 0.0))
            d_str = str(tr.get("run_date", ""))

            if action == "BUY" and qty > 0:
                # Lookup model prediction probability at or before buy date
                if tkr not in preds_cache:
                    try:
                        preds_cache[tkr] = repository.get_predictions(tkr)
                    except Exception:
                        preds_cache[tkr] = pd.DataFrame()
                p_df = preds_cache[tkr]
                score: Optional[float] = None
                if not p_df.empty and "date" in p_df.columns and "probability" in p_df.columns:
                    priors = p_df[p_df["date"] <= d_str]
                    if not priors.empty:
                        score = round(float(priors["probability"].iloc[-1]), 2)
                    else:
                        score = round(float(p_df["probability"].iloc[0]), 2)

                buy_lots.append({
                    "date": d_str,
                    "shares": qty,
                    "price": price,
                    "cost": float(tr.get("cost", 0.0)),
                    "score": score,
                })

            elif action == "SELL" and qty > 0:
                sell_qty = qty
                sell_pnl = float(tr.get("net_pnl", 0.0))

                # Lookup reason from matching order
                raw_reason = order_reason_map.get((d_str, tkr, "SELL"))
                exit_reason = _normalize_exit_reason(raw_reason, sell_pnl)

                while sell_qty > 1e-6 and buy_lots:
                    lot = buy_lots[0]
                    matched = min(sell_qty, lot["shares"])
                    fraction = matched / qty if qty > 0 else 1.0

                    if abs(sell_pnl) > 1e-4:
                        pnl_d = round(sell_pnl * fraction, 2)
                    else:
                        pnl_d = round(matched * (price - lot["price"]), 2)

                    pnl_pct = round(((price - lot["price"]) / lot["price"]) * 100.0, 2) if lot["price"] > 0 else 0.0

                    closed_records.append({
                        "Date Bought": lot["date"],
                        "Date Sold": d_str,
                        "Ticker": tkr,
                        "Entry Price": round(lot["price"], 2),
                        "Exit Price": round(price, 2),
                        "Shares": round(matched, 4),
                        "Profit / Loss ($)": pnl_d,
                        "Profit / Loss (%)": pnl_pct,
                        "Exit Reason": exit_reason,
                        "Model Score at Entry": lot["score"],
                    })

                    lot["shares"] -= matched
                    sell_qty -= matched
                    if lot["shares"] <= 1e-6:
                        buy_lots.pop(0)

    if not closed_records:
        empty_df = pd.DataFrame(columns=cols)
        return empty_df, defaults_summary, empty_df

    # Sort by Date Sold descending (newest first)
    closed_records.sort(key=lambda r: (r["Date Sold"], r["Date Bought"]), reverse=True)
    closed_df = pd.DataFrame(closed_records)

    if "Profit / Loss ($)" in closed_df.columns and "Entry Price" in closed_df.columns:
        closed_df["Risk/Reward"] = closed_df.apply(
            lambda r: round(r["Profit / Loss ($)"] / (r["Entry Price"] * 0.08), 2)
            if r["Entry Price"] > 0 else 0.0, axis=1
        )

    # Compute Summary metrics
    total_trades = len(closed_records)
    total_pnl = round(sum(r["Profit / Loss ($)"] for r in closed_records), 2)
    winning_trades = [r for r in closed_records if r["Profit / Loss ($)"] > 0]
    losing_trades = [r for r in closed_records if r["Profit / Loss ($)"] < 0]
    win_rate = round((len(winning_trades) / total_trades) * 100.0, 1) if total_trades > 0 else 0.0
    avg_gain = round(sum(r["Profit / Loss ($)"] for r in winning_trades) / len(winning_trades), 2) if winning_trades else 0.0
    avg_loss = round(sum(r["Profit / Loss ($)"] for r in losing_trades) / len(losing_trades), 2) if losing_trades else 0.0
    best_trade = max(r["Profit / Loss ($)"] for r in closed_records) if closed_records else 0.0
    worst_trade = min(r["Profit / Loss ($)"] for r in closed_records) if closed_records else 0.0

    summary_metrics = {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }

    # Summary row formatted for CSV export and bottom table display
    summary_row = {
        "Date Bought": "SUMMARY",
        "Date Sold": f"{total_trades} trades completed",
        "Ticker": f"Win Rate: {win_rate:.1f}%",
        "Entry Price": f"Avg Gain: ${avg_gain:+.2f}",
        "Exit Price": f"Avg Loss: ${avg_loss:+.2f}",
        "Shares": "",
        "Profit / Loss ($)": total_pnl,
        "Profit / Loss (%)": "",
        "Exit Reason": f"Best: ${best_trade:+.2f} | Worst: ${worst_trade:+.2f}",
        "Model Score at Entry": "",
    }

    export_df = pd.concat([closed_df, pd.DataFrame([summary_row])], ignore_index=True)

    return closed_df, summary_metrics, export_df


# ── Section 6 Item 3: Live Price Ticker Loaders ──────────────────────────────

from dashboard.live_ticker import (
    fetch_single_ticker_live_price,
    get_live_quotes_for_held_stocks,
    is_us_market_open,
)


def get_held_positions_correlation_data(market: str = "US") -> pd.DataFrame:
    """
    Computes/fetches pairwise correlation matrix for currently held portfolio stocks.
    Returns a square DataFrame of correlations (index=tickers, columns=tickers).
    """
    snap = repository.get_latest_portfolio_snapshot(market=market)
    if not snap or "positions" not in snap or not snap["positions"]:
        return pd.DataFrame()

    positions = snap["positions"]
    held_tickers = [
        t.strip().upper() for t, pos in positions.items()
        if float(pos.get("quantity", 0.0)) > 0
    ]

    if len(held_tickers) < 2:
        return pd.DataFrame()

    from src.risk.correlation import compute_and_save_correlation_matrix
    today_str = date.today().strftime("%Y-%m-%d")
    return compute_and_save_correlation_matrix(date_str=today_str, tickers=held_tickers)


# ── Feature Importance Loaders (Section 9 Item 2) ─────────────────────────────

def get_feature_importance_data(model_type: str = "primary") -> list:
    """
    Load the latest feature importance scores from the DB for a given model_type.
    Falls back to JSON file if DB is empty.
    Returns list of {feature_name, importance_score, date, model_type}.
    """
    try:
        from src.db import repository
        rows = repository.get_feature_importances(model_type=model_type)
        if rows:
            return rows
        # Fallback to JSON sidecar
        from src.ml.importance import load_importances_json
        payload = load_importances_json(model_type=model_type)
        if payload and "importances" in payload:
            date_str = payload.get("date", "unknown")
            return [
                {"feature_name": k, "importance_score": v, "date": date_str, "model_type": model_type}
                for k, v in payload["importances"].items()
            ]
    except Exception:
        pass
    return []


def get_feature_importance_history_data(model_type: str = "primary", top_n: int = 5, last_n: int = 5) -> list:
    """
    Load feature importance time-series for the top-N features over the last N retrains.
    Returns list of {date, feature_name, importance_score, model_type}.
    """
    try:
        from src.db import repository
        return repository.get_feature_importance_history(
            model_type=model_type,
            top_n_features=top_n,
            last_n_retrains=last_n,
        )
    except Exception:
        return []


# ── Section 10 Item 1: Leaderboard Data Loaders ─────────────────────────────

def get_leaderboard_summary_data() -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load leaderboard rankings for all active strategy variants.
    Returns (rankings_list, winning_strategy_dict).
    """
    try:
        from src.trading.leaderboard import get_leaderboard_rankings
        rankings = get_leaderboard_rankings()
        winning = rankings[0] if rankings else None
        return rankings, winning
    except Exception:
        return [], None


def get_leaderboard_equity_curves() -> pd.DataFrame:
    """
    Load all strategy snapshots and format as a DataFrame for Altair multi-line plotting.
    DataFrame columns: ['date', 'Strategy', 'portfolio_value']
    """
    try:
        from src.db import repository
        variants_list = repository.get_strategy_variants()
        var_map = {v["id"]: v["name"] for v in variants_list}

        snaps = repository.get_strategy_snapshots()
        if not snaps:
            return pd.DataFrame()

        rows = []
        for s in snaps:
            sid = s["strategy_id"]
            name = var_map.get(sid, f"Strategy {sid}")
            rows.append({
                "date": s["date"],
                "Strategy": name,
                "portfolio_value": s["portfolio_value"],
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Section 10 Item 3: Tax Report Data Loader ─────────────────────────────

def run_tax_report_trigger(year: int = 2026, country: str = "India") -> Dict[str, Any]:
    """
    Executes annual tax report generation and returns report metadata and file paths.
    """
    from src.reports.tax_report import generate_tax_report
    return generate_tax_report(year=year, country=country)


# ── Section 10 Option C: Chatbot Data Loader ─────────────────────────────

def ask_chatbot_trigger(user_message: str, chat_history: Optional[List[Dict[str, str]]] = None, market: str = "US") -> str:
    """
    Triggers the Gemini Chatbot response for a user prompt in active market context.
    """
    from src.chatbot.gemini_chat import ask_gemini_chatbot
    return ask_gemini_chatbot(user_message=user_message, chat_history=chat_history, market=market)


# ── Performance Metrics & Walk Forward History Data Loaders ───────────────────

def get_performance_metrics_data(market: str = "US") -> Dict[str, Any]:
    """
    Computes performance metrics: Sharpe Ratio, Max Drawdown, Calmar Ratio,
    Sortino Ratio, Win Rate %, Profit Factor from snapshots and trades tables.
    """
    metrics = {
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": 0.0,
        "sortino_ratio": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
    }

    import numpy as np

    try:
        equity_df = get_equity_history_df(market=market)
        if not equity_df.empty and len(equity_df) >= 2:
            equity_df = equity_df.sort_values("date").reset_index(drop=True)
            values = equity_df["total_value"].astype(float)

            # Daily returns
            daily_returns = values.pct_change().dropna()

            # 1. Sharpe Ratio
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                metrics["sharpe_ratio"] = round(float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)), 2)

            # 2. Max Drawdown
            cummax = values.cummax()
            drawdowns = (cummax - values) / cummax
            max_dd = float(drawdowns.max()) if not drawdowns.empty else 0.0
            metrics["max_drawdown_pct"] = round(max_dd * 100.0, 2)

            # Annualized Return for Calmar
            start_val = values.iloc[0]
            end_val = values.iloc[-1]
            try:
                d0 = pd.to_datetime(equity_df["date"].iloc[0])
                d1 = pd.to_datetime(equity_df["date"].iloc[-1])
                days = (d1 - d0).days
            except Exception:
                days = 0

            if days > 0 and start_val > 0:
                cagr = (end_val / start_val) ** (365.25 / max(days, 1)) - 1.0
            elif start_val > 0:
                cagr = (end_val - start_val) / start_val
            else:
                cagr = 0.0

            # 3. Calmar Ratio
            if max_dd > 0:
                metrics["calmar_ratio"] = round(float(cagr / max_dd), 2)
            else:
                metrics["calmar_ratio"] = round(float(cagr * 100) if cagr > 0 else 0.0, 2)

            # 4. Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    metrics["sortino_ratio"] = round(float((daily_returns.mean() / downside_std) * np.sqrt(252)), 2)
            elif len(downside_returns) == 1:
                downside_std = abs(float(downside_returns.iloc[0]))
                if downside_std > 0:
                    metrics["sortino_ratio"] = round(float((daily_returns.mean() / downside_std) * np.sqrt(252)), 2)
            elif len(daily_returns) > 0 and daily_returns.mean() > 0:
                metrics["sortino_ratio"] = round(metrics["sharpe_ratio"], 2)

    except Exception as exc:
        logger.warning("Error calculating performance metrics: %s", exc)

    # Trades metrics: Win Rate % and Profit Factor
    try:
        closed_df, summary_metrics, _ = get_closed_trade_history(market=market)
        metrics["win_rate_pct"] = round(float(summary_metrics.get("win_rate", 0.0)), 1)

        if not closed_df.empty and "Profit / Loss ($)" in closed_df.columns:
            pnls = closed_df["Profit / Loss ($)"].astype(float)
            gross_profit = float(pnls[pnls > 0].sum())
            gross_loss = abs(float(pnls[pnls < 0].sum()))
            if gross_loss > 0:
                metrics["profit_factor"] = round(gross_profit / gross_loss, 2)
            elif gross_profit > 0:
                metrics["profit_factor"] = round(gross_profit, 2)
            else:
                metrics["profit_factor"] = 0.0
    except Exception as exc:
        logger.warning("Error calculating trade metrics: %s", exc)

    return metrics


def get_walk_forward_history_data(model_type: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches walk-forward validation history from the database.
    """
    try:
        return repository.get_walk_forward_history(model_type=model_type)
    except Exception as exc:
        logger.warning("Failed to load walk-forward history: %s", exc)
        return pd.DataFrame(columns=[
            "id", "trained_at", "model_type", "fold_number",
            "train_start", "train_end", "test_start", "test_end",
            "accuracy", "roc_auc", "brier_score", "n_samples",
        ])


def get_drawdown_series(market: str = "US") -> pd.DataFrame:
    try:
        equity_df = get_equity_history_df(market=market)
        if equity_df.empty or len(equity_df) < 2:
            return pd.DataFrame()
        equity_df = equity_df.sort_values("date").reset_index(drop=True)
        values = equity_df["total_value"].astype(float)
        cummax = values.cummax()
        drawdown = ((values - cummax) / cummax) * 100
        return pd.DataFrame({"date": equity_df["date"], "drawdown_pct": drawdown})
    except Exception:
        return pd.DataFrame()


def get_ohlcv_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    Returns OHLCV data for a specific ticker from the database.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., "AAPL")
    days : int
        Number of trading days to retrieve (default: 30)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume.
        Returns empty DataFrame if no data found.
    """
    try:
        df = repository.get_market_data(ticker)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        required_cols = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required_cols):
            return pd.DataFrame()

        df = df[required_cols].sort_values("date").tail(days).reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("get_ohlcv_data failed for %s: %s", ticker, exc)
        return pd.DataFrame()
