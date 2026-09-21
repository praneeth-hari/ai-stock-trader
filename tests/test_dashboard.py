"""
tests/test_dashboard.py — Phase 12 Streamlit Dashboard Unit and Integration Tests.

Verifies:
  1. Default portfolio summary loads from clean DB matching settings.initial_capital ($50.00).
  2. Portfolio summary with positions accurately reflects mark-to-market equity and cash reserve %.
  3. Historical equity DataFrame correctly parses snapshots into chronologically ordered time-series.
  4. Trades and orders loaders return formatted DataFrames matching DB records.
  5. System events loader retrieves and formats event logs.
  6. Market regime and predictions loader returns regime status dictionary and rankings DataFrame.
  7. Dashboard app module compiles and imports without syntax errors.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.settings import settings
from src.db import repository


@pytest.fixture()
def clean_db(tmp_path):
    """Provides a fresh isolated SQLite database for testing."""
    db_file = tmp_path / "test_dash.db"
    db_url = f"sqlite:///{db_file}"
    repository._engine = None
    orig_url = repository.settings.db_url

    try:
        repository.settings.__dict__["db_url"] = db_url
        repository.create_all_tables()
        yield db_url
    finally:
        repository._engine = None
        repository.settings.__dict__["db_url"] = orig_url


def test_1_default_portfolio_summary_clean_db(clean_db):
    """Clean DB should default to settings.initial_capital ($50.00) and zero positions."""
    from dashboard.data_loader import get_portfolio_summary

    summary = get_portfolio_summary()
    assert summary["cash"] == pytest.approx(settings.initial_capital, abs=1e-4)
    assert summary["total_equity"] == pytest.approx(settings.initial_capital, abs=1e-4)
    assert summary["invested_value"] == 0.0
    assert summary["cash_reserve_pct"] == 100.0
    assert summary["open_positions_count"] == 0
    assert summary["positions"] == []


def test_2_portfolio_summary_with_positions(clean_db):
    """Snapshot with active positions should correctly compute MTM and reserve %."""
    from dashboard.data_loader import get_portfolio_summary

    # Save a snapshot with 1 position: 0.0785 AAPL @ $180 entry, now $190
    positions = {
        "AAPL": {
            "shares": 0.0785,
            "entry_price": 180.0,
            "current_price": 190.0,
            "market_value": 14.915,
            "unrealized_pnl": 0.785,
            "unrealized_pnl_pct": 5.56,
        }
    }
    cash = 35.8417
    total_val = 50.7567

    repository.save_portfolio_snapshot(
        run_date="2023-11-21",
        cash=cash,
        total_value=total_val,
        positions=positions,
    )

    summary = get_portfolio_summary()
    assert summary["run_date"] == "2023-11-21"
    assert summary["cash"] == pytest.approx(35.8417, abs=1e-4)
    assert summary["total_equity"] == pytest.approx(50.7567, abs=1e-4)
    assert summary["invested_value"] == pytest.approx(50.7567 - 35.8417, abs=1e-4)
    assert summary["open_positions_count"] == 1
    assert summary["cash_reserve_pct"] == pytest.approx(35.8417 / 50.7567 * 100.0, abs=1e-2)

    pos = summary["positions"][0]
    assert pos["ticker"] == "AAPL"
    assert pos["shares"] == pytest.approx(0.0785, abs=1e-4)
    assert pos["entry_price"] == 180.0
    assert pos["current_price"] == 190.0
    assert pos["unrealized_pnl"] == pytest.approx(0.785, abs=1e-3)


def test_3_equity_history_df(clean_db):
    """Verifies historical snapshots are parsed chronologically."""
    from dashboard.data_loader import get_equity_history_df

    repository.save_portfolio_snapshot("2023-11-20", 35.84, 49.97)
    repository.save_portfolio_snapshot("2023-11-21", 35.84, 50.75)
    repository.save_portfolio_snapshot("2023-11-22", 47.83, 47.83)

    df = get_equity_history_df()
    assert len(df) == 3
    assert list(df.columns) == ["date", "cash", "total_value", "invested"]
    assert df["total_value"].iloc[0] == pytest.approx(49.97, abs=1e-2)
    assert df["total_value"].iloc[1] == pytest.approx(50.75, abs=1e-2)
    assert df["total_value"].iloc[2] == pytest.approx(47.83, abs=1e-2)


def test_4_trades_and_orders_loaders(clean_db):
    """Verifies trade and order query loaders format DataFrames accurately."""
    from dashboard.data_loader import get_orders_df, get_recent_trades_df

    repository.save_order("2023-11-20", "AAPL", "BUY", 0.0785, 180.0, "QUALIFIED_BUY")
    repository.save_trade("2023-11-20", "AAPL", "BUY", 0.0785, 180.0, 0.0283, 0.0)

    orders = get_orders_df(limit=10)
    assert len(orders) == 1
    assert orders["ticker"].iloc[0] == "AAPL"
    assert orders["action"].iloc[0] == "BUY"
    assert orders["shares"].iloc[0] == pytest.approx(0.0785, abs=1e-4)

    trades = get_recent_trades_df(limit=10)
    assert len(trades) == 1
    assert trades["ticker"].iloc[0] == "AAPL"
    assert trades["action"].iloc[0] == "BUY"
    assert trades["fee"].iloc[0] == pytest.approx(0.0283, abs=1e-4)


def test_5_system_events_loader(clean_db):
    """Verifies system event log loading and formatting."""
    from dashboard.data_loader import get_system_events_df

    repository.log_event("INFO", "daily_pipeline", "Pipeline started for 2023-11-20")
    repository.log_event("WARNING", "risk_engine", "Cash reserve low")

    events = get_system_events_df(limit=10)
    assert len(events) == 2
    assert "daily_pipeline" in events["component"].values
    assert "risk_engine" in events["component"].values


def test_6_regime_and_predictions_loader(clean_db):
    """Verifies regime & predictions loader produces valid data structures.

    Section 5 added sector, sentiment, and earnings columns to preds_df.
    Test checks that required base columns are present (superset check).
    """
    from dashboard.data_loader import get_market_regime_and_predictions

    regime, preds = get_market_regime_and_predictions()
    assert "status" in regime
    assert "is_risk_on" in regime
    assert isinstance(preds, pd.DataFrame)
    # Core columns always required
    required_cols = {"ticker", "probability", "qualification", "action_signal"}
    assert required_cols.issubset(set(preds.columns)), (
        f"Missing required columns: {required_cols - set(preds.columns)}"
    )
    # Section 5 intelligence columns are also expected (may be present)
    # We accept them as additive — no regression


def test_7_dashboard_app_compiles():
    """Verifies that dashboard/app.py can be compiled and parsed without syntax errors."""
    app_path = "d:/ai-stock-trader/dashboard/app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compiled = compile(code, app_path, "exec")
    assert compiled is not None


def test_8_dashboard_app_full_render_execution(clean_db):
    """
    Verifies that dashboard/app.py executes its full rendering loop via Streamlit's AppTest.
    This catches missing imports, NameError, and unhandled runtime exceptions across all tabs.
    """
    from streamlit.testing.v1 import AppTest

    fake_spy = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=250, freq="B").strftime("%Y-%m-%d"),
        "open": [400.0] * 250,
        "high": [405.0] * 250,
        "low": [395.0] * 250,
        "close": [402.0] * 250,
        "volume": [50000000.0] * 250,
        "ticker": ["SPY"] * 250,
    })

    with patch("src.data.market_data.fetch_ticker_data", return_value=fake_spy):
        at = AppTest.from_file("dashboard/app.py", default_timeout=15)
        at.run()
        assert len(at.exception) == 0, f"Dashboard execution raised exceptions: {at.exception}"


def test_9_triggers_execution_and_schema(clean_db):
    """Verifies run_daily_paper_cycle_trigger and run_backtest_trigger execute and return correct schema."""
    from dashboard.data_loader import run_daily_paper_cycle_trigger, run_backtest_trigger
    from src.pipeline.daily_pipeline import DailyPipelineResult
    from src.backtest.backtest import BacktestResult, BacktestMetrics

    # Test run_daily_paper_cycle_trigger with mock
    mock_pipeline_res = DailyPipelineResult(
        run_date="2024-01-02",
        tickers_fetched=1,
        tickers_valid=1,
        tickers_skipped=0,
        regime="RISK-ON",
        orders_generated=1,
        fills=[],
        pending_orders=[],
        cash=10000.0,
        total_equity=10000.0,
        errors=[],
    )
    with patch("src.pipeline.daily_pipeline.run_daily_pipeline", return_value=mock_pipeline_res):
        cycle_res = run_daily_paper_cycle_trigger(run_date="2024-01-02")
        assert cycle_res["run_date"] == "2024-01-02"
        assert cycle_res["fills_count"] == 0
        assert cycle_res["orders_count"] == 1
        assert cycle_res["equity"] == 10000.0
        assert "Daily Pipeline Audit" in cycle_res["audit_markdown"]

    # Test run_backtest_trigger with mock
    mock_metrics = BacktestMetrics(
        start_date="2024-01-02",
        end_date="2024-06-28",
        trading_days=120,
        starting_capital=10000.0,
        final_equity=10500.0,
        total_net_return_pct=5.0,
        annualized_return_pct=10.0,
        spy_total_return_pct=4.0,
        spy_annualized_return_pct=8.0,
        alpha_pct=1.0,
        max_drawdown_pct=-3.0,
        spy_max_drawdown_pct=-5.0,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        total_trades=5,
        winning_trades=3,
        losing_trades=2,
        win_rate_pct=60.0,
        profit_factor=1.5,
        avg_trade_pnl_pct=1.0,
        avg_holding_days=5.0,
        avg_cash_pct=40.0,
        alarm_triggered=False,
    )
    mock_bt_res = BacktestResult(metrics=mock_metrics, daily_snapshots=[], trades=[])
    with patch("src.backtest.backtest.run_strategy_backtest", return_value=mock_bt_res), \
         patch("src.data.market_data.fetch_ticker_data", return_value=pd.DataFrame({"date": ["2024-01-02"], "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0], "volume": [1000.0], "ticker": ["SPY"]})), \
         patch("src.ml.evaluate.load_active_model"):
        bt_out = run_backtest_trigger(start_date="2024-01-02", end_date="2024-06-28", tickers=["SPY"])
        assert bt_out["cagr"] == 10.0
        assert bt_out["benchmark_cagr"] == 8.0
        assert bt_out["total_trades"] == 5
        assert bt_out["win_rate"] == 60.0
        assert "equity_curve" in bt_out


def test_10_v2_model_drift_loader(clean_db):
    """Verifies get_model_drift_summary returns expected fields and handles low data cleanly."""
    from dashboard.data_loader import get_model_drift_summary

    drift_summary = get_model_drift_summary()
    assert "status" in drift_summary
    assert "psi" in drift_summary
    assert "ks_pvalue" in drift_summary
    assert "recommendation" in drift_summary
    assert "disclaimer" in drift_summary
    assert drift_summary["status"] in ("STABLE", "MONITOR", "DRIFT_ALERT", "INSUFFICIENT_DATA", "ERROR")


def test_11_v2_portfolio_intelligence_loaders(clean_db):
    """Verifies sector analysis and diversification score loaders return consistent structures."""
    from dashboard.data_loader import (
        get_portfolio_diversification_summary,
        get_portfolio_sectors_summary,
    )

    sec_summary = get_portfolio_sectors_summary()
    assert "total_equity" in sec_summary
    assert "cash" in sec_summary
    assert "cash_reserve_pct" in sec_summary
    assert "holdings_sectors" in sec_summary
    assert "trading_universe_sectors" in sec_summary
    assert isinstance(sec_summary["holdings_sectors"], list)

    div_summary = get_portfolio_diversification_summary()
    assert "score" in div_summary
    assert "rating" in div_summary
    assert "sector_hhi" in div_summary
    assert "effective_sectors" in div_summary
    assert "pillar_breakdown" in div_summary
    assert "formula_explanation" in div_summary
    assert 0.0 <= div_summary["score"] <= 100.0


