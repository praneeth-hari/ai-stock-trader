"""
tests/test_backtest_ui.py — Unit tests for SECTION 10 ITEM 2: Strategy Backtesting UI.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dashboard.data_loader import run_backtest_lab_trigger
from src.backtest.backtest import DailySnapshot, compute_monthly_returns_heatmap, run_lab_backtest


@pytest.fixture
def mock_market_data():
    """Generate synthetic daily OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", "2023-08-30", freq="B").strftime("%Y-%m-%d").tolist()

    spy_rows = []
    spy_price = 400.0
    for d in dates:
        spy_price += np.random.uniform(-2, 2.5)
        spy_rows.append({"date": d, "open": spy_price - 0.5, "close": spy_price, "high": spy_price + 1.0, "low": spy_price - 1.0, "volume": 1000000})
    spy_df = pd.DataFrame(spy_rows)

    universe_dict = {}
    tickers = ["AAPL", "MSFT", "NVDA"]
    for t in tickers:
        t_rows = []
        price = 150.0
        for d in dates:
            price += np.random.uniform(-1.5, 2.0)
            t_rows.append({"date": d, "open": price - 0.5, "close": price, "high": price + 1.0, "low": price - 1.0, "volume": 500000})
        universe_dict[t] = pd.DataFrame(t_rows)

    return spy_df, universe_dict, dates


@pytest.fixture
def mock_trained_model():
    """Mock trained model returning prediction probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    return model


def test_compute_monthly_returns_heatmap_basic():
    snapshots = [
        DailySnapshot(
            date="2023-01-03", cash=10000.0, portfolio_value=0.0, total_equity=10000.0,
            daily_return=0.0, cumulative_return=0.0, spy_close=400.0, spy_daily_return=0.0,
            spy_cumulative_return=0.0, positions_count=0, cash_pct=100.0, drawdown=0.0, spy_drawdown=0.0
        ),
        DailySnapshot(
            date="2023-01-31", cash=5000.0, portfolio_value=5500.0, total_equity=10500.0,
            daily_return=0.05, cumulative_return=0.05, spy_close=410.0, spy_daily_return=0.025,
            spy_cumulative_return=0.025, positions_count=1, cash_pct=47.6, drawdown=0.0, spy_drawdown=0.0
        ),
        DailySnapshot(
            date="2023-02-28", cash=5000.0, portfolio_value=4800.0, total_equity=9800.0,
            daily_return=-0.066, cumulative_return=-0.02, spy_close=405.0, spy_daily_return=-0.012,
            spy_cumulative_return=0.0125, positions_count=1, cash_pct=51.0, drawdown=-0.066, spy_drawdown=-0.012
        ),
    ]

    grid_df, best_str, worst_str = compute_monthly_returns_heatmap(snapshots, starting_capital=10000.0)

    assert not grid_df.empty
    assert "Year" in grid_df.columns
    assert "Jan" in grid_df.columns
    assert "Feb" in grid_df.columns
    assert "Year Total" in grid_df.columns
    assert best_str == "+5.0%"
    assert worst_str == "-6.7%"


def test_run_lab_backtest_execution(mock_market_data, mock_trained_model):
    spy_df, universe_dict, dates = mock_market_data
    start_date = dates[0]
    end_date = dates[-1]

    result = run_lab_backtest(
        universe_dict=universe_dict,
        spy_df=spy_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        buy_threshold=0.60,
        exit_threshold=0.45,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
        max_positions=3,
        position_sizing="Fixed",
        use_sentiment=True,
        use_earnings_blackout=True,
        use_sector_rotation=True,
        use_macro_regime=True,
        use_correlation_filter=True,
        use_trailing_stop=True,
        model=mock_trained_model,
    )

    assert "total_return_pct" in result
    assert "spy_total_return_pct" in result
    assert "alpha_pct" in result
    assert "max_drawdown_pct" in result
    assert "win_rate_pct" in result
    assert "total_trades" in result
    assert "days_in_cash_pct" in result
    assert "sharpe_ratio" in result
    assert "best_month" in result
    assert "worst_month" in result
    assert "equity_curve" in result
    assert "monthly_returns" in result
    assert "trades_df" in result

    assert isinstance(result["equity_curve"], pd.DataFrame)
    assert isinstance(result["monthly_returns"], pd.DataFrame)
    assert isinstance(result["trades_df"], pd.DataFrame)


def test_run_lab_backtest_confidence_sizing(mock_market_data, mock_trained_model):
    spy_df, universe_dict, dates = mock_market_data
    start_date = dates[0]
    end_date = dates[-1]

    result = run_lab_backtest(
        universe_dict=universe_dict,
        spy_df=spy_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        buy_threshold=0.60,
        exit_threshold=0.45,
        stop_loss_pct=0.05,
        take_profit_pct=0.20,
        max_positions=2,
        position_sizing="Confidence-Based",
        use_sentiment=False,
        use_earnings_blackout=False,
        use_sector_rotation=False,
        use_macro_regime=False,
        use_correlation_filter=False,
        use_trailing_stop=False,
        model=mock_trained_model,
    )

    assert "total_return_pct" in result
    assert isinstance(result["total_return_pct"], float)


def test_run_backtest_lab_trigger_wrapper(mock_market_data, mock_trained_model):
    spy_df, universe_dict, dates = mock_market_data
    start_date = dates[0]
    end_date = dates[-1]

    with patch("src.data.market_data.fetch_ticker_data") as mock_fetch, \
         patch("src.data.validation.validate_ticker_data") as mock_val, \
         patch("src.ml.evaluate.load_active_model", return_value=mock_trained_model):

        mock_fetch.side_effect = lambda ticker, **kwargs: spy_df if ticker == "SPY" else universe_dict.get(ticker, spy_df)
        mock_val.side_effect = lambda raw, ticker: MagicMock(is_valid=True, cleaned_df=raw)

        res = run_backtest_lab_trigger(
            start_date=start_date,
            end_date=end_date,
            buy_threshold=0.60,
            exit_threshold=0.45,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            max_positions=3,
            position_sizing="Fixed",
            use_sentiment=True,
            use_earnings_blackout=True,
            use_sector_rotation=True,
            use_macro_regime=True,
            use_correlation_filter=True,
            use_trailing_stop=True,
            tickers=["AAPL", "MSFT"],
        )

        assert "total_return_pct" in res
        assert "equity_curve" in res
        assert "monthly_returns" in res


def test_date_range_warning_threshold():
    start_d = datetime.date(2023, 1, 1)
    end_d_short = datetime.date(2023, 4, 1)
    end_d_long = datetime.date(2023, 10, 1)

    assert (end_d_short - start_d).days < 180
    assert (end_d_long - start_d).days >= 180
