"""
tests/test_performance_charts.py — Section 6 Item 1: Visual Performance Charts Unit Tests.

Verifies:
  1. Portfolio vs SPY data loader handles empty DB with initial capital defaults.
  2. Portfolio vs SPY data loader accurately scales SPY benchmark from market data and flags underperformance.
  3. Portfolio vs SPY chart builder compiles into an Altair LayerChart with red underperformance shading.
  4. Daily returns histogram data loader handles empty/single snapshot edge cases gracefully.
  5. Daily returns histogram data loader properly bins positive (green) and negative (red) return days.
  6. Daily returns histogram builder constructs bars with vertical zero rule.
  7. Win/loss trades data loader returns empty DataFrame when no closed trades exist.
  8. Win/loss trades data loader extracts SELL trades with accurate PnL amounts and Win/Loss signs.
  9. Win/loss trade chart builder compiles into layered Altair chart with baseline rule.
 10. Sector allocation donut data loader handles 100% cash state.
 11. Sector allocation donut data loader deconstructs active positions into sectors + cash slice totaling 100%.
 12. Sector allocation donut chart builder produces valid Altair arc chart with inner radius.
 13. End-to-end rendering of the Performance tab in Streamlit AppTest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import settings
from dashboard.charts import (
    build_daily_returns_histogram,
    build_portfolio_vs_spy_chart,
    build_sector_allocation_donut,
    build_win_loss_trade_chart,
)
from dashboard.data_loader import (
    get_daily_returns_histogram_data,
    get_portfolio_vs_spy_chart_data,
    get_sector_allocation_donut_data,
    get_win_loss_trades_chart_data,
)
from src.db import repository


@pytest.fixture()
def clean_db(tmp_path):
    """Provides a fresh isolated SQLite database for performance chart tests."""
    db_file = tmp_path / "test_perf_charts.db"
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


# ── 1. Portfolio vs SPY Benchmark Tests ───────────────────────────────────────

def test_1_portfolio_vs_spy_empty_db(clean_db):
    """Empty DB returns default 1-day point with initial capital and no deficit."""
    df = get_portfolio_vs_spy_chart_data()
    assert not df.empty
    assert len(df) == 1
    assert "My Portfolio" in df.columns
    assert "SPY Benchmark" in df.columns
    assert "underperforming" in df.columns
    assert "deficit" in df.columns
    assert df["My Portfolio"].iloc[0] == pytest.approx(settings.initial_capital, abs=1e-2)
    assert df["SPY Benchmark"].iloc[0] == pytest.approx(settings.initial_capital, abs=1e-2)
    assert not df["underperforming"].iloc[0]
    assert df["deficit"].iloc[0] == 0.0


def test_2_portfolio_vs_spy_underperformance_flag(clean_db):
    """When SPY rises and portfolio stays flat or declines, underperforming is True and deficit is positive."""
    # Seed SPY data
    spy_data = pd.DataFrame([
        {"date": "2026-09-10", "ticker": "SPY", "open": 500.0, "high": 505.0, "low": 499.0, "close": 500.0, "volume": 1e6},
        {"date": "2026-09-11", "ticker": "SPY", "open": 500.0, "high": 560.0, "low": 500.0, "close": 550.0, "volume": 1e6},  # +10%
    ])
    repository.save_market_data(spy_data)

    # Seed Portfolio snapshots
    repository.save_portfolio_snapshot(run_date="2026-09-10", cash=10000.0, total_value=10000.0)
    repository.save_portfolio_snapshot(run_date="2026-09-11", cash=9800.0, total_value=9800.0)

    df = get_portfolio_vs_spy_chart_data()
    assert len(df) == 2

    # Day 1: Equal
    assert df["My Portfolio"].iloc[0] == 10000.0
    assert df["SPY Benchmark"].iloc[0] == 10000.0
    assert not df["underperforming"].iloc[0]
    assert df["deficit"].iloc[0] == 0.0

    # Day 2: Portfolio $9,800 vs SPY $11,000 (+10%)
    assert df["My Portfolio"].iloc[1] == 9800.0
    assert df["SPY Benchmark"].iloc[1] == pytest.approx(11000.0, abs=1.0)
    assert bool(df["underperforming"].iloc[1]) is True
    assert df["deficit"].iloc[1] == pytest.approx(1200.0, abs=1.0)


def test_3_build_portfolio_vs_spy_chart_layers():
    """Verifies build_portfolio_vs_spy_chart generates a valid Altair layer chart with shading."""
    df = pd.DataFrame([
        {"date": "2026-09-10", "My Portfolio": 10000.0, "SPY Benchmark": 10000.0, "underperforming": False, "deficit": 0.0},
        {"date": "2026-09-11", "My Portfolio": 9500.0, "SPY Benchmark": 10500.0, "underperforming": True, "deficit": 1000.0},
    ])
    chart = build_portfolio_vs_spy_chart(df)
    assert chart is not None
    json_spec = chart.to_json()
    assert "Portfolio vs SPY Benchmark" in json_spec
    # Verify both lines and red underperformance color exist in spec
    assert "#2962ff" in json_spec  # My Portfolio blue
    assert "#ff9800" in json_spec  # SPY Benchmark amber
    assert "#ef5350" in json_spec  # Red underperformance area/points

    # Empty test
    assert build_portfolio_vs_spy_chart(pd.DataFrame()) is None


# ── 2. Daily Returns Histogram Tests ──────────────────────────────────────────

def test_4_daily_returns_histogram_empty_or_single_snapshot(clean_db):
    """Empty DB or single snapshot returns empty DataFrame with proper column schema."""
    df_empty = get_daily_returns_histogram_data()
    assert df_empty.empty
    assert list(df_empty.columns) == ["bin_center", "bin_label", "count", "sign"]

    repository.save_portfolio_snapshot("2026-09-10", 10000.0, 10000.0)
    df_single = get_daily_returns_histogram_data()
    assert df_single.empty


def test_5_daily_returns_histogram_positive_and_negative_bins(clean_db):
    """Multiple snapshots generate positive and negative return bins with matching signs."""
    # Sequence of 4 snapshots -> 3 returns: +2.0%, -1.0%, +3.0%
    repository.save_portfolio_snapshot("2026-09-10", 10000.0, 10000.0)
    repository.save_portfolio_snapshot("2026-09-11", 10200.0, 10200.0)  # +2.0%
    repository.save_portfolio_snapshot("2026-09-12", 10098.0, 10098.0)  # -1.0%
    repository.save_portfolio_snapshot("2026-09-13", 10400.94, 10400.94)  # +3.0%

    df = get_daily_returns_histogram_data()
    assert not df.empty
    assert sum(df["count"]) == 3

    # Negative bins should strictly have sign == 'Negative'
    neg_rows = df[df["sign"] == "Negative"]
    assert len(neg_rows) >= 1
    assert all(r["bin_center"] < 0 for _, r in neg_rows.iterrows())

    # Positive bins should strictly have sign == 'Positive'
    pos_rows = df[df["sign"] == "Positive"]
    assert len(pos_rows) >= 1
    assert all(r["bin_center"] >= 0 for _, r in pos_rows.iterrows())


def test_6_build_daily_returns_histogram_layers():
    """Verifies build_daily_returns_histogram creates Altair chart with zero rule and color encoding."""
    df = pd.DataFrame([
        {"bin_center": -0.8, "bin_label": "-1.0% to -0.6%", "count": 2, "sign": "Negative"},
        {"bin_center": 0.5, "bin_label": "+0.3% to +0.7%", "count": 5, "sign": "Positive"},
    ])
    chart = build_daily_returns_histogram(df)
    assert chart is not None
    json_spec = chart.to_json()
    assert "Distribution of Daily Returns" in json_spec
    assert "#26a69a" in json_spec  # Green
    assert "#ef5350" in json_spec  # Red

    assert build_daily_returns_histogram(pd.DataFrame()) is None


# ── 3. Win/Loss Trade Bar Chart Tests ─────────────────────────────────────────

def test_7_win_loss_trades_empty_db(clean_db):
    """Empty DB or only BUY trades returns empty DataFrame with trade schema."""
    df = get_win_loss_trades_chart_data()
    assert df.empty
    assert list(df.columns) == ["trade_id", "date", "ticker", "net_pnl", "sign", "trade_label"]

    # Only BUY trades (open positions, not closed)
    repository.save_trade(
        run_date="2026-09-10", ticker="AAPL", action="BUY",
        quantity=5.0, fill_price=150.0, cost=1.5, net_pnl=0.0
    )
    df_buys = get_win_loss_trades_chart_data()
    assert df_buys.empty


def test_8_win_loss_trades_with_closed_trades(clean_db):
    """SELL trades are extracted with correct net PnL dollar amounts and Win/Loss signs."""
    repository.save_trade(
        run_date="2026-09-10", ticker="AAPL", action="BUY",
        quantity=10.0, fill_price=150.0, cost=3.0, net_pnl=0.0
    )
    # Win trade
    repository.save_trade(
        run_date="2026-09-15", ticker="AAPL", action="SELL",
        quantity=10.0, fill_price=165.0, cost=3.3, net_pnl=143.70
    )
    # Loss trade
    repository.save_trade(
        run_date="2026-09-16", ticker="NVDA", action="SELL",
        quantity=5.0, fill_price=120.0, cost=1.2, net_pnl=-45.20
    )

    df = get_win_loss_trades_chart_data()
    assert len(df) == 2

    trade_1 = df.iloc[0]
    assert trade_1["ticker"] == "AAPL"
    assert trade_1["net_pnl"] == pytest.approx(143.70, abs=1e-2)
    assert trade_1["sign"] == "Win"

    trade_2 = df.iloc[1]
    assert trade_2["ticker"] == "NVDA"
    assert trade_2["net_pnl"] == pytest.approx(-45.20, abs=1e-2)
    assert trade_2["sign"] == "Loss"


def test_9_build_win_loss_trade_chart():
    """Verifies build_win_loss_trade_chart compiles with green/red bars and zero rule."""
    df = pd.DataFrame([
        {"trade_id": 1, "date": "2026-09-15", "ticker": "AAPL", "net_pnl": 120.0, "sign": "Win", "trade_label": "#1 AAPL (2026-09-15)"},
        {"trade_id": 2, "date": "2026-09-16", "ticker": "TSLA", "net_pnl": -80.0, "sign": "Loss", "trade_label": "#2 TSLA (2026-09-16)"},
    ])
    chart = build_win_loss_trade_chart(df)
    assert chart is not None
    json_spec = chart.to_json()
    assert "Realized PnL per Closed Trade" in json_spec
    assert "#26a69a" in json_spec  # Green
    assert "#ef5350" in json_spec  # Red

    assert build_win_loss_trade_chart(pd.DataFrame()) is None


# ── 4. Sector Allocation Donut Chart Tests ────────────────────────────────────

def test_10_sector_allocation_donut_empty_db(clean_db):
    """Empty DB defaults to 100% Cash Reserve slice."""
    df = get_sector_allocation_donut_data()
    assert len(df) == 1
    assert df["sector"].iloc[0] == "Cash Reserve"
    assert df["pct"].iloc[0] == 100.0
    assert df["value"].iloc[0] == pytest.approx(settings.initial_capital, abs=1e-2)


def test_11_sector_allocation_donut_with_positions(clean_db):
    """Portfolio with positions partitions market value across sectors plus cash slice."""
    # Seed snapshot with AAPL (Technology) and META (Communications)
    positions = {
        "AAPL": {"shares": 10.0, "current_price": 200.0, "market_value": 2000.0},
        "META": {"shares": 5.0, "current_price": 600.0, "market_value": 3000.0},
    }
    repository.save_portfolio_snapshot(
        run_date="2026-09-19",
        cash=5000.0,
        total_value=10000.0,
        positions=positions,
    )

    df = get_sector_allocation_donut_data()
    assert len(df) >= 3  # Cash + Technology + Communications

    sec_map = dict(zip(df["sector"], df["value"]))
    pct_map = dict(zip(df["sector"], df["pct"]))

    assert "Cash Reserve" in sec_map
    assert sec_map["Cash Reserve"] == pytest.approx(5000.0, abs=1e-2)
    assert pct_map["Cash Reserve"] == pytest.approx(50.0, abs=1e-1)

    assert "Technology" in sec_map
    assert sec_map["Technology"] == pytest.approx(2000.0, abs=1e-2)
    assert pct_map["Technology"] == pytest.approx(20.0, abs=1e-1)

    # Total weights should sum to 100%
    assert sum(df["pct"]) == pytest.approx(100.0, abs=0.5)


def test_12_build_sector_allocation_donut():
    """Verifies build_sector_allocation_donut creates an Altair arc chart with innerRadius."""
    df = pd.DataFrame([
        {"sector": "Cash Reserve", "value": 5000.0, "pct": 50.0},
        {"sector": "Technology", "value": 3000.0, "pct": 30.0},
        {"sector": "Communications", "value": 2000.0, "pct": 20.0},
    ])
    chart = build_sector_allocation_donut(df)
    assert chart is not None
    json_spec = chart.to_json()
    assert "Portfolio Sector & Cash Allocation" in json_spec
    assert '"innerRadius": 65' in json_spec or '"innerRadius":65' in json_spec

    assert build_sector_allocation_donut(pd.DataFrame()) is None


# ── 5. End-to-End Streamlit App Test ──────────────────────────────────────────

def test_13_performance_tab_renders_in_app(clean_db):
    """Verifies that the Performance tab and its charts render through Streamlit AppTest without errors."""
    from unittest.mock import patch
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
        assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"
        # Confirm tab header exists
        tab_titles = [t.label for t in at.tabs]
        assert any("Performance" in t for t in tab_titles), f"Performance tab not found in tabs: {tab_titles}"


def test_14_performance_metrics_calculation(clean_db):
    """Verifies calculation of Sharpe Ratio, Max Drawdown, Calmar Ratio, Sortino Ratio, Win Rate %, and Profit Factor."""
    from dashboard.data_loader import get_performance_metrics_data

    # Empty DB default check
    m_empty = get_performance_metrics_data()
    assert m_empty["sharpe_ratio"] == 0.0
    assert m_empty["max_drawdown_pct"] == 0.0
    assert m_empty["calmar_ratio"] == 0.0
    assert m_empty["sortino_ratio"] == 0.0
    assert m_empty["win_rate_pct"] == 0.0
    assert m_empty["profit_factor"] == 0.0

    # Seed equity snapshots with a growth trajectory
    dates = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"]
    vals = [10000.0, 10200.0, 10100.0, 10400.0, 10500.0]
    for d, v in zip(dates, vals):
        repository.save_portfolio_snapshot(run_date=d, cash=v * 0.5, total_value=v)

    # Seed trades
    repository.save_trade("2026-09-01", "AAPL", "BUY", 10.0, 150.0, 1500.0, net_pnl=0.0)
    repository.save_trade("2026-09-03", "AAPL", "SELL", 10.0, 170.0, 1700.0, net_pnl=200.0)
    repository.save_trade("2026-09-02", "MSFT", "BUY", 5.0, 300.0, 1500.0, net_pnl=0.0)
    repository.save_trade("2026-09-04", "MSFT", "SELL", 5.0, 290.0, 1450.0, net_pnl=-50.0)

    m = get_performance_metrics_data()
    assert m["sharpe_ratio"] > 0
    assert m["max_drawdown_pct"] > 0
    assert m["sortino_ratio"] > 0
    assert m["win_rate_pct"] == 50.0  # 1 win / 1 loss
    assert m["profit_factor"] == pytest.approx(4.0, abs=0.1)  # 200 / 50


def test_drawdown_series_returns_dataframe():
    from dashboard.data_loader import get_drawdown_series
    result = get_drawdown_series()
    assert isinstance(result, pd.DataFrame)


# ── Candlestick Chart Tests ──────────────────────────────────────────────────────

def test_15_build_candlestick_chart_structure(clean_db):
    """Verifies build_candlestick_chart creates a valid Plotly Figure with candlestick and volume traces."""
    from dashboard.charts import build_candlestick_chart

    ohlcv = pd.DataFrame({
        "date": pd.date_range("2026-09-01", periods=10, freq="B"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "volume": [1_000_000] * 10,
    })

    fig = build_candlestick_chart(ohlcv, "AAPL")
    assert fig is not None
    assert hasattr(fig, "data")
    # Should have at least candlestick + volume = 2 traces
    assert len(fig.data) >= 2

    # Check trace types
    trace_types = [trace.type for trace in fig.data]
    assert "candlestick" in trace_types
    assert "bar" in trace_types

    # Check layout properties
    assert "AAPL" in fig.layout.title.text
    assert fig.layout.height == 500
    assert fig.layout.xaxis.rangeslider.visible is False


def test_16_get_ohlcv_data_returns_correct_shape(clean_db):
    """Verifies get_ohlcv_data returns correctly shaped DataFrame from DB."""
    from dashboard.data_loader import get_ohlcv_data
    from src.db import repository

    # Seed market data
    mkt_data = pd.DataFrame([
        {"date": "2026-09-01", "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000_000},
        {"date": "2026-09-02", "ticker": "AAPL", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1_200_000},
        {"date": "2026-09-03", "ticker": "AAPL", "open": 101.5, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1_100_000},
    ])
    repository.save_market_data(mkt_data)

    df = get_ohlcv_data("AAPL", days=30)
    assert not df.empty
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    # Should be sorted by date
    assert df["date"].iloc[0] < df["date"].iloc[-1]

    # Test with days limit
    df_limited = get_ohlcv_data("AAPL", days=2)
    assert len(df_limited) == 2

    # Non-existent ticker returns empty
    df_empty = get_ohlcv_data("NONEXISTENT", days=30)
    assert df_empty.empty
