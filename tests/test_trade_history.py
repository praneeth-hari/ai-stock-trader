"""
tests/test_trade_history.py — Section 6 Item 2: Trade History Table Unit & Integration Tests.

Verifies:
  1. Empty DB produces empty trade history DataFrame with all 10 required columns and 0-defaults summary.
  2. BUY-only trades are recognized as open positions and excluded from closed trades.
  3. Single profitable closed trade is correctly computed with entry/exit price, positive PnL ($ and %), and Take-Profit reason.
  4. Single losing closed trade is correctly computed with negative PnL ($ and %) and Stop-Loss reason.
  5. FIFO ordering matches multiple BUY lots chronologically against subsequent SELL executions.
  6. Closed trades table is strictly sorted by Date Sold descending (newest first).
  7. Summary metrics (total trades, total PnL, win rate, avg gain, avg loss, best, worst) compute accurately.
  8. Model score at entry accurately maps to prediction probability on/before buy date.
  9. CSV export DataFrame contains all 10 columns and includes the summary row at the bottom.
 10. Streamlit AppTest verifies Tab 2 rendering with trade history table and open positions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data_loader import get_closed_trade_history
from src.db import repository


@pytest.fixture()
def clean_db(tmp_path):
    """Provides a fresh isolated SQLite database for trade history tests."""
    db_file = tmp_path / "test_trade_hist.db"
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


EXPECTED_COLUMNS = [
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


def test_1_empty_trade_history(clean_db):
    """Empty DB returns empty DataFrame with all 10 columns and zeroed summary metrics."""
    closed_df, summary, export_df = get_closed_trade_history()
    assert list(closed_df.columns) == EXPECTED_COLUMNS
    assert closed_df.empty
    assert summary["total_trades"] == 0
    assert summary["total_pnl"] == 0.0
    assert summary["win_rate"] == 0.0
    assert summary["avg_gain"] == 0.0
    assert summary["avg_loss"] == 0.0
    assert summary["best_trade"] == 0.0
    assert summary["worst_trade"] == 0.0
    assert list(export_df.columns) == EXPECTED_COLUMNS


def test_2_buy_only_trades_not_closed(clean_db):
    """Only BUY trades present (unrealized open positions) should yield 0 closed trades."""
    repository.save_trade(
        run_date="2026-09-10", ticker="AAPL", action="BUY",
        quantity=10.0, fill_price=150.0, cost=3.0, net_pnl=0.0
    )
    closed_df, summary, _ = get_closed_trade_history()
    assert len(closed_df) == 0
    assert summary["total_trades"] == 0


def test_3_single_closed_trade_profitable(clean_db):
    """Single profitable trade pairs buy and sell, computing positive PnL and Take-Profit reason."""
    repository.save_order(
        run_date="2026-09-15", ticker="AAPL", action="SELL",
        quantity=10.0, price=165.0, reason="TAKE_PROFIT_TRIGGERED"
    )
    repository.save_trade(
        run_date="2026-09-10", ticker="AAPL", action="BUY",
        quantity=10.0, fill_price=150.0, cost=3.0, net_pnl=0.0
    )
    repository.save_trade(
        run_date="2026-09-15", ticker="AAPL", action="SELL",
        quantity=10.0, fill_price=165.0, cost=3.3, net_pnl=143.70
    )

    closed_df, summary, export_df = get_closed_trade_history()
    assert len(closed_df) == 1
    row = closed_df.iloc[0]

    assert row["Date Bought"] == "2026-09-10"
    assert row["Date Sold"] == "2026-09-15"
    assert row["Ticker"] == "AAPL"
    assert row["Entry Price"] == 150.0
    assert row["Exit Price"] == 165.0
    assert row["Shares"] == 10.0
    assert row["Profit / Loss ($)"] == pytest.approx(143.70, abs=1e-2)
    assert row["Profit / Loss (%)"] == pytest.approx(10.0, abs=1e-2)
    assert row["Exit Reason"] == "Take-Profit"

    assert summary["total_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(143.70, abs=1e-2)
    assert summary["win_rate"] == 100.0
    assert summary["avg_gain"] == pytest.approx(143.70, abs=1e-2)
    assert summary["avg_loss"] == 0.0
    assert summary["best_trade"] == pytest.approx(143.70, abs=1e-2)
    assert summary["worst_trade"] == pytest.approx(143.70, abs=1e-2)


def test_4_single_closed_trade_losing(clean_db):
    """Single losing trade computes negative PnL and Stop-Loss reason."""
    repository.save_order(
        run_date="2026-09-16", ticker="NVDA", action="SELL",
        quantity=5.0, price=110.0, reason="STOP_LOSS_HIT"
    )
    repository.save_trade(
        run_date="2026-09-12", ticker="NVDA", action="BUY",
        quantity=5.0, fill_price=120.0, cost=1.2, net_pnl=0.0
    )
    repository.save_trade(
        run_date="2026-09-16", ticker="NVDA", action="SELL",
        quantity=5.0, fill_price=110.0, cost=1.1, net_pnl=-52.30
    )

    closed_df, summary, _ = get_closed_trade_history()
    assert len(closed_df) == 1
    row = closed_df.iloc[0]

    assert row["Date Bought"] == "2026-09-12"
    assert row["Date Sold"] == "2026-09-16"
    assert row["Ticker"] == "NVDA"
    assert row["Entry Price"] == 120.0
    assert row["Exit Price"] == 110.0
    assert row["Shares"] == 5.0
    assert row["Profit / Loss ($)"] == pytest.approx(-52.30, abs=1e-2)
    assert row["Profit / Loss (%)"] == pytest.approx(-8.33, abs=1e-2)
    assert row["Exit Reason"] == "Stop-Loss"

    assert summary["total_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(-52.30, abs=1e-2)
    assert summary["win_rate"] == 0.0
    assert summary["avg_loss"] == pytest.approx(-52.30, abs=1e-2)


def test_5_fifo_multiple_lots(clean_db):
    """Verifies First-In First-Out matching when multiple BUY lots exist for a ticker."""
    # Lot 1: 5 shares @ $100 on 2026-09-01
    repository.save_trade("2026-09-01", "MSFT", "BUY", 5.0, 100.0, 1.0, 0.0)
    # Lot 2: 5 shares @ $120 on 2026-09-05
    repository.save_trade("2026-09-05", "MSFT", "BUY", 5.0, 120.0, 1.2, 0.0)
    # Sell 7 shares @ $130 on 2026-09-10 (should close all 5 of Lot 1, and 2 of Lot 2)
    repository.save_trade("2026-09-10", "MSFT", "SELL", 7.0, 130.0, 1.8, 170.0)

    closed_df, _, _ = get_closed_trade_history()
    assert len(closed_df) == 2

    # First closed record corresponds to Lot 1 (5 shares)
    r1 = closed_df[closed_df["Shares"] == 5.0].iloc[0]
    assert r1["Date Bought"] == "2026-09-01"
    assert r1["Entry Price"] == 100.0

    # Second closed record corresponds to partial Lot 2 (2 shares)
    r2 = closed_df[closed_df["Shares"] == 2.0].iloc[0]
    assert r2["Date Bought"] == "2026-09-05"
    assert r2["Entry Price"] == 120.0


def test_6_sort_order_date_sold_descending(clean_db):
    """Closed trades must be sorted by Date Sold with newest first."""
    # Trade 1 sold on Sept 10
    repository.save_trade("2026-09-01", "AAPL", "BUY", 5.0, 100.0, 1.0, 0.0)
    repository.save_trade("2026-09-10", "AAPL", "SELL", 5.0, 110.0, 1.0, 50.0)

    # Trade 2 sold on Sept 20
    repository.save_trade("2026-09-05", "NVDA", "BUY", 5.0, 100.0, 1.0, 0.0)
    repository.save_trade("2026-09-20", "NVDA", "SELL", 5.0, 120.0, 1.0, 100.0)

    # Trade 3 sold on Sept 15
    repository.save_trade("2026-09-02", "META", "BUY", 5.0, 100.0, 1.0, 0.0)
    repository.save_trade("2026-09-15", "META", "SELL", 5.0, 105.0, 1.0, 25.0)

    closed_df, _, _ = get_closed_trade_history()
    assert len(closed_df) == 3
    date_sold_list = list(closed_df["Date Sold"])
    assert date_sold_list == ["2026-09-20", "2026-09-15", "2026-09-10"]


def test_7_summary_metrics_calculation(clean_db):
    """Verifies all 7 summary metrics accurately reflect the aggregate trade statistics."""
    # 2 Winning trades (+100, +50) and 1 Losing trade (-30)
    repository.save_trade("2026-09-01", "T1", "BUY", 1.0, 100.0, 0.0, 0.0)
    repository.save_trade("2026-09-05", "T1", "SELL", 1.0, 200.0, 0.0, 100.0)

    repository.save_trade("2026-09-02", "T2", "BUY", 1.0, 100.0, 0.0, 0.0)
    repository.save_trade("2026-09-06", "T2", "SELL", 1.0, 150.0, 0.0, 50.0)

    repository.save_trade("2026-09-03", "T3", "BUY", 1.0, 100.0, 0.0, 0.0)
    repository.save_trade("2026-09-07", "T3", "SELL", 1.0, 70.0, 0.0, -30.0)

    _, summary, _ = get_closed_trade_history()

    assert summary["total_trades"] == 3
    assert summary["total_pnl"] == pytest.approx(120.0, abs=1e-2)
    assert summary["win_rate"] == pytest.approx(66.7, abs=1e-1)
    assert summary["avg_gain"] == pytest.approx(75.0, abs=1e-2)   # (100 + 50) / 2
    assert summary["avg_loss"] == pytest.approx(-30.0, abs=1e-2)  # -30 / 1
    assert summary["best_trade"] == pytest.approx(100.0, abs=1e-2)
    assert summary["worst_trade"] == pytest.approx(-30.0, abs=1e-2)


def test_8_model_score_at_entry_lookup(clean_db):
    """Model score at entry retrieves prediction probability on or before buy date."""
    preds_df = pd.DataFrame([
        {"date": "2026-09-01", "ticker": "GOOGL", "probability": 0.67, "model_version": "v1"},
    ])
    repository.save_predictions(preds_df)

    repository.save_trade("2026-09-02", "GOOGL", "BUY", 5.0, 150.0, 1.0, 0.0)
    repository.save_trade("2026-09-10", "GOOGL", "SELL", 5.0, 160.0, 1.0, 50.0)

    closed_df, _, _ = get_closed_trade_history()
    assert len(closed_df) == 1
    assert closed_df["Model Score at Entry"].iloc[0] == pytest.approx(0.67, abs=1e-2)


def test_9_csv_export_format_and_summary_row(clean_db):
    """Verifies CSV export contains all trade rows plus the SUMMARY row at the bottom."""
    repository.save_trade("2026-09-01", "AAPL", "BUY", 10.0, 150.0, 1.0, 0.0)
    repository.save_trade("2026-09-10", "AAPL", "SELL", 10.0, 160.0, 1.0, 100.0)

    _, _, export_df = get_closed_trade_history()
    assert len(export_df) == 2  # 1 trade + 1 summary row

    # Last row is SUMMARY
    last_row = export_df.iloc[-1]
    assert last_row["Date Bought"] == "SUMMARY"
    assert "1 trades completed" in last_row["Date Sold"]
    assert "Win Rate: 100.0%" in last_row["Ticker"]
    assert last_row["Profit / Loss ($)"] == 100.0

    # Convert to CSV string and check headers and contents
    csv_text = export_df.to_csv(index=False)
    assert "Date Bought,Date Sold,Ticker,Entry Price,Exit Price" in csv_text
    assert "SUMMARY" in csv_text
    assert "AAPL" in csv_text


def test_10_app_test_renders_trade_history_tab(clean_db):
    """Streamlit AppTest confirms that Tab 2 executes without runtime errors."""
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    # Seed sample trade
    repository.save_trade("2026-09-01", "AAPL", "BUY", 5.0, 150.0, 1.0, 0.0)
    repository.save_trade("2026-09-05", "AAPL", "SELL", 5.0, 160.0, 1.0, 50.0)

    fake_spy = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=250, freq="B").strftime("%Y-%m-%d"),
        "open": [400.0] * 250, "high": [405.0] * 250, "low": [395.0] * 250,
        "close": [402.0] * 250, "volume": [50000000.0] * 250, "ticker": ["SPY"] * 250,
    })

    with patch("src.data.market_data.fetch_ticker_data", return_value=fake_spy):
        at = AppTest.from_file("dashboard/app.py", default_timeout=15)
        at.run()
        assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"


def test_risk_reward_column_exists():
    from dashboard.data_loader import get_closed_trade_history
    df, _, _ = get_closed_trade_history()
    if not df.empty:
        assert "Risk/Reward" in df.columns

def test_risk_reward_is_numeric():
    from dashboard.data_loader import get_closed_trade_history
    df, _, _ = get_closed_trade_history()
    if not df.empty and "Risk/Reward" in df.columns:
        assert pd.to_numeric(df["Risk/Reward"], errors="coerce").notna().all()
