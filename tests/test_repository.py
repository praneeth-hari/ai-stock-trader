"""
tests/test_repository.py — Phase 2b round-trip verification for all tables.

Every test uses an isolated in-memory SQLite database (never the production
trader.db). The in-memory DB is created fresh per test class via a pytest
fixture that patches settings.db_url and resets the engine singleton.

Round-trip structure for every table:
  1. create_all_tables() — schema creation
  2. save_*(...)         — write
  3. get_*(...)          — read back
  4. assert             — data integrity confirmed
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# ── Fixture: isolated in-memory DB per test ────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """
    For each test, patch settings.db_url to a fresh SQLite file in tmp_path
    and reset the repository's engine singleton so nothing leaks between tests.
    This guarantees no cross-test contamination and never touches trader.db.
    """
    db_path = tmp_path / "test_trader.db"
    db_url = f"sqlite:///{db_path}"

    # We patch both settings.db_url AND reset the module-level _engine sentinel.
    import src.db.repository as repo_module
    from config.settings import settings

    original_url = settings.db_url
    settings.db_url = db_url
    repo_module._engine = None          # reset singleton so get_engine() rebuilds

    yield db_url

    # Teardown: restore original url and reset engine again
    settings.db_url = original_url
    repo_module._engine = None


# ── Import after fixture so patching takes effect ─────────────────────────────

def _repo():
    import src.db.repository as r
    return r


# ── Schema creation ────────────────────────────────────────────────────────────

class TestSchemaCreation:
    def test_create_all_tables_is_idempotent(self):
        """Calling create_all_tables() twice must not raise."""
        repo = _repo()
        repo.create_all_tables()
        repo.create_all_tables()   # second call must be a no-op, not an error


# ── Market data ────────────────────────────────────────────────────────────────

class TestMarketData:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_single_ticker(self):
        """Write 3 OHLCV rows for AAPL, read them back, confirm schema and values."""
        repo = _repo()
        df_in = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAPL", "open": 185.0, "high": 187.0, "low": 184.0, "close": 186.5, "volume": 50_000_000},
            {"date": "2024-01-03", "ticker": "AAPL", "open": 186.5, "high": 188.0, "low": 185.5, "close": 187.2, "volume": 45_000_000},
            {"date": "2024-01-04", "ticker": "AAPL", "open": 187.2, "high": 189.0, "low": 186.0, "close": 188.0, "volume": 48_000_000},
        ])
        inserted = repo.save_market_data(df_in)
        assert inserted == 3

        df_out = repo.get_market_data("AAPL")
        assert len(df_out) == 3
        assert list(df_out.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
        assert df_out["close"].tolist() == [186.5, 187.2, 188.0]
        assert (df_out["ticker"] == "AAPL").all()

    def test_date_range_filter(self):
        """get_market_data with start/end filters returns only the matching rows."""
        repo = _repo()
        df_in = pd.DataFrame([
            {"date": f"2024-01-0{i}", "ticker": "SPY", "open": 470.0, "high": 472.0, "low": 469.0, "close": 471.0, "volume": 80_000_000}
            for i in range(2, 9)
        ])
        repo.save_market_data(df_in)
        df_out = repo.get_market_data("SPY", start_date="2024-01-04", end_date="2024-01-06")
        assert len(df_out) == 3
        assert df_out["date"].tolist() == ["2024-01-04", "2024-01-05", "2024-01-06"]

    def test_duplicate_rows_skipped(self):
        """Saving the same (date, ticker) twice must insert only once."""
        repo = _repo()
        df = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "MSFT", "open": 370.0, "high": 372.0, "low": 369.0, "close": 371.0, "volume": 20_000_000},
        ])
        first = repo.save_market_data(df)
        second = repo.save_market_data(df)
        assert first == 1
        assert second == 0    # skipped — already exists
        assert len(repo.get_market_data("MSFT")) == 1

    def test_empty_dataframe_returns_zero(self):
        """Saving an empty DataFrame must return 0 and raise no errors."""
        repo = _repo()
        inserted = repo.save_market_data(pd.DataFrame())
        assert inserted == 0


# ── Features ──────────────────────────────────────────────────────────────────

class TestFeatures:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_feature_blob(self):
        """Write a feature vector, read it back, confirm blob is preserved exactly."""
        repo = _repo()
        df_in = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAPL", "rsi_14": 62.5, "macd_signal": 0.35, "vol_ratio": 1.12},
            {"date": "2024-01-03", "ticker": "AAPL", "rsi_14": 58.1, "macd_signal": 0.20, "vol_ratio": 0.95},
        ])
        upserted = repo.save_features(df_in)
        assert upserted == 2

        df_out = repo.get_features("AAPL")
        assert len(df_out) == 2
        assert "rsi_14" in df_out.columns
        assert "macd_signal" in df_out.columns
        assert abs(df_out.loc[0, "rsi_14"] - 62.5) < 1e-6
        assert abs(df_out.loc[1, "macd_signal"] - 0.20) < 1e-6

    def test_feature_upsert_updates_existing(self):
        """Saving the same (date, ticker) twice should overwrite, not duplicate."""
        repo = _repo()
        df1 = pd.DataFrame([{"date": "2024-01-02", "ticker": "MSFT", "rsi_14": 50.0}])
        df2 = pd.DataFrame([{"date": "2024-01-02", "ticker": "MSFT", "rsi_14": 65.0}])
        repo.save_features(df1)
        repo.save_features(df2)
        df_out = repo.get_features("MSFT")
        assert len(df_out) == 1
        assert abs(df_out.loc[0, "rsi_14"] - 65.0) < 1e-6


# ── Predictions ────────────────────────────────────────────────────────────────

class TestPredictions:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_probability(self):
        """Write probabilities, read back and confirm values are preserved."""
        repo = _repo()
        df_in = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "NVDA", "probability": 0.73, "model_version": "v1"},
            {"date": "2024-01-03", "ticker": "NVDA", "probability": 0.61, "model_version": "v1"},
        ])
        upserted = repo.save_predictions(df_in)
        assert upserted == 2

        df_out = repo.get_predictions("NVDA")
        assert len(df_out) == 2
        assert abs(df_out.loc[0, "probability"] - 0.73) < 1e-6
        assert df_out["model_version"].tolist() == ["v1", "v1"]

    def test_default_model_version_is_v1(self):
        """Saving without model_version should default to 'v1'."""
        repo = _repo()
        df_in = pd.DataFrame([{"date": "2024-01-02", "ticker": "TSLA", "probability": 0.55}])
        repo.save_predictions(df_in)
        df_out = repo.get_predictions("TSLA")
        assert df_out.loc[0, "model_version"] == "v1"


# ── Portfolio ─────────────────────────────────────────────────────────────────

class TestPortfolio:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_snapshot(self):
        """Write a snapshot, read it back by run_date, confirm all fields."""
        repo = _repo()
        positions = {"AAPL": {"quantity": 10, "avg_cost": 185.0, "current_price": 188.0, "value": 1880.0}}
        repo.save_portfolio_snapshot(
            run_date="2024-01-02",
            cash=8000.0,
            total_value=9880.0,
            positions=positions,
        )
        snap = repo.get_portfolio_snapshot("2024-01-02")
        assert snap is not None
        assert snap["cash"] == 8000.0
        assert snap["total_value"] == 9880.0
        assert snap["positions"]["AAPL"]["quantity"] == 10

    def test_get_latest_returns_most_recent(self):
        """get_latest_portfolio_snapshot must return the most recent run_date."""
        repo = _repo()
        repo.save_portfolio_snapshot("2024-01-02", cash=9000.0, total_value=9000.0)
        repo.save_portfolio_snapshot("2024-01-03", cash=8800.0, total_value=9200.0)
        repo.save_portfolio_snapshot("2024-01-04", cash=8500.0, total_value=9500.0)
        latest = repo.get_latest_portfolio_snapshot()
        assert latest is not None
        assert latest["run_date"] == "2024-01-04"
        assert latest["cash"] == 8500.0

    def test_snapshot_upsert_updates_existing(self):
        """Saving the same run_date twice should overwrite, not duplicate."""
        repo = _repo()
        repo.save_portfolio_snapshot("2024-01-02", cash=9000.0, total_value=9000.0)
        repo.save_portfolio_snapshot("2024-01-02", cash=8000.0, total_value=9500.0)
        snap = repo.get_portfolio_snapshot("2024-01-02")
        assert snap["cash"] == 8000.0     # updated, not duplicated

    def test_missing_run_date_returns_none(self):
        """get_portfolio_snapshot for a non-existent date returns None."""
        assert _repo().get_portfolio_snapshot("1999-01-01") is None


# ── Orders ────────────────────────────────────────────────────────────────────

class TestOrders:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_order(self):
        """Write an order, read it back, confirm all fields preserved."""
        repo = _repo()
        row_id = repo.save_order(
            run_date="2024-01-02",
            ticker="AAPL",
            action="BUY",
            quantity=5.0,
            price=186.5,
            reason="Signal: p=0.73 > threshold 0.60 | regime: bullish",
        )
        assert row_id == 1

        orders = repo.get_orders("2024-01-02")
        assert len(orders) == 1
        o = orders[0]
        assert o["ticker"] == "AAPL"
        assert o["action"] == "BUY"
        assert abs(o["quantity"] - 5.0) < 1e-6
        assert abs(o["price"] - 186.5) < 1e-6
        assert "p=0.73" in o["reason"]

    def test_multiple_orders_same_run_date(self):
        """Multiple orders on the same run date are all preserved."""
        repo = _repo()
        repo.save_order("2024-01-02", "AAPL", "BUY", 5.0, 186.5, "signal")
        repo.save_order("2024-01-02", "MSFT", "SELL", 3.0, 371.0, "stop-loss 8%")
        repo.save_order("2024-01-02", "NVDA", "HOLD", 0.0, 590.0, "regime: bearish")
        orders = repo.get_orders("2024-01-02")
        assert len(orders) == 3
        tickers = [o["ticker"] for o in orders]
        assert set(tickers) == {"AAPL", "MSFT", "NVDA"}


# ── Trades ────────────────────────────────────────────────────────────────────

class TestTrades:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_buy_trade(self):
        """
        Write a BUY fill. net_pnl=0.0 for buys (position is open, no realized P&L).
        cost = §1.6 fee = 0.2% of fill_price * quantity.
        """
        repo = _repo()
        quantity = 5.0
        fill_price = 186.5
        cost = round(fill_price * quantity * 0.002, 6)    # §1.6: 0.2%

        row_id = repo.save_trade(
            run_date="2024-01-02",
            ticker="AAPL",
            action="BUY",
            quantity=quantity,
            fill_price=fill_price,
            cost=cost,
            net_pnl=0.0,
        )
        assert row_id == 1

        trades = repo.get_trades("2024-01-02")
        assert len(trades) == 1
        t = trades[0]
        assert t["action"] == "BUY"
        assert abs(t["fill_price"] - 186.5) < 1e-6
        assert abs(t["cost"] - cost) < 1e-6
        assert t["net_pnl"] == 0.0

    def test_round_trip_sell_trade_with_pnl(self):
        """Write a SELL fill with a realized net P&L after costs."""
        repo = _repo()
        quantity = 5.0
        fill_price = 200.0
        avg_cost_price = 185.0
        gross_pnl = (fill_price - avg_cost_price) * quantity   # 75.0
        cost = round(fill_price * quantity * 0.002, 6)         # sell-side cost only
        net_pnl = round(gross_pnl - cost, 6)

        repo.save_trade(
            run_date="2024-01-10",
            ticker="AAPL",
            action="SELL",
            quantity=quantity,
            fill_price=fill_price,
            cost=cost,
            net_pnl=net_pnl,
        )
        trades = repo.get_trades("2024-01-10")
        assert len(trades) == 1
        t = trades[0]
        assert t["action"] == "SELL"
        assert t["net_pnl"] > 0     # profitable trade
        assert abs(t["net_pnl"] - net_pnl) < 1e-6


# ── Event log ─────────────────────────────────────────────────────────────────

class TestEventLog:
    def setup_method(self):
        _repo().create_all_tables()

    def test_round_trip_log_event(self):
        """Write a structured log event and read it back with all fields intact."""
        repo = _repo()
        repo.log_event(
            level="WARNING",
            component="validation",
            message="AAPL: 2 unexplained missing trading days",
            details={"ticker": "AAPL", "missing_dates": ["2024-01-04", "2024-01-05"]},
        )
        events = repo.get_events()
        assert len(events) == 1
        e = events[0]
        assert e["level"] == "WARNING"
        assert e["component"] == "validation"
        assert "AAPL" in e["message"]
        assert e["details"]["ticker"] == "AAPL"
        assert len(e["details"]["missing_dates"]) == 2

    def test_level_filter(self):
        """get_events(level='ERROR') should return only ERROR-level entries."""
        repo = _repo()
        repo.log_event("INFO", "pipeline", "Daily run started", None)
        repo.log_event("ERROR", "validation", "BOGUS: negative price", {"ticker": "BOGUS"})
        repo.log_event("WARNING", "risk_engine", "Cash below 15% threshold", None)
        repo.log_event("ERROR", "paper_broker", "Fill rejected: insufficient cash", None)

        errors = repo.get_events(level="ERROR")
        assert len(errors) == 2
        assert all(e["level"] == "ERROR" for e in errors)

    def test_component_filter(self):
        """get_events(component='validation') should return only validation entries."""
        repo = _repo()
        repo.log_event("INFO", "validation", "SPY passed all checks", None)
        repo.log_event("ERROR", "risk_engine", "Position limit exceeded", None)
        repo.log_event("WARNING", "validation", "MSFT: 1 gap day", {"ticker": "MSFT"})

        val_events = repo.get_events(component="validation")
        assert len(val_events) == 2
        assert all(e["component"] == "validation" for e in val_events)

    def test_multiple_events_ordered_newest_first(self):
        """Events returned by get_events() are ordered newest first."""
        repo = _repo()
        for i in range(5):
            repo.log_event("INFO", "test", f"Event {i}", {"seq": i})
        events = repo.get_events(limit=5)
        seqs = [e["details"]["seq"] for e in events]
        assert seqs == sorted(seqs, reverse=True)   # descending

    def test_null_details_allowed(self):
        """log_event with details=None must not raise."""
        repo = _repo()
        repo.log_event("INFO", "pipeline", "No details needed", None)
        events = repo.get_events()
        assert events[0]["details"] is None


class TestWalkForwardResults:
    def test_save_and_get_walk_forward_results(self):
        """Verify saving fold metrics and reading back history."""
        repo = _repo()
        repo.create_all_tables()

        records = [
            {
                "model_type": "primary",
                "fold_number": 0,
                "train_start": "2021-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2023-06-30",
                "accuracy": 0.65,
                "roc_auc": 0.72,
                "brier_score": 0.20,
                "n_samples": 126,
            },
            {
                "model_type": "primary",
                "fold_number": 1,
                "train_start": "2021-07-01",
                "train_end": "2023-06-30",
                "test_start": "2023-07-01",
                "test_end": "2023-12-31",
                "accuracy": 0.68,
                "roc_auc": 0.75,
                "brier_score": 0.18,
                "n_samples": 126,
            },
        ]

        count = repo.save_walk_forward_results(records)
        assert count == 2

        history_df = repo.get_walk_forward_history("primary")
        assert not history_df.empty
        assert len(history_df) == 2
        assert list(history_df["fold_number"]) == [0, 1]
        assert float(history_df["roc_auc"].iloc[1]) == 0.75


def test_kill_switch_state_persists():
    from src.db import repository
    repository.create_all_tables()
    repository.save_kill_switch_state(True, "Test reason")
    enabled, reason = repository.load_kill_switch_state()
    assert enabled is True
    assert reason == "Test reason"
    # cleanup
    repository.save_kill_switch_state(False, "")


def test_kill_switch_default_off():
    from src.db import repository
    enabled, reason = repository.load_kill_switch_state()
    assert isinstance(enabled, bool)
    assert isinstance(reason, str)

