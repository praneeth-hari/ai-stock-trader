"""
tests/test_trailing_stop.py - Unit tests for Section 8 Item 3: Trailing Stop Loss.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from config.settings import settings
from src.portfolio.portfolio import OrderSpec
from src.ranking.ranking import RankedOpportunity, RankingResult
from src.risk.risk_engine import ExitReason, HeldPosition, evaluate_portfolio_risk
from src.trading.paper_broker import PaperBroker, Position


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    test_db = f"sqlite:///{tmp_path}/test_trailing.db"
    monkeypatch.setattr(settings, "db_url", test_db)
    from src.db import repository
    repository.create_all_tables()


def _make_spy_df(is_risk_on: bool = True) -> pd.DataFrame:
    dates = pd.date_range(end="2024-06-14", periods=250, freq="D")
    base_price = 450.0 if is_risk_on else 350.0
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": [base_price] * 250,
        "high": [base_price + 2.0] * 250,
        "low": [base_price - 2.0] * 250,
        "close": [base_price + (10.0 if is_risk_on else -50.0)] * 250,
        "volume": [50_000_000] * 250,
        "ticker": ["SPY"] * 250,
    })


def _ranking_hold_only(ticker: str = "AAPL", prob: float = 0.55) -> RankingResult:
    opp = RankedOpportunity(
        ticker=ticker, date="2024-06-14", probability=prob, rank=1,
        conviction_tier="NEUTRAL", is_buy_eligible=False, is_exit_signal=False,
    )
    return RankingResult(
        date="2024-06-14", regime_risk_on=True, total_evaluated=1,
        ranked_opportunities=[opp], top_buy_candidates=[],
    )


class TestPositionInit:

    def test_trailing_stop_set_on_init_default_8pct(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        assert pos.highest_price_since_entry == pytest.approx(100.0)
        assert pos.trailing_stop_price == pytest.approx(100.0 * 0.92, abs=0.001)

    def test_highest_price_initialised_to_entry_price(self):
        pos = Position(ticker="MSFT", quantity=5.0, entry_price=200.0,
                       entry_date="2024-06-14", entry_fee=0.40)
        assert pos.highest_price_since_entry == pytest.approx(200.0)

    def test_explicit_highest_and_trail_preserved(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(
            ticker="GOOG", quantity=1.0, entry_price=100.0,
            entry_date="2024-06-14", entry_fee=0.20,
            highest_price_since_entry=130.0, trailing_stop_price=119.60,
        )
        assert pos.highest_price_since_entry == pytest.approx(130.0)
        assert pos.trailing_stop_price == pytest.approx(119.60)

    def test_trailing_stop_configurable_via_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.10)
        pos = Position(ticker="NVDA", quantity=2.0, entry_price=500.0,
                       entry_date="2024-06-14", entry_fee=1.0)
        assert pos.trailing_stop_price == pytest.approx(500.0 * 0.90, abs=0.001)


class TestPositionUpdateHigh:

    def test_new_high_moves_trail_up(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        moved = pos.update_high_and_trailing_stop(110.0)
        assert moved is True
        assert pos.highest_price_since_entry == pytest.approx(110.0)
        assert pos.trailing_stop_price == pytest.approx(110.0 * 0.92, abs=0.001)

    def test_lower_price_does_not_move_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        pos.update_high_and_trailing_stop(110.0)
        trail_after_high = pos.trailing_stop_price
        moved = pos.update_high_and_trailing_stop(95.0)
        assert moved is False
        assert pos.trailing_stop_price == pytest.approx(trail_after_high)

    def test_same_price_does_not_move_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        moved = pos.update_high_and_trailing_stop(100.0)
        assert moved is False

    def test_multiple_new_highs_advance_trail_incrementally(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        for price in [105.0, 110.0, 120.0, 130.0]:
            pos.update_high_and_trailing_stop(price)
        assert pos.highest_price_since_entry == pytest.approx(130.0)
        assert pos.trailing_stop_price == pytest.approx(130.0 * 0.92, abs=0.001)

    def test_trail_update_emits_log(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        pos = Position(ticker="TST", quantity=10.0, entry_price=100.0,
                       entry_date="2024-06-14", entry_fee=0.20)
        with caplog.at_level(logging.INFO, logger="src.trading.paper_broker"):
            pos.update_high_and_trailing_stop(120.0)
        assert "TRAILING_STOP_UPDATED" in caplog.text


class TestHeldPositionUpdatePrice:

    def test_new_price_high_moves_trail_in_held_position(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        hp = HeldPosition(ticker="AAPL", quantity=10.0, avg_cost=100.0,
                          highest_price_since_entry=100.0, trailing_stop_price=92.0)
        hp.update_price(115.0)
        assert hp.highest_price_since_entry == pytest.approx(115.0)
        assert hp.trailing_stop_price == pytest.approx(115.0 * 0.92, abs=0.001)

    def test_price_drop_does_not_lower_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        hp = HeldPosition(ticker="AAPL", quantity=10.0, avg_cost=100.0,
                          highest_price_since_entry=120.0, trailing_stop_price=110.40)
        hp.update_price(95.0)
        assert hp.trailing_stop_price == pytest.approx(110.40)

    def test_none_price_does_not_affect_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        hp = HeldPosition(ticker="AAPL", quantity=10.0, avg_cost=100.0,
                          highest_price_since_entry=110.0, trailing_stop_price=101.20)
        hp.update_price(None)
        assert hp.trailing_stop_price == pytest.approx(101.20)


@pytest.fixture
def trailing_enabled(monkeypatch):
    """Trailing exits are locked OFF in V1; these tests exercise the optional feature explicitly."""
    monkeypatch.setattr(settings, "trailing_stop_enabled", True)


@pytest.mark.usefixtures("trailing_enabled")
class TestRiskEngineTrailingStopExit:

    def test_trailing_stop_triggers_sell_when_price_at_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 10, "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"AAPL": 119.60},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len(res.exit_orders) == 1
        assert res.exit_orders[0].action == "SELL"
        assert res.exit_orders[0].reason == ExitReason.STOP_LOSS.value

    def test_trailing_stop_triggers_sell_below_trail(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"TSLA": {"quantity": 5.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 12, "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"TSLA": 115.0},
            ranking_result=_ranking_hold_only("TSLA"), spy_df=spy)
        assert len(res.exit_orders) == 1
        assert res.exit_orders[0].action == "SELL"

    def test_trailing_stop_does_not_trigger_when_above_trail(self, monkeypatch):
        """Price above trail and below take-profit: position should be held unchanged."""
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        # Entry $200, highest $210 (5% up), trail = 210 * 0.92 = $193.20
        # Current price $207 — above trail ($193.20) and PnL = 3.5% (below 15% tp)
        positions = {"NVDA": {"quantity": 5.0, "avg_cost": 200.0, "entry_date": "2024-05-01",
                               "holding_days": 8, "highest_price_since_entry": 210.0, "trailing_stop_price": 193.20}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"NVDA": 207.0},
            ranking_result=_ranking_hold_only("NVDA"), spy_df=spy)
        assert len(res.exit_orders) == 0
        assert res.held_unchanged[0].ticker == "NVDA"

    def test_plain_8pct_stop_loss_still_works_when_no_high(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"META": {"quantity": 5.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 6, "highest_price_since_entry": 100.0, "trailing_stop_price": 92.0}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"META": 91.50},
            ranking_result=_ranking_hold_only("META"), spy_df=spy)
        assert len(res.exit_orders) == 1
        assert res.exit_orders[0].reason == ExitReason.STOP_LOSS.value

    def test_trailing_stop_sell_includes_trail_in_details(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"AMZN": {"quantity": 3.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 10, "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"AMZN": 118.0},
            ranking_result=_ranking_hold_only("AMZN"), spy_df=spy)
        assert len(res.exit_orders) == 1
        details = res.exit_orders[0].details or ""
        assert "trailing stop" in details.lower() or "119.60" in details

    def test_trailing_stop_triggers_log_message(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 10, "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}}
        with caplog.at_level(logging.WARNING, logger="src.risk.risk_engine"):
            evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
                current_positions=positions, current_prices={"AAPL": 118.0},
                ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert "TRAILING_STOP_TRIGGERED" in caplog.text

    def test_locked_in_profit_scenario(self, monkeypatch):
        """Trail locks in a gain: price holds above trail then drops just below -> SELL with +PnL."""
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        # Entry $200, high $225 (12.5% up), trail = 225 * 0.92 = $207.00
        # Day 1: price $215 -> above trail ($207), PnL = 7.5% (below 15% tp) -> HOLD
        # Day 2: price $206 -> below trail ($207) -> SELL with positive PnL (+3% vs $200 entry)
        base_pos = {"quantity": 10.0, "avg_cost": 200.0, "entry_date": "2024-05-01",
                    "holding_days": 10, "highest_price_since_entry": 225.0, "trailing_stop_price": 207.0}
        res_hold = evaluate_portfolio_risk(run_date="2024-06-13", current_cash=5000.0,
            current_positions={"AAPL": base_pos}, current_prices={"AAPL": 215.0},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len(res_hold.exit_orders) == 0
        res_sell = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions={"AAPL": base_pos}, current_prices={"AAPL": 206.0},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len(res_sell.exit_orders) == 1
        assert res_sell.exit_orders[0].pnl_pct > 0



class TestPaperBrokerTrailingStop:

    def setup_method(self, method):
        """Clear broker state before each test - use isolated temp DB."""
        import os
        import tempfile
        from src.db import repository
        from config.settings import settings
        self._temp_db = tempfile.mktemp(suffix=".db")
        self._original_db = settings.db_url
        settings.__dict__["db_url"] = f"sqlite:///{self._temp_db}"
        repository._engine = None
        repository.create_all_tables()

    def teardown_method(self, method):
        """Cleanup after each test."""
        import os
        from src.db import repository
        from config.settings import settings
        repository._engine = None
        settings.__dict__["db_url"] = self._original_db
        try:
            os.unlink(self._temp_db)
        except Exception:
            pass

    def _make_buy_order(self, ticker="AAPL", shares=10.0, price=100.0):
        return OrderSpec(
            date="2024-06-14", ticker=ticker, action="BUY", order_type="MARKET",
            shares=shares, reference_price=price, gross_value=shares * price,
            estimated_fee=shares * price * 0.002, net_amount=shares * price * 1.002,
            reason="QUALIFIED_CONVICTION_BUY", confidence_tier="FULL",
        )

    def test_buy_initialises_trailing_stop(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        broker = PaperBroker()
        broker.load_state()
        broker.positions.clear()
        assert broker.positions == {}
        fill = broker.execute_order(self._make_buy_order(), fill_price=100.0, run_date="2024-06-14")
        assert fill is not None
        pos = broker.positions["AAPL"]
        assert pos.trailing_stop_price == pytest.approx(pos.highest_price_since_entry * 0.92, abs=0.01)

    def test_record_snapshot_advances_trail_on_new_high(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        broker = PaperBroker()
        broker.load_state()
        broker.positions["AAPL"] = Position(ticker="AAPL", quantity=10.0, entry_price=100.0,
            entry_date="2024-06-14", entry_fee=0.20, highest_price_since_entry=100.0, trailing_stop_price=92.0)
        broker.record_snapshot("2024-06-14", {"AAPL": 115.0})
        pos = broker.positions["AAPL"]
        assert pos.highest_price_since_entry == pytest.approx(115.0)
        assert pos.trailing_stop_price == pytest.approx(115.0 * 0.92, abs=0.01)

    def test_record_snapshot_does_not_lower_trail_on_drop(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        broker = PaperBroker()
        broker.load_state()
        broker.positions["MSFT"] = Position(ticker="MSFT", quantity=5.0, entry_price=100.0,
            entry_date="2024-06-14", entry_fee=0.10, highest_price_since_entry=130.0, trailing_stop_price=119.60)
        broker.record_snapshot("2024-06-14", {"MSFT": 120.0})
        pos = broker.positions["MSFT"]
        assert pos.highest_price_since_entry == pytest.approx(130.0)
        assert pos.trailing_stop_price == pytest.approx(119.60)

    def test_record_snapshot_persists_trailing_stop_in_blob(self, monkeypatch):
        from src.db import repository
        repository.create_all_tables()
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        broker = PaperBroker()
        broker.load_state()
        broker.positions["GOOG"] = Position(ticker="GOOG", quantity=1.0, entry_price=150.0,
            entry_date="2024-06-14", entry_fee=0.30, highest_price_since_entry=160.0, trailing_stop_price=147.20)
        broker.record_snapshot("2024-06-14", {"GOOG": 155.0})
        snap = repository.get_latest_portfolio_snapshot()
        assert snap is not None
        pos_data = snap["positions"]["GOOG"]
        assert pos_data["trailing_stop_price"] > 0
        assert pos_data["highest_price_since_entry"] > 0


@pytest.mark.usefixtures("trailing_enabled")
class TestTrailingStopEdgeCases:

    def test_trailing_stop_not_triggered_when_price_just_above(self, monkeypatch):
        """Price one cent above the trailing stop (and below take-profit) -> held."""
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        # Entry $200, highest $210, trail = $193.20; price $193.21 -> just above trail, PnL=-3.4% (no tp)
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 200.0, "entry_date": "2024-05-01",
                               "holding_days": 10, "highest_price_since_entry": 210.0, "trailing_stop_price": 193.20}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"AAPL": 193.21},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len(res.exit_orders) == 0

    def test_trailing_stop_no_double_sell(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01",
                               "holding_days": 10, "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"AAPL": 118.0},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len([o for o in res.exit_orders if o.ticker == "AAPL"]) == 1

    def test_position_without_trailing_stop_data_uses_entry_price(self, monkeypatch):
        monkeypatch.setattr(settings, "trailing_stop_pct", 0.08)
        spy = _make_spy_df()
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01", "holding_days": 10}}
        res = evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions=positions, current_prices={"AAPL": 91.0},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=spy)
        assert len(res.exit_orders) == 1
        assert res.exit_orders[0].action == "SELL"


class TestV1TrailingStopLockedOff:
    """V1 locks trailing stops OFF: paper trading (which tracks a high-water mark) and the backtest
    (which does not) must make the same fixed -8%-from-entry stop decision with the right reason."""

    # Paper-broker position after a run-up to $130 (trail $119.60) vs. the backtest's plain position.
    PAPER = {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01", "holding_days": 10,
             "highest_price_since_entry": 130.0, "trailing_stop_price": 119.60}
    BACKTEST = {"quantity": 10.0, "avg_cost": 100.0, "entry_date": "2024-05-01", "holding_days": 10}

    def _decide(self, pos, price):
        return evaluate_portfolio_risk(run_date="2024-06-14", current_cash=5000.0,
            current_positions={"AAPL": dict(pos)}, current_prices={"AAPL": price},
            ranking_result=_ranking_hold_only("AAPL"), spy_df=_make_spy_df())

    def test_locked_default_is_off(self):
        assert settings.trailing_stop_enabled is False

    def test_high_water_mark_does_not_trigger_exit(self, caplog):
        # $110 is below the $119.60 trail but only +10% from entry: no stop, no take-profit.
        with caplog.at_level(logging.WARNING, logger="src.risk.risk_engine"):
            paper, backtest = self._decide(self.PAPER, 110.0), self._decide(self.BACKTEST, 110.0)
        assert paper.exit_orders == [] and backtest.exit_orders == []
        assert "TRAILING_STOP_TRIGGERED" not in caplog.text

    def test_fixed_stop_same_decision_and_reason_for_paper_and_backtest(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.risk.risk_engine"):
            decisions = [self._decide(pos, 91.5) for pos in (self.PAPER, self.BACKTEST)]
        for res in decisions:
            assert [(o.action, o.reason) for o in res.exit_orders] == [("SELL", ExitReason.STOP_LOSS.value)]
            assert "Fixed stop-loss" in res.exit_orders[0].details
        assert "STOP_LOSS_TRIGGERED" in caplog.text and "TRAILING_STOP_TRIGGERED" not in caplog.text

        # Just above the -8% floor: both hold.
        for pos in (self.PAPER, self.BACKTEST):
            assert self._decide(pos, 92.5).exit_orders == []
