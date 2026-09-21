"""
tests/test_risk.py — Phase 8 Risk Management & Sizing Unit & Integration Tests.

TEST SUITE OBJECTIVES:
  1. Stop-Loss Isolation: Holding down -8.1% triggers immediate SELL.
  2. Take-Profit Isolation: Holding up +15.1% triggers immediate SELL.
  3. Signal-Exit Isolation: Holding with P = 0.44 triggers immediate SELL.
  4. Regime Filter Isolation: SPY < 200d MA blocks all new buys (Risk-OFF).
  5. Position Cap Isolation: 3 open positions blocks 4th candidate (MAX_POSITIONS_REACHED).
  6. Exact Position Sizing & Cash Reserve Formula:
     Asserts exact formula (equity * 0.85) / 3 and tests the $50 equity, 1-slot open scenario.
  7. Skip Already-Held and Advance: Rank #1 is held; candidate #2 is approved into open slot.
  8. Min Trade Size Veto: Available allocation < $10.00 is vetoed.
  9. Held Stock Bad Data Immunity (§1.7):
     Held stock missing valid data is NOT evaluated for exit; held untouched as SKIPPED_INVALID_DATA_HELD_UNCHANGED.
  10. Multi-Rule Interaction:
     Stock hits Stop-Loss on the same day market goes Risk-OFF (SELL executes, buys vetoed, cash held).
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.ranking.ranking import RankedOpportunity, RankingResult, TIER_BUY, TIER_NEUTRAL, TIER_EXIT
from src.risk.risk_engine import (
    ExitReason,
    HeldPosition,
    RiskAssessmentResult,
    RiskDecision,
    RiskVetoReason,
    evaluate_portfolio_risk,
)


def _make_dummy_ranking(opportunities: list[tuple[str, float]], date: str = "2024-06-14") -> RankingResult:
    """Helper to build a RankingResult from a list of (ticker, prob) tuples."""
    ranked = []
    buys = []
    for rank, (ticker, p) in enumerate(opportunities, start=1):
        tier = TIER_BUY if p >= settings.buy_bar else (TIER_EXIT if p < settings.signal_exit else TIER_NEUTRAL)
        opp = RankedOpportunity(
            ticker=ticker,
            date=date,
            probability=p,
            rank=rank,
            conviction_tier=tier,
            is_buy_eligible=(p >= settings.buy_bar),
            is_exit_signal=(p < settings.signal_exit),
        )
        ranked.append(opp)
        if p >= settings.buy_bar:
            buys.append(opp)

    return RankingResult(
        date=date,
        regime_risk_on=True,
        total_evaluated=len(ranked),
        ranked_opportunities=ranked,
        top_buy_candidates=buys,
    )


def _make_spy_df(is_risk_on: bool, date: str = "2024-06-14") -> pd.DataFrame:
    """Helper to create SPY data above or below 200-day MA."""
    n = 250
    dates = pd.bdate_range(end=date, periods=n).strftime("%Y-%m-%d").tolist()
    # Flat 100.0 with 200d MA ~ 100.0
    closes = [100.0] * n
    closes[-1] = 110.0 if is_risk_on else 85.0
    return pd.DataFrame({"date": dates, "ticker": "SPY", "close": closes})


class TestExitRulesInIsolation:
    """Verify each exit rule triggers independently and cleanly."""

    def test_1_stop_loss_triggers_sell_at_minus_8_pct(self):
        """Holding down -8.1% must trigger immediate SELL regardless of model probability."""
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0}}
        prices = {"AAPL": 91.90}  # -8.10% PnL
        # Model gives AAPL high probability (0.70), but stop loss must override
        ranking = _make_dummy_ranking([("AAPL", 0.70)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=500.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.exit_orders) == 1
        exit_ord = res.exit_orders[0]
        assert exit_ord.ticker == "AAPL"
        assert exit_ord.action == "SELL"
        assert exit_ord.approved is True
        assert exit_ord.reason == ExitReason.STOP_LOSS.value
        assert exit_ord.pnl_pct == pytest.approx(-0.081, rel=1e-3)
        assert res.positions_after == 0

    def test_2_take_profit_triggers_sell_at_plus_15_pct(self):
        """Holding up +15.1% must trigger immediate SELL to lock in gains."""
        positions = {"MSFT": {"quantity": 5.0, "avg_cost": 200.0}}
        prices = {"MSFT": 230.20}  # +15.10% PnL
        ranking = _make_dummy_ranking([("MSFT", 0.65)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=500.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.exit_orders) == 1
        exit_ord = res.exit_orders[0]
        assert exit_ord.ticker == "MSFT"
        assert exit_ord.action == "SELL"
        assert exit_ord.approved is True
        assert exit_ord.reason == ExitReason.TAKE_PROFIT.value
        assert exit_ord.pnl_pct == pytest.approx(0.151, rel=1e-3)
        assert res.positions_after == 0

    def test_3_signal_exit_triggers_sell_below_0_45(self):
        """Holding with healthy PnL (+2%) must trigger SELL if model probability degrades < 0.45."""
        positions = {"JPM": {"quantity": 8.0, "avg_cost": 150.0}}
        prices = {"JPM": 153.0}  # +2.0% PnL (not stop-loss or take-profit)
        # Model probability degraded to 0.42 (< 0.45)
        ranking = _make_dummy_ranking([("JPM", 0.42)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=500.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.exit_orders) == 1
        exit_ord = res.exit_orders[0]
        assert exit_ord.ticker == "JPM"
        assert exit_ord.action == "SELL"
        assert exit_ord.approved is True
        assert exit_ord.reason == ExitReason.SIGNAL_EXIT.value
        assert res.positions_after == 0


class TestEntryRulesAndVetoMechanisms:
    """Verify position caps, cash reserve preservation, regime veto, and candidate iteration."""

    def test_4_regime_filter_blocks_all_buys_when_risk_off(self):
        """When SPY < 200d MA, all buy candidates are vetoed with REGIME_FILTER_RISK_OFF."""
        ranking = _make_dummy_ranking([("AAPL", 0.75), ("MSFT", 0.68), ("NVDA", 0.62)])
        prices = {"AAPL": 150.0, "MSFT": 300.0, "NVDA": 100.0}
        spy = _make_spy_df(is_risk_on=False)  # Risk-OFF

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=1000.0,
            current_positions={},
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert res.regime_risk_on is False
        assert len(res.buy_orders) == 0
        assert len(res.vetoed_orders) == 3
        for v in res.vetoed_orders:
            assert v.veto_reason == RiskVetoReason.REGIME_FILTER_RISK_OFF
            assert v.approved is False

    def test_5_position_cap_blocks_fourth_stock(self):
        """When 3 positions are held, a 4th candidate with P >= 0.60 is vetoed with MAX_POSITIONS_REACHED."""
        positions = {
            "STOCK_A": {"quantity": 10.0, "avg_cost": 50.0},
            "STOCK_B": {"quantity": 10.0, "avg_cost": 50.0},
            "STOCK_C": {"quantity": 10.0, "avg_cost": 50.0},
        }
        prices = {"STOCK_A": 50.0, "STOCK_B": 50.0, "STOCK_C": 50.0, "STOCK_D": 50.0}
        # STOCK_A, B, C are neutral holds; STOCK_D is a top buy candidate
        ranking = _make_dummy_ranking([("STOCK_D", 0.78), ("STOCK_A", 0.50), ("STOCK_B", 0.50), ("STOCK_C", 0.50)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=2000.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert res.positions_before == 3
        assert res.positions_after == 3
        assert len(res.buy_orders) == 0
        assert len(res.vetoed_orders) == 1
        assert res.vetoed_orders[0].ticker == "STOCK_D"
        assert res.vetoed_orders[0].veto_reason == RiskVetoReason.MAX_POSITIONS_REACHED

    def test_6_exact_position_sizing_and_cash_reserve_formula(self):
        """
        Verify exact sizing formula:
          target_position_size = (total_equity * (1 - cash_reserve)) / max_positions
                               = (total_equity * 0.85) / 3
        Scenario: $50 total equity, 1 slot open (2 already filled).
          Expected allocation = (50.0 * 0.85) / 3 = $14.1667
          Cash reserve floor = $50.0 * 0.15 = $7.50
        """
        # 2 existing positions worth $14.16 each ($28.32 total), plus $21.68 cash = $50.00 total equity
        positions = {
            "STOCK_1": {"quantity": 1.0, "avg_cost": 14.16},
            "STOCK_2": {"quantity": 1.0, "avg_cost": 14.16},
        }
        prices = {"STOCK_1": 14.16, "STOCK_2": 14.16, "CANDIDATE": 14.16}
        ranking = _make_dummy_ranking([("CANDIDATE", 0.80), ("STOCK_1", 0.50), ("STOCK_2", 0.50)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=21.68,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        # Confirm 1 buy order approved
        assert len(res.buy_orders) == 1
        buy_ord = res.buy_orders[0]
        assert buy_ord.ticker == "CANDIDATE"
        assert buy_ord.approved is True

        # Formula check: (50.0 * 0.85) / 3 = 14.16666...
        expected_allocation = (res.total_equity * (1.0 - settings.cash_reserve)) / settings.max_positions
        assert buy_ord.allocated_amount == pytest.approx(expected_allocation, abs=0.02)
        assert buy_ord.allocated_amount == pytest.approx(14.17, abs=0.02)

        # Confirm cash reserve floor (15% of $50 = $7.50) is strictly preserved
        cash_floor = res.total_equity * settings.cash_reserve
        assert res.final_cash >= cash_floor - 0.01

    def test_7_skip_already_held_and_advance(self):
        """If Rank #1 is already held, Risk skips it and advances to approve Candidate #2."""
        positions = {"AAPL": {"quantity": 10.0, "avg_cost": 100.0}}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        # AAPL is Rank 1, MSFT is Rank 2
        ranking = _make_dummy_ranking([("AAPL", 0.75), ("MSFT", 0.68)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=1000.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        # AAPL skipped / vetoed with ALREADY_HELD
        assert any(v.ticker == "AAPL" and v.veto_reason == RiskVetoReason.ALREADY_HELD for v in res.vetoed_orders)
        # MSFT successfully approved into slot #2
        assert len(res.buy_orders) == 1
        assert res.buy_orders[0].ticker == "MSFT"
        assert res.positions_after == 2

    def test_8_min_trade_size_veto(self):
        """If available allocation < $10.00, buy order is vetoed with MIN_TRADE_SIZE_VIOLATION."""
        # Total equity = $20.00. 1 slot target = (20 * 0.85) / 3 = $5.67 < $10.00 min trade size
        ranking = _make_dummy_ranking([("TINY_STOCK", 0.70)])
        prices = {"TINY_STOCK": 5.0}
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=20.0,
            current_positions={},
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.buy_orders) == 0
        assert len(res.vetoed_orders) == 1
        assert res.vetoed_orders[0].ticker == "TINY_STOCK"
        assert res.vetoed_orders[0].veto_reason == RiskVetoReason.MIN_TRADE_SIZE_VIOLATION


class TestSpecialEdgeCasesAndInteractions:
    """Verify data failure immunity (§1.7) and multi-rule compound interactions."""

    def test_9_held_stock_skipped_invalid_data_held_unchanged(self):
        """
        §1.7 Rule: If a currently-held stock fails validation or has missing data today,
        Risk Engine must NOT evaluate stop-loss/take-profit/signal-exit.
        It is held untouched as SKIPPED_INVALID_DATA_HELD_UNCHANGED, and slot remains occupied.
        """
        positions = {
            "GOOD_STOCK": {"quantity": 10.0, "avg_cost": 100.0},
            "BAD_DATA_STOCK": {"quantity": 10.0, "avg_cost": 100.0},
        }
        # BAD_DATA_STOCK has no price or is explicitly listed in invalid_or_missing_tickers
        prices = {"GOOD_STOCK": 100.0}
        ranking = _make_dummy_ranking([("GOOD_STOCK", 0.50)])
        spy = _make_spy_df(is_risk_on=True)

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=500.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
            invalid_or_missing_tickers={"BAD_DATA_STOCK"},
        )

        # Neither stock should be sold
        assert len(res.exit_orders) == 0
        # BAD_DATA_STOCK is recorded in held_unchanged with exact status
        skipped = [h for h in res.held_unchanged if h.ticker == "BAD_DATA_STOCK"]
        assert len(skipped) == 1
        assert skipped[0].reason == ExitReason.SKIPPED_INVALID_DATA_HELD_UNCHANGED.value
        assert skipped[0].action == "HOLD"
        # Slot remains occupied (2 positions before, 2 positions after)
        assert res.positions_after == 2

    def test_10_multi_rule_interaction_stop_loss_in_risk_off(self):
        """
        Compound interaction:
        Stock A hits Stop-Loss (-8.5%) on the EXACT same day SPY breaks below 200d MA (Risk-OFF).
        Expected:
          - Stop-Loss executes cleanly, selling Stock A and returning cash.
          - Regime filter vetoes all buy candidates.
          - Freed cash is held safely in reserve; portfolio moves towards cash.
        """
        positions = {"STOCK_A": {"quantity": 10.0, "avg_cost": 100.0}}
        prices = {"STOCK_A": 91.50, "CANDIDATE_B": 50.0}  # -8.5% PnL
        ranking = _make_dummy_ranking([("CANDIDATE_B", 0.78)])
        spy = _make_spy_df(is_risk_on=False)  # Market went Risk-OFF!

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=500.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        # 1. Stop-Loss executes
        assert len(res.exit_orders) == 1
        assert res.exit_orders[0].ticker == "STOCK_A"
        assert res.exit_orders[0].reason == ExitReason.STOP_LOSS.value
        assert res.exit_orders[0].approved is True

        # 2. Candidate B is vetoed by Regime Filter
        assert len(res.buy_orders) == 0
        assert len(res.vetoed_orders) == 1
        assert res.vetoed_orders[0].ticker == "CANDIDATE_B"
        assert res.vetoed_orders[0].veto_reason == RiskVetoReason.REGIME_FILTER_RISK_OFF

        # 3. Cash increased from sale, positions dropped to 0
        assert res.final_cash > res.initial_cash
        assert res.positions_after == 0
