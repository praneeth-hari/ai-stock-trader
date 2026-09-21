"""
tests/test_position_sizing.py — Unit tests for Section 8 Item 2: Position Sizing by Confidence.

Covers:
  - Confidence tiers:
      * Score 0.60 - 0.65 → 50% size ("HALF"), log "HALF_POSITION: score just above bar"
      * Score 0.65 - 0.75 → 75% size ("THREE_QUARTER"), log "THREE_QUARTER_POSITION: moderate confidence"
      * Score 0.75 - 1.00 → 100% size ("FULL"), log "FULL_POSITION: high confidence"
  - Cash reserve floor enforcement (skip if half position breaches 15% floor)
  - Integration with existing modifiers (Macro, Earnings caution, PSI drift)
  - OrderSpec and Position confidence_tier recording
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import settings
from src.portfolio.portfolio import OrderSpec, allocate_portfolio
from src.ranking.ranking import TIER_BUY, TIER_EXIT, TIER_NEUTRAL, RankedOpportunity, RankingResult
from src.risk.risk_engine import RiskVetoReason, evaluate_portfolio_risk
from src.trading.paper_broker import PaperBroker, Position


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """Use an isolated SQLite database for each test."""
    test_db = f"sqlite:///{tmp_path}/test_sizing.db"
    monkeypatch.setattr(settings, "db_url", test_db)
    from src.db import repository
    repository.create_all_tables()


def _make_dummy_ranking(opportunities: list[tuple[str, float]], date: str = "2024-06-14") -> RankingResult:
    """Helper to build a RankingResult from a list of (ticker, probability) tuples."""
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


def _make_spy_df(is_risk_on: bool = True) -> pd.DataFrame:
    """Generate SPY DataFrame guaranteeing Risk-ON or Risk-OFF."""
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


# ── 1. Confidence Sizing Tiers Tests ──────────────────────────────────────────

class TestConfidenceSizingTiers:

    def test_half_position_tier_score_between_60_and_65(self, caplog):
        """Score 0.62 (0.60 <= P < 0.65) yields 50% size ($1,416.67) and HALF tier."""
        ranking = _make_dummy_ranking([("HALF_STOCK", 0.62)])
        spy = _make_spy_df(is_risk_on=True)
        prices = {"HALF_STOCK": 100.0}

        with caplog.at_level("INFO"):
            res = evaluate_portfolio_risk(
                run_date="2024-06-14",
                current_cash=10000.0,
                current_positions={},
                current_prices=prices,
                ranking_result=ranking,
                spy_df=spy,
            )

        assert len(res.buy_orders) == 1
        buy_ord = res.buy_orders[0]
        assert buy_ord.confidence_tier == "HALF"
        # 50% of normal slot size (($10000 * 0.85) / 3 = $2833.33) = $1416.67
        expected = ((10000.0 * 0.85) / 3.0) * 0.50
        assert buy_ord.allocated_amount == pytest.approx(expected, abs=0.01)
        assert "HALF_POSITION: score just above bar" in caplog.text

    def test_three_quarter_position_tier_score_between_65_and_75(self, caplog):
        """Score 0.70 (0.65 <= P < 0.75) yields 75% size ($2,125.00) and THREE_QUARTER tier."""
        ranking = _make_dummy_ranking([("TQ_STOCK", 0.70)])
        spy = _make_spy_df(is_risk_on=True)
        prices = {"TQ_STOCK": 100.0}

        with caplog.at_level("INFO"):
            res = evaluate_portfolio_risk(
                run_date="2024-06-14",
                current_cash=10000.0,
                current_positions={},
                current_prices=prices,
                ranking_result=ranking,
                spy_df=spy,
            )

        assert len(res.buy_orders) == 1
        buy_ord = res.buy_orders[0]
        assert buy_ord.confidence_tier == "THREE_QUARTER"
        # 75% of normal slot size = $2125.00
        expected = ((10000.0 * 0.85) / 3.0) * 0.75
        assert buy_ord.allocated_amount == pytest.approx(expected, abs=0.01)
        assert "THREE_QUARTER_POSITION: moderate confidence" in caplog.text

    def test_full_position_tier_score_above_75(self, caplog):
        """Score 0.80 (P >= 0.75) yields 100% size ($2,833.33) and FULL tier."""
        ranking = _make_dummy_ranking([("FULL_STOCK", 0.80)])
        spy = _make_spy_df(is_risk_on=True)
        prices = {"FULL_STOCK": 100.0}

        with caplog.at_level("INFO"):
            res = evaluate_portfolio_risk(
                run_date="2024-06-14",
                current_cash=10000.0,
                current_positions={},
                current_prices=prices,
                ranking_result=ranking,
                spy_df=spy,
            )

        assert len(res.buy_orders) == 1
        buy_ord = res.buy_orders[0]
        assert buy_ord.confidence_tier == "FULL"
        expected = (10000.0 * 0.85) / 3.0
        assert buy_ord.allocated_amount == pytest.approx(expected, abs=0.01)
        assert "FULL_POSITION: high confidence" in caplog.text


# ── 2. Cash Reserve Floor Breach Tests ────────────────────────────────────────

class TestCashReserveFloorBreach:

    def test_half_position_skips_when_cash_floor_would_be_breached(self):
        """If half position target size exceeds spendable cash, trade is skipped (vetoed)."""
        ranking = _make_dummy_ranking([("TIGHT_CASH_STOCK", 0.62)])
        spy = _make_spy_df(is_risk_on=True)
        prices = {"TIGHT_CASH_STOCK": 100.0}

        positions = {
            "HELD_1": {"quantity": 41.0, "avg_cost": 100.0},
            "HELD_2": {"quantity": 41.0, "avg_cost": 100.0},
        }
        prices.update({"HELD_1": 100.0, "HELD_2": 100.0})

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=1800.0,
            current_positions=positions,
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.buy_orders) == 0
        assert len(res.vetoed_orders) == 1
        assert res.vetoed_orders[0].veto_reason == RiskVetoReason.CASH_RESERVE_VIOLATION


# ── 3. Combined Modifiers Tests ───────────────────────────────────────────────

class TestCombinedModifiers:

    def test_earnings_caution_plus_half_confidence_tier(self):
        """Earnings caution (50%) + HALF confidence (50%) = 25% of normal slot size."""
        opp = RankedOpportunity(
            ticker="EARN_HALF",
            date="2024-06-14",
            probability=0.62,
            rank=1,
            conviction_tier="BUY_CANDIDATE",
            is_buy_eligible=True,
            earnings_status="CAUTION",
            days_until_earnings=4,
        )
        ranking = RankingResult(
            date="2024-06-14",
            regime_risk_on=True,
            total_evaluated=1,
            ranked_opportunities=[opp],
            top_buy_candidates=[opp],
        )
        spy = _make_spy_df(is_risk_on=True)
        prices = {"EARN_HALF": 100.0}

        res = evaluate_portfolio_risk(
            run_date="2024-06-14",
            current_cash=10000.0,
            current_positions={},
            current_prices=prices,
            ranking_result=ranking,
            spy_df=spy,
        )

        assert len(res.buy_orders) == 1
        buy_ord = res.buy_orders[0]
        # Normal = $2,833.33. Combined = 0.5 * 0.5 * $2,833.33 = $708.33
        expected = ((10000.0 * 0.85) / 3.0) * 0.50 * 0.50
        assert buy_ord.allocated_amount == pytest.approx(expected, abs=0.01)
        assert buy_ord.confidence_tier == "HALF"


# ── 4. Integration Tests (OrderSpec & PaperBroker) ────────────────────────────

class TestIntegration:

    def test_order_spec_confidence_tier_serialisation(self):
        order = OrderSpec(
            date="2024-06-14",
            ticker="AAPL",
            action="BUY",
            order_type="MARKET",
            shares=10.0,
            reference_price=150.0,
            gross_value=1500.0,
            estimated_fee=3.0,
            net_amount=1503.0,
            reason="QUALIFIED_CONVICTION_BUY",
            confidence_tier="HALF",
        )
        d = order.to_dict()
        assert d["confidence_tier"] == "HALF"

    def test_paper_broker_records_confidence_tier(self):
        broker = PaperBroker()
        broker.load_state()

        order = OrderSpec(
            date="2024-06-14",
            ticker="MSFT",
            action="BUY",
            order_type="MARKET",
            shares=5.0,
            reference_price=200.0,
            gross_value=1000.0,
            estimated_fee=2.0,
            net_amount=1002.0,
            reason="QUALIFIED_CONVICTION_BUY",
            confidence_tier="THREE_QUARTER",
        )

        fill = broker.execute_order(order, fill_price=200.0, run_date="2024-06-14")
        assert fill is not None
        assert "MSFT" in broker.positions
        pos = broker.positions["MSFT"]
        assert pos.confidence_tier == "THREE_QUARTER"
