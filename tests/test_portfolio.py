"""
tests/test_portfolio.py — Phase 9 Portfolio Allocation Engine Unit & Integration Tests.

TEST SUITE OBJECTIVES:
  1. Buy Order Fee-Inclusive Outflow Convention:
     Asserts allocated_amount ($14.16) covers both share cost and the 0.2% fee,
     so total_outflow <= allocated_amount.
  2. Sell Order Net-Proceeds Inflow Convention:
     Asserts liquidation proceeds returned to cash are net of 0.2% fee (shares * price * 0.998).
  3. 4-Decimal Floor Rounding Safety:
     Asserts fractional shares are floored at 4 decimals and never round up to breach the allocation.
  4. Exact $50 Scale Multi-Action Scenario:
     Starting equity $50, 1 hold (AAPL), 1 stop-loss exit (JPM), 1 buy (MSFT).
     Asserts exact order specs, ending cash, and 15% cash reserve compliance.
  5. Zero-Order / Inactive Day Scenario:
     When Risk Engine outputs only HOLDs, portfolio state carries forward with 0 orders.
  6. End-to-End Pipeline Integration:
     Feeds real RiskAssessmentResult directly into allocate_portfolio and verifies pro-forma reconciliation.
"""

import math
import pytest
from config.settings import settings
from src.portfolio.portfolio import (
    OrderSpec,
    PortfolioAllocationResult,
    allocate_portfolio,
    calculate_fractional_shares,
)
from src.risk.risk_engine import (
    ExitReason,
    HeldPosition,
    RiskAssessmentResult,
    RiskDecision,
    evaluate_portfolio_risk,
)
from src.ranking.ranking import RankedOpportunity, RankingResult, TIER_BUY, TIER_NEUTRAL


def test_1_buy_fee_inclusive_convention():
    """
    Test 1: Confirms for a BUY, allocated_amount represents total cash outflow INCLUDING 0.2% fee.
    """
    alloc = 14.16
    price = 187.32
    fee_rate = 0.002

    shares, gross_value, fee = calculate_fractional_shares(alloc, price, fee_rate)

    # 4-decimal precision
    assert len(str(shares).split(".")[-1]) <= 4
    # Outflow must be within budget
    total_outflow = round(gross_value + fee, 4)
    assert total_outflow <= alloc
    # Fee matches 0.2%
    assert math.isclose(fee, round(gross_value * fee_rate, 4), abs_tol=1e-4)


def test_2_sell_net_proceeds_convention():
    """
    Test 2: Confirms for a SELL, cash inflow is net of 0.2% fee (gross * 0.998).
    """
    date = "2024-08-30"
    current_cash = 20.00
    positions = {"JPM": {"quantity": 0.0590, "avg_cost": 240.0, "current_price": 218.0}}
    prices = {"JPM": 218.0}

    # Dummy risk assessment with 1 approved SELL
    sell_decision = RiskDecision(
        ticker="JPM",
        action="SELL",
        approved=True,
        reason="STOP_LOSS",
        details="Stop-loss triggered",
        allocated_amount=0.0,
        quantity=0.0590,
    )
    risk_assessment = RiskAssessmentResult(
        date=date,
        total_equity=current_cash + (0.0590 * 218.0),
        initial_cash=current_cash,
        final_cash=current_cash + (0.0590 * 218.0 * 0.998),
        regime_risk_on=True,
        positions_before=1,
        positions_after=0,
        exit_orders=[sell_decision],
        buy_orders=[],
    )

    res = allocate_portfolio(risk_assessment, current_cash, positions, prices, fee_rate=0.002)

    assert len(res.orders) == 1
    order = res.orders[0]
    assert order.action == "SELL"
    assert order.ticker == "JPM"
    assert order.shares == 0.0590

    expected_gross = round(0.0590 * 218.0, 4)  # 12.862
    expected_fee = round(expected_gross * 0.002, 4)  # 0.0257
    expected_net = round(expected_gross - expected_fee, 4)  # 12.8363 (exact 0.998)

    assert math.isclose(order.gross_value, expected_gross, abs_tol=1e-4)
    assert math.isclose(order.estimated_fee, expected_fee, abs_tol=1e-4)
    assert math.isclose(order.net_amount, expected_net, abs_tol=1e-4)
    assert math.isclose(res.projected_cash, round(current_cash + expected_net, 4), abs_tol=1e-4)
    assert "JPM" not in res.projected_positions


def test_3_four_decimal_floor_rounding_safety():
    """
    Test 3: Confirms floor rounding prevents over-allocation on high-priced assets.
    """
    alloc = 14.16
    price = 417.23
    fee_rate = 0.002

    shares, gross, fee = calculate_fractional_shares(alloc, price, fee_rate)

    # Raw shares: (14.16 / 1.002) / 417.23 = 14.1317365 / 417.23 = 0.033870...
    # Floored to 4 decimals = 0.0338
    assert shares == 0.0338
    assert (gross + fee) <= alloc


def test_4_exact_50_dollar_scale_scenario():
    """
    Test 4: Full $50 starting scale with:
      - 1 holding in profit (AAPL @ $210 cost, price $229 -> HOLD)
      - 1 holding in stop-loss (JPM @ $240 cost, price $218 -> STOP-LOSS SELL)
      - 1 approved buy (MSFT @ $417.23, $14.16 allocation -> BUY)
      - Asserts exact post-allocation cash >= 15% reserve floor ($7.50).
    """
    date = "2024-08-30"
    current_cash = 21.67
    positions = {
        "AAPL": {"quantity": 0.0675, "avg_cost": 210.0, "current_price": 229.0},
        "JPM": {"quantity": 0.0590, "avg_cost": 240.0, "current_price": 218.0},
    }
    prices = {"AAPL": 229.0, "JPM": 218.0, "MSFT": 417.23}

    sell_jpm = RiskDecision(
        ticker="JPM",
        action="SELL",
        approved=True,
        reason="STOP_LOSS",
        details="Stop loss exit",
        allocated_amount=0.0,
        quantity=0.0590,
    )
    hold_aapl = RiskDecision(
        ticker="AAPL",
        action="HOLD",
        approved=True,
        reason="HOLDING_ACTIVE",
        details="Active holding",
        allocated_amount=0.0,
        quantity=0.0675,
    )
    buy_msft = RiskDecision(
        ticker="MSFT",
        action="BUY",
        approved=True,
        reason="QUALIFIED_CONVICTION_BUY",
        details="Equal-weight buy",
        allocated_amount=14.16,
        quantity=0.0338,
    )

    total_equity = current_cash + (0.0675 * 229.0) + (0.0590 * 218.0)  # ~49.99
    risk_assessment = RiskAssessmentResult(
        date=date,
        total_equity=total_equity,
        initial_cash=current_cash,
        final_cash=20.34,
        regime_risk_on=True,
        positions_before=2,
        positions_after=2,
        exit_orders=[sell_jpm],
        buy_orders=[buy_msft],
        held_unchanged=[hold_aapl],
    )

    res = allocate_portfolio(risk_assessment, current_cash, positions, prices)

    # 2 orders generated: SELL JPM, BUY MSFT
    assert len(res.orders) == 2
    sell_order = next(o for o in res.orders if o.ticker == "JPM")
    buy_order = next(o for o in res.orders if o.ticker == "MSFT")

    assert sell_order.action == "SELL"
    assert buy_order.action == "BUY"
    assert buy_order.shares == 0.0338

    # Cash reconciliation:
    # starting_cash + net_sell_inflow - buy_outflow
    expected_sell_net = round(0.0590 * 218.0 * 0.998, 4)
    expected_buy_outflow = round((0.0338 * 417.23) * 1.002, 4)
    expected_cash = round(current_cash + expected_sell_net - expected_buy_outflow, 4)

    assert math.isclose(res.projected_cash, expected_cash, abs_tol=1e-4)
    assert res.cash_reserve_maintained
    assert res.projected_cash >= 7.50

    # Positions check
    assert "JPM" not in res.projected_positions
    assert "AAPL" in res.projected_positions
    assert "MSFT" in res.projected_positions
    assert len(res.projected_positions) == 2


def test_5_zero_orders_when_no_active_decisions():
    """
    Test 5: Confirms when no buys or sells are approved, 0 orders generated and balances unchanged.
    """
    date = "2024-08-30"
    current_cash = 50.00
    positions = {}
    prices = {"AAPL": 229.0}

    risk_assessment = RiskAssessmentResult(
        date=date,
        total_equity=50.00,
        initial_cash=50.00,
        final_cash=50.00,
        regime_risk_on=True,
        positions_before=0,
        positions_after=0,
        exit_orders=[],
        buy_orders=[],
    )

    res = allocate_portfolio(risk_assessment, current_cash, positions, prices)
    assert len(res.orders) == 0
    assert res.projected_cash == 50.00
    assert res.projected_equity == 50.00
    assert res.cash_reserve_maintained


def test_6_integration_risk_engine_to_portfolio_allocation():
    """
    Test 6: Full pipeline integration passing evaluate_portfolio_risk output directly into allocate_portfolio.
    """
    date = "2024-08-30"
    current_cash = 21.67
    positions = {
        "AAPL": {"quantity": 0.0675, "avg_cost": 210.0},
        "JPM": {"quantity": 0.0590, "avg_cost": 240.0},
    }
    prices = {"AAPL": 229.0, "JPM": 218.0, "MSFT": 417.23}

    ranking = RankingResult(
        date=date,
        regime_risk_on=True,
        total_evaluated=1,
        ranked_opportunities=[
            RankedOpportunity("MSFT", date, 0.68, 1, TIER_BUY, 0.05, 1.02, is_buy_eligible=True)
        ],
        top_buy_candidates=[
            RankedOpportunity("MSFT", date, 0.68, 1, TIER_BUY, 0.05, 1.02, is_buy_eligible=True)
        ],
    )

    risk_res = evaluate_portfolio_risk(
        run_date=date,
        current_cash=current_cash,
        current_positions=positions,
        current_prices=prices,
        ranking_result=ranking,
    )

    # Allocate directly from risk engine output
    alloc_res = allocate_portfolio(risk_res, current_cash, positions, prices)

    assert len(alloc_res.orders) == 2
    assert alloc_res.cash_reserve_maintained
    assert alloc_res.projected_cash >= alloc_res.cash_reserve_floor
    markdown = alloc_res.to_markdown_summary()
    assert "Portfolio Allocation Summary" in markdown
    assert "MSFT" in markdown
    assert "JPM" in markdown
