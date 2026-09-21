"""
tests/test_risk_rules_section2.py — Test Section 2 Risk & Safety Rules (Items 5 through 9).
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.ranking.ranking import RankedOpportunity, RankingResult
from src.risk.risk_engine import (
    ExitReason,
    RiskVetoReason,
    check_macro_circuit_breaker,
    compute_adaptive_thresholds,
    evaluate_portfolio_risk,
)


def _make_dummy_spy(prices: list[float]) -> pd.DataFrame:
    """Helper to create dummy SPY dataframe."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B").strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [80_000_000.0] * len(prices),
        "ticker": "SPY",
    })


def _dummy_ranking(opportunities: list[tuple[str, float]]) -> RankingResult:
    ranked = [
        RankedOpportunity(
            ticker=t,
            date="2024-01-30",
            probability=p,
            rank=i + 1,
            conviction_tier="BUY_CANDIDATE" if p >= 0.60 else "NEUTRAL_HOLD",
            is_buy_eligible=p >= 0.60,
        )
        for i, (t, p) in enumerate(opportunities)
    ]
    return RankingResult(
        date="2024-01-30",
        regime_risk_on=True,
        total_evaluated=len(ranked),
        ranked_opportunities=ranked,
    )


# ── Item 5: Adaptive Conviction Thresholds ─────────────────────────────────────

def test_adaptive_thresholds_low_vol():
    """Low volatility regime (< 12% realized or VIX < 15) keeps standard 0.60 / 0.45."""
    buy_bar, sig_exit, regime, vol = compute_adaptive_thresholds(vix_value=12.0)
    assert regime == "LOW_VOLATILITY"
    assert buy_bar == 0.60
    assert sig_exit == 0.45


def test_adaptive_thresholds_normal_vol():
    """Normal volatility regime (12-20% realized or VIX 15-25) scales to 0.63 / 0.47."""
    buy_bar, sig_exit, regime, vol = compute_adaptive_thresholds(vix_value=18.0)
    assert regime == "NORMAL_VOLATILITY"
    assert buy_bar == 0.63
    assert sig_exit == 0.47


def test_adaptive_thresholds_high_vol():
    """High volatility regime (> 20% realized or VIX > 25) scales to 0.67 / 0.50."""
    buy_bar, sig_exit, regime, vol = compute_adaptive_thresholds(vix_value=28.0)
    assert regime == "HIGH_VOLATILITY"
    assert buy_bar == 0.67
    assert sig_exit == 0.50


def test_adaptive_buy_bar_filters_candidate_in_high_vol():
    """Candidate with P=0.64 passes in low vol (0.60) but is excluded in high vol (0.67)."""
    ranking = _dummy_ranking([("AAPL", 0.64)])
    current_prices = {"AAPL": 150.0}

    # Low vol: buy approved
    res_low = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices=current_prices,
        ranking_result=ranking,
        vix_value=12.0,
    )
    assert len(res_low.buy_orders) == 1
    assert res_low.buy_orders[0].ticker == "AAPL"

    # High vol: P=0.64 < 0.67 active buy bar -> no buy
    res_high = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices=current_prices,
        ranking_result=ranking,
        vix_value=28.0,
    )
    assert len(res_high.buy_orders) == 0


# ── Item 6: 5-Day Minimum Holding Period ──────────────────────────────────────

def test_signal_exit_suppressed_within_first_5_days():
    """When held < 5 days, signal exit (P < 0.45) is suppressed to prevent churn."""
    positions = {
        "MSFT": {"quantity": 10.0, "avg_cost": 300.0, "holding_days": 2}
    }
    # MSFT conviction drops to 0.40 (< 0.45), price flat (pnl = 0%)
    ranking = _dummy_ranking([("MSFT", 0.40)])
    prices = {"MSFT": 300.0}

    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=7000.0,
        current_positions=positions,
        current_prices=prices,
        ranking_result=ranking,
        vix_value=12.0,
    )
    assert len(res.exit_orders) == 0
    assert len(res.held_unchanged) == 1
    assert res.held_unchanged[0].reason == ExitReason.MINIMUM_HOLD_OVERRIDE.value


def test_signal_exit_triggers_after_5_days():
    """When held >= 5 days, signal exit executes normally."""
    positions = {
        "MSFT": {"quantity": 10.0, "avg_cost": 300.0, "holding_days": 5}
    }
    ranking = _dummy_ranking([("MSFT", 0.40)])
    prices = {"MSFT": 300.0}

    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=7000.0,
        current_positions=positions,
        current_prices=prices,
        ranking_result=ranking,
        vix_value=12.0,
    )
    assert len(res.exit_orders) == 1
    assert res.exit_orders[0].reason == ExitReason.SIGNAL_EXIT.value


def test_stop_loss_and_take_profit_always_trigger_regardless_of_holding_days():
    """Hard -8% stop-loss and +15% take-profit MUST trigger on Day 1 or 2."""
    # Stop-loss on Day 1
    pos_loss = {"NVDA": {"quantity": 5.0, "avg_cost": 500.0, "holding_days": 1}}
    prices_loss = {"NVDA": 450.0}  # -10% drop
    res_loss = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=7500.0,
        current_positions=pos_loss,
        current_prices=prices_loss,
        ranking_result=_dummy_ranking([]),
    )
    assert len(res_loss.exit_orders) == 1
    assert res_loss.exit_orders[0].reason == ExitReason.STOP_LOSS.value

    # Take-profit on Day 2
    pos_gain = {"NVDA": {"quantity": 5.0, "avg_cost": 500.0, "holding_days": 2}}
    prices_gain = {"NVDA": 585.0}  # +17% gain
    res_gain = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=7500.0,
        current_positions=pos_gain,
        current_prices=prices_gain,
        ranking_result=_dummy_ranking([]),
    )
    assert len(res_gain.exit_orders) == 1
    assert res_gain.exit_orders[0].reason == ExitReason.TAKE_PROFIT.value


# ── Item 7: Tiered Bad Data Response for Held Positions ────────────────────────

def test_tiered_bad_data_day1_untouched():
    """Day 1 of bad data: held untouched."""
    positions = {"AAPL": {"quantity": 10.0, "avg_cost": 150.0, "bad_data_days": 0}}
    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=8500.0,
        current_positions=positions,
        current_prices={},  # Missing/invalid
        ranking_result=_dummy_ranking([]),
        invalid_or_missing_tickers={"AAPL"},
    )
    assert len(res.held_unchanged) == 1
    assert res.held_unchanged[0].reason == ExitReason.SKIPPED_INVALID_DATA_HELD_UNCHANGED.value


def test_tiered_bad_data_day2_near_stop_loss_urgent_alert():
    """Day 2 of bad data near stop-loss floor (-6.5% PnL): triggers urgent alert."""
    # Last known price was 140.0 (avg cost 150.0 -> -6.67% PnL, within 2% of -8% stop loss)
    positions = {"AAPL": {"quantity": 10.0, "avg_cost": 150.0, "bad_data_days": 1}}
    prices = {"AAPL": 140.0}
    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=8500.0,
        current_positions=positions,
        current_prices=prices,
        ranking_result=_dummy_ranking([]),
        invalid_or_missing_tickers={"AAPL"},
    )
    assert len(res.held_unchanged) == 1
    assert res.held_unchanged[0].reason == ExitReason.BAD_DATA_NEAR_STOP_LOSS_ALERT.value


def test_tiered_bad_data_day3_locks_position():
    """Day 3+ of consecutive bad data: marked as REVIEW REQUIRED — DO NOT TRADE."""
    positions = {"AAPL": {"quantity": 10.0, "avg_cost": 150.0, "bad_data_days": 2}}
    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=8500.0,
        current_positions=positions,
        current_prices={},
        ranking_result=_dummy_ranking([]),
        invalid_or_missing_tickers={"AAPL"},
    )
    assert len(res.held_unchanged) == 1
    assert res.held_unchanged[0].reason == ExitReason.REVIEW_REQUIRED_DO_NOT_TRADE.value


# ── Item 8: Macro Crash Circuit Breakers ───────────────────────────────────────

def test_macro_circuit_breaker_5d_crash():
    """SPY drop > 7% over 5 days halts new buys."""
    # SPY drops from 500 to 460 (-8.0%) over 6 days
    spy_closes = [500.0, 495.0, 490.0, 480.0, 470.0, 460.0]
    spy_df = _make_dummy_spy(spy_closes)

    is_active, breaker_type, details, halt_days = check_macro_circuit_breaker(spy_df)
    assert is_active is True
    assert breaker_type == "MACRO_CIRCUIT_BREAKER_5D"
    assert halt_days == 5

    ranking = _dummy_ranking([("GOOGL", 0.70)])
    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices={"GOOGL": 140.0},
        ranking_result=ranking,
        spy_df=spy_df,
    )
    assert len(res.buy_orders) == 0
    assert len(res.vetoed_orders) == 1
    assert res.vetoed_orders[0].veto_reason == RiskVetoReason.MACRO_CIRCUIT_BREAKER_ACTIVE


def test_macro_circuit_breaker_20d_crash():
    """SPY drop > 15% over 20 days halts new buys for 10 days."""
    spy_closes = [500.0] * 10 + [420.0] * 12  # -16.0% drop
    spy_df = _make_dummy_spy(spy_closes)

    is_active, breaker_type, details, halt_days = check_macro_circuit_breaker(spy_df)
    assert is_active is True
    assert breaker_type == "MACRO_CIRCUIT_BREAKER_20D"
    assert halt_days == 10


# ── Item 9: PSI Drift Protocol ─────────────────────────────────────────────────

def test_psi_moderate_drift_cuts_sizing_in_half():
    """PSI between 0.10 and 0.25 scales target position size by 50%."""
    ranking = _dummy_ranking([("GOOGL", 0.80)])
    prices = {"GOOGL": 100.0}

    # Baseline run (no drift): target allocation = ($10,000 * 0.85) / 3 = $2,833.33
    res_normal = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices=prices,
        ranking_result=ranking,
        psi_value=0.05,
    )
    assert len(res_normal.buy_orders) == 1
    assert round(res_normal.buy_orders[0].allocated_amount, 2) == 2833.33

    # Moderate drift (PSI = 0.15): allocation scaled by 50% -> $1,416.67
    res_drift = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices=prices,
        ranking_result=ranking,
        psi_value=0.15,
    )
    assert len(res_drift.buy_orders) == 1
    assert round(res_drift.buy_orders[0].allocated_amount, 2) == 1416.67


def test_psi_severe_drift_pauses_all_buys():
    """PSI >= 0.25 pauses all new buy signals."""
    ranking = _dummy_ranking([("GOOGL", 0.70), ("AMZN", 0.68)])
    prices = {"GOOGL": 100.0, "AMZN": 150.0}

    res = evaluate_portfolio_risk(
        run_date="2024-01-30",
        current_cash=10000.0,
        current_positions={},
        current_prices=prices,
        ranking_result=ranking,
        psi_value=0.30,
    )
    assert len(res.buy_orders) == 0
    assert len(res.vetoed_orders) == 2
    assert all(d.veto_reason == RiskVetoReason.DRIFT_ALERT_PAUSE_BUYS for d in res.vetoed_orders)
