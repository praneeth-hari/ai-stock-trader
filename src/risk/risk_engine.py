"""
src/risk/risk_engine.py — Phase 8 Risk Management & Portfolio Sizing Engine.

PURPOSE
-------
Enforce the project's capital preservation rules (§1.1, §1.4, §1.8) with unilateral
veto power between ML ranking and trade execution (CLAUDE.md Rule 2).

ARCHITECTURAL PRINCIPLES & EXECUTION ORDER
-------------------------------------------
1. Exit Evaluation First:
   Every held position is audited against current prices and model signals before
   any new buy candidate is considered:
     a. Data validity check: If today's data is invalid/missing, position is
        held untouched with "SKIPPED_INVALID_DATA_HELD_UNCHANGED" (§1.7).
     b. Stop-Loss (-8.0%): Mandatory SELL at market open. Trumps all else.
     c. Take-Profit (+15.0%): Mandatory SELL to lock in gains.
     d. Signal-Exit (P < 0.45): Mandatory SELL on directional breakdown.
   Liquidations free portfolio slots and return settled cash (net 0.2% cost).

2. Regime Filter (SPY < 200-day MA):
   If SPY is below its 200-day moving average (Risk-OFF), all new BUY proposals
   are vetoed immediately with REGIME_FILTER_RISK_OFF. Existing exits execute normally.

3. Candidate Allocation Loop (Uncapped List):
   Iterate through Ranking's uncapped candidate list:
     a. Ticker already held: Skipped (no doubling down, advances to next rank).
     b. 3-position cap reached: Vetoed with MAX_POSITIONS_REACHED.
     c. Target allocation = (total_equity * (1 - cash_reserve)) / max_positions.
        Fixed target per slot (28.33% of equity for 15% reserve and 3 slots).
     d. Cash reserve check: Cash cannot drop below total_equity * cash_reserve (15%).
        If spendable cash < target, allocation clamps to spendable cash.
     e. Min trade size ($10.00): If allocation < $10, vetoed with MIN_TRADE_SIZE_VIOLATION.
     f. Approved BUY decrements open slots and available cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.ranking.ranking import RankedOpportunity, RankingResult, check_market_regime

logger = logging.getLogger(__name__)


# ── Enums & Dataclasses ────────────────────────────────────────────────────────

class RiskVetoReason(str, Enum):
    REGIME_FILTER_RISK_OFF = "REGIME_FILTER_RISK_OFF"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    CASH_RESERVE_VIOLATION = "CASH_RESERVE_VIOLATION"
    MIN_TRADE_SIZE_VIOLATION = "MIN_TRADE_SIZE_VIOLATION"
    BELOW_BUY_BAR = "BELOW_BUY_BAR"
    ALREADY_HELD = "ALREADY_HELD"
    MACRO_CIRCUIT_BREAKER_ACTIVE = "MACRO_CIRCUIT_BREAKER_ACTIVE"
    DRIFT_ALERT_PAUSE_BUYS = "DRIFT_ALERT_PAUSE_BUYS"
    SENTIMENT_VETO = "SENTIMENT_VETO"
    EARNINGS_BLACKOUT = "EARNINGS_BLACKOUT"
    CORRELATION_VETO = "CORRELATION_VETO"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    SKIPPED_INVALID_DATA_HELD_UNCHANGED = "SKIPPED_INVALID_DATA_HELD_UNCHANGED"
    MINIMUM_HOLD_OVERRIDE = "MINIMUM_HOLD_OVERRIDE"
    BAD_DATA_NEAR_STOP_LOSS_ALERT = "BAD_DATA_NEAR_STOP_LOSS_ALERT"
    REVIEW_REQUIRED_DO_NOT_TRADE = "REVIEW_REQUIRED_DO_NOT_TRADE"
    EARNINGS_DAY_HOLD_OVERRIDE = "EARNINGS_DAY_HOLD_OVERRIDE"


@dataclass
class HeldPosition:
    """Represents an active stock position held in the portfolio."""
    ticker: str
    quantity: float
    avg_cost: float
    entry_date: Optional[str] = None
    holding_days: int = 0
    bad_data_days: int = 0
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    highest_price_since_entry: float = 0.0
    trailing_stop_price: float = 0.0

    def update_price(self, current_price: Optional[float]) -> None:
        if self.highest_price_since_entry <= 0:
            self.highest_price_since_entry = self.avg_cost
        trail_pct = getattr(settings, "trailing_stop_pct", 0.08)
        if self.trailing_stop_price <= 0:
            self.trailing_stop_price = round(self.highest_price_since_entry * (1.0 - trail_pct), 4)

        if current_price is not None and not np.isnan(current_price) and current_price > 0:
            self.current_price = float(current_price)
            self.market_value = float(self.quantity * self.current_price)
            self.unrealized_pnl = float(self.market_value - (self.quantity * self.avg_cost))
            self.unrealized_pnl_pct = float((self.current_price / self.avg_cost) - 1.0)

            # Update highest price and trailing stop if new high
            if self.current_price > self.highest_price_since_entry:
                old_high = self.highest_price_since_entry
                old_trail = self.trailing_stop_price
                self.highest_price_since_entry = float(self.current_price)
                new_trail = round(self.highest_price_since_entry * (1.0 - trail_pct), 4)
                if new_trail > self.trailing_stop_price:
                    self.trailing_stop_price = new_trail
                    logger.info(
                        "TRAILING_STOP_UPDATED: %s New high: $%.2f Trail moved: $%.2f (was $%.2f)",
                        self.ticker, self.highest_price_since_entry, self.trailing_stop_price, old_trail,
                    )
        else:
            self.current_price = None
            self.market_value = float(self.quantity * self.avg_cost)
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0


@dataclass
class RiskDecision:
    """Single decision emitted by the Risk Engine (SELL, BUY, VETO, or HOLD)."""
    ticker: str
    action: str                       # "BUY", "SELL", "HOLD", "VETO"
    approved: bool
    quantity: float = 0.0
    price: float = 0.0
    allocated_amount: float = 0.0
    pnl_pct: Optional[float] = None
    reason: Optional[str] = None
    veto_reason: Optional[RiskVetoReason] = None
    details: Optional[str] = None
    confidence_tier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ticker": self.ticker,
            "action": self.action,
            "approved": self.approved,
            "quantity": round(self.quantity, 4),
            "price": round(self.price, 4),
            "allocated_amount": round(self.allocated_amount, 2),
            "pnl_pct": round(self.pnl_pct, 4) if self.pnl_pct is not None else None,
            "reason": self.reason,
            "veto_reason": self.veto_reason.value if self.veto_reason else None,
            "details": self.details,
        }
        if self.confidence_tier:
            d["confidence_tier"] = self.confidence_tier
        return d


@dataclass
class RiskAssessmentResult:
    """Complete portfolio audit and decision container for a run date."""
    date: str
    regime_risk_on: bool
    initial_cash: float
    final_cash: float
    total_equity: float
    positions_before: int
    positions_after: int
    exit_orders: List[RiskDecision] = field(default_factory=list)
    buy_orders: List[RiskDecision] = field(default_factory=list)
    vetoed_orders: List[RiskDecision] = field(default_factory=list)
    held_unchanged: List[RiskDecision] = field(default_factory=list)
    active_buy_bar: float = 0.60
    active_signal_exit: float = 0.45
    volatility_regime: str = "NORMAL_VOLATILITY"
    circuit_breaker_active: bool = False
    psi_value: Optional[float] = None

    @property
    def approved_orders(self) -> List[RiskDecision]:
        """All actionable trade orders (Exits + Approved Buys)."""
        return self.exit_orders + self.buy_orders

    def to_markdown_summary(self) -> str:
        """Render a concise markdown summary of all decisions."""
        regime_str = "RISK-ON (SPY >= 200d MA)" if self.regime_risk_on else "RISK-OFF (SPY < 200d MA)"
        lines = [
            f"### Risk Engine Assessment — {self.date}",
            f"**Regime**: `{regime_str}` | **Vol Regime**: `{self.volatility_regime}` | **Total Equity**: `${self.total_equity:.2f}` | "
            f"**Cash**: `${self.initial_cash:.2f} -> ${self.final_cash:.2f}`",
            f"**Thresholds**: Buy Bar=`{self.active_buy_bar:.2f}` | Signal Exit=`{self.active_signal_exit:.2f}`",
            f"**Positions**: `{self.positions_before}` before -> `{self.positions_after}` after (Cap: {settings.max_positions})\n",
            "| Ticker | Action | Approved? | Qty | Alloc ($) | PnL % | Reason / Details |",
            "|---|---|---|---|---|---|---|",
        ]

        for d in self.exit_orders:
            pnl_str = f"{d.pnl_pct*100.0:+.2f}%" if d.pnl_pct is not None else "N/A"
            lines.append(f"| **{d.ticker}** | `SELL` | **YES** | {d.quantity:.2f} | ${d.allocated_amount:.2f} | {pnl_str} | {d.reason} ({d.details}) |")

        for d in self.held_unchanged:
            lines.append(f"| **{d.ticker}** | `HOLD` | N/A | {d.quantity:.2f} | $0.00 | N/A | {d.reason} |")

        for d in self.buy_orders:
            lines.append(f"| **{d.ticker}** | `BUY` | **YES** | {d.quantity:.2f} | ${d.allocated_amount:.2f} | N/A | {d.reason} ({d.details}) |")

        for d in self.vetoed_orders:
            lines.append(f"| **{d.ticker}** | `VETO` | **NO** | 0.00 | $0.00 | N/A | `{d.veto_reason}`: {d.details} |")

        return "\n".join(lines)


# ── Adaptive Threshold & Circuit Breaker Helpers ──────────────────────────────

def compute_adaptive_thresholds(
    spy_df: Optional[pd.DataFrame] = None,
    vix_value: Optional[float] = None,
) -> Tuple[float, float, str, float]:
    """
    Computes active buy bar and signal exit conviction thresholds based on market volatility (Item 5).
    - Low vol (VIX < 15 / SPY vol < 12% annualized): Buy 0.60, Signal Exit 0.45
    - Normal vol (VIX 15-25 / SPY vol 12-20% annualized): Buy 0.63, Signal Exit 0.47
    - High vol (VIX > 25 / SPY vol > 20% annualized): Buy 0.67, Signal Exit 0.50

    Returns (buy_bar, signal_exit, regime_name, effective_vol).
    """
    if vix_value is not None:
        effective_vol = float(vix_value) / 100.0
    elif spy_df is not None and len(spy_df) >= 5:
        pct = spy_df["close"].pct_change().dropna().tail(20)
        effective_vol = float(pct.std() * np.sqrt(252))
    else:
        effective_vol = 0.16  # default normal volatility (16% annualized)

    low_bound = (settings.vix_low_threshold / 100.0) if settings.vix_low_threshold > 1.0 else settings.vix_low_threshold
    high_bound = (settings.vix_high_threshold / 100.0) if settings.vix_high_threshold > 1.0 else settings.vix_high_threshold

    if effective_vol < low_bound:
        buy_bar = settings.adaptive_buy_bar_low
        sig_exit = settings.adaptive_signal_exit_low
        regime_name = "LOW_VOLATILITY"
    elif effective_vol <= high_bound:
        buy_bar = settings.adaptive_buy_bar_normal
        sig_exit = settings.adaptive_signal_exit_normal
        regime_name = "NORMAL_VOLATILITY"
    else:
        buy_bar = settings.adaptive_buy_bar_high
        sig_exit = settings.adaptive_signal_exit_high
        regime_name = "HIGH_VOLATILITY"

    logger.info(
        "ACTIVE CONVICTION THRESHOLDS: %s (Realized Vol=%.1f%%) -> Buy Bar=%.2f, Signal Exit=%.2f",
        regime_name, effective_vol * 100.0, buy_bar, sig_exit,
    )
    return buy_bar, sig_exit, regime_name, effective_vol


def check_macro_circuit_breaker(
    spy_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, Optional[str], Optional[str], int]:
    """
    Evaluates rolling SPY returns for macro crash circuit breakers (Item 8).
    - SPY 5-day drop > 7% -> 5-day buy halt
    - SPY 20-day drop > 15% -> 10-day buy halt

    Returns (is_active, breaker_type, details_str, halt_days).
    """
    if spy_df is None or len(spy_df) < 6:
        return False, None, None, 0

    closes = spy_df["close"].values
    if len(closes) >= 6:
        ret_5d = (closes[-1] / closes[-6]) - 1.0
        if ret_5d <= -settings.macro_cb_5d_drop_pct:
            return (
                True,
                "MACRO_CIRCUIT_BREAKER_5D",
                f"SPY dropped {ret_5d*100.0:.2f}% over 5 days (threshold: -{settings.macro_cb_5d_drop_pct*100.0:.0f}%)",
                settings.macro_cb_5d_halt_days,
            )

    if len(closes) >= 21:
        ret_20d = (closes[-1] / closes[-21]) - 1.0
        if ret_20d <= -settings.macro_cb_20d_drop_pct:
            return (
                True,
                "MACRO_CIRCUIT_BREAKER_20D",
                f"SPY dropped {ret_20d*100.0:.2f}% over 20 days (threshold: -{settings.macro_cb_20d_drop_pct*100.0:.0f}%)",
                settings.macro_cb_20d_halt_days,
            )

    return False, None, None, 0


# ── Core Risk Engine Logic ─────────────────────────────────────────────────────

def evaluate_portfolio_risk(
    run_date: str,
    current_cash: float,
    current_positions: Dict[str, Dict[str, Any]],
    current_prices: Dict[str, float],
    ranking_result: RankingResult,
    spy_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    invalid_or_missing_tickers: Optional[Set[str]] = None,
    psi_value: Optional[float] = None,
    vix_value: Optional[float] = None,
    circuit_breaker_active: bool = False,
    macro_size_multiplier: float = 1.0,
    earnings_result: Optional[Any] = None,
    price_histories: Optional[Dict[str, pd.DataFrame]] = None,
) -> RiskAssessmentResult:
    """
    Execute full daily risk evaluation and portfolio sizing with Section 2 enhancements:
    - Adaptive conviction thresholds (Item 5)
    - 5-day minimum holding period (Item 6)
    - Tiered bad data handling for held stocks (Item 7)
    - Macro crash circuit breakers (Item 8)
    - PSI drift sizing and pause protocol (Item 9)

    Parameters
    ----------
    run_date : str
        Decision date ('YYYY-MM-DD').
    current_cash : float
        Available settled cash in portfolio.
    current_positions : Dict[str, Dict[str, Any]]
        Dict of currently held positions, e.g.:
        {'AAPL': {'quantity': 10, 'avg_cost': 150.0, 'current_price': 160.0, 'holding_days': 3}}
    current_prices : Dict[str, float]
        Today's valid prices for tickers.
    ranking_result : RankingResult
        Uncapped ranking output from Phase 7 ranking engine.
    spy_df : Optional[pd.DataFrame]
        SPY market data for regime verification and circuit breakers.
    features_df : Optional[pd.DataFrame]
        Features DataFrame for regime verification.
    invalid_or_missing_tickers : Optional[Set[str]]
        Set of ticker symbols whose data failed Phase 2 validation or is missing today.
    psi_value : Optional[float]
        Current Population Stability Index metric from drift monitoring.
    vix_value : Optional[float]
        Current market VIX level (or None to infer from SPY realized volatility).
    circuit_breaker_active : bool
        External circuit breaker override flag.
    """
    invalid_tickers = {t.upper() for t in (invalid_or_missing_tickers or set())}
    regime_on = check_market_regime(spy_df=spy_df, features_df=features_df, as_of_date=run_date)

    # ── Item 5: Compute Adaptive Conviction Thresholds ─────────────────────────
    active_buy_bar, active_signal_exit, vol_regime, realized_vol = compute_adaptive_thresholds(
        spy_df=spy_df, vix_value=vix_value
    )

    # ── Item 8: Evaluate Macro Crash Circuit Breakers ──────────────────────────
    cb_triggered, cb_type, cb_details, cb_halt_days = check_macro_circuit_breaker(spy_df=spy_df)
    is_macro_circuit_breaker = circuit_breaker_active or cb_triggered

    # ── 1. Parse Existing Holdings ─────────────────────────────────────────────
    holdings: Dict[str, HeldPosition] = {}
    for sym, pos_data in current_positions.items():
        ticker_clean = sym.strip().upper()
        qty = float(pos_data.get("quantity", 0.0))
        cost = float(pos_data.get("avg_cost", 0.0))
        entry_d = pos_data.get("entry_date")
        if "holding_days" in pos_data and pos_data["holding_days"] is not None:
            h_days = int(pos_data["holding_days"])
        elif entry_d is not None and run_date is not None:
            try:
                b_days = len(pd.bdate_range(entry_d, run_date)) - 1
                h_days = max(0, b_days)
            except Exception:
                h_days = settings.min_holding_days
        else:
            h_days = settings.min_holding_days
        bad_days = int(pos_data.get("bad_data_days", 0))

        if qty > 0:
            highest_p = float(pos_data.get("highest_price_since_entry", pos_data.get("highest_price", cost)))
            if highest_p <= 0 or np.isnan(highest_p):
                highest_p = cost
            trail_pct = getattr(settings, "trailing_stop_pct", 0.08)
            initial_trail = round(highest_p * (1.0 - trail_pct), 4)
            trailing_stop_p = float(pos_data.get("trailing_stop_price", pos_data.get("trailing_stop", initial_trail)))
            if trailing_stop_p <= 0 or np.isnan(trailing_stop_p):
                trailing_stop_p = initial_trail

            hp = HeldPosition(
                ticker=ticker_clean,
                quantity=qty,
                avg_cost=cost,
                entry_date=entry_d,
                holding_days=h_days,
                bad_data_days=bad_days,
                highest_price_since_entry=highest_p,
                trailing_stop_price=trailing_stop_p,
            )
            # Check price validity
            price = current_prices.get(ticker_clean)
            if price is not None and not np.isnan(price) and price > 0 and (ticker_clean not in invalid_tickers):
                hp.update_price(price)
            else:
                hp.update_price(None)
            holdings[ticker_clean] = hp

    # Initial equity computation
    invested_equity_before = sum(h.market_value for h in holdings.values())
    total_equity = float(current_cash + invested_equity_before)

    working_cash = float(current_cash)
    active_held_tickers: Set[str] = set()
    liquidated_today: Set[str] = set()
    exit_orders: List[RiskDecision] = []
    held_unchanged: List[RiskDecision] = []

    # Map today's model probability for held stocks
    prob_by_ticker: Dict[str, float] = {
        opp.ticker.upper(): opp.probability for opp in ranking_result.ranked_opportunities
    }

    # ── 2. Stage 1: Evaluate Existing Holdings (Exits First) ───────────────────
    for ticker, hp in holdings.items():
        pos_data = current_positions.get(ticker, {})

        # ── Item 7: Tiered Bad Data Handling for Held Stocks ───────────────────
        if hp.current_price is None or (ticker in invalid_tickers):
            bad_data_count = hp.bad_data_days + 1
            last_price = current_prices.get(ticker) or hp.avg_cost
            last_pnl_pct = ((last_price / hp.avg_cost) - 1.0) if hp.avg_cost > 0 else 0.0

            if bad_data_count == 1:
                # Day 1: Hold untouched, log warning
                logger.warning(
                    "TIERED BAD DATA [DAY 1] for %s: Retaining position untouched.",
                    ticker,
                )
                reason = ExitReason.SKIPPED_INVALID_DATA_HELD_UNCHANGED.value
                details = "Day 1 bad/missing data: held safely without changes."

            elif bad_data_count == 2:
                # Day 2: Check last known price against -8% stop loss (within 2% margin -> pnl <= -6.0%)
                if last_pnl_pct <= -0.06:
                    logger.critical(
                        "TIERED BAD DATA [DAY 2 - URGENT ALERT] for %s: Last known price is within 2%% "
                        "of stop-loss floor (PnL=%.2f%%). Human review required!",
                        ticker, last_pnl_pct * 100.0,
                    )
                    reason = ExitReason.BAD_DATA_NEAR_STOP_LOSS_ALERT.value
                    details = (
                        f"Day 2 bad data: last known price is within 2% of -8% stop-loss floor "
                        f"({last_pnl_pct*100.0:+.2f}%). Flagged for urgent human review."
                    )
                else:
                    logger.warning(
                        "TIERED BAD DATA [DAY 2] for %s: Consecutive bad data day. Retaining position untouched.",
                        ticker,
                    )
                    reason = "SKIPPED_INVALID_DATA_DAY_2"
                    details = f"Day 2 bad data: position held untouched (last known PnL: {last_pnl_pct*100.0:+.2f}%)."

            else:
                # Day 3+: Mark position as 'REVIEW REQUIRED — DO NOT TRADE'
                logger.critical(
                    "TIERED BAD DATA [DAY %d - POSITION LOCKED] for %s: Marked as REVIEW REQUIRED — DO NOT TRADE.",
                    bad_data_count, ticker,
                )
                reason = ExitReason.REVIEW_REQUIRED_DO_NOT_TRADE.value
                details = f"{bad_data_count} consecutive days of invalid data. Position locked: REVIEW REQUIRED — DO NOT TRADE."

            decision = RiskDecision(
                ticker=ticker,
                action="HOLD",
                approved=True,
                quantity=hp.quantity,
                price=hp.avg_cost,
                reason=reason,
                details=details,
            )
            held_unchanged.append(decision)
            active_held_tickers.add(ticker)
            continue

        pnl_pct = hp.unrealized_pnl_pct
        current_price = hp.current_price
        p_val = prob_by_ticker.get(ticker, 0.50)  # Default neutral if unranked

        stop_loss_thresh = -abs(settings.stop_loss)
        take_profit_thresh = abs(settings.take_profit)

        # Section 5, Item 2: Earnings Day Hold Protection
        # If earnings are today: HOLD existing position (do not sell into earnings panic — wait for clean data next day)
        if earnings_result and hasattr(earnings_result, "has_hold_protection") and earnings_result.has_hold_protection(ticker):
            logger.info("EARNINGS_DAY_HOLD_OVERRIDE: %s reports earnings today. Holding position through announcement.", ticker)
            active_held_tickers.add(ticker)
            held_unchanged.append(
                RiskDecision(
                    ticker=ticker,
                    action="HOLD",
                    approved=True,
                    quantity=hp.quantity,
                    price=current_price,
                    pnl_pct=pnl_pct,
                    reason=ExitReason.EARNINGS_DAY_HOLD_OVERRIDE.value,
                    details="Position held through earnings announcement day.",
                )
            )
            continue

        # Rule 1b: Trailing Stop-Loss — TRIGGERS when current_price <= trailing_stop_price
        is_trailing_stop_hit = (current_price <= hp.trailing_stop_price) or (pnl_pct <= stop_loss_thresh)
        if is_trailing_stop_hit:
            gross_proceeds = hp.quantity * current_price
            est_cost = gross_proceeds * settings.simulated_cost_per_trade
            net_proceeds = gross_proceeds - est_cost
            working_cash += net_proceeds
            liquidated_today.add(ticker)

            pnl_dollars = (current_price - hp.avg_cost) * hp.quantity
            pnl_dollars_str = f"+${pnl_dollars:,.2f}" if pnl_dollars >= 0 else f"-${abs(pnl_dollars):,.2f}"
            logger.warning(
                "TRAILING_STOP_TRIGGERED: %s Sold at: $%.2f Entry: $%.2f Gain locked: %s (%+.1f%%)",
                ticker, current_price, hp.avg_cost, pnl_dollars_str, pnl_pct * 100.0,
            )
            exit_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="SELL",
                    approved=True,
                    quantity=hp.quantity,
                    price=current_price,
                    allocated_amount=net_proceeds,
                    pnl_pct=pnl_pct,
                    reason=ExitReason.STOP_LOSS.value,
                    details=f"Trailing stop-loss exit triggered: price ${current_price:.2f} <= trailing stop ${hp.trailing_stop_price:.2f}.",
                )
            )
            continue

        # Rule 1c: Take-Profit (+15%) — ALWAYS TRIGGERS regardless of holding period!
        if pnl_pct >= take_profit_thresh:
            gross_proceeds = hp.quantity * current_price
            est_cost = gross_proceeds * settings.simulated_cost_per_trade
            net_proceeds = gross_proceeds - est_cost
            working_cash += net_proceeds
            liquidated_today.add(ticker)

            logger.info(
                "TAKE-PROFIT TRIGGERED: %s up %.2f%% (threshold: %.1f%%). Mandatory SELL to lock in gains.",
                ticker, pnl_pct * 100.0, take_profit_thresh * 100.0,
            )
            exit_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="SELL",
                    approved=True,
                    quantity=hp.quantity,
                    price=current_price,
                    allocated_amount=net_proceeds,
                    pnl_pct=pnl_pct,
                    reason=ExitReason.TAKE_PROFIT.value,
                    details=f"Take-profit exit triggered: PnL {pnl_pct*100.0:+.2f}% >= {take_profit_thresh*100.0:.1f}%.",
                )
            )
            continue

        # Rule 1d & Item 6: Signal-Exit (P < active_signal_exit) with 5-Day Minimum Hold Check
        if p_val < active_signal_exit:
            if hp.holding_days < settings.min_holding_days:
                # 5-day minimum holding rule suppresses transient probability drops!
                logger.info(
                    "MINIMUM_HOLD_OVERRIDE: %s held %d/%d days. Suppressed signal exit (P=%.4f < %.2f) to prevent churn.",
                    ticker, hp.holding_days, settings.min_holding_days, p_val, active_signal_exit,
                )
                active_held_tickers.add(ticker)
                held_unchanged.append(
                    RiskDecision(
                        ticker=ticker,
                        action="HOLD",
                        approved=True,
                        quantity=hp.quantity,
                        price=current_price,
                        pnl_pct=pnl_pct,
                        reason=ExitReason.MINIMUM_HOLD_OVERRIDE.value,
                        details=(
                            f"Held {hp.holding_days}/{settings.min_holding_days} days. "
                            f"Signal exit (P={p_val:.4f} < {active_signal_exit:.2f}) suppressed."
                        ),
                    )
                )
                continue

            gross_proceeds = hp.quantity * current_price
            est_cost = gross_proceeds * settings.simulated_cost_per_trade
            net_proceeds = gross_proceeds - est_cost
            working_cash += net_proceeds
            liquidated_today.add(ticker)

            logger.info(
                "SIGNAL-EXIT TRIGGERED: %s model conviction P=%.4f < %.2f (held %d days). Mandatory SELL.",
                ticker, p_val, active_signal_exit, hp.holding_days,
            )
            exit_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="SELL",
                    approved=True,
                    quantity=hp.quantity,
                    price=current_price,
                    allocated_amount=net_proceeds,
                    pnl_pct=pnl_pct,
                    reason=ExitReason.SIGNAL_EXIT.value,
                    details=f"Signal-exit triggered: model conviction {p_val:.4f} < {active_signal_exit:.2f}.",
                )
            )
            continue

        # Position retained within normal bounds
        active_held_tickers.add(ticker)
        held_unchanged.append(
            RiskDecision(
                ticker=ticker,
                action="HOLD",
                approved=True,
                quantity=hp.quantity,
                price=current_price,
                pnl_pct=pnl_pct,
                reason="HOLDING_ACTIVE",
                details=f"Within risk bounds: PnL {pnl_pct*100.0:+.2f}%, P={p_val:.4f}.",
            )
        )

    # ── 3. Stage 2: Sizing Formula & New Buy Evaluation ────────────────────────
    target_slot_size = ((total_equity * (1.0 - settings.cash_reserve)) / float(settings.max_positions)) * float(macro_size_multiplier)
    cash_reserve_floor = total_equity * settings.cash_reserve

    # Item 9: PSI Drift Protocol
    is_drift_pause = False
    if psi_value is not None:
        if psi_value >= settings.psi_drift_alert_threshold:
            is_drift_pause = True
            logger.critical(
                "DRIFT_ALERT_PAUSE_BUYS: Model PSI=%.4f >= %.2f. Halting all new buy signals until retraining.",
                psi_value, settings.psi_drift_alert_threshold,
            )
        elif psi_value >= settings.psi_monitor_threshold:
            target_slot_size = target_slot_size * settings.psi_monitor_size_multiplier
            logger.warning(
                "MODERATE_DRIFT_WARNING: Model PSI=%.4f in [%.2f, %.2f). Reducing position sizing by 50%% to $%.2f.",
                psi_value, settings.psi_monitor_threshold, settings.psi_drift_alert_threshold, target_slot_size,
            )

    buy_orders: List[RiskDecision] = []
    vetoed_orders: List[RiskDecision] = []

    open_slots = settings.max_positions - len(active_held_tickers)

    candidates = (
        ranking_result.top_buy_candidates
        if ranking_result.top_buy_candidates
        else [opp for opp in ranking_result.ranked_opportunities if opp.probability >= min(settings.buy_bar, active_buy_bar)]
    )

    # Process candidates sequentially
    for opp in candidates:
        ticker = opp.ticker.upper()
        p = opp.probability

        # Check Regime Filter first (§1.8)
        if not regime_on:
            logger.warning(
                "RISK VETO: Buy order for %s (P=%.4f) VETOED by 200-day Market Regime Filter (Risk-OFF).",
                ticker, p,
            )
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.REGIME_FILTER_RISK_OFF,
                    details=f"Market regime is Risk-OFF (SPY < 200d MA). All buys vetoed.",
                )
            )
            continue

        # Item 8: Check Macro Crash Circuit Breaker
        if is_macro_circuit_breaker:
            cb_msg = cb_details or "Macro crash circuit breaker active"
            logger.warning("RISK VETO: Buy order for %s (P=%.4f) VETOED by Macro Crash Circuit Breaker.", ticker, p)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.MACRO_CIRCUIT_BREAKER_ACTIVE,
                    details=f"Macro circuit breaker active: {cb_msg}. New buys halted for {cb_halt_days} days.",
                )
            )
            continue

        # Section 5, Item 1: Sentiment Veto Check
        if getattr(opp, "is_sentiment_veto", False):
            msg = f"SENTIMENT_VETO: {ticker} — negative news detected"
            logger.warning("RISK VETO: %s", msg)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.SENTIMENT_VETO,
                    details=msg,
                )
            )
            continue

        # Section 5, Item 2: Earnings Blackout Check (1-2 days before earnings)
        if getattr(opp, "is_earnings_blackout", False):
            msg = f"EARNINGS_BLACKOUT: {ticker} reports in {getattr(opp, 'days_until_earnings', 'N/A')} days"
            logger.warning("RISK VETO: %s", msg)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.EARNINGS_BLACKOUT,
                    details=msg,
                )
            )
            continue

        # Item 9: Check Model Drift Pause
        if is_drift_pause:
            logger.warning("RISK VETO: Buy order for %s (P=%.4f) VETOED by Drift Alert Pause Protocol.", ticker, p)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.DRIFT_ALERT_PAUSE_BUYS,
                    details=f"Severe model drift (PSI={psi_value:.4f} >= {settings.psi_drift_alert_threshold:.2f}). All new buys paused.",
                )
            )
            continue

        # Item 5: Check adaptive buy bar
        if p < active_buy_bar:
            logger.info("Ticker %s (P=%.4f) below active buy bar %.2f. Skipping candidate.", ticker, p, active_buy_bar)
            continue

        # Check if already held or liquidated today (no doubling down / no same-day rebuy)
        if ticker in active_held_tickers or ticker in liquidated_today:
            logger.info("Ticker %s already held or exited today. Skipping candidate to evaluate next rank.", ticker)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.ALREADY_HELD,
                    details=f"{ticker} is already held or was liquidated on exit today.",
                )
            )
            continue

        # Check Position Cap (Max 3 positions)
        if open_slots <= 0:
            logger.info("RISK VETO: Buy order for %s (P=%.4f) VETOED. Max positions (%d) reached.", ticker, p, settings.max_positions)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.MAX_POSITIONS_REACHED,
                    details=f"Portfolio already at maximum capacity of {settings.max_positions} positions.",
                )
            )
            continue

        # Section 8b: Correlation Filter Check
        from src.risk.correlation import evaluate_candidate_correlation
        corr_eval = evaluate_candidate_correlation(
            candidate_ticker=ticker,
            held_tickers=list(active_held_tickers),
            price_histories=price_histories,
        )
        if not corr_eval.allowed:
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    veto_reason=RiskVetoReason.CORRELATION_VETO,
                    details=corr_eval.details,
                )
            )
            continue

        # Sizing & Cash Reserve Floor Check (with Earnings Caution 50% sizing if 3-5 days)
        cand_target_size = target_slot_size
        if getattr(opp, "earnings_status", "OK") == "CAUTION" or (
            getattr(opp, "days_until_earnings", None) is not None
            and settings.earnings_caution_min_days <= opp.days_until_earnings <= settings.earnings_caution_max_days
        ):
            cand_target_size *= settings.earnings_caution_size_multiplier
            logger.info("EARNINGS_CAUTION: %s allocation reduced by 50%% to $%.2f due to upcoming earnings",
                        ticker, cand_target_size)

        # Confidence Sizing (Section 8 Item 2)
        if p < settings.confidence_half_position_max:
            conf_multiplier = 0.50
            confidence_tier = "HALF"
            logger.info("HALF_POSITION: score just above bar")
        elif p < settings.confidence_three_quarter_max:
            conf_multiplier = 0.75
            confidence_tier = "THREE_QUARTER"
            logger.info("THREE_QUARTER_POSITION: moderate confidence")
        else:
            conf_multiplier = 1.00
            confidence_tier = "FULL"
            logger.info("FULL_POSITION: high confidence")

        cand_target_size *= conf_multiplier

        spendable_cash = max(0.0, working_cash - cash_reserve_floor)

        # Cash Floor Breach Check: If target size exceeds spendable cash, skip trade entirely
        if spendable_cash < cand_target_size:
            logger.warning(
                "RISK VETO: Buy order for %s (P=%.4f, tier=%s) VETOED. Position size ($%.2f) > spendable cash ($%.2f).",
                ticker, p, confidence_tier, cand_target_size, spendable_cash,
            )
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    allocated_amount=spendable_cash,
                    confidence_tier=confidence_tier,
                    veto_reason=RiskVetoReason.CASH_RESERVE_VIOLATION,
                    details=f"Position size ${cand_target_size:.2f} ({confidence_tier}) would breach cash reserve floor.",
                )
            )
            continue

        allocated_amount = cand_target_size

        # Minimum Trade Size Check ($10.00)
        if allocated_amount < settings.min_trade_size:
            logger.warning(
                "RISK VETO: Buy order for %s (P=%.4f) VETOED. Available allocation ($%.2f) < min trade size ($%.2f).",
                ticker, p, allocated_amount, settings.min_trade_size,
            )
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    allocated_amount=allocated_amount,
                    confidence_tier=confidence_tier,
                    veto_reason=RiskVetoReason.MIN_TRADE_SIZE_VIOLATION,
                    details=f"Allocatable cash ${allocated_amount:.2f} below min trade size ${settings.min_trade_size:.2f}.",
                )
            )
            continue

        # Resolve price and share count
        price = current_prices.get(ticker)
        if price is None or price <= 0 or np.isnan(price):
            logger.warning("RISK VETO: Buy order for %s VETOED. Missing or non-positive price ($%s).", ticker, price)
            vetoed_orders.append(
                RiskDecision(
                    ticker=ticker,
                    action="VETO",
                    approved=False,
                    confidence_tier=confidence_tier,
                    veto_reason=RiskVetoReason.CASH_RESERVE_VIOLATION,
                    details=f"Invalid pricing data for {ticker}.",
                )
            )
            continue

        # Fraction of shares or whole shares
        shares = allocated_amount / price

        # Deduct cash and fill slot
        working_cash -= allocated_amount
        open_slots -= 1
        active_held_tickers.add(ticker)

        logger.info(
            "RISK APPROVED: Buy %s (P=%.4f, Tier=%s) -> Alloc: $%.2f, Shares: %.4f @ $%.2f.",
            ticker, p, confidence_tier, allocated_amount, shares, price,
        )
        buy_orders.append(
            RiskDecision(
                ticker=ticker,
                action="BUY",
                approved=True,
                quantity=shares,
                price=price,
                allocated_amount=allocated_amount,
                confidence_tier=confidence_tier,
                reason="QUALIFIED_CONVICTION_BUY",
                details=f"Ranked opportunity (P={p:.4f} >= {active_buy_bar:.2f}, tier={confidence_tier}). Position sized accordingly.",
            )
        )

    positions_after = len(active_held_tickers)

    return RiskAssessmentResult(
        date=run_date,
        regime_risk_on=regime_on,
        initial_cash=float(current_cash),
        final_cash=round(float(working_cash), 2),
        total_equity=round(float(total_equity), 2),
        positions_before=len(holdings),
        positions_after=positions_after,
        exit_orders=exit_orders,
        buy_orders=buy_orders,
        vetoed_orders=vetoed_orders,
        held_unchanged=held_unchanged,
        active_buy_bar=active_buy_bar,
        active_signal_exit=active_signal_exit,
        volatility_regime=vol_regime,
        circuit_breaker_active=is_macro_circuit_breaker,
        psi_value=psi_value,
    )
