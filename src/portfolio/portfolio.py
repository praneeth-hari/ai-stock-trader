"""
src/portfolio/portfolio.py — Phase 9 Portfolio Allocation Engine.

Translates approved Risk Decisions (actions, allocations, stop-losses) into
concrete, broker-executable Order Specifications with exact fractional share
quantities, strict floor-rounding, and simulated 0.2% transaction fee accounting.

FEE & CASH CONVENTION (§1.6):
  1. BUY Order Outflow Convention (Fee-Inclusive):
     The Risk Engine's allocated_amount represents the MAXIMUM total cash outflow,
     INCLUSIVE of the 0.2% simulated transaction cost (settings.simulated_cost_per_trade = 0.002).
     Capital dedicated to shares: S = allocated_amount / (1 + fee_rate)
     Raw fractional shares: shares_raw = S / current_price
     4-decimal floor rounding: shares = floor(shares_raw * 10000) / 10000
     Gross share value: gross_value = shares * current_price
     Actual fee: fee = gross_value * fee_rate
     Total cash outflow = gross_value + fee <= allocated_amount.
     This ensures a buy NEVER exceeds the allocated budget or breaches the statutory 15% cash reserve.

  2. SELL Order Inflow Convention (Net Proceeds):
     When a position is liquidated (Stop-Loss, Take-Profit, or Signal-Exit), all held
     shares are sold at current_price.
     Gross proceeds: gross_proceeds = shares * current_price
     Actual fee: fee = gross_proceeds * fee_rate
     Net cash inflow returned to portfolio cash = gross_proceeds - fee = gross_proceeds * (1 - fee_rate)
     (e.g., gross_proceeds * 0.998).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.risk.risk_engine import ExitReason, RiskAssessmentResult, RiskDecision

logger = logging.getLogger(__name__)


@dataclass
class OrderSpec:
    """
    Standardized, broker-executable order specification for Phase 11 Paper Broker.
    """
    date: str
    ticker: str
    action: str              # "BUY" or "SELL"
    order_type: str          # "MARKET"
    shares: float            # Fractional shares rounded to 4 decimals (floor)
    reference_price: float   # Benchmark/close price at decision time
    gross_value: float       # shares * reference_price
    estimated_fee: float     # gross_value * fee_rate (0.2%)
    net_amount: float        # Cash impact: outflow for BUY (+), inflow for SELL (+)
    reason: str              # Decision reason (e.g., STOP_LOSS, QUALIFIED_CONVICTION_BUY)
    adv: Optional[float] = None  # Optional 20-day Average Daily Volume for slippage modeling
    confidence_tier: Optional[str] = None  # HALF, THREE_QUARTER, FULL

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "date": self.date,
            "ticker": self.ticker,
            "action": self.action,
            "order_type": self.order_type,
            "shares": self.shares,
            "reference_price": round(self.reference_price, 4),
            "gross_value": round(self.gross_value, 4),
            "estimated_fee": round(self.estimated_fee, 4),
            "net_amount": round(self.net_amount, 4),
            "reason": self.reason,
        }
        if self.adv is not None:
            d["adv"] = round(self.adv, 2)
        if self.confidence_tier is not None:
            d["confidence_tier"] = self.confidence_tier
        return d


@dataclass
class PortfolioAllocationResult:
    """
    Comprehensive result of the Portfolio Allocation step.
    Contains concrete orders and pro-forma post-allocation portfolio state.
    """
    date: str
    starting_cash: float
    projected_cash: float
    starting_equity: float
    projected_equity: float
    orders: List[OrderSpec] = field(default_factory=list)
    projected_positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cash_reserve_floor: float = 0.0
    cash_reserve_maintained: bool = True

    def to_markdown_summary(self) -> str:
        """Render a readable markdown summary of orders and projected portfolio state."""
        lines = [
            f"### Portfolio Allocation Summary — {self.date}",
            f"**Starting Cash**: `${self.starting_cash:.2f}` | **Projected Cash**: `${self.projected_cash:.2f}`",
            f"**Starting Equity**: `${self.starting_equity:.2f}` | **Projected Equity**: `${self.projected_equity:.2f}`",
            f"**Cash Reserve Floor (15%)**: `${self.cash_reserve_floor:.2f}` | **Reserve Maintained**: `{'YES' if self.cash_reserve_maintained else 'NO'}`",
            f"**Orders Generated**: `{len(self.orders)}`\n",
        ]

        if self.orders:
            lines.append("| Ticker | Action | Type | Shares | Ref Price | Gross ($) | Fee (0.2%) | Net Cash Impact | Reason |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for o in self.orders:
                sign = "-" if o.action == "BUY" else "+"
                lines.append(
                    f"| **{o.ticker}** | `{o.action}` | {o.order_type} | {o.shares:.4f} | "
                    f"${o.reference_price:.2f} | ${o.gross_value:.2f} | ${o.estimated_fee:.2f} | "
                    f"{sign}${o.net_amount:.2f} | {o.reason} |"
                )
        else:
            lines.append("_No orders generated. Portfolio remains unchanged._")

        lines.append("\n**Projected Positions**:")
        if self.projected_positions:
            lines.append("| Ticker | Shares | Avg Cost | Current Price | Market Value ($) |")
            lines.append("|---|---|---|---|---|")
            for t, pos in self.projected_positions.items():
                mv = pos["quantity"] * pos["current_price"]
                lines.append(
                    f"| **{t}** | {pos['quantity']:.4f} | ${pos['avg_cost']:.2f} | "
                    f"${pos['current_price']:.2f} | ${mv:.2f} |"
                )
        else:
            lines.append("_100% Cash._")

        return "\n".join(lines)


def calculate_fractional_shares(
    dollar_amount: float,
    current_price: float,
    fee_rate: float = settings.simulated_cost_per_trade,
) -> tuple[float, float, float]:
    """
    Calculate fractional shares using 4-decimal floor rounding.

    Returns:
        (shares, gross_value, estimated_fee)
        where gross_value + estimated_fee <= dollar_amount.
    """
    if current_price <= 0 or dollar_amount <= 0:
        return 0.0, 0.0, 0.0

    # Max budget for shares excluding fee: S = dollar_amount / (1 + fee_rate)
    max_share_spend = dollar_amount / (1.0 + fee_rate)
    raw_shares = max_share_spend / current_price

    # Floor to 4 decimal places: math.floor(x * 10000) / 10000
    shares = math.floor(raw_shares * 10000.0) / 10000.0
    gross_value = round(shares * current_price, 4)
    estimated_fee = round(gross_value * fee_rate, 4)

    # In the rare floating edge case where gross + fee > dollar_amount by 1e-4, decrement by 1 least-significant unit
    if (gross_value + estimated_fee) > (dollar_amount + 1e-6) and shares > 0.0001:
        shares = round(shares - 0.0001, 4)
        gross_value = round(shares * current_price, 4)
        estimated_fee = round(gross_value * fee_rate, 4)

    return shares, gross_value, estimated_fee


def allocate_portfolio(
    risk_assessment: RiskAssessmentResult,
    current_cash: float,
    current_positions: Dict[str, Dict[str, float]],
    current_prices: Dict[str, float],
    fee_rate: float = settings.simulated_cost_per_trade,
) -> PortfolioAllocationResult:
    """
    Converts approved Risk decisions into concrete OrderSpec objects and reconciles pro-forma balances.

    Processing sequence:
      1. Liquidations (SELL): Sells entire held position. Cash increases by net proceeds (gross - fee).
      2. Purchases (BUY): Allocated cash purchases fractional shares. Cash decreases by gross + fee.
      3. Active holdings (HOLD/SKIPPED): Carried forward untouched.
      4. Integrity check: Verifies ending cash >= statutory 15% cash reserve floor.
    """
    date = getattr(risk_assessment, "date", getattr(risk_assessment, "run_date", ""))
    starting_cash = float(current_cash)
    starting_equity = float(risk_assessment.total_equity)

    orders: List[OrderSpec] = []
    projected_positions: Dict[str, Dict[str, float]] = {}
    cash = starting_cash

    # Copy current positions as the base
    for ticker, pos in current_positions.items():
        qty = float(pos.get("quantity", 0.0))
        if qty > 0:
            price = current_prices.get(ticker, float(pos.get("avg_cost", 0.0)))
            projected_positions[ticker] = {
                "quantity": qty,
                "avg_cost": float(pos.get("avg_cost", price)),
                "current_price": price,
            }

    # Step 1: Process SELL decisions (from exit_orders)
    exit_decisions = getattr(risk_assessment, "exit_orders", [d for d in getattr(risk_assessment, "decisions", []) if d.action == "SELL"])
    for decision in exit_decisions:
        if decision.approved:
            ticker = decision.ticker
            if ticker in projected_positions:
                held_qty = projected_positions[ticker]["quantity"]
                price = current_prices.get(ticker, projected_positions[ticker]["current_price"])
                gross_proceeds = round(held_qty * price, 4)
                fee = round(gross_proceeds * fee_rate, 4)
                net_inflow = round(gross_proceeds - fee, 4)

                order = OrderSpec(
                    date=date,
                    ticker=ticker,
                    action="SELL",
                    order_type="MARKET",
                    shares=held_qty,
                    reference_price=price,
                    gross_value=gross_proceeds,
                    estimated_fee=fee,
                    net_amount=net_inflow,
                    reason=decision.reason or "SELL",
                )
                orders.append(order)
                cash = round(cash + net_inflow, 4)
                del projected_positions[ticker]
                logger.info(
                    f"Order generated: SELL {held_qty:.4f} {ticker} @ ${price:.2f}. "
                    f"Gross: ${gross_proceeds:.2f}, Fee: ${fee:.2f}, Net Cash Inflow: +${net_inflow:.2f} ({decision.reason})"
                )

    # Step 2: Process BUY decisions (from buy_orders)
    buy_decisions = getattr(risk_assessment, "buy_orders", [d for d in getattr(risk_assessment, "decisions", []) if d.action == "BUY"])
    for decision in buy_decisions:
        if decision.approved:
            ticker = decision.ticker
            price = current_prices.get(ticker, 0.0)
            alloc = decision.allocated_amount

            if price <= 0 or alloc < settings.min_trade_size:
                logger.warning(f"Skipping buy order for {ticker}: invalid price (${price:.2f}) or alloc (${alloc:.2f})")
                continue

            # Enforce 15% cash reserve
            investable_cash = cash * (1 - settings.cash_reserve)
            cost = alloc
            if cost > investable_cash:
                logger.warning(
                    "Cash reserve breach prevented — cost=%.2f "
                    "exceeds investable cash=%.2f (15%% reserve enforced)",
                    cost, investable_cash,
                )
                continue

            shares, gross_value, fee = calculate_fractional_shares(alloc, price, fee_rate)
            total_outflow = round(gross_value + fee, 4)

            if shares <= 0:
                logger.warning(f"Calculated 0 shares for {ticker} with allocation ${alloc:.2f} @ ${price:.2f}")
                continue

            conf_tier = getattr(decision, "confidence_tier", None)
            order = OrderSpec(
                date=date,
                ticker=ticker,
                action="BUY",
                order_type="MARKET",
                shares=shares,
                reference_price=price,
                gross_value=gross_value,
                estimated_fee=fee,
                net_amount=total_outflow,
                reason=decision.reason or "BUY",
                confidence_tier=conf_tier,
            )
            orders.append(order)
            cash = round(cash - total_outflow, 4)

            # Update positions
            projected_positions[ticker] = {
                "quantity": shares,
                "avg_cost": price,
                "current_price": price,
                "confidence_tier": conf_tier or "FULL",
            }
            logger.info(
                f"Order generated: BUY {shares:.4f} {ticker} @ ${price:.2f}. "
                f"Gross: ${gross_value:.2f}, Fee: ${fee:.2f}, Total Outflow: -${total_outflow:.2f} <= Alloc: ${alloc:.2f}"
            )

    # Step 3: Compute final projected equity and verify cash reserve
    portfolio_mv = sum(pos["quantity"] * pos["current_price"] for pos in projected_positions.values())
    projected_equity = round(cash + portfolio_mv, 4)
    cash_reserve_floor = round(projected_equity * settings.cash_reserve, 4)
    cash_reserve_maintained = cash >= (cash_reserve_floor - 0.01)

    return PortfolioAllocationResult(
        date=date,
        starting_cash=round(starting_cash, 4),
        projected_cash=round(cash, 4),
        starting_equity=round(starting_equity, 4),
        projected_equity=round(projected_equity, 4),
        orders=orders,
        projected_positions=projected_positions,
        cash_reserve_floor=cash_reserve_floor,
        cash_reserve_maintained=cash_reserve_maintained,
    )


def get_strategy_dna() -> str:
    from config.settings import settings
    import json
    from pathlib import Path
    meta_path = settings.data_models_dir / "active_model_metadata.json"
    model_type = "LR"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            model_type = "LR" if "logistic" in meta.get("model_type", "").lower() else "HGB"
        except Exception:
            pass
    return (
        f"DNA:{model_type}"
        f"|BUY@{settings.buy_bar}"
        f"|EXIT@{settings.signal_exit}"
        f"|SL{int(settings.stop_loss*100)}%"
        f"|TP{int(settings.take_profit*100)}%"
        f"|MAX{settings.max_positions}"
        f"|COST{settings.simulated_cost_per_trade*100}%"
        f"|TRAIL{int(settings.trailing_stop_pct*100)}%"
    )
