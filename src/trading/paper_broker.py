"""
src/trading/paper_broker.py — Phase 11 Paper Broker.

PURPOSE
-------
Simulates broker-side execution for the paper-trading loop (§1.9 — no real money).
The Paper Broker is the ONLY place that:
  1. Holds the cash balance and current open positions in memory (loaded from DB on start).
  2. Executes OrderSpec instructions from the Portfolio Allocation Engine.
  3. Applies 0.2% simulated transaction cost on every fill (§1.6).
  4. Persists every fill, trade record, and daily portfolio snapshot to SQLite via repository.py.

FEE CONVENTIONS (matching Phase 9 / Portfolio Engine exactly):
  BUY:  Total cash outflow = gross_spend + fee (fee = shares * fill_price * 0.002).
        allocated_amount from risk engine is the budget cap; gross_spend + fee must not exceed it.
  SELL: Net cash inflow = gross_proceeds - fee (fee = shares * fill_price * 0.002).
        PnL is computed net of BOTH entry fee (from original buy) and exit fee.

ARCHITECTURE RULE:
  This module NEVER makes trading decisions. It only executes what the Portfolio Allocation
  Engine (Phase 9) has already authorised. The Risk Engine's veto authority is upheld at
  the previous layer — the broker executes blindly and records faithfully.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.db import repository
from src.portfolio.portfolio import OrderSpec

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Represents a single open paper position."""
    ticker: str
    quantity: float          # Fractional shares held (4-decimal precision)
    entry_price: float       # Average fill price at entry (used for PnL calc)
    entry_date: str          # 'YYYY-MM-DD' of the buy fill
    entry_fee: float         # Entry commission paid (gross * 0.002)
    confidence_tier: Optional[str] = "FULL"  # HALF, THREE_QUARTER, FULL
    highest_price_since_entry: float = 0.0
    trailing_stop_price: float = 0.0

    def __post_init__(self):
        if self.highest_price_since_entry <= 0:
            self.highest_price_since_entry = self.entry_price
        if self.trailing_stop_price <= 0:
            trail_pct = getattr(settings, "trailing_stop_pct", 0.08)
            self.trailing_stop_price = round(self.highest_price_since_entry * (1.0 - trail_pct), 4)

    def update_high_and_trailing_stop(self, current_price: float) -> bool:
        """Update highest price since entry and trailing stop if new high reached."""
        if current_price > self.highest_price_since_entry:
            old_high = self.highest_price_since_entry
            old_trail = self.trailing_stop_price
            self.highest_price_since_entry = float(current_price)
            trail_pct = getattr(settings, "trailing_stop_pct", 0.08)
            new_trail = round(self.highest_price_since_entry * (1.0 - trail_pct), 4)
            if new_trail > self.trailing_stop_price:
                self.trailing_stop_price = new_trail
                logger.info(
                    "TRAILING_STOP_UPDATED: %s New high: $%.2f Trail moved: $%.2f (was $%.2f)",
                    self.ticker, self.highest_price_since_entry, self.trailing_stop_price, old_trail,
                )
                return True
        return False


@dataclass
class FillResult:
    """
    Record of a completed broker fill.

    For BUYs:  net_pnl = 0.0 (unrealised; PnL is booked on the matching sell).
    For SELLs: net_pnl = gross_proceeds - exit_fee - (entry_cost + entry_fee).
    """
    run_date: str
    ticker: str
    action: str            # 'BUY' or 'SELL'
    shares: float
    fill_price: float
    gross_value: float     # shares * fill_price
    fee: float             # 0.2% of gross_value
    cash_impact: float     # Cash change: negative for BUY, positive for SELL
    net_pnl: float         # 0.0 for buys; realised P&L (after both fees) for sells
    reason: str
    slippage_cost: float = 0.0   # Market impact slippage cost ($)
    effective_price: float = 0.0 # Price adjusted for slippage


def compute_slippage(
    shares: float,
    base_price: float,
    action: str = "BUY",
    adv: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Simulate execution slippage based on trade size vs Average Daily Volume (ADV).

    Tiers (§ Item 15):
      - trade size / ADV <= 1%  (<= 0.01): 0.05% (0.0005)
      - 1% < trade size / ADV <= 5% (0.01 - 0.05): 0.15% (0.0015)
      - trade size / ADV > 5%   (> 0.05):  0.30% (0.0030)

    If ADV is None or <= 0, defaults to low-tier (0.05%).

    Returns:
      (effective_price, slippage_rate, slippage_cost)
      where:
        effective_price = base_price * (1 + slippage_rate) for BUY
                        = base_price * (1 - slippage_rate) for SELL
        slippage_cost   = round(base_price * slippage_rate * shares, 4)
    """
    if shares <= 0 or base_price <= 0:
        return base_price, 0.0, 0.0

    if adv is not None and adv > 0:
        ratio = shares / adv
        if ratio <= 0.01:
            rate = settings.slippage_tier_low_pct   # 0.0005 (0.05%)
        elif ratio <= 0.05:
            rate = settings.slippage_tier_mid_pct   # 0.0015 (0.15%)
        else:
            rate = settings.slippage_tier_high_pct  # 0.0030 (0.30%)
    else:
        rate = settings.slippage_tier_low_pct       # 0.0005 (0.05%)

    slippage_cost = round(base_price * rate * shares, 4)
    if action.upper() == "BUY":
        effective_price = round(base_price * (1.0 + rate), 4)
    else:
        effective_price = round(base_price * (1.0 - rate), 4)

    return effective_price, rate, slippage_cost


# ── PaperBroker ───────────────────────────────────────────────────────────────

class PaperBroker:
    """
    Simulated paper broker that tracks cash and positions, executes fills,
    and persists all activity to the project database.

    Lifecycle
    ---------
    1. Instantiate: ``broker = PaperBroker()``
    2. Call ``broker.load_state()`` once at the start of each daily run
       to restore the most recent portfolio snapshot from DB.
    3. Call ``broker.execute_order(order, fill_price, run_date)`` for each
       OrderSpec returned by the Portfolio Allocation Engine.
    4. After all orders are executed, call ``broker.record_snapshot(run_date, prices)``
       to persist the end-of-day mark-to-market snapshot.
    """

    def __init__(self) -> None:
        # Initialised to defaults; load_state() overwrites from DB.
        self.cash: float = settings.initial_capital
        self.positions: Dict[str, Position] = {}
        self.total_slippage_cost: float = 0.0
        self._state_loaded: bool = False

    # ── State management ──────────────────────────────────────────────────────

    def load_state(self) -> None:
        """
        Restore the latest portfolio snapshot from DB.

        If no snapshot exists (first ever run), defaults to:
          - Cash = settings.initial_capital
          - Positions = {} (empty)
          - total_slippage_cost = 0.0
        Positions are reconstructed from the snapshot's JSON positions blob,
        which was written by a prior record_snapshot() call.
        """
        repository.create_all_tables()
        snap = repository.get_latest_portfolio_snapshot()

        if snap is None:
            self.cash = round(settings.initial_capital, 4)
            self.positions = {}
            self.total_slippage_cost = 0.0
            logger.info(
                "PaperBroker: No prior snapshot found. Initialising at $%.2f capital.",
                self.cash,
            )
        else:
            self.cash = round(float(snap["cash"]), 4)
            self.total_slippage_cost = round(float(snap.get("total_slippage_cost", 0.0) or 0.0), 4)
            raw_positions = snap.get("positions") or {}
            self.positions = {}
            for ticker, pos_data in raw_positions.items():
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=float(pos_data["quantity"]),
                    entry_price=float(pos_data["entry_price"]),
                    entry_date=str(pos_data["entry_date"]),
                    entry_fee=float(pos_data.get("entry_fee", 0.0)),
                )
            logger.info(
                "PaperBroker: Restored snapshot from %s. Cash=%.4f, Slippage=%.4f, Positions=%s.",
                snap["run_date"],
                self.cash,
                self.total_slippage_cost,
                list(self.positions.keys()),
            )

        self._state_loaded = True

    @property
    def total_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Return total equity: cash + mark-to-market position value.
        Uses entry_price as a fallback when current_prices is not supplied
        (e.g., mid-day before close prices are known).
        """
        position_value = sum(
            p.quantity * p.entry_price for p in self.positions.values()
        )
        return round(self.cash + position_value, 4)

    def mark_to_market(self, current_prices: Dict[str, float]) -> float:
        """
        Compute total equity using live (or end-of-day close) prices.
        Returns total equity (cash + positions at current_prices).
        """
        position_value = sum(
            p.quantity * current_prices.get(t, p.entry_price)
            for t, p in self.positions.items()
        )
        return round(self.cash + position_value, 4)

    # ── Order Execution ───────────────────────────────────────────────────────

    def execute_order(
        self,
        order: OrderSpec,
        fill_price: float,
        run_date: str,
        adv: Optional[float] = None,
        simulate_slippage: Optional[bool] = None,
    ) -> Optional[FillResult]:
        """
        Execute a single OrderSpec at the given fill_price.

        Fills are at market-open prices (the Lag Rule: orders generated at T-1 close
        are filled at T open). The caller supplies fill_price; this method does not
        look up prices itself.

        Parameters
        ----------
        order : OrderSpec
            The order from the Portfolio Allocation Engine (Phase 9).
        fill_price : float
            The actual fill price (typically T's market open).
        run_date : str
            The execution date 'YYYY-MM-DD' (Day T).
        adv : float, optional
            20-day Average Daily Volume for trade size vs ADV slippage modeling.
        simulate_slippage : bool, optional
            Whether to enforce slippage calculation even if ADV is None (defaults to low tier).

        Returns
        -------
        FillResult on success, None if the order was rejected (logged as WARNING).
        """
        if not self._state_loaded:
            raise RuntimeError("PaperBroker.load_state() must be called before execute_order().")

        ticker = order.ticker
        fee_rate = settings.simulated_cost_per_trade
        order_adv = adv if adv is not None else getattr(order, "adv", None)

        if order.action == "BUY":
            return self._execute_buy(order, fill_price, fee_rate, run_date, adv=order_adv, simulate_slippage=simulate_slippage)
        elif order.action == "SELL":
            return self._execute_sell(order, fill_price, fee_rate, run_date, adv=order_adv, simulate_slippage=simulate_slippage)
        else:
            logger.warning("PaperBroker: Unknown order action '%s' for %s. Skipping.", order.action, ticker)
            return None

    def _execute_buy(
        self,
        order: OrderSpec,
        fill_price: float,
        fee_rate: float,
        run_date: str,
        adv: Optional[float] = None,
        simulate_slippage: Optional[bool] = None,
    ) -> Optional[FillResult]:
        """
        Execute a BUY order.

        Uses the shares quantity already calculated by the Portfolio Allocation Engine
        (Phase 9), which guarantees 4-decimal floor rounding and fee-inclusive sizing.

        Total outflow = shares * fill_price + (shares * fill_price * fee_rate)
        Cash guard: rejects if insufficient cash (defensive layer, allocation engine
        should have already checked this).
        """
        ticker = order.ticker

        if ticker in self.positions:
            logger.warning(
                "PaperBroker BUY REJECTED: %s already held. No doubling down.", ticker,
            )
            return None

        shares = order.shares  # 4-decimal floor-rounded by Phase 9
        if shares <= 0:
            logger.warning("PaperBroker BUY REJECTED: %s zero shares. Skipping.", ticker)
            return None

        # Compute slippage if ADV is supplied or simulation explicitly requested
        if adv is not None or simulate_slippage:
            effective_price, slip_rate, slippage_cost = compute_slippage(shares, fill_price, "BUY", adv=adv)
        else:
            effective_price = fill_price
            slippage_cost = 0.0

        gross_spend = round(shares * effective_price, 4)
        fee = round(gross_spend * fee_rate, 4)
        total_outflow = round(gross_spend + fee, 4)

        if self.cash < total_outflow:
            logger.warning(
                "PaperBroker BUY REJECTED: %s — insufficient cash. Need $%.4f, have $%.4f.",
                ticker, total_outflow, self.cash,
            )
            return None

        # Deduct cash & accumulate slippage
        self.cash = round(self.cash - total_outflow, 4)
        self.total_slippage_cost = round(self.total_slippage_cost + slippage_cost, 4)

        # Record position
        conf_tier = getattr(order, "confidence_tier", None) or "FULL"
        trail_pct = getattr(settings, "trailing_stop_pct", 0.08)
        initial_high = effective_price
        initial_trail = round(initial_high * (1.0 - trail_pct), 4)
        self.positions[ticker] = Position(
            ticker=ticker,
            quantity=shares,
            entry_price=effective_price,
            entry_date=run_date,
            entry_fee=fee,
            confidence_tier=conf_tier,
            highest_price_since_entry=initial_high,
            trailing_stop_price=initial_trail,
        )

        logger.info(
            "PaperBroker BUY EXECUTED: %s %.4f shares @ $%.4f (base: $%.4f, slip: $%.4f). Gross: $%.4f, Fee: $%.4f, Outflow: $%.4f. Cash remaining: $%.4f.",
            ticker, shares, effective_price, fill_price, slippage_cost, gross_spend, fee, total_outflow, self.cash,
        )

        # Persist to DB
        repository.save_order(
            run_date=run_date,
            ticker=ticker,
            action="BUY",
            quantity=shares,
            price=fill_price,
            reason=order.reason,
        )
        repository.save_trade(
            run_date=run_date,
            ticker=ticker,
            action="BUY",
            quantity=shares,
            fill_price=effective_price,
            cost=fee,
            net_pnl=0.0,  # PnL is booked on matching SELL
            slippage_cost=slippage_cost,
        )
        repository.log_event(
            level="INFO",
            component="paper_broker",
            message=f"BUY {ticker} {shares:.4f} @ ${effective_price:.4f} (slip: ${slippage_cost:.4f})",
            details={
                "run_date": run_date,
                "ticker": ticker,
                "shares": shares,
                "base_price": fill_price,
                "effective_price": effective_price,
                "slippage_cost": slippage_cost,
                "gross_spend": gross_spend,
                "fee": fee,
                "total_outflow": total_outflow,
                "cash_after": self.cash,
                "reason": order.reason,
            },
        )

        return FillResult(
            run_date=run_date,
            ticker=ticker,
            action="BUY",
            shares=shares,
            fill_price=effective_price,
            gross_value=gross_spend,
            fee=fee,
            cash_impact=-total_outflow,
            net_pnl=0.0,
            reason=order.reason,
            slippage_cost=slippage_cost,
            effective_price=effective_price,
        )

    def _execute_sell(
        self,
        order: OrderSpec,
        fill_price: float,
        fee_rate: float,
        run_date: str,
        adv: Optional[float] = None,
        simulate_slippage: Optional[bool] = None,
    ) -> Optional[FillResult]:
        """
        Execute a SELL order.

        Sells the full position held. Net cash inflow = gross_proceeds * (1 - fee_rate).
        Realised net PnL = net_inflow - (entry_cost + entry_fee).
        """
        ticker = order.ticker

        if ticker not in self.positions:
            logger.warning(
                "PaperBroker SELL REJECTED: %s not in positions. Cannot sell what we don't hold.",
                ticker,
            )
            return None

        pos = self.positions[ticker]
        shares = pos.quantity

        if adv is not None or simulate_slippage:
            effective_price, slip_rate, slippage_cost = compute_slippage(shares, fill_price, "SELL", adv=adv)
        else:
            effective_price = fill_price
            slippage_cost = 0.0

        gross_proceeds = round(shares * effective_price, 4)
        fee = round(gross_proceeds * fee_rate, 4)
        net_inflow = round(gross_proceeds - fee, 4)

        # PnL: net_inflow minus the original cash outflow (entry cost + entry fee)
        entry_cost = round(shares * pos.entry_price, 4)
        net_pnl = round(net_inflow - entry_cost - pos.entry_fee, 4)

        # Return cash & accumulate slippage
        self.cash = round(self.cash + net_inflow, 4)
        self.total_slippage_cost = round(self.total_slippage_cost + slippage_cost, 4)
        del self.positions[ticker]

        logger.info(
            "PaperBroker SELL EXECUTED: %s %.4f shares @ $%.4f (base: $%.4f, slip: $%.4f). Gross: $%.4f, Fee: $%.4f, Net inflow: $%.4f, Net PnL: %+.4f. Cash: $%.4f.",
            ticker, shares, effective_price, fill_price, slippage_cost, gross_proceeds, fee, net_inflow, net_pnl, self.cash,
        )

        # Persist to DB
        repository.save_order(
            run_date=run_date,
            ticker=ticker,
            action="SELL",
            quantity=shares,
            price=fill_price,
            reason=order.reason,
        )
        repository.save_trade(
            run_date=run_date,
            ticker=ticker,
            action="SELL",
            quantity=shares,
            fill_price=effective_price,
            cost=fee,
            net_pnl=net_pnl,
            slippage_cost=slippage_cost,
        )
        repository.log_event(
            level="INFO",
            component="paper_broker",
            message=f"SELL {ticker} {shares:.4f} @ ${effective_price:.4f} ({order.reason})",
            details={
                "run_date": run_date,
                "ticker": ticker,
                "shares": shares,
                "base_price": fill_price,
                "effective_price": effective_price,
                "slippage_cost": slippage_cost,
                "gross_proceeds": gross_proceeds,
                "fee": fee,
                "net_inflow": net_inflow,
                "net_pnl": net_pnl,
                "cash_after": self.cash,
                "reason": order.reason,
            },
        )

        return FillResult(
            run_date=run_date,
            ticker=ticker,
            action="SELL",
            shares=shares,
            fill_price=effective_price,
            gross_value=gross_proceeds,
            fee=fee,
            cash_impact=net_inflow,
            net_pnl=net_pnl,
            reason=order.reason,
            slippage_cost=slippage_cost,
            effective_price=effective_price,
        )

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        run_date: str,
        current_prices: Dict[str, float],
    ) -> float:
        """
        Mark positions to market using current (close) prices and persist the
        portfolio snapshot to DB.

        Parameters
        ----------
        run_date : str
            'YYYY-MM-DD' for this snapshot.
        current_prices : dict
            {ticker: close_price}. Missing tickers fall back to entry_price.

        Returns
        -------
        total_equity : float
            Cash + marked portfolio value for logging / reporting.
        """
        position_value = 0.0
        positions_blob: Dict[str, Any] = {}

        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos.entry_price)
            if price > 0:
                pos.update_high_and_trailing_stop(price)
            mkt_value = round(pos.quantity * price, 4)
            position_value += mkt_value
            positions_blob[ticker] = {
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "entry_date": pos.entry_date,
                "entry_fee": pos.entry_fee,
                "current_price": round(price, 4),
                "market_value": mkt_value,
                "confidence_tier": getattr(pos, "confidence_tier", "FULL"),
                "highest_price_since_entry": round(pos.highest_price_since_entry, 4),
                "trailing_stop_price": round(pos.trailing_stop_price, 4),
            }

        total_equity = round(self.cash + position_value, 4)

        repository.save_portfolio_snapshot(
            run_date=run_date,
            cash=self.cash,
            total_value=total_equity,
            positions=positions_blob,
            total_slippage_cost=self.total_slippage_cost,
        )

        logger.info(
            "PaperBroker snapshot %s: Cash=%.4f, Positions=%.4f, Total=%.4f, TotalSlippage=%.4f.",
            run_date, self.cash, position_value, total_equity, self.total_slippage_cost,
        )

        return total_equity
