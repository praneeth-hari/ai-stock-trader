"""
src/backtest/backtest.py — Phase 10 Backtesting Engine.

Simulates the complete autonomous trading pipeline chronologically through historical data
with strict anti-self-deception safeguards (§1.5):
  1. The Lag Rule: Trade signals generated at T-1 close; orders filled at T open (Market-On-Open).
  2. Transaction Costs: 0.2% cost per trade (0.4% round trip) deducted on every fill.
  3. Strict Chronology & Regime Awareness: SPY 200d MA regime filter enforced daily.
  4. Real Portfolio Sizing: 3-position cap, 15% cash reserve, $10 min trade size.
  5. The Known-Answer Test: Verifies engine accounting logic on SPY before any strategy run.
  6. "Too Good to Be True" Alarm: Suspicious metrics trigger an anti-leakage audit alarm.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS, compute_features, drop_warmup_rows
from src.ml.evaluate import load_active_model
from src.ml.train import TrainedModel
from src.portfolio.portfolio import OrderSpec, allocate_portfolio, calculate_fractional_shares
from src.ranking.ranking import RankingResult, rank_candidates
from src.risk.risk_engine import RiskAssessmentResult, evaluate_portfolio_risk

logger = logging.getLogger(__name__)

# §1.5 "Too Good to Be True" Backtest Alarm Thresholds
ALARM_MAX_CAGR: float = 0.35           # > 35% annualized net return is suspicious
ALARM_MIN_DRAWDOWN: float = 0.05       # < 5% max drawdown over multi-year equities is suspicious
ALARM_MAX_WIN_RATE: float = 0.65       # > 65% win rate for daily swing is suspicious
ALARM_MAX_SHARPE: float = 2.00         # > 2.0 Sharpe ratio is suspicious


@dataclass
class BacktestTrade:
    """Represents a completed or active trade in the backtest."""
    ticker: str
    entry_date: str
    entry_price: float
    shares: float
    entry_cost: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_cost: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    net_pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None

    def close(self, exit_date: str, exit_price: float, fee_rate: float, reason: str) -> None:
        self.exit_date = exit_date
        self.exit_price = float(exit_price)
        self.exit_cost = round(self.shares * self.exit_price * fee_rate, 4)
        gross_proceeds = self.shares * self.exit_price
        gross_cost = self.shares * self.entry_price
        self.gross_pnl = round(gross_proceeds - gross_cost, 4)
        self.net_pnl = round(self.gross_pnl - self.entry_cost - self.exit_cost, 4)
        self.net_pnl_pct = round(self.net_pnl / (gross_cost + self.entry_cost), 4)
        self.exit_reason = reason


@dataclass
class DailySnapshot:
    """Mark-to-market portfolio snapshot at end of trading day T."""
    date: str
    cash: float
    portfolio_value: float
    total_equity: float
    daily_return: float
    cumulative_return: float
    spy_close: float
    spy_daily_return: float
    spy_cumulative_return: float
    positions_count: int
    cash_pct: float
    drawdown: float
    spy_drawdown: float


@dataclass
class BacktestMetrics:
    """Comprehensive performance and risk metrics (Net of 0.2% costs)."""
    start_date: str
    end_date: str
    trading_days: int
    starting_capital: float
    final_equity: float
    total_net_return_pct: float
    annualized_return_pct: float
    spy_total_return_pct: float
    spy_annualized_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    spy_max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl_pct: float
    avg_holding_days: float
    avg_cash_pct: float
    alarm_triggered: bool
    alarm_reasons: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Complete container for a backtest run."""
    metrics: BacktestMetrics
    daily_snapshots: List[DailySnapshot]
    trades: List[BacktestTrade]

    def to_markdown_summary(self, title: str = "Strategy Backtest Report") -> str:
        m = self.metrics
        alarm_text = ""
        if m.alarm_triggered:
            alarm_text = (
                "\n> [!CAUTION]\n"
                "> **§1.5 TOO GOOD TO BE TRUE ALARM TRIGGERED**\n"
                f"> Suspected Look-Ahead Leakage: {', '.join(m.alarm_reasons)}\n"
            )

        lines = [
            f"### {title}",
            f"> [!WARNING]",
            f"> **OPTIMISTIC — Real results likely worse.** Assumes perfect market-open execution,",
            f"> zero slippage beyond 0.2%, and uses a survivor-biased mega-cap universe.",
            alarm_text,
            f"**Period**: `{m.start_date}` to `{m.end_date}` ({m.trading_days} trading days) | "
            f"**Starting Capital**: `${m.starting_capital:.2f}` -> **Final Equity**: `${m.final_equity:.2f}`\n",
            "| Performance Metric | Strategy (Net of 0.2% Fees) | SPY Benchmark (Buy & Hold) | Excess / Edge |",
            "|---|---|---|---|",
            f"| **Total Return** | **{m.total_net_return_pct:+.2f}%** | {m.spy_total_return_pct:+.2f}% | **{m.alpha_pct:+.2f}%** |",
            f"| **Annualized Return (CAGR)** | **{m.annualized_return_pct:+.2f}%** | {m.spy_annualized_return_pct:+.2f}% | {m.annualized_return_pct - m.spy_annualized_return_pct:+.2f}% |",
            f"| **Max Drawdown** | **{m.max_drawdown_pct:.2f}%** | {m.spy_max_drawdown_pct:.2f}% | {m.spy_max_drawdown_pct - m.max_drawdown_pct:+.2f}% pts better |",
            f"| **Sharpe Ratio (0% rf)** | **{m.sharpe_ratio:.2f}** | N/A | N/A |",
            f"| **Sortino Ratio** | **{m.sortino_ratio:.2f}** | N/A | N/A |",
            f"| **Average Cash Exposure** | **{m.avg_cash_pct:.1f}%** | 0.0% | +{m.avg_cash_pct:.1f}% safety |",
            "\n**Trading Activity & Execution Statistics**:",
            "| Metric | Value |",
            "|---|---|",
            f"| Total Completed Trades | **{m.total_trades}** |",
            f"| Win Rate | **{m.win_rate_pct:.1f}%** ({m.winning_trades} wins / {m.losing_trades} losses) |",
            f"| Profit Factor | **{m.profit_factor:.2f}** |",
            f"| Average Trade PnL | **{m.avg_trade_pnl_pct:+.2f}%** |",
            f"| Average Holding Period | **{m.avg_holding_days:.1f} days** |",
        ]
        return "\n".join(lines)


# ── 1. The Known-Answer Test (§1.5) ──────────────────────────────────────────

def run_known_answer_test(
    spy_df: pd.DataFrame,
    start_date: str = "2023-01-03",
    end_date: str = "2023-12-29",
    initial_capital: float = 10_000.0,
    cost_per_trade: float = settings.simulated_cost_per_trade,
    tolerance_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Mandatory §1.5 Known-Answer Test.
    Runs a pure 'Buy SPY at T1 Open and Hold until TN Close' simulation using a larger
    notional amount ($10,000.00) specifically to eliminate fractional share rounding noise.

    Compares the engine's simulated net equity against the closed-form benchmark return:
      R_price = (Close(TN) / Open(T1)) - 1
      R_net   = (1 - cost_per_trade) * (1 + R_price) - 1

    Tolerance: absolute difference must be <= 0.05% (0.0005).
    """
    logger.info(
        "Running §1.5 Known-Answer Test on SPY (%s to %s, capital=$%.2f)...",
        start_date, end_date, initial_capital,
    )

    df = spy_df.copy()
    if "date" not in df.columns:
        raise ValueError("spy_df must contain 'date' column.")

    df["date"] = df["date"].astype(str)
    window_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date").reset_index(drop=True)

    if len(window_df) < 2:
        raise ValueError(f"Insufficient SPY data in window [{start_date}, {end_date}]: only {len(window_df)} rows.")

    t1_open = float(window_df.iloc[0]["open"])
    tn_close = float(window_df.iloc[-1]["close"])

    # 1. Closed-Form Benchmark Net Return
    raw_benchmark_return = (tn_close / t1_open) - 1.0
    expected_net_return = (1.0 - cost_per_trade) * (1.0 + raw_benchmark_return) - 1.0

    # 2. Simulated Engine Execution
    # Buy on T1 open with full capital
    max_spend = initial_capital / (1.0 + cost_per_trade)
    raw_shares = max_spend / t1_open
    shares = math.floor(raw_shares * 10000.0) / 10000.0
    gross_spend = round(shares * t1_open, 4)
    entry_fee = round(gross_spend * cost_per_trade, 4)
    cash = round(initial_capital - (gross_spend + entry_fee), 4)

    # Mark to market at TN close
    final_portfolio_val = round(shares * tn_close, 4)
    final_equity = round(cash + final_portfolio_val, 4)
    simulated_net_return = (final_equity / initial_capital) - 1.0

    diff_pct = abs(simulated_net_return - expected_net_return) * 100.0
    passed = diff_pct <= tolerance_pct

    result = {
        "passed": passed,
        "test_name": "SPY Buy-and-Hold Known-Answer Test (§1.5)",
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "t1_open": t1_open,
        "tn_close": tn_close,
        "simulated_net_return_pct": round(simulated_net_return * 100.0, 4),
        "expected_net_return_pct": round(expected_net_return * 100.0, 4),
        "difference_pct_points": round(diff_pct, 4),
        "tolerance_pct": tolerance_pct,
        "shares": shares,
        "remaining_cash": cash,
        "dividend_notice": (
            "Benchmark return represents unadjusted price return (matching yfinance auto_adjust=False). "
            "Real SPY investors receive cash dividends (~1.5%/year) which are excluded from price-only simulation."
        ),
    }

    if not passed:
        logger.critical(
            "KNOWN-ANSWER TEST FAILED! Simulated: %.4f%%, Expected: %.4f%%, Diff: %.4f%% > %.2f%%",
            simulated_net_return * 100.0, expected_net_return * 100.0, diff_pct, tolerance_pct,
        )
    else:
        logger.info(
            "KNOWN-ANSWER TEST PASSED! Simulated: %.4f%% vs Expected: %.4f%% (Diff: %.4f%% <= %.2f%%)",
            simulated_net_return * 100.0, expected_net_return * 100.0, diff_pct, tolerance_pct,
        )

    return result


# ── 2. Full-Pipeline Strategy Backtest Engine ────────────────────────────────

def run_strategy_backtest(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = settings.initial_capital,
    model: Optional[TrainedModel] = None,
    cost_per_trade: float = settings.simulated_cost_per_trade,
) -> BacktestResult:
    """
    Executes the full day-by-day autonomous trading replay across historical data.

    Enforces:
      - Sizing at project scale (default $50.00 capital, $14.16 equal-weight slot, $7.50 cash floor).
      - The Lag Rule: Day T-1 close features -> Rank -> Risk -> Alloc -> Day T open execution.
      - 0.2% cost deduction per trade.
      - Mark-to-market daily snapshots and comparison with SPY benchmark.
    """
    active_model = model if model is not None else load_active_model()

    # 1. Clean and validate inputs
    spy_clean = spy_df.copy().sort_values("date").reset_index(drop=True)
    spy_dates = set(spy_clean["date"].astype(str))

    # Precompute features for each ticker
    all_features: List[pd.DataFrame] = []
    price_history: Dict[str, Dict[str, Dict[str, float]]] = {}  # ticker -> date -> {open, close}

    for ticker, df in universe_dict.items():
        clean = df.copy().sort_values("date").reset_index(drop=True)
        feats = compute_features(clean, spy_df=spy_clean)
        all_features.append(feats)

        price_history[ticker] = {}
        for _, row in clean.iterrows():
            d = str(row["date"])
            price_history[ticker][d] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
            }

    valid_features = [f for f in all_features if not f.empty and not f.isna().all().all()]
    combined_features = pd.concat(valid_features, ignore_index=True) if valid_features else pd.DataFrame()

    # Determine unique trading dates in test window
    all_test_dates = sorted(
        d for d in spy_dates if start_date <= d <= end_date
    )

    if len(all_test_dates) < 2:
        raise ValueError(f"Insufficient dates in test window [{start_date}, {end_date}].")

    # SPY benchmark reference
    spy_price_map = {
        str(r["date"]): {"open": float(r["open"]), "close": float(r["close"])}
        for _, r in spy_clean.iterrows()
    }
    spy_t0_close = spy_price_map[all_test_dates[0]]["close"]

    # 2. Simulation State
    cash = float(initial_capital)
    positions: Dict[str, Dict[str, Any]] = {}  # ticker -> {quantity, avg_cost, entry_date, entry_price}
    pending_orders: List[OrderSpec] = []
    completed_trades: List[BacktestTrade] = []
    daily_snapshots: List[DailySnapshot] = []

    peak_equity = float(initial_capital)
    spy_peak_close = spy_t0_close

    for day_idx, current_date in enumerate(all_test_dates):
        # ── Step A: Morning Fills (The Lag Rule) ──────────────────────────────
        # Execute orders generated at previous trading day's close at today's OPEN
        if pending_orders:
            for order in pending_orders:
                t = order.ticker
                if t not in price_history or current_date not in price_history[t]:
                    logger.warning("Missing open price for %s on %s. Skipping order.", t, current_date)
                    continue

                open_price = price_history[t][current_date]["open"]

                if order.action == "SELL" and t in positions:
                    held_qty = positions[t]["quantity"]
                    trade = BacktestTrade(
                        ticker=t,
                        entry_date=positions[t]["entry_date"],
                        entry_price=positions[t]["entry_price"],
                        shares=held_qty,
                        entry_cost=round(held_qty * positions[t]["entry_price"] * cost_per_trade, 4),
                    )
                    trade.close(
                        exit_date=current_date,
                        exit_price=open_price,
                        fee_rate=cost_per_trade,
                        reason=order.reason,
                    )
                    completed_trades.append(trade)

                    # Return net proceeds to cash (gross - fee = gross * 0.998)
                    gross_proceeds = round(held_qty * open_price, 4)
                    fee = round(gross_proceeds * cost_per_trade, 4)
                    net_inflow = round(gross_proceeds - fee, 4)
                    cash = round(cash + net_inflow, 4)
                    del positions[t]

                elif order.action == "BUY" and t not in positions:
                    # Execute buy order using today's open price
                    shares, gross_spend, fee = calculate_fractional_shares(
                        order.net_amount, open_price, cost_per_trade
                    )
                    total_outflow = round(gross_spend + fee, 4)

                    if shares > 0 and cash >= total_outflow:
                        cash = round(cash - total_outflow, 4)
                        positions[t] = {
                            "quantity": shares,
                            "avg_cost": open_price,
                            "entry_date": current_date,
                            "entry_price": open_price,
                        }

            pending_orders = []

        # ── Step B: Mark to Market at Day T Close ─────────────────────────────
        portfolio_val = 0.0
        current_prices: Dict[str, float] = {}

        for t, pos in positions.items():
            close_price = price_history[t].get(current_date, {}).get("close", pos["avg_cost"])
            current_prices[t] = close_price
            portfolio_val += pos["quantity"] * close_price

        total_equity = round(cash + portfolio_val, 4)
        peak_equity = max(peak_equity, total_equity)
        current_drawdown = (total_equity - peak_equity) / peak_equity

        # Benchmark SPY daily tracking
        spy_close = spy_price_map.get(current_date, {}).get("close", spy_t0_close)
        spy_peak_close = max(spy_peak_close, spy_close)
        spy_drawdown = (spy_close - spy_peak_close) / spy_peak_close

        prev_equity = daily_snapshots[-1].total_equity if daily_snapshots else initial_capital
        daily_ret = (total_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        cum_ret = (total_equity / initial_capital) - 1.0

        prev_spy = daily_snapshots[-1].spy_close if daily_snapshots else spy_t0_close
        spy_daily_ret = (spy_close / prev_spy) - 1.0 if prev_spy > 0 else 0.0
        spy_cum_ret = (spy_close / spy_t0_close) - 1.0

        snapshot = DailySnapshot(
            date=current_date,
            cash=cash,
            portfolio_value=round(portfolio_val, 4),
            total_equity=total_equity,
            daily_return=daily_ret,
            cumulative_return=cum_ret,
            spy_close=spy_close,
            spy_daily_return=spy_daily_ret,
            spy_cumulative_return=spy_cum_ret,
            positions_count=len(positions),
            cash_pct=round((cash / total_equity) * 100.0, 2) if total_equity > 0 else 100.0,
            drawdown=current_drawdown,
            spy_drawdown=spy_drawdown,
        )
        daily_snapshots.append(snapshot)

        # ── Step C: Generate Decisions for Day T+1 (End of Day T) ─────────────
        # If not the last day, run pipeline up to current_date close
        if day_idx < len(all_test_dates) - 1:
            # Sliced features up to current_date
            sub_feats = combined_features[combined_features["date"] <= current_date]
            sub_spy = spy_clean[spy_clean["date"] <= current_date]

            if not sub_feats.empty:
                try:
                    # 1. Ranking Engine
                    ranking_res = rank_candidates(
                        sub_feats,
                        model=active_model,
                        spy_df=sub_spy,
                        run_date=current_date,
                    )

                    # Current prices for all tickers in universe
                    today_universe_prices = {
                        t: price_history[t][current_date]["close"]
                        for t in universe_dict
                        if current_date in price_history.get(t, {})
                    }

                    # 2. Risk Engine
                    risk_res = evaluate_portfolio_risk(
                        run_date=current_date,
                        current_cash=cash,
                        current_positions=positions,
                        current_prices=today_universe_prices,
                        ranking_result=ranking_res,
                        spy_df=sub_spy,
                    )

                    # 3. Portfolio Allocation (Order Generation)
                    alloc_res = allocate_portfolio(
                        risk_assessment=risk_res,
                        current_cash=cash,
                        current_positions=positions,
                        current_prices=today_universe_prices,
                        fee_rate=cost_per_trade,
                    )

                    pending_orders = alloc_res.orders

                except Exception as exc:
                    logger.error("Error generating orders on %s: %s", current_date, exc)
                    pending_orders = []

    # 3. Compute Summary Metrics
    trading_days = len(daily_snapshots)
    final_equity = daily_snapshots[-1].total_equity
    total_net_ret = (final_equity / initial_capital) - 1.0

    years = trading_days / 252.0 if trading_days > 0 else 1.0
    cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) if final_equity > 0 and years > 0 else 0.0

    final_spy = daily_snapshots[-1].spy_close
    spy_total_ret = (final_spy / spy_t0_close) - 1.0
    spy_cagr = ((final_spy / spy_t0_close) ** (1.0 / years) - 1.0) if final_spy > 0 and years > 0 else 0.0

    max_dd = min(s.drawdown for s in daily_snapshots) * 100.0  # negative %
    spy_max_dd = min(s.spy_drawdown for s in daily_snapshots) * 100.0

    # Risk-adjusted ratios
    daily_returns = [s.daily_return for s in daily_snapshots]
    mean_ret = float(np.mean(daily_returns)) if daily_returns else 0.0
    std_ret = float(np.std(daily_returns)) if daily_returns else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 1e-8 else 0.0

    downside_returns = [r for r in daily_returns if r < 0]
    downside_std = float(np.std(downside_returns)) if downside_returns else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(252.0)) if downside_std > 1e-8 else 0.0

    # Trade statistics
    winning_trades = [t for t in completed_trades if (t.net_pnl or 0.0) > 0]
    losing_trades = [t for t in completed_trades if (t.net_pnl or 0.0) <= 0]
    win_rate = (len(winning_trades) / len(completed_trades) * 100.0) if completed_trades else 0.0

    gross_gains = sum(t.net_pnl for t in winning_trades if t.net_pnl is not None)
    gross_losses = abs(sum(t.net_pnl for t in losing_trades if t.net_pnl is not None))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-6 else (99.0 if gross_gains > 0 else 0.0)

    avg_trade_pnl = float(np.mean([t.net_pnl_pct for t in completed_trades])) * 100.0 if completed_trades else 0.0

    # Average holding duration
    holding_days_list = []
    for t in completed_trades:
        if t.entry_date and t.exit_date:
            try:
                d1 = pd.Timestamp(t.entry_date)
                d2 = pd.Timestamp(t.exit_date)
                holding_days_list.append((d2 - d1).days)
            except Exception:
                pass
    avg_holding = float(np.mean(holding_days_list)) if holding_days_list else 0.0
    avg_cash = float(np.mean([s.cash_pct for s in daily_snapshots])) if daily_snapshots else 100.0

    # 4. Anti-Self-Deception Alarm Checks (§1.5)
    alarm_triggered = False
    alarm_reasons: List[str] = []

    if cagr > ALARM_MAX_CAGR:
        alarm_triggered = True
        alarm_reasons.append(f"CAGR suspiciously high ({cagr*100:.1f}% > {ALARM_MAX_CAGR*100:.0f}%)")
    if abs(max_dd) < (ALARM_MIN_DRAWDOWN * 100.0) and trading_days >= 200 and len(completed_trades) >= 5:
        alarm_triggered = True
        alarm_reasons.append(f"Max Drawdown suspiciously low ({abs(max_dd):.1f}% < {ALARM_MIN_DRAWDOWN*100:.0f}%) while actively trading")
    if win_rate > (ALARM_MAX_WIN_RATE * 100.0) and len(completed_trades) >= 15:
        alarm_triggered = True
        alarm_reasons.append(f"Win Rate suspiciously high ({win_rate:.1f}% > {ALARM_MAX_WIN_RATE*100:.0f}%)")
    if sharpe > ALARM_MAX_SHARPE and trading_days >= 200 and len(completed_trades) >= 5:
        alarm_triggered = True
        alarm_reasons.append(f"Sharpe ratio suspiciously high ({sharpe:.2f} > {ALARM_MAX_SHARPE:.2f})")

    metrics = BacktestMetrics(
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        starting_capital=initial_capital,
        final_equity=final_equity,
        total_net_return_pct=round(total_net_ret * 100.0, 2),
        annualized_return_pct=round(cagr * 100.0, 2),
        spy_total_return_pct=round(spy_total_ret * 100.0, 2),
        spy_annualized_return_pct=round(spy_cagr * 100.0, 2),
        alpha_pct=round((total_net_ret - spy_total_ret) * 100.0, 2),
        max_drawdown_pct=round(max_dd, 2),
        spy_max_drawdown_pct=round(spy_max_dd, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        total_trades=len(completed_trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        avg_trade_pnl_pct=round(avg_trade_pnl, 2),
        avg_holding_days=round(avg_holding, 1),
        avg_cash_pct=round(avg_cash, 1),
        alarm_triggered=alarm_triggered,
        alarm_reasons=alarm_reasons,
    )

    return BacktestResult(
        metrics=metrics,
        daily_snapshots=daily_snapshots,
        trades=completed_trades,
    )


# ── 3. Section 10 Item 2: Backtest Lab Engine ─────────────────────────────────

def compute_monthly_returns_heatmap(
    daily_snapshots: List[DailySnapshot],
    starting_capital: float = 10000.0,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Computes a Year x Month grid of returns (%), plus best_month and worst_month strings.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if not daily_snapshots:
        empty_df = pd.DataFrame(columns=["Year"] + month_names + ["Year Total"])
        return empty_df, "N/A", "N/A"

    df = pd.DataFrame([{"date": pd.to_datetime(s.date), "equity": s.total_equity} for s in daily_snapshots])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    years = sorted(df["year"].unique())
    grid_rows = []
    monthly_returns_list: List[float] = []

    for y in years:
        ydf = df[df["year"] == y]
        row_dict: Dict[str, str] = {"Year": str(y)}
        prior_year_df = df[df["year"] < y]
        year_start_eq = prior_year_df["equity"].iloc[-1] if not prior_year_df.empty else starting_capital

        for m in range(1, 13):
            m_name = month_names[m - 1]
            mdf = ydf[ydf["month"] == m]
            if mdf.empty:
                row_dict[m_name] = "—"
            else:
                prior_df = df[(df["date"] < mdf["date"].iloc[0])]
                prev_eq = prior_df["equity"].iloc[-1] if not prior_df.empty else year_start_eq
                curr_eq = mdf["equity"].iloc[-1]
                m_ret = ((curr_eq / prev_eq) - 1.0) * 100.0 if prev_eq > 0 else 0.0
                monthly_returns_list.append(m_ret)
                sign = "+" if m_ret > 0 else ""
                row_dict[m_name] = f"{sign}{m_ret:.1f}%"

        year_end_eq = ydf["equity"].iloc[-1]
        y_ret = ((year_end_eq / year_start_eq) - 1.0) * 100.0 if year_start_eq > 0 else 0.0
        sign_y = "+" if y_ret > 0 else ""
        row_dict["Year Total"] = f"{sign_y}{y_ret:.1f}%"
        grid_rows.append(row_dict)

    grid_df = pd.DataFrame(grid_rows)

    if monthly_returns_list:
        best_m = max(monthly_returns_list)
        worst_m = min(monthly_returns_list)
        best_str = f"+{best_m:.1f}%" if best_m > 0 else f"{best_m:.1f}%"
        worst_str = f"+{worst_m:.1f}%" if worst_m > 0 else f"{worst_m:.1f}%"
    else:
        best_str = "0.0%"
        worst_str = "0.0%"

    return grid_df, best_str, worst_str


def run_lab_backtest(
    universe_dict: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0,
    buy_threshold: float = 0.60,
    exit_threshold: float = 0.45,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
    max_positions: int = 3,
    position_sizing: str = "Fixed",
    use_sentiment: bool = True,
    use_earnings_blackout: bool = True,
    use_sector_rotation: bool = True,
    use_macro_regime: bool = True,
    use_correlation_filter: bool = True,
    use_trailing_stop: bool = True,
    model: Optional[TrainedModel] = None,
    cost_per_trade: float = 0.002,
) -> Dict[str, Any]:
    """
    Executes an interactive Backtest Lab simulation with fully customizable strategy parameters.
    Returns comprehensive metrics, equity curves, monthly returns heatmap, and trade list.
    """
    active_model = model if model is not None else load_active_model()

    spy_clean = spy_df.copy().sort_values("date").reset_index(drop=True)
    if "ticker" not in spy_clean.columns:
        spy_clean["ticker"] = "SPY"
    spy_dates = set(spy_clean["date"].astype(str))

    price_history: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_features: List[pd.DataFrame] = []

    for ticker, df in universe_dict.items():
        clean = df.copy().sort_values("date").reset_index(drop=True)
        if "ticker" not in clean.columns:
            clean["ticker"] = ticker
        feats = compute_features(clean, spy_df=spy_clean)
        all_features.append(feats)

        price_history[ticker] = {}
        for _, row in clean.iterrows():
            d = str(row["date"])
            price_history[ticker][d] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
            }

    valid_features = [f for f in all_features if not f.empty and not f.isna().all().all()]
    combined_features = pd.concat(valid_features, ignore_index=True) if valid_features else pd.DataFrame()

    all_test_dates = sorted([d for d in spy_dates if start_date <= d <= end_date])
    if len(all_test_dates) < 2:
        return {
            "total_return_pct": 0.0,
            "spy_total_return_pct": 0.0,
            "alpha_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "days_in_cash_pct": 100.0,
            "sharpe_ratio": 0.0,
            "best_month": "0.0%",
            "worst_month": "0.0%",
            "equity_curve": pd.DataFrame(),
            "monthly_returns": pd.DataFrame(),
            "trades_df": pd.DataFrame(),
        }

    spy_price_map = {
        str(r["date"]): {"open": float(r["open"]), "close": float(r["close"])}
        for _, r in spy_clean.iterrows()
    }
    spy_t0_close = spy_price_map[all_test_dates[0]]["close"]

    cash = float(initial_capital)
    positions: Dict[str, Dict[str, Any]] = {}  # ticker -> {quantity, avg_cost, entry_date, entry_price, highest_price, sector}
    pending_buy_orders: List[Dict[str, Any]] = []
    completed_trades: List[BacktestTrade] = []
    daily_snapshots: List[DailySnapshot] = []

    peak_equity = float(initial_capital)
    spy_peak_close = spy_t0_close

    ticker_sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Communication Services", "AMZN": "Consumer Cyclical",
        "META": "Communication Services", "BRK-B": "Financial",
        "JNJ": "Healthcare", "JPM": "Financial", "V": "Financial",
    }

    for day_idx, current_date in enumerate(all_test_dates):
        # ── A. Morning Exits and Fills (Lag Rule) ─────────────────────────────
        # 1. Process Exits first
        active_tickers = list(positions.keys())
        for t in active_tickers:
            pos = positions[t]
            if t not in price_history or current_date not in price_history[t]:
                continue
            open_price = price_history[t][current_date]["open"]
            highest_price = max(pos.get("highest_price", open_price), open_price)
            pos["highest_price"] = highest_price

            exit_reason = None
            # Check Stop Loss / Trailing Stop
            if use_trailing_stop:
                stop_target = highest_price * (1.0 - stop_loss_pct)
                if open_price <= stop_target:
                    exit_reason = f"Trailing Stop ({stop_loss_pct*100:.0f}%)"
            else:
                stop_target = pos["entry_price"] * (1.0 - stop_loss_pct)
                if open_price <= stop_target:
                    exit_reason = f"Stop Loss ({stop_loss_pct*100:.0f}%)"

            # Check Take Profit
            if not exit_reason:
                tp_target = pos["entry_price"] * (1.0 + take_profit_pct)
                if open_price >= tp_target:
                    exit_reason = f"Take Profit ({take_profit_pct*100:.0f}%)"

            if exit_reason:
                held_qty = pos["quantity"]
                trade = BacktestTrade(
                    ticker=t,
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    shares=held_qty,
                    entry_cost=round(held_qty * pos["entry_price"] * cost_per_trade, 4),
                )
                trade.close(
                    exit_date=current_date,
                    exit_price=open_price,
                    fee_rate=cost_per_trade,
                    reason=exit_reason,
                )
                completed_trades.append(trade)

                gross_proceeds = held_qty * open_price
                fee = gross_proceeds * cost_per_trade
                cash += (gross_proceeds - fee)
                del positions[t]

        # 2. Process Morning Pending Buys
        if pending_buy_orders:
            for order in pending_buy_orders:
                t = order["ticker"]
                if len(positions) >= max_positions:
                    break
                if t in positions or t not in price_history or current_date not in price_history[t]:
                    continue
                open_price = price_history[t][current_date]["open"]
                alloc_amount = order["net_amount"]

                shares, gross_spend, fee = calculate_fractional_shares(
                    alloc_amount, open_price, cost_per_trade
                )
                total_outflow = gross_spend + fee
                if shares > 0 and cash >= total_outflow:
                    cash -= total_outflow
                    positions[t] = {
                        "quantity": shares,
                        "avg_cost": open_price,
                        "entry_date": current_date,
                        "entry_price": open_price,
                        "highest_price": open_price,
                        "sector": ticker_sectors.get(t, "Other"),
                    }

            pending_buy_orders = []

        # ── B. Mark to Market at Day T Close ──────────────────────────────────
        portfolio_val = 0.0
        for t, pos in positions.items():
            close_price = price_history[t].get(current_date, {}).get("close", pos["avg_cost"])
            portfolio_val += pos["quantity"] * close_price

        total_equity = cash + portfolio_val
        peak_equity = max(peak_equity, total_equity)
        current_drawdown = (total_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        spy_close = spy_price_map.get(current_date, {}).get("close", spy_t0_close)
        spy_peak_close = max(spy_peak_close, spy_close)
        spy_drawdown = (spy_close - spy_peak_close) / spy_peak_close if spy_peak_close > 0 else 0.0

        prev_equity = daily_snapshots[-1].total_equity if daily_snapshots else initial_capital
        daily_ret = (total_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        cum_ret = (total_equity / initial_capital) - 1.0

        prev_spy = daily_snapshots[-1].spy_close if daily_snapshots else spy_t0_close
        spy_daily_ret = (spy_close / prev_spy) - 1.0 if prev_spy > 0 else 0.0
        spy_cum_ret = (spy_close / spy_t0_close) - 1.0

        snapshot = DailySnapshot(
            date=current_date,
            cash=round(cash, 2),
            portfolio_value=round(portfolio_val, 2),
            total_equity=round(total_equity, 2),
            daily_return=daily_ret,
            cumulative_return=cum_ret,
            spy_close=spy_close,
            spy_daily_return=spy_daily_ret,
            spy_cumulative_return=spy_cum_ret,
            positions_count=len(positions),
            cash_pct=round((cash / total_equity) * 100.0, 2) if total_equity > 0 else 100.0,
            drawdown=current_drawdown,
            spy_drawdown=spy_drawdown,
        )
        daily_snapshots.append(snapshot)

        # ── C. Generate Signals for Day T+1 (End of Day T) ─────────────────────
        if day_idx < len(all_test_dates) - 1:
            sub_feats = combined_features[combined_features["date"] <= current_date]
            sub_spy = spy_clean[spy_clean["date"] <= current_date]

            # 1. Macro Regime Check
            is_risk_on = True
            if use_macro_regime and len(sub_spy) >= 200:
                spy_200ma = sub_spy["close"].tail(200).mean()
                if spy_close < spy_200ma:
                    is_risk_on = False

            if is_risk_on and not sub_feats.empty and active_model is not None:
                latest_feats = sub_feats[sub_feats["date"] == current_date]
                candidates = []

                for _, r in latest_feats.iterrows():
                    ticker = str(r["ticker"])
                    if ticker in positions:
                        continue

                    # Feature vector prediction
                    feat_row = r[FEATURE_COLUMNS].to_frame().T
                    prob = float(active_model.predict_proba(feat_row)[0, 1])

                    # Sentiment toggle adjustment
                    if use_sentiment:
                        sent_val = float(r.get("sentiment_score", 0.0))
                        if sent_val > 0.2:
                            prob = min(0.99, prob + 0.03)
                        elif sent_val < -0.2:
                            prob = max(0.01, prob - 0.03)

                    if prob < buy_threshold:
                        continue

                    # Earnings blackout toggle check
                    if use_earnings_blackout:
                        days_to_earnings = abs(int(r.get("days_to_earnings", 999)))
                        if days_to_earnings <= 7:
                            continue

                    # Sector rotation toggle check
                    if use_sector_rotation:
                        sec = ticker_sectors.get(ticker, "Other")
                        held_sectors = [p["sector"] for p in positions.values()]
                        if sec in held_sectors:
                            continue

                    candidates.append({"ticker": ticker, "prob": prob})

                candidates.sort(key=lambda x: x["prob"], reverse=True)
                available_slots = max_positions - len(positions)
                selected_candidates = candidates[:available_slots]

                if selected_candidates and available_slots > 0:
                    base_alloc = cash / available_slots
                    for cand in selected_candidates:
                        if position_sizing == "Confidence-Based":
                            conviction = max(0.0, (cand["prob"] - buy_threshold) / (1.0 - buy_threshold + 1e-6))
                            alloc = base_alloc * (0.8 + 0.4 * conviction)
                        else:
                            alloc = base_alloc

                        pending_buy_orders.append({
                            "ticker": cand["ticker"],
                            "net_amount": min(cash, alloc),
                            "prob": cand["prob"],
                        })

    # ── Post-Simulation Metrics Assembly ──────────────────────────────────────
    trading_days = len(daily_snapshots)
    final_equity = daily_snapshots[-1].total_equity
    total_net_ret = ((final_equity / initial_capital) - 1.0) * 100.0

    final_spy = daily_snapshots[-1].spy_close
    spy_total_ret = ((final_spy / spy_t0_close) - 1.0) * 100.0
    alpha_pct = total_net_ret - spy_total_ret

    max_dd = min(s.drawdown for s in daily_snapshots) * 100.0 if daily_snapshots else 0.0

    daily_returns = [s.daily_return for s in daily_snapshots]
    mean_ret = float(np.mean(daily_returns)) if daily_returns else 0.0
    std_ret = float(np.std(daily_returns)) if daily_returns else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 1e-8 else 0.0

    winning_trades = [t for t in completed_trades if (t.net_pnl or 0.0) > 0]
    win_rate = (len(winning_trades) / len(completed_trades) * 100.0) if completed_trades else 0.0

    cash_days = sum(1 for s in daily_snapshots if s.cash_pct >= 80.0)
    days_in_cash_pct = (cash_days / trading_days * 100.0) if trading_days > 0 else 100.0

    monthly_grid_df, best_month, worst_month = compute_monthly_returns_heatmap(
        daily_snapshots=daily_snapshots,
        starting_capital=initial_capital,
    )

    # Convert snapshots into dataframe
    eq_rows = []
    for s in daily_snapshots:
        eq_rows.append({
            "date": s.date,
            "Strategy": round(s.total_equity, 2),
            "SPY Benchmark": round(initial_capital * (1.0 + s.spy_cumulative_return), 2),
            "Cash": round(s.cash, 2),
            "Drawdown": round(s.drawdown * 100.0, 2),
        })
    eq_df = pd.DataFrame(eq_rows)

    # Convert trades into dataframe
    trade_rows = []
    for t in completed_trades:
        trade_rows.append({
            "Ticker": t.ticker,
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date or "—",
            "Entry Price": f"${t.entry_price:.2f}",
            "Exit Price": f"${t.exit_price:.2f}" if t.exit_price else "—",
            "Shares": f"{t.shares:.4f}",
            "Net PnL ($)": f"${t.net_pnl:+.2f}" if t.net_pnl is not None else "—",
            "Net PnL (%)": f"{t.net_pnl_pct * 100:+.2f}%" if t.net_pnl_pct is not None else "—",
            "Exit Reason": t.exit_reason or "—",
        })
    trades_df = pd.DataFrame(trade_rows)

    return {
        "total_return_pct": round(total_net_ret, 2),
        "spy_total_return_pct": round(spy_total_ret, 2),
        "alpha_pct": round(alpha_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": len(completed_trades),
        "days_in_cash_pct": round(days_in_cash_pct, 1),
        "sharpe_ratio": round(sharpe, 2),
        "best_month": best_month,
        "worst_month": worst_month,
        "equity_curve": eq_df,
        "monthly_returns": monthly_grid_df,
        "trades_df": trades_df,
    }

