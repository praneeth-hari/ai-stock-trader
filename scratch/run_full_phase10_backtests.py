"""
scratch/run_full_phase10_backtests.py — Full Phase 10 Backtesting Execution.

Executes:
  1. The Mandatory Known-Answer Test on SPY (2023, $10,000 notional scale).
  2. The 4 Regime Walk-Forward Strategy Backtests at the REAL $50.00 scale.
  3. The Full Contiguous Multi-Year Strategy Backtest (2021–2024) at the REAL $50.00 scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, ".")

import pandas as pd
from config.settings import settings
from src.data.market_data import fetch_ticker_data
from src.data.validation import validate_ticker_data
from src.backtest.backtest import run_known_answer_test, run_strategy_backtest

print("=" * 80)
print("PHASE 10: AUTONOMOUS TRADING PIPELINE COMPREHENSIVE BACKTEST REPORT")
print("=" * 80)

# Fetch universe data covering 2020 through 2024 (ensuring full warmup buffer)
tickers = ["AAPL", "MSFT", "JPM"]
start_full = "2021-07-09"
end_full = "2024-08-30"

print(f"\n[1/3] Fetching and validating historical data for SPY + {tickers}...")
raw_spy = fetch_ticker_data("SPY", start_date=start_full, end_date="2024-09-01")
val_spy = validate_ticker_data(raw_spy, ticker="SPY").cleaned_df

universe_dict = {}
for t in tickers:
    raw = fetch_ticker_data(t, start_date=start_full, end_date="2024-09-01")
    val = validate_ticker_data(raw, ticker=t).cleaned_df
    universe_dict[t] = val

print(f"Historical data ready: SPY ({len(val_spy)} rows), Universe ({[f'{t}: {len(df)} rows' for t, df in universe_dict.items()]})")

# ── Part 1: Mandatory Known-Answer Test ────────────────────────────────────────
print("\n" + "=" * 80)
print("PART 1: THE MANDATORY §1.5 KNOWN-ANSWER TEST (SPY BUY & HOLD 2023)")
print("=" * 80)
known_res = run_known_answer_test(
    spy_df=val_spy,
    start_date="2023-01-03",
    end_date="2023-12-29",
    initial_capital=10_000.0,
    cost_per_trade=0.002,
    tolerance_pct=0.05,
)

print(f"Status: {'PASSED [OK]' if known_res['passed'] else 'FAILED [CRITICAL]'}")
print(f"Period: {known_res['start_date']} to {known_res['end_date']}")
print(f"Notional Capital: ${known_res['initial_capital']:.2f}")
print(f"T1 Open: ${known_res['t1_open']:.2f} | TN Close: ${known_res['tn_close']:.2f}")
print(f"Simulated Net Return: {known_res['simulated_net_return_pct']:.4f}%")
print(f"Expected Net Return:  {known_res['expected_net_return_pct']:.4f}%")
print(f"Difference:           {known_res['difference_pct_points']:.4f}% points (Tolerance: <= {known_res['tolerance_pct']}%)")
print(f"Notes: {known_res['dividend_notice']}")

if not known_res["passed"]:
    print("\n[CRITICAL ERROR] Known-Answer Test Failed! Halting backtests.")
    sys.exit(1)

# ── Part 2: 4 Regime Walk-Forward Strategy Backtests ($50 Scale) ───────────────
print("\n" + "=" * 80)
print("PART 2: WALK-FORWARD STRATEGY BACKTESTS ACROSS 4 REGIMES ($50 SCALE)")
print("=" * 80)

regime_windows = [
    ("Window 1: 2021 Bull Market", "2021-07-09", "2021-12-31"),
    ("Window 2: 2022 Bear Market (Critical Test)", "2022-01-03", "2022-12-30"),
    ("Window 3: 2023 Tech Recovery", "2023-01-03", "2023-12-29"),
    ("Window 4: 2024 Late Cycle", "2024-01-02", "2024-08-23"),
]

window_results = []
for name, start, end in regime_windows:
    print(f"\n--- Running {name} ({start} to {end}) at $50 scale ---")
    res = run_strategy_backtest(
        universe_dict=universe_dict,
        spy_df=val_spy,
        start_date=start,
        end_date=end,
        initial_capital=50.0,
    )
    window_results.append((name, res))
    print(res.to_markdown_summary(name))

# ── Part 3: Contiguous Multi-Year Strategy Backtest ($50 Scale) ────────────────
print("\n" + "=" * 80)
print("PART 3: FULL CONTIGUOUS MULTI-YEAR BACKTEST (2021–2024, $50 SCALE)")
print("=" * 80)

full_res = run_strategy_backtest(
    universe_dict=universe_dict,
    spy_df=val_spy,
    start_date=start_full,
    end_date=end_full,
    initial_capital=50.0,
)

print(full_res.to_markdown_summary("Full Contiguous Backtest (2021–2024)"))

# Print closed trades breakdown if any
if full_res.trades:
    print("\nCompleted Trades Audit Trail (First 15 trades):")
    print("| Ticker | Entry Date | Exit Date | Entry Price | Exit Price | Shares | Net PnL ($) | Net PnL (%) | Reason |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in full_res.trades[:15]:
        print(
            f"| {t.ticker} | {t.entry_date} | {t.exit_date} | ${t.entry_price:.2f} | "
            f"${t.exit_price or 0:.2f} | {t.shares:.4f} | ${t.net_pnl or 0:.2f} | "
            f"{(t.net_pnl_pct or 0)*100:+.2f}% | {t.exit_reason} |"
        )
else:
    print("\nCompleted Trades: 0 (System maintained 100% cash / cash preservation during test period).")
