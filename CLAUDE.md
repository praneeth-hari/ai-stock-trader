# CLAUDE.md — Always-On Rules for This Project

Read this every session. The full spec is in `PROJECT_PLAN.md`. This file is the short version of the rules that must **never** be broken. If anything here conflicts with a request, follow these rules and say so.

## What this project is
A **paper-trading (simulated money) US-stock system** for learning and capital preservation. **V1 has NO real money and NO broker integration.** Do not write real-money or broker code under any circumstances in V1 — it is gated behind a deliberate checklist in `PROJECT_PLAN.md` §12.

## The one goal
**Do not lose money.** Break-even is acceptable; profit is a bonus. When in doubt, protect the capital. Cash is a valid position.

## Locked strategy values (do not change without explicit instruction)
- **Starting capital:** **$10,000.00** paper money (locked §1.1; Phase 0–13 development and initial validation used $50.00 micro-scale, updated to $10,000.00 for V1 operation).
- **Active model:** **Baseline** (`logistic_regression_baseline_v1` — retained for capital preservation over Primary HistGBM).
- **Market:** US stocks. **Benchmark:** SPY. **Tickers:** standard (`AAPL`, `NVDA`…), no suffix.
- **Prediction label:** P(stock rises **> +1% over the next 5 trading days**).
- **Buy bar:** probability **≥ 0.60**. **Signal exit:** probability **< 0.45**.
- **Stop-loss:** **−8%**. **Take-profit:** **+15%**. **Trailing stop:** OFF in V1.
- **Exit priority (safety first):** Stop-loss → Take-profit → Signal exit → Hold.
- **Positions:** **3**, ~equal weight. **Cash reserve:** ~**15%**. **Min trade:** **$10**.
- **Direction:** long-only (no shorting).
- **Simulated cost:** **0.2% per trade** (0.4% round trip) — apply in every simulated fill.
- **Regime filter:** if SPY is **below** its **200-day** average → **no new buys** (still manage/sell existing).
- **Fractional shares:** allowed in the simulator.

## Non-negotiable engineering rules
1. **`config/settings.py` is the only place that reads env/secrets.** Never hardcode secrets. Never read env elsewhere.
2. **ML never calls the broker directly** — the Risk Engine always sits between them and can veto any trade.
3. **No look-ahead bias, ever.** Obey the lag rule everywhere: decide on data available *before* the trade (prior close), fill at the next open. Never use a day's own close to decide that day's trade.
4. **Net, never gross.** Costs always applied. Never report gross returns.
5. **Backtests must show SPY buy-and-hold beside them** and carry an "optimistic — real results likely worse" label.
6. **Known-answer test is mandatory** in the backtester: a "just buy SPY and hold" run must reproduce SPY's real return. A spectacular result is a **suspected bug** — hunt leakage before trusting it.
7. **Validation gate before any decision:** reject impossible values, flag ±50% spikes and gaps, **skip-and-log** bad stocks in isolation; never fill fake data, never crash the whole run. Save raw pulls unmodified.
8. **Fail loud, log everything** — every prediction, decision, skip, and error.
9. **Build phase by phase and test each phase** (see `PROJECT_PLAN.md` §7). Do not batch-generate untested code across many files.
10. **No guaranteed-profit claims** anywhere.
11. **No cloud/AWS in V1** — runs free on the local machine.

## When unsure
Prefer the more conservative, capital-preserving choice. Ask before deviating from any locked value above.
