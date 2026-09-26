# V1 Freeze Report — Forward Paper Trading

**Date:** 2026-09-26 · **Status:** FROZEN for forward paper trading · Paper money only, no broker.

## 1. Why it is frozen

Historical tuning has stopped. No configuration showed a durable edge over SPY. The "entry-only" sector variant failed its pre-registered test: it passed in 2 of 4 training windows on 2007–2016, and 3 were required.

Final benchmark for the configured strategy (2-year walk-forward, net of 0.2% per trade; optimistic, real results likely worse):

| Period | Frozen strategy (CAGR / max DD / Sharpe) | SPY buy & hold | SPY only when ≥ 200-day SMA |
| --- | --- | --- | --- |
| 2007–2016 | 1.09% / −20.7% / 0.18 | 4.66% / −56.5% / 0.32 | 3.56% / −21.1% / 0.37 |
| 2017–2026 | 1.26% / −31.4% / 0.19 | 13.40% / −34.1% / 0.78 | 7.83% / −25.1% / 0.69 |

- **TSLA dependence.** TSLA was 32–35% of the strategy's trades. Excluding TSLA, the strategy lost money in both periods (−$1,959 and −$703).
- **Expectation.** Forward results will probably trail SPY. The purpose of forward trading is to measure, not to prove.
- **What the backtest does and does not cover.** The historical figures test the walk-forward *method* (a new model every 6 months). They do not test the exact frozen model file, which was trained on 2019–2023 data and can't be honestly backtested over those years. Forward trading is the first real test of this file.

## 2. Exact frozen configuration

### Code

- V1 commit: `ed810d8adbbca13e6b2484c4184064ac7a49c838` (648 tests passed; see section 6 for what it contains). The hash is recorded in a follow-up commit that changes only this report.
- Every effective setting equals its code default (checked 2026-09-26), so `.env` only repeats `config/settings.py` and a clean checkout runs the same configuration. `.env` stays uncommitted because it holds secrets.

### Model

- File: `data/models/active_model.joblib`, sha256 `09a8f688630438e200519d246008918a947d4208295ad7da54fd9256588963dc`. The file is gitignored, so the hash is the record; keep a backup copy of the file.
- Name: `logistic_regression_baseline_v1`, source `logistic_regression_baseline_v1_20260912_192600.joblib`.
- Training window: 2019-12-13 to 2023-12-01, with a 5-day purge gap.
- Setup: StandardScaler, logistic regression (lbfgs), `class_weight=balanced`, seed 42.
- Label: price rises more than 1% over the next 5 trading days.
- 15 features: price_to_ma20, price_to_ma50, price_to_ma200, ma5_above_ma20, roc_5, roc_21, rsi_14, macd_line, macd_signal, macd_hist, vol_20, atr_14, vol_ratio_5_20, vol_spike, rel_strength_21.

### Universe and capital

- 25 US tickers: AAPL MSFT NVDA GOOGL META VZ CMCSA AMZN TSLA HD JPM V BAC UNH JNJ LLY CAT GE UPS PG COST KO XOM CVX COP.
- Benchmark SPY. Long-only, fractional shares allowed.
- $10,000 paper capital, 0.2% cost per trade.

### Entry

- Final P = model P + sector modifier + sentiment modifier.
  - Sector: +0.04 for the top-2 and −0.04 for the bottom-3 of 8 sectors, ranked by 20-day return.
  - Sentiment: +0.03 or −0.05, and a veto below −0.6.
- Buy if P ≥ 0.60. The bar rises by 0.03 when the macro regime is RESTRICTIVE.
- No new buys while SPY's close is below its 200-day SMA.
- Earnings: no buys 1–2 days before earnings, half size 3–5 days before.

### Sizing

- 3 positions, roughly equal weight, 15% cash reserve, $10 minimum trade.
- Confidence tiers: 50% size if P < 0.65, 75% if P < 0.75, 100% if P ≥ 0.75.
- Macro size multiplier: 1.0 FAVORABLE, 0.8 NEUTRAL, 0.6 RESTRICTIVE.

### Exits, in priority order

1. Stop-loss at −8%.
2. Take-profit at +15%.
3. Signal exit when P < 0.45 / 0.47 / 0.50, for SPY 20-day volatility < 15% / 15–25% / > 25%. Only allowed after a 5-day minimum hold.

Trailing stop is OFF.

Note: CLAUDE.md lists the signal exit as a flat 0.45. The code uses the volatility-adaptive values above; that was already the case before the freeze and is recorded here, not changed.

### Other risk controls

- Correlation (60-day): warn at ≥ 0.70; block above 0.85 for the same sector and above 0.90 for different sectors.
- Macro circuit breaker settings: 5-day SPY drop 7% → 5-day halt; 20-day drop 15% → 10-day halt.
- PSI drift: monitor at 0.10 (half size), alert at 0.25.

## 3. Rules during forward trading

- **Don't change any value above.**
- **The model is locked in code.** `model_freeze_enabled=True` (with `frozen_model_sha256` set to the hash above) means:
  - promotion and rollback of the production model are refused;
  - the daily pipeline refuses to trade, and reports an error, if the model file is missing or its hash differs. It will never train and promote a replacement on its own;
  - monthly retraining still runs and saves candidates, but reports them as "awaiting approval" instead of promoting them.
- **Ending the freeze** is a deliberate human decision: set `MODEL_FREEZE_ENABLED=false` and log why.
- **Allowed changes:** only bug fixes that don't change decisions. Log each one with its date.
- **Never tested on history:** sentiment, earnings calendar, macro circuit breaker and PSI drift are live-only and were never backtested.

## 4. What to measure (decided in advance)

- **Monthly:**
  - Equity vs SPY buy & hold and vs SPY ≥ 200-day SMA timing, both starting on the same date with $10,000 and 0.2% costs.
  - Max drawdown, number of trades and average cash %.
  - TSLA's share of trades and of P&L.
- **Verdict at 12 months, not earlier:** if the strategy doesn't beat SPY ≥ 200-day timing on both Sharpe and max drawdown, conclude the ML layer adds no value in V1. Twelve months is still a small sample, so treat any result with caution.

## 5. Where it runs: this machine only

**Decision (2026-09-26):** this machine's Task Scheduler is the single V1 trading runner. The model file is not added to Git.

| Automated path | Trades? | Forward-paper period |
| --- | --- | --- |
| Task `AI-Stock-Trader-Daily` → `run_trader.bat` → scheduler → pipeline (Mon–Fri 2:00 AM IST) | Yes, local SQLite portfolio | **Enabled**: the only trading path |
| Task `AI-Stock-Trader-Monthly-Retrain` → `python -m src.ml.retrain --force` | No | Enabled; candidates are saved but never promoted (freeze plus approval gate) |
| Tasks `DailyStatus`, `Morning-HealthCheck`, `Weekly-Summary` | No | Enabled (reporting only) |
| GitHub `daily_trading.yml` (remote database, unfrozen model) | Yes | **Disabled**: no schedule, and the job is hard-off even when started manually |
| GitHub `monthly_retrain.yml`, `weekly_summary.yml` (remote database) | No | Schedules paused; manual runs still possible |

- **Same-day duplicates:** `run_daily_pipeline` skips a second run for the same market and date. The scheduler's `--force` only bypasses weekend/holiday/market-hours checks, not this guard. The dashboard's manual "run cycle" button also goes through the guard.
- **GitHub workflow changes take effect only after they are pushed.** Until then, GitHub keeps running the old schedules from `origin/main`.

## 6. What the V1 commit contains

- **Included:** all modified production code in `config/`, `src/` and `dashboard/`, and all modified tests. This is the exact code that the backtests and the full test suite ran on; the market-isolation code is interwoven and inactive for the US universe. Also included: `tests/conftest.py`, `tests/test_market_isolation_fixes.py`, `tests/test_model_freeze.py`, `.gitignore`, `.gitattributes` and this report.
- **Excluded (left uncommitted):**
  - `run_india_trader.bat` and the workflow's second (India) cron. India is outside V1 scope, and that cron would re-run the US pipeline a second time each day.
  - `scratch/candidate_evaluation_results.json` (experiment output).
  - `architecture_visualization.html` (unrelated).
  - `reports/tax_2026.csv` and `reports/tax_2026.pdf` (generated output).
- **Backtest-only tooling** (in `src/backtest/backtest.py`; not used by live trading): point-in-time correlation price histories, `macro_cache_path` for deterministic macro data, and `apply_sector_rotation`. Defaults are unchanged. Experiment scripts and results stay in the session scratchpad.
