# AI Stock Trader

A **paper-trading (simulated money) system for US stocks**, built for learning and capital preservation.

> **Disclaimer.** This project is for education only and is not financial advice. It produces probabilistic estimates, not guarantees. V1 uses **simulated money only, with no broker integration**. Backtest results are optimistic, and real results are likely worse.

## Goal

**Do not lose money.** Break-even is acceptable and profit is a bonus. When in doubt the system stays in cash.

The frozen V1 strategy trails SPY buy-and-hold in historical walk-forward tests (see [V1_FREEZE_REPORT.md](V1_FREEZE_REPORT.md)). Forward paper trading exists to measure the strategy honestly, not to prove it works.

## How it works

1. **Data**: pulls daily US price history, saves the raw pull unmodified, and runs a validation gate. The gate rejects impossible values, flags ±50% spikes and gaps, and skips and logs bad tickers without filling fake data.
2. **Features and model**: technical, sentiment, macro and sector features feed a model. The active model is `logistic_regression_baseline_v1`. It predicts the probability that a stock rises more than 1% over the next 5 trading days.
3. **Risk engine**: sits between the model and the paper broker and can veto any trade. The model never trades directly.
4. **Paper broker**: simulates fills at the next open with a 0.2% cost per trade.
5. **Reporting**: provides a Streamlit dashboard, a decision log, a SPY 200-day shadow benchmark, weekly summaries, and Telegram and email alerts.

### Locked strategy values

| Setting | Value |
|---|---|
| Starting capital | $10,000 (paper) |
| Buy bar / signal exit | probability ≥ 0.60 / < 0.45 |
| Stop-loss / take-profit | −8% / +15% (trailing stop off) |
| Exit priority | Stop-loss, then take-profit, then signal exit, then hold |
| Positions / cash reserve | 3 equal-weight / ~15% |
| Direction | Long-only |
| Simulated cost | 0.2% per trade |
| Regime filter | No new buys when SPY is below its 200-day average |

### Honest-backtesting rules

- No look-ahead: decide on the prior close and fill at the next open.
- Costs are always applied, and only net returns are reported.
- Every backtest is shown beside SPY buy-and-hold.
- A known-answer test checks that "buy SPY and hold" reproduces SPY's real return.
- A suspiciously good result is treated as a bug to investigate.

## Project layout

```
config/       settings.py: the only place that reads env vars and secrets
src/
  data/       market data fetching and validation gate
  features/   feature engineering
  ml/         dataset, training, tuning, evaluation, drift, explainability
  risk/       risk engine and correlation checks
  trading/    paper broker, leaderboard, SPY shadow benchmark
  portfolio/  portfolio state
  ranking/    candidate ranking
  screening/  universe and screening
  intelligence/  earnings, macro, sector rotation, sentiment
  backtest/   walk-forward backtester
  pipeline/   daily pipeline, scheduler, run lock
  reports/    decision log, weekly summary, tax report
  alerts/     Telegram and email alerts
  db/         SQLAlchemy models and repository
  chatbot/    Gemini chat helper
dashboard/    Streamlit dashboard
tests/        pytest suite
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in your values
```

Secrets such as Telegram, Gemini, Alpha Vantage and the database URL belong in `.env`, which is gitignored. Only `config/settings.py` reads them. See [GITHUB_SETUP.md](GITHUB_SETUP.md) for the Supabase and secrets walkthrough.

## Usage

```bash
python -m src.pipeline.scheduler --force   # run the daily pipeline once
streamlit run streamlit_app.py             # open the dashboard
pytest                                     # run the test suite
```

Windows helper scripts: `run_trader.bat` (daily run with logging), `run_daily_status.bat`, `run_health_check.bat`, `run_retrain.bat`, `run_weekly_summary.bat`, `start_dashboard.bat`, and `setup_task_scheduler.ps1` (schedule the daily run).

## Documentation

- [PROJECT_PLAN.md](PROJECT_PLAN.md): master plan and locked decisions
- [CLAUDE.md](CLAUDE.md): always-on engineering rules
- [SIMPLE_GUIDE.md](SIMPLE_GUIDE.md): plain-English walkthrough of each phase
- [CORE_DOCUMENTATION.md](CORE_DOCUMENTATION.md): technical reference
- [V1_FREEZE_REPORT.md](V1_FREEZE_REPORT.md): frozen V1 configuration and benchmark results

## Status

V1 is frozen for forward paper trading. Real-money trading is out of scope until the checklist in `PROJECT_PLAN.md` §12 is satisfied.
