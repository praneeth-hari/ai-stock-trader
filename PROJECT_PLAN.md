# AI Stock Prediction & Paper-Trading System — Master Project Plan (V1)

> **Disclaimer.** This system produces probabilistic estimates, **not** guarantees, and nothing here is financial advice. All trading in V1 runs in a **paper (simulated) environment only** — there is no real money and no broker integration. Backtest performance does not guarantee future results. Real-money trading, and the India-vs-US market decision that comes with it, is explicitly **out of scope for V1** and is parked until the "Before Any Live Trading" gate (Section 12).

---

## 0. How to Use This Document

This is the single master roadmap. Follow it **phase by phase, top to bottom.** Each phase states what to build, why, what "done" looks like, and how to test it before moving on. **Do not skip phases** — later ones assume earlier ones are built and tested.

This plan is written to be handed to **Claude Code** (or built by hand). The companion file `CLAUDE.md` holds the short, always-on rules that keep the build aligned with the locked decisions below.

**The guiding philosophy of this entire project:** *When in doubt, protect the capital.* Cut losses fast, let winners run, sit in cash when uncertain, trust nothing without a known-answer test, assume reality is worse than the backtest, and never let real money near an unproven system. Every decision below serves that philosophy.

---

## 1. LOCKED DECISIONS (the heart of this plan)

These were decided deliberately and are **fixed for V1.** Everything else in the document must respect them. If any later section appears to contradict a value here, **this section wins.**

### 1.1 Goal & success bar

| Item | Decision |
|---|---|
| **Primary goal** | **Capital preservation.** Do not lose money; roughly break-even is acceptable; profit is a bonus. |
| **Starting capital** | **$10,000.00** (paper trading) — updated for V1 operation; Phase 0–13 development & backtesting used $50.00 micro-scale. |
| **Promoted model** | **Baseline (`logistic_regression_baseline_v1`)** — retained over Primary (HistGBM) for capital preservation during market drawdowns. |
| **Success = PASS** | All must hold: (1) final value ≥ starting value after all costs; (2) achieved *while actually trading* a reasonable amount, not by hiding in cash; (3) did not lose money during a bad market stretch (e.g. 2022); (4) any *gain* survives the anti-self-deception checks (§1.5). |
| **Benchmark** | Always report results **next to SPY buy-and-hold.** Staying flat while SPY falls is a *win* for this goal — you can only see that by showing both. |

### 1.2 Market & execution (V1)

| Item | Decision |
|---|---|
| **Market** | **US stocks.** |
| **Benchmark / market filter** | **SPY** (and the S&P 500 200-day trend). |
| **Tickers** | Standard US format (`AAPL`, `NVDA`, `MSFT`, …). No suffix. |
| **Fractional shares** | **Yes — in the simulator.** |
| **Real money / broker** | **NONE in V1.** Paper trading and backtesting only. |

### 1.3 The prediction target (the model's label)

| Item | Decision |
|---|---|
| **What the model predicts** | Probability that a stock will **rise more than +1% over the next 5 trading days.** |
| **Horizon** | **5 trading days** (~1 week) — low-noise, low-churn. |
| **Win threshold** | **+1%** — deliberately set **above** round-trip cost so a predicted "win" is a *profitable* win, not noise. |

### 1.4 Entry, exit & sizing rules

| Rule | Value | Purpose |
|---|---|---|
| **Buy bar** | model probability **≥ 0.60** | only act on real conviction |
| **Signal exit** | model probability **< 0.45** | exit when the reason to hold is gone |
| **Dead zone** | 0.45–0.60 (hold, no action) | prevents flip-flop churn |
| **Stop-loss** | **−8%** from entry | caps the loss on any bad pick — main capital protector |
| **Take-profit** | **+15%** from entry | banks gains before they evaporate |
| **Trailing stop** | **OFF in V1** | kept simple; may add post-V1 |
| **Exit priority** | Stop-loss → Take-profit → Signal exit → else Hold | **safety checked first, always** |
| **Positions held** | **3 stocks**, ~equal weight | modest diversification without over-spreading |
| **Cash reserve** | **~15%** always uninvested | dry powder + buffer |
| **Min trade size** | **$10** | keeps per-trade costs a small fraction |
| **Direction** | **Long-only** — no shorting | shorting imports unlimited-loss risk; fleeing to cash is the plan |

### 1.5 Anti-self-deception rules (trust & honest backtesting)

These exist so a backtest cannot lie to you. **A suspiciously good result is treated as a bug to hunt, not a success to celebrate.**

- **Benchmark is king** — every backtest shown next to SPY buy-and-hold.
- **Chronological splits only** — train on older data, test on newer, *never* shuffle time.
- **Walk-forward validation** — slide the train/test window forward repeatedly; a real edge persists across windows, a fluke does not. *(This is also the regime-robustness test.)*
- **The lag rule (kills look-ahead bias)** — decide on data available *before* the trade: use yesterday's close to decide, execute at today's open. Never use a day's own close to decide to trade that same day.
- **"Too good to be true" alarm** — spectacular returns almost always mean data leakage. Investigate, don't celebrate.
- **Known-answer test** — a "just buy SPY and hold" backtest must reproduce SPY's actual return. If it doesn't, the engine has a bug — fix it before trusting any other result.
- **Survivorship honesty** — validate the engine on **SPY first**, then broaden to a wider universe. Label every report: *"optimistic — real results likely worse."*
- **Net, never gross** — always subtract costs; never report gross returns.

### 1.6 Costs (V1 paper assumption)

| Item | Decision |
|---|---|
| **Simulated cost** | **0.2% per trade** (0.4% round trip), applied to every backtest & paper trade. |
| **Design consequence** | Prefer holding over churning; every trade is a toll. |
| **Broker fees** | N/A in V1 (paper). Real-broker/fee/forex modelling deferred to §12. |

### 1.7 Data reliability (the validation gate)

Raw data is never trusted. Every pull passes a **validation gate** before any decision uses it:

- **Sanity bounds** — reject price ≤ 0, `High` < `Low`, `Close` outside [`Low`,`High`], volume < 0.
- **Spike detection** — flag any single-day move beyond ±50% for review.
- **Gap detection** — compare expected vs received trading days; flag holes.
- **Fail loud, never silent** — bad data for a stock → **skip that stock for the day and log why.** Never guess, never fill fake numbers, never crash the whole run.
- **Isolation** — one bad stock never affects the others; a held stock with bad data is held untouched and flagged (never act on numbers you can't trust).
- **Raw snapshot** — save every raw pull unmodified (`data/raw/`) before cleaning; never overwrite raw.
- **Watch signal** — many skips at once = the data source is degrading; pause and check, don't push through.

### 1.8 Regime filter (risk-on / risk-off)

| Item | Decision |
|---|---|
| **Signal** | Is SPY **above** its 200-day average (rough "market healthy?" gauge)? |
| **Risk-OFF** (SPY below 200-day) | **No new buys.** Existing positions still managed/sold normally. |
| **Risk-ON** (SPY above 200-day) | Buying allowed. |
| **Known trade-off** | The 200-day is lagging; it may whipsaw or miss a crash's start. Accepted — it trades a little missed upside for a lot of downside protection, which fits the goal. |
| **Training** | Train the model across **multiple regimes** (include a bad year like 2022), not one calm stretch. |
| **Staleness** | Retrain periodically on fresh data; watch for performance drift over time. |

### 1.9 Operations & safety

| Item | Decision |
|---|---|
| **Hosting** | **Run free on your own laptop.** Cloud (AWS) is **skipped indefinitely.** |
| **Maintenance** | Logs + staleness alarm are the early-warning system; a ~10-min weekly check. Dependencies pinned (`requirements.txt`). |
| **Before real money** (all deferred) | Weeks of paper history first → **human approves every real trade** → **kill-switch** → start with losable money. |

---

## 2. Before Any Live Trading — DEFERRED (do not build in V1)

Captured here so it is not forgotten, and explicitly **excluded** from the V1 build. Revisit only when genuinely ready to consider real money (months away):

- **India-vs-US market choice** for real money.
- **Costs differ by market.** US real brokers ≈ $0 commission. **India is not zero:** STT ~0.1% buy + ~0.1% sell, plus exchange/SEBI/stamp/GST, plus a **flat DP fee (~₹15) on every sell** that makes very small trades unviable and raises the practical minimum account size. India also **largely lacks fractional shares** for local stocks (whole shares only).
- **US-from-India plumbing:** RBI Liberalized Remittance Scheme (LRS), ₹→$ forex conversion cost (~0.5–2%), currency (₹/$) risk in the returns math, platform choice (INDmoney / Vested / NSE IX Global Access), and US tax paperwork (W-8BEN, dividend withholding).
- **Human-approval layer + kill-switch** for live orders.
- **Upgrade to reliable paid data** (Alpha Vantage / Polygon / Twelve Data — free tiers exist) before risking money on scraped data.

**None of the above affects the V1 build.** The design is deliberately currency- and market-agnostic under the hood: only the ticker format and the benchmark differ between markets.

---

## 3. Reality Check (read before building)

- **The model is a weak edge, not a crystal ball.** Daily/weekly stock-direction models typically land around ~52–55% accuracy — barely better than a coin flip. That is *normal* and can still be useful inside a disciplined risk framework. Do not expect more.
- **This will probably not make money.** The honest, valuable outcome is a *ruthlessly honest backtest* that tells you whether the strategy preserves capital. If it can't beat "do nothing / hold cash / hold SPY," that is a finding worth having — it saved you real money.
- **The real cost is time.** The build is ~13 phases; the original estimate is ~30 days of near-daily effort for one person. Right-size your pace to what you can sustain.

---

## 4. Architecture

Layers are separated so each can be built, tested, and replaced independently. **The ML layer never talks to the broker directly — the Risk Engine always sits in between.** That separation is a core safety rule.

```
                         Dashboard (Streamlit)
                                │  Direct Python/DB
                                ▼
                      src.db.repository
                                ▲
                                │
                      Orchestrator / daily pipeline
                                │
    ┌──────────┬──────────┬─────┴─────┬───────────┬──────────────┬───────────────┐
    ▼          ▼          ▼           ▼           ▼              ▼               ▼
 Market    Data      Feature       ML        Ranking       Risk Engine    Portfolio
 Data   Validation  Engineering  Predict    (score/sort)   (can VETO)      Manager
    │          │          │           │           │              │               │
    └──────────┴──────────┴───────────┴───────────┴──────────────┴───────────────┘
                                │
                                ▼
                          Paper Broker  ──►  SQLite / PostgreSQL (source of truth)
                                ▲
                    Scheduler (cron / APScheduler)
                    Logging & Monitoring (cross-cutting)
```

**Component responsibilities (V1/V2):**

| Layer | Responsibility |
|---|---|
| Dashboard (Streamlit) | Standalone operator inspection: read-only views of predictions, portfolio, trades, backtests, risk metrics + trigger buttons. Queries `src.db.repository` directly. |
| Backend API (FastAPI) | *(Built then removed in V2.x — deprecated in favor of unified, direct Streamlit dashboard to eliminate network overhead).* |
| Market Data | Pull OHLCV via `yfinance`; normalize to a common schema; snapshot raw. |
| Data Validation | The §1.7 validation gate: sanity/spike/gap checks, skip-and-log. |
| Feature Engineering | Compute technical indicators from clean data (RSI, MACD, moving averages, momentum, volatility, etc.). |
| ML Predict | Load trained model; output P(rise > +1% over 5 days) per stock. |
| Ranking | Combine probability with simple factors (e.g. momentum/liquidity) into one score; sort. |
| Risk Engine | Enforce §1.4 sizing/exposure/stop/target/reserve **and** the §1.8 regime filter. **Can veto any trade.** |
| Portfolio Manager | Turn approved candidates + capital into concrete (fractional) share quantities. |
| Paper Broker | Simulate fills at the correct lagged price, apply §1.6 costs, hold virtual cash/positions, record trades. |
| Backtesting Engine | Replay history through the *same* pipeline in simulation mode; enforce §1.5. |
| SQLite / PostgreSQL | Single source of truth for data, features, predictions, portfolio, orders, trades, logs. |
| Scheduler | Trigger the daily pipeline via APScheduler. |
| Logging/Monitoring | Structured logs for every prediction, decision, and error; drift/staleness watch. |
| Config | Centralized `.env`-driven settings (all §1 values live here). |

**Communication:** Streamlit queries `src.db.repository` directly in Python. All engines read/write through the thin repository layer, never scattered raw SQL. The Scheduler triggers the Orchestrator, which calls each engine in sequence and persists results at each step so a mid-pipeline failure doesn't lose earlier work.

---

## 5. Technology Stack (all free / open-source)

| Purpose | Tool |
|---|---|
| Language | Python 3.11+ |
| Market data | `yfinance` |
| Data handling | `pandas`, `numpy` |
| ML | `scikit-learn`, `xgboost` (and/or `lightgbm`) |
| Technical indicators | `pandas`/`numpy` (or `ta`) |
| Database | SQLite (dev) / PostgreSQL + SQLAlchemy |
| Dashboard | Streamlit (standalone operator UI) |
| Scheduling | APScheduler (or system cron) |
| Config | pydantic-settings + `.env` |
| Testing | pytest |
| Version control | Git |

**No paid software is required anywhere in V1.** Only a computer, internet, and time.

---

## 6. Folder Structure

```
ai-stock-trader/
├── config/
│   ├── __init__.py
│   └── settings.py            # ALL settings load here (the only reader of env)
├── src/
│   ├── data/
│   │   ├── market_data.py     # yfinance pull + raw snapshot
│   │   └── validation.py      # the validation gate (§1.7)
│   ├── features/
│   │   └── features.py        # technical indicators
│   ├── ml/
│   │   ├── dataset.py         # build labelled dataset (label = rise>+1% over 5d)
│   │   ├── train.py           # train model (chronological split)
│   │   └── predict.py         # load model, output probabilities
│   ├── ranking/
│   │   └── ranking.py         # score & sort
│   ├── risk/
│   │   └── risk_engine.py     # sizing, stops, reserve, regime filter, VETO
│   ├── portfolio/
│   │   └── portfolio.py       # capital → fractional share quantities
│   ├── trading/
│   │   └── paper_broker.py    # simulated fills, costs, positions, trades
│   ├── backtest/
│   │   └── backtest.py        # replay pipeline in sim mode; enforce §1.5
│   ├── pipeline/
│   │   └── daily_pipeline.py  # orchestrator: data→…→trade
│   └── db/
│       └── repository.py      # thin DB access layer
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   └── data_loader.py         # DB data loader for UI
├── tests/                     # pytest tests, one area per module
├── data/
│   ├── raw/                   # untouched raw pulls (git-ignored)
│   ├── processed/             # cleaned data (git-ignored)
│   └── models/                # saved model files (git-ignored)
├── .env.example
├── .env                       # local secrets/settings (git-ignored)
├── .gitignore
├── requirements.txt
├── CLAUDE.md                  # always-on rules for Claude Code
└── PROJECT_PLAN.md            # this document
```

---

## 7. Development Phases

Each phase: **Goal → Build → Done when → Test.** Build in order. Every phase must respect Section 1.

> **Note on ordering:** Features (Phase 3) come *before* the ML dataset (Phase 4) because the dataset is built *from* features. The database (Phase 2b) is introduced early but can start as simple file/CSV storage and be upgraded to PostgreSQL when convenient — do not let a database block progress on the strategy.

### Phase 0 — Environment Setup ✅ (DONE)
- **Goal:** A runnable project skeleton with centralized config.
- **Build:** `config/settings.py` (loads all §1 values from `.env`), `.env.example`, `.env`, `.gitignore`, `requirements.txt`, folder structure.
- **Done when:** `python -c "from config.settings import settings; print(settings.initial_capital)"` prints the configured capital with no error.
- **Test:** the command above returns the expected number.
- **Status:** Already built and verified.

### Phase 1 — Market Data
- **Goal:** Reliably pull US OHLCV data and snapshot it raw.
- **Build:** `src/data/market_data.py` — pull daily OHLCV for a list of US tickers via `yfinance`; save each raw pull unmodified to `data/raw/`; normalize into a common DataFrame schema (date, open, high, low, close, volume, ticker).
- **Done when:** You can pull a small universe (e.g. SPY + 5–10 large caps) for a date range and see a clean, normalized table, with the raw file saved.
- **Test:** pull SPY for a known period; row count and last close look sane; raw snapshot exists on disk.

### Phase 2 — Data Validation (the gate)
- **Goal:** No decision ever runs on untrusted data.
- **Build:** `src/data/validation.py` implementing the full §1.7 gate: sanity bounds, spike detection (±50%), gap detection, **skip-and-log** per stock, isolation, held-stock-with-bad-data handling. Emit a clear per-run log of what was skipped and why.
- **Done when:** Feeding deliberately broken data (a negative price, a 400% spike, a missing week) results in that stock being skipped and logged, while good stocks pass untouched.
- **Test:** unit tests with crafted bad rows; assert the right stocks are excluded and the log explains why.

### Phase 2b — Storage / Database
- **Goal:** A single source of truth for data, predictions, portfolio, trades, logs.
- **Build:** `src/db/repository.py` — start with a simple, clean interface (CSV/SQLite acceptable early), upgradeable to PostgreSQL + SQLAlchemy. Tables/entities: market_data, features, predictions, portfolio, orders, trades, logs.
- **Done when:** Each engine reads/writes only through the repository, never scattered SQL.
- **Test:** round-trip write→read for each entity.

### Phase 3 — Feature Engineering
- **Goal:** Turn clean prices into model inputs.
- **Build:** `src/features/features.py` — compute technical indicators per stock: moving averages (e.g. 20/50/200-day), RSI, MACD, momentum, rolling volatility, price-vs-MA distance, volume features. **Respect the lag rule** — every feature for a given decision date uses only data available *before* that date.
- **Done when:** For a stock you get a tidy feature table aligned by date, with no forward-looking leakage.
- **Test:** spot-check an indicator against a hand calculation; assert no feature uses same-day close to predict same-day action.

### Phase 4 — ML Dataset (the label)
- **Goal:** A labelled, chronologically-ordered training set.
- **Build:** `src/ml/dataset.py` — join features with the **label: 1 if the stock rises > +1% over the next 5 trading days, else 0.** Keep strict time order. No shuffling.
- **Done when:** Dataset has features + binary label, ordered by date, with the future-looking label correctly computed (and only used as the target, never as a feature).
- **Test:** verify a few labels by hand against raw prices; confirm the label horizon is exactly 5 trading days and threshold +1%.

### Phase 5 — Model Training
- **Goal:** A trained classifier that outputs P(rise > +1% over 5 days).
- **Build:** `src/ml/train.py` — train XGBoost/LightGBM (or start with logistic regression as a baseline) using a **chronological split** (older = train, newer = test). Train across multiple regimes (include a down year). Save the model to `data/models/`.
- **Done when:** Model trains, saves, and reloads; test-set output is a probability per row.
- **Test:** confirm split is by time (not random); confirm a saved model reloads and predicts.

### Phase 6 — Model Evaluation (honesty first)
- **Goal:** Know if the model has any real, honest edge.
- **Build:** `src/ml/` evaluation — accuracy, precision/recall, and probability calibration on the **unseen** test period. Expect ~52–55% and treat anything spectacular as suspected leakage (§1.5).
- **Done when:** You have honest test-period metrics and have actively checked for leakage.
- **Test:** run the "too-good alarm" — if metrics look amazing, hunt the bug before proceeding.

### Phase 7 — Stock Ranking
- **Goal:** Turn probabilities into an ordered opportunity list.
- **Build:** `src/ranking/ranking.py` — combine model probability with simple factors (e.g. momentum, liquidity) into one score; sort the universe.
- **Done when:** Given a day's predictions, you get a sorted candidate list.
- **Test:** assert the highest-probability, healthiest names rank at the top.

### Phase 8 — Risk Engine (the safety gate) ✅ (DONE)
- **Goal:** The wall between "model wants to buy" and "money moves." Enforces every §1.4 and §1.8 rule and **can veto any trade.**
- **Build:** `src/risk/risk_engine.py` — enforce: buy bar ≥ 0.60; **exit priority stop(−8%) → target(+15%) → signal(<0.45) → hold**; max 3 positions; ~15% cash reserve; $10 min trade; **regime filter** (no new buys when SPY < 200-day); long-only.
- **Done when:** The engine correctly approves/vetoes trades against all rules, including refusing new buys in risk-off.
- **Test:** unit tests for each rule: a −9% position is force-sold; a 0.58 signal is rejected; buys are blocked when SPY is below its 200-day.
- **Status:** Complete. 10/10 risk unit & interaction tests passed; 114/114 full project tests passed.

### Phase 9 — Portfolio Allocation ✅ (DONE)
- **Goal:** Convert approved candidates + capital into concrete (fractional) share quantities with strict 4-decimal floor-rounding and 0.2% fee accounting.
- **Build:** `src/portfolio/portfolio.py` — fee-inclusive buy outflows (`gross + fee <= alloc`), net-proceeds sell inflows (`gross * 0.998`), 4-decimal floor-rounding, pro-forma balance reconciliation, and 15% statutory cash reserve compliance.
- **Done when:** Given $50 micro-scale (historical development scale) and approved names, it outputs exact fractional quantities and broker-executable `OrderSpec` objects that obey all sizing rules.
- **Test:** with $50 historical test scale, 1 hold, 1 stop-loss exit, 1 candidate buy, assert ~$14.16 buy allocation, exact net cash flows, and cash reserve maintained >= $7.50.
- **Status:** Complete. 6/6 unit & integration tests passed; 120/120 full project tests passed.

### Phase 10 — Backtesting Engine (the most important deliverable) ✅ (DONE)
- **Goal:** A ruthlessly honest replay of the whole pipeline on history with anti-self-deception rules (§1.5).
- **Build:** `src/backtest/backtest.py` — replay data → features → predict → rank → risk → allocate → fill at open (the lag rule), **0.2% cost per trade**, benchmark vs SPY always shown, survivorship honesty (SPY-first), net-not-gross, "too-good = bug", and the Known-Answer Test.
- **Done when:** The Known-Answer Test passes (0.0004% diff <= 0.05%), and full walk-forward + contiguous multi-year backtests run at historical $50 scale (and scale-agnostic engine), reporting net returns, drawdowns, trade logs, and the optimistic label.
- **Test (critical):** the **known-answer test** — SPY Buy & Hold 2023 with $10,000 capital matched closed-form expected return to within 0.0004% points.
- **Status:** Complete. 3/3 backtest tests passed; 123/123 full project tests passed.

### Phase 11 — Paper Trading Loop
- **Goal:** Run the strategy forward on live-ish data with simulated money.
- **Build:** `src/trading/paper_broker.py` + wire into `src/pipeline/daily_pipeline.py` — each day: pull → validate → feature → predict → rank → risk → allocate → simulate fills (correct lag, 0.2% cost) → persist. Cash is a valid position; a quiet day is success.
- **Done when:** The daily pipeline runs end-to-end on paper and updates a simulated portfolio with a full log.
- **Test:** run several simulated days; verify positions, cash, and logs evolve correctly and all rules hold.

### Phase 12 — Dashboard
- **Goal:** See what the system is doing.
- **Build:** `dashboard/app.py` (Streamlit) — read-only views: current portfolio, recent trades, latest predictions/ranking, backtest equity curve vs SPY, risk/skip logs. A couple of trigger buttons (run backtest, run one paper cycle).
- **Done when:** You can watch the system's state and history at a glance.
- **Test:** dashboard reflects the DB truthfully after a paper cycle.
- **Status:** Complete. 7/7 dashboard tests passed; 136/136 full project tests passed.

### Phase 13 — Automation
- **Goal:** Run the daily pipeline on schedule, unattended, **for free on your laptop.**
- **Build:** APScheduler or cron to trigger `daily_pipeline.py` (§1.9: run before US market open to use the prior finalized close, act at the open — or after close queuing for next open). Pin dependencies.
- **Done when:** The pipeline runs on schedule and logs each run.
- **Test:** scheduled run produces the same result as a manual run; failures are logged loudly.
- **Status:** Complete. 5/5 scheduler tests passed; 141/141 full project tests passed.

### Phase 14 — Cloud Deployment — **SKIPPED in V1**
- Deliberately out of scope. Runs free on your laptop. Revisit only if you ever want true 24/7 unattended operation (free/cheap tiers exist).

### Phase 15 — Real Broker Integration — **DEFERRED (see §2)**
- Out of scope for V1. Gated behind: weeks of paper history → human approves every trade → kill-switch → losable money → the India-vs-US / forex / tax decisions.


---

## 8. Code Implementation Order (dependency-safe)

1. `config/settings.py` ✅
2. `src/data/market_data.py` (Phase 1) ✅
3. `src/data/validation.py` (Phase 2) ✅
4. `src/db/repository.py` (Phase 2b) ✅
5. `src/features/features.py` (Phase 3) ✅
6. `src/ml/dataset.py` (Phase 4) ✅
7. `src/ml/train.py` (Phase 5) ✅
8. `src/ml/predict.py` + evaluation (Phase 6) ✅
9. `src/ranking/ranking.py` (Phase 7) ✅
10. `src/risk/risk_engine.py` (Phase 8) ✅
11. `src/portfolio/portfolio.py` (Phase 9) ✅
12. `src/backtest/backtest.py` (Phase 10) ✅
13. `src/trading/paper_broker.py` (Phase 11 core, used by 10) ✅
14. `src/pipeline/daily_pipeline.py` (Phase 11) ✅
15. `api/main.py` (thin, optional early)
16. `dashboard/app.py` (Phase 12) ✅
17. `src/pipeline/scheduler.py` (Phase 13) ✅

Build one file, test it, then move on. Do not batch-generate untested code.

---

## 9. MVP Definition (what "V1 done" means)

V1 is complete when **all** of these are true:

1. Data for a US universe pulls and passes the validation gate (skip-and-log works).
2. Features compute with no look-ahead leakage.
3. A model trains on a chronological split and outputs P(rise > +1% / 5 days), with honest test metrics.
4. The Risk Engine enforces every §1.4/§1.8 rule and can veto (incl. risk-off no-buy).
5. The Backtesting Engine passes the **known-answer SPY test**, reports **net** returns **beside SPY**, and carries the "optimistic" label.
6. The paper-trading loop runs end-to-end, updating a simulated portfolio with full logs.
7. The dashboard shows portfolio, trades, predictions, and backtest-vs-SPY.
8. The daily pipeline runs on a schedule, for free, on your laptop.
9. **No real-money code exists anywhere.**

**V1 explicitly does NOT include:** real money, any broker, cloud hosting, shorting, trailing stops, options/derivatives, or the India/forex real-money layer.

---

## 10. Post-V1 Ideas (all optional, none in V1)

Trailing stops; a larger/smarter universe; better data source; news/sentiment features; model retraining automation & drift dashboards; more sophisticated position sizing; the real-money layer (with its §2 gate). Decide these only after V1 has proven itself on paper.

---

## 11. Important Engineering Rules

1. **Section 1 is law.** If code and §1 disagree, §1 wins.
2. **ML never touches the broker directly** — the Risk Engine is always between them.
3. **All config comes from `config/settings.py`** — nothing else reads the environment directly; secrets live in `.env`, never hardcoded.
4. **Never commit `.env`, raw data, or model files** (they're git-ignored).
5. **Fail loud, log everything** — every prediction, trade decision, skip, and error is logged.
6. **No look-ahead, ever** — obey the lag rule in features, dataset, backtest, and paper loop.
7. **Net, never gross** — costs (0.2%/trade) apply in every simulated fill.
8. **A great backtest is a suspect** — run the known-answer test; hunt leakage before trusting results.
9. **Cash is a valid position** — the system is allowed (and expected) to hold cash when nothing clears the bar or the market is risk-off.
10. **Paper only in V1** — no real-money path is written until the §2 gate is deliberately opened.
11. **Build phase by phase, test each** — later phases assume earlier ones are done and verified.
12. **No guaranteed-profit claims** — anywhere, ever.

---

## 12. Before-Any-Live-Trading Gate (single source of truth for deferral)

Do **not** write any real-money or broker code until every item is deliberately worked through:
- [ ] Weeks of successful paper-trading history reviewed.
- [ ] Human-approval layer designed (system proposes, you approve each trade).
- [ ] Kill-switch implemented and tested (halt all trading; optionally flatten to cash).
- [ ] Market chosen for real money (US vs India) with eyes open.
- [ ] If US-from-India: LRS process, forex cost, currency risk, platform, W-8BEN understood.
- [ ] If India: STT + flat DP fees + whole-share (no fractional) constraints modelled; minimum viable account size recalculated.
- [ ] Reliable (ideally paid) data source in place.
- [ ] Starting real amount is money you can afford to lose entirely.

Until then, V1 stays paper-only.

---

## 13. Master Build Checklist

- [x] Phase 0 — Environment setup (verified)
- [x] Phase 1 — Market data
- [x] Phase 2 — Data validation gate
- [x] Phase 2b — Storage / repository
- [x] Phase 3 — Feature engineering (no leakage)
- [x] Phase 4 — ML dataset (label = rise>+1% / 5 days)
- [x] Phase 5 — Model training (chronological split)
- [x] Phase 6 — Honest evaluation (+ leakage hunt)
- [x] Phase 7 — Ranking
- [x] Phase 8 — Risk engine (all rules + regime filter + veto)
- [x] Phase 9 — Portfolio allocation (3 pos, ~15% cash, $10 min, fractional)
- [x] Phase 10 — Backtesting (**known-answer SPY test**, net-vs-SPY, optimistic label)
- [x] Phase 11 — Paper-trading loop (end-to-end)
- [ ] Phase 12 — Dashboard
- [ ] Phase 13 — Automation (free, on laptop)
- [ ] Phase 14 — Cloud — SKIPPED
- [ ] Phase 15 — Real broker — DEFERRED (§12 gate)

---

## 14. Suggested Pace

The original estimate is ~30 days of near-daily effort for one person; treat that as a guide, not a deadline. A sustainable rhythm (e.g. one phase per sitting, testing each) matters more than speed. A stalled-but-correct build beats a fast-but-untrusted one. Right-size to the time you actually have.

---

*End of V1 Master Plan. Companion file: `CLAUDE.md` (always-on rules for Claude Code).*
