# Simple Guide — AI Stock Trader (Plain English)

A friendly guide to what this system actually does, written without coding jargon.

---

## Phase 0 — Giving the System a "Settings Brain"

### What can the system do now that it couldn't before?
Before this phase, there was nothing here — just an empty folder. Now, the project has a central **"settings brain"** (`config/settings.py`).

Think of this like establishing the house rules before playing a board game. Instead of scattering numbers and rules randomly all over the place where someone could change or break them by accident, the system now has one single, secure control room that holds every rule:

- **How much virtual cash we start with** (e.g., $10,000 in paper money; initial development started at $50 micro-scale).
- **Our safety brakes**: Never let a single stock loss go past 8% before cutting it, take profits if a stock hits 15%, and always keep at least 15% of our portfolio in cash as an emergency cushion.
- **Our shopping list**: Which US stocks we are allowed to look at (like Apple, Microsoft, Nvidia).
- **Trading fees**: Reminding the system that every trade costs money (0.2%), so we don't trick ourselves into thinking buying and selling constantly is free.

Nothing is trading yet, but the system now has clear, unbreakable boundaries that everything built next will automatically follow.

---

## Phase 1 — Giving the System "Eyes" to Look at the Stock Market

### What can the system do now that it couldn't before?
In Phase 0, the system only knew the rules. Now, in Phase 1, it has **eyes**: it can reach out to the US stock market and pull down real price history (Open, High, Low, Close, Volume) for any stock we ask for.

Here is what happens when it looks at stock data:

1. **It takes a permanent snapshot first**: Before doing anything else, it saves an exact, untouched photocopy of what it received into a date-stamped folder (`data/raw/YYYY-MM-DD/`). Even if something goes wrong later, our original evidence of what the market was doing that day is never erased or overwritten.
2. **It puts the data in standard uniform**: Market data often arrives messy or formatted differently depending on the ticker. The system cleans up the columns into a tidy, uniform table (`date`, `open`, `high`, `low`, `close`, `volume`, `ticker`).
3. **It raises a caution flag**: Crucially, the system puts a big **"UNVALIDATED"** caution tape on this raw data. Just because we fetched numbers from the internet doesn't mean we believe them yet — a glitch could give us a negative stock price or a fake 500% jump. The system knows it is forbidden to make any trading decisions until the inspector (the Phase 2 validation gate) checks every single row.

---

## Phase 2 — Hiring a Strict Quality Inspector

### What can the system do now that it couldn't before?
In Phase 1, the system could receive market data, but it was totally unguarded — it had no way to distinguish good data from garbage. Phase 2 adds a **quality inspector** who stands at the gate and checks every number before it is allowed anywhere near a trading decision.

The inspector works in a very specific order:

1. **Impossible number check (grounds for immediate rejection)**: The inspector knows certain things are physically impossible in real markets — a stock price can't be negative, the day's highest price can't be lower than the lowest price, and "shares traded" can't be a negative number. If any of these are spotted, that stock is **immediately rejected** for the entire day, a clear reason is written in the log, and all other stocks continue on unaffected.

2. **"Did it teleport?" check (grounds for rejection)**: If a stock's price moves more than 50% overnight, the inspector treats it as suspected bad data — not a real market event. In reality, a stock like Apple or Microsoft has virtually never moved 50% in a single day. This almost always means a missing decimal point, an unadjusted stock split, or a data error.

3. **"Are there gaps?" check (just a warning, not rejection)**: The inspector also checks for missing days. But importantly, it uses the **official US stock market holiday calendar** to distinguish expected closures (weekends, Martin Luther King Day, Thanksgiving, etc.) from genuinely missing data. A missing Thursday that should have been a trading day raises a flag; a missing Monday that is a holiday does not. Gaps are a concern but don't throw away the data we do have — they just get noted.

4. **One bad apple doesn't spoil the barrel**: If Apple's data is corrupt, Microsoft and Nvidia are unaffected. Each stock is inspected completely independently.

5. **Special rule for stocks we already own**: If a stock we currently hold has bad data, the system doesn't panic-sell it. It issues a high-priority alert: *"Don't touch this position — hold it exactly as-is until clean data arrives."*

The system can now receive market data AND ensure only clean, trustworthy data ever reaches any decision-making component.

---

## Phase 2b — Building the Filing Cabinet

### What can the system do now that it couldn't before?
Until now, the system could fetch market data and check it for quality — but everything was happening in memory, like doing calculations on a whiteboard that gets erased the moment you walk away. Phase 2b adds a **permanent filing cabinet**: a real database that stores everything the system ever sees, decides, or does.

Think of this like a bank building's record room. Every document has an official place:

- **Drawer 1 — Prices**: Every clean, validated stock price goes here, organized by stock and date. A price can only be filed once per day per stock — no duplicates, no overwriting.
- **Drawer 2 — Report cards**: The feature numbers the system computes about each stock (momentum, volatility, etc.) get filed here. If updated information comes in, the old report card is replaced.
- **Drawer 3 — Gut-feelings**: The model's probability estimate ("I'm 73% confident this stock will rise") for each stock each day.
- **Drawer 4 — Daily balance sheet**: At the end of every day, a snapshot of how much cash we have and what positions we hold.
- **Drawer 5 — Decision log**: Every buy/sell/hold decision the system makes, with a written reason — like a trade blotter.
- **Drawer 6 — Fill confirmation**: Every simulated "trade executed" record, with the price, fee charged, and profit or loss.
- **Drawer 7 — Audit trail**: Every significant event — every check passed, every skip, every error — logged permanently so nothing is ever done silently.

Two important properties of this filing cabinet:

1. **Only one key exists.** There's a single "librarian" (`repository.py`) that any part of the system must go through to read or write. No component can bypass it and scribble directly in the drawers. This prevents chaos.

2. **It starts simple, upgrades cleanly.** Right now it uses a lightweight local database (SQLite — one file on disk, no server needed). When this project grows and needs a server-grade database, we change a single line of configuration and everything works the same — no rewriting.

---

## Phase 3 — Teaching the System to Read the Market

### What can the system do now that it couldn't before?

Until now, the system could receive and store price data — but it couldn't *interpret* it. Phase 3 gives the system a set of "reading glasses": 19 measurements it extracts from raw price and volume history to form a view of each stock's momentum, trend, and risk.

Think of it like a doctor running tests before making a diagnosis. Before seeing a patient, the doctor collects blood pressure, temperature, heart rate, and test results. Only once those measurements are in hand does the doctor form a judgment. Our system does the same thing before making any trading decision.

The 19 measurements include things like:

- **How is the stock trending?** Is it above or below its 200-day average? Is the short-term average rising faster than the long-term one (a "golden cross" — classically bullish)?
- **How much momentum does it have?** Has it risen 5% in the last week? 21% in the last month?
- **Is it overbought or oversold?** RSI (Relative Strength Index) is a classic indicator that flags when a stock has risen too far too fast (overbought) or fallen too hard (oversold).
- **What's the trend strength signal?** MACD tells you whether short-term momentum is accelerating or fading relative to the longer trend.
- **How volatile is it?** A wild stock that swings 3% a day is riskier than one that moves 0.3%.
- **Is there unusual buying activity?** A sudden spike in trading volume is often a meaningful signal.
- **Is it beating the market?** We compare each stock's recent performance to SPY (the S&P 500 index fund) to see if it's outperforming or lagging.

**The most important property: no cheating.**

Every measurement is computed using only data that would have been available the night before a decision. The system cannot "peek" at tomorrow's price to make today's measurement look better. We built and ran five adversarial tests that specifically try to detect if any measurement accidentally uses future data — all five confirm there's no leakage.

**Extra history is fetched automatically.**

The 200-day moving average needs 200 days of data before it can produce its first valid reading. So whenever we ask for data starting from, say, January 2024, the system automatically fetches an extra ~10 months of history going back to March 2023. Those extra rows are used only for warm-up and are discarded before training.

---

## Phase 4 — Creating the Training Flashcards (The Target Label)

### What can the system do now that it couldn't before?

Until now, the system could collect raw numbers (Phase 1), verify they weren't corrupt (Phase 2), store them safely (Phase 2b), and compute 19 market measurements (Phase 3). But an AI cannot learn how to trade unless it has **flashcards with an answer key**.

Phase 4 builds this training set. Think of each day for each stock as a study flashcard:
- **The Front of the Card (Questions / Features)**: All 19 measurements from Phase 3 (momentum, trend, volatility, RSI, etc.) as of the close of day T.
- **The Back of the Card (The Answer / Target Label)**: Did this stock rise by **more than +1.0%** over the next **5 trading days**?
  - If yes: **1** (a profitable win).
  - If no (flat, dropped, or only gained 0.8%): **0** (not a win).

### Why this target?
A 1% threshold is deliberately set above our simulated trading fee (0.4% round-trip). That way, the AI is taught to hunt for moves that actually generate net profit, rather than chasing market noise.

### The Safety Rules Built into Phase 4:

1. **No peeking across stocks (Per-Ticker Isolation)**:
   When building flashcards for multiple stocks (like Apple and Microsoft together), each stock's future prices are kept strictly in their own separate lane. Apple's future price is *never* accidentally grabbed from Microsoft's data.

2. **The "Unwritten Future" (Tail Truncation)**:
   For the most recent 5 trading days in our database, day T+5 hasn't happened yet in the real world! We cannot possibly know whether the stock will rise or fall. Those 5 days are cleanly set aside and dropped from the training set. We never guess, fabricate, or fill them with fake answers.

3. **No writing answers on the front of the card (Zero Contamination)**:
   The future price and forward return are strictly used to determine the 1 or 0 answer on the back. They are never allowed into the feature columns on the front. Our automated leakage tests verify that changing future prices does not alter past features by even a single trillionth of a cent.

4. **Honest Class Balance**:
   If a test deck had 95% "Yes" answers, an AI could get an "A" just by blindly guessing "Yes" every time without learning anything. Phase 4 actively monitors the balance of 1s and 0s and logs a warning if either side becomes too dominant.

---

## Phase 5 — Training the AI Brains (Dual Models & The Embargo Gap)

### What can the system do now that it couldn't before?

Until now, we had flashcards with questions on the front (19 market indicators) and answers on the back (did the stock rise > 1% over the next 5 days?). Phase 5 is where the AI actually **studies those cards to learn how to predict the future**.

Think of this like preparing a student for a final exam:
1. **Study Period (Training Data)**: We give the student older flashcards (the first 80% of our timeline) so it can discover patterns (e.g., "when RSI is below 30 and volume spikes, this stock usually rises").
2. **The Final Exam (Test Data)**: We test the student on newer, unseen flashcards (the last 20% of our timeline) to see if it actually learned real principles or just memorized old answers.

### The Safety Rules Built into Phase 5:

1. **The 5-Day "Blackout Curtain" (The Embargo Gap)**:
   Because our flashcard answers look 5 trading days into the future, any flashcard from the very end of the study period would have an answer key computed using price data from *inside* the final exam!
   To prevent this subtle cheating, the system automatically draws a 5-day blackout curtain between the study period and the exam. Those 5 buffer days are completely thrown away so no exam answers can leak into the study room.

2. **Two Students in the Classroom (Dual Models)**:
   We train two different AI models side-by-side:
   - **The Simple Student (Baseline — Logistic Regression)**: A straightforward linear model that acts as our benchmark. It’s transparent, simple, and hard to fool.
   - **The Clever Student (Primary — Gradient Boosting)**: A tree-based model (`HistGradientBoostingClassifier`) that can spot complex non-linear combinations (like "moving average crossed AND volatility is low").
   If the clever student can't clearly outperform the simple student, we know it’s just hallucinating noise.

3. **No Automatic Hiring (Human Review Required)**:
   The software does *not* automatically promote either model to trade your money. Both models are marked as **candidates**. In Phase 6, you inspect their report cards, calibration curves, and crash-resilience before personally deciding which one to promote.

4. **The "Ruler Never Changes" (Fixed Scaler)**:
   The simple student needs prices and indicators normalized onto a standard 0-to-1 scale. That "ruler" (`StandardScaler`) is measured **once** on the study period, saved permanently inside the model file, and never re-adjusted during tests or live trading.

---

## Phase 6 — The Honesty Test (Evaluating the Brains Across Market Seasons)

### What can the system do now that it couldn't before?

Until now, we could train models on history — but testing a model on one single quiet period can deceive you. Phase 6 is the system's **rigorous, multi-year report card**. It forces our two AI models to walk through 4.5 years of real market history, testing them across completely different market seasons:
1. **The 2021 Stimulus Bull Run**: Easy money, strong uptrend.
2. **The 2022 Inflation & Rate-Hike Crash**: A brutal bear market where most stocks cratered.
3. **The 2023 Tech Rebound**: The AI boom recovery.
4. **The 2024 Mature Market**: Late-cycle market conditions.

### The Honesty Rules Built into Phase 6:

1. **Edge Over Base Rate (Not Just Raw Percentages)**:
   If a teacher gives a test where 65% of the questions are "True", a student getting 62% didn't learn anything — they actually did worse than guessing!
   Phase 6 calculates the **Base Rate** (the actual percentage of winning moves) for *each specific window* and compares it directly against high-conviction calls ($P \ge 60\%$). A model is only credited with **real edge** if its conviction win rate beats that window's base rate.

2. **The 2022 Bear Market Veto (Capital Preservation)**:
   Our #1 rule is protecting money (§1.1). In 2022, when the market crashed, winning moves dropped to just 37.8%.
   Our simple baseline model (`LogisticRegression`) stepped up: it identified high-conviction trades with a **44.7% win rate** — beating the grim market base rate by **+6.9 percentage points**! Meanwhile, the complex tree model struggled. Any model that blows up during a crash is disqualified from trading real paper capital.

3. **The Small-Sample Trap & Minimum Volume Rule**:
   - Imagine someone flips a coin 18 times and gets 15 heads (83%). You wouldn't bet your life savings that their coin is magical — small samples produce random lucky streaks all the time!
   - In Window 1 (2021), our baseline model got 15 out of 18 trades right (83.3%), showing a huge "+41.3 point edge." But 18 trades is far too small a sample to draw conclusions from.
   - We added a strict safety rule: any window with fewer than 30 high-conviction trades is explicitly stamped as **"LOW CONFIDENCE — N too small"**. It cannot be treated as proven edge.

4. **Weighted Averages (No Equal Votes for Unequal Samples)**:
   - If an AI trades 18 times in one window with +41.3 pts edge, and 302 times in another window with +6.9 pts edge, taking a simple flat average gives equal weight to a tiny lucky sample and a massive stress test.
   - We now weight the aggregate edge strictly by the number of trades taken ($N$). The 302-trade 2022 bear market carries over 16 times more weight than the 18-trade 2021 window, bringing the baseline's honest aggregate edge to **+8.1 pts** (down from the inflated unweighted +11.1 pts).

5. **Trading Activity: Genuine Edge vs. Sitting on the Bench**:
   - A trader who never takes a trade will never lose money, but they won't build wealth either. We now track total trading opportunities across all 4.5 years:
     - **Primary Model (Gradient Boosting)**: Actively plays the game across all market seasons — fired **824 signals** (~183 trades/year across 3 stocks). All 4 windows had plenty of sample size ($N \ge 30$), delivering a modest sample-weighted edge of **+1.0 pt**.
     - **Baseline Model (Logistic Regression)**: Extremely cautious. Fired **329 total signals**, but **302 of them occurred during the 2022 bear market crash alone**! In the other 3.5 years combined, it only traded 27 times. Its capital preservation in bull markets was achieved primarily by **sitting on cash**, while in crashes it traded heavily with strong genuine edge (+6.9 pts).

6. **The "Too Good to Be True" Fire Alarm (§1.5)**:
   - Real-world stock prediction is barely better than a coin flip (~52% to 55%). If any test produces an accuracy over 62% or an AUC over 65%, the system immediately sounds a **Leakage Alarm**. We treat amazing numbers as a bug to investigate, never as a victory.

7. **The Human Hires the Trader**:
   - The software never makes the final hiring call. You review the sample-size-honest scorecard, weigh the active regular trader against the hyper-selective crash trader, and explicitly run the promotion command (`promote_model`) to name the active trader for Phase 7.

---

## Phase 7 — The Leaderboard (Ranking the Best Opportunities)

### What can the system do now that it couldn't before?

Until now, our AI could look at stocks individually and estimate their chances of rising. But knowing that Apple has a 55% chance and Microsoft has a 65% chance doesn't tell a portfolio manager what to do today. Phase 7 is our **daily leaderboard**: it gathers every validated stock in our universe, scores them using the active AI model, breaks any ties, and sorts them into a crystal-clear opportunity list.

### The Rules Built into Phase 7:

1. **The Conviction Tiers (§1.4)**:
   - **Buy Candidates ($P \ge 60\%$)**: Only stocks that cross our strict 60% conviction bar qualify for new purchases.
   - **The Dead Zone ($45\% \le P < 60\%$)**: Marked as `NEUTRAL_HOLD`. We never buy new shares here, but if we already own the stock, we don't panic-sell either.
   - **Exit Candidates ($P < 45\%$)**: The positive momentum thesis is broken. If held, the Risk Engine will evaluate it for an exit.

2. **Smart Tie-Breaking (Healthiest Wins)**:
   - If two stocks are tied with the same probability (say, both at 65%), the system doesn't flip a coin. It checks:
     1. **Relative Strength (21-day)**: Which stock has outperformed the S&P 500 (SPY) more over the past month?
     2. **Moving Average Buffer (50-day)**: Which stock has a healthier price cushion above its medium-term trend?
     3. Alphabetical ticker as a deterministic fallback.

3. **Never Pre-Cap at 3 (The Risk Engine is the Boss)**:
   - Our portfolio rule is to hold at most 3 stocks (§1.4). But **Ranking never cuts the list down to 3**.
   - If 6 stocks cross the 60% bar, Ranking reports all 6. Why? Because Phase 8's Risk Engine must make the final decision! For example, if your top 2 ranked stocks are already in your portfolio, the Risk Engine needs to see candidates #3, #4, and #5 to allocate new capital.

4. **Cash is a Valid Decision**:
   - If no stock in the market crosses the 60% bar, the buy list is completely empty ($0$ candidates). The engine explicitly recommends: **HOLD CASH**. We never force a trade when conviction is missing.

---

## Long-Term Stock Screener (A Transparent 3–5+ Year Health Checklist)

### What is this and why does it exist?
Everything else in this project is built for **short-term (5-day) momentum trading** using machine learning. 

This screener is the exact opposite: it is an independent, **long-term (3 to 5+ year) fundamental health checklist** for looking at companies the way a business owner or value investor would.

### Why didn't we use an AI model for this?
We deliberately chose **NOT** to use a machine learning model or neural network here. Here is why:
1. **No cheating or fake precision**: Free financial databases do not have 30+ years of clean, point-in-time accounting history. Trying to train an AI to forecast 5-year stock returns on today's surviving winners would create severe survivorship bias and an illusion of predictive power.
2. **Total honesty and transparency**: Instead of an opaque AI "black box" outputting a mysterious number, this screener is an open checklist of common-sense financial health rules that anyone can verify with their own eyes.

### The 5 Health Pillars (0–100 Points)
Each stock is evaluated across 5 distinct pillars, each worth up to 20 points:

1. **Valuation & Price Discipline (20 pts)**: Is the stock reasonably priced relative to earnings and book value, or is it priced for perfection?
2. **Profitability & Moat (20 pts)**: Does the company earn high returns on shareholder equity (ROE $\ge 15\%$) and keep healthy operating margins?
3. **Solvency & Debt Safety (20 pts)**: Is the balance sheet safe? Can the company comfortably cover its short-term bills, and is its debt load manageable?
4. **Cash Flow Reality (20 pts)**: Does the business generate real cash in the bank (positive Free Cash Flow), or are its "profits" just accounting paper games?
5. **Growth & Stewardship (20 pts)**: Has the company grown its sales over the past year, and does it handle dividends responsibly without paying out more than it earns?

### Special Rules for Banks and Financial Companies
Standard non-financial companies get penalized if they carry massive debt or have zero free cash flow. But banks (like JPMorgan Chase) are in the business of holding customer deposits (which count as liabilities) and lending money. 

Applying industrial debt formulas to a bank is like judging a submarine on how well it flies! 

Our screener automatically recognizes banks and evaluates them using proper financial metrics: **Price-to-Book (P/B)**, **Return on Equity (ROE)**, and **Return on Assets (ROA)**, while cleanly marking industrial debt and cash flow metrics as `N/A (Bank)` without penalizing them.

### What happens if data is missing?
Financial data providers sometimes miss numbers. To keep things fair and transparent:
- **One missing item**: That single pillar scores 0 points, gets stamped with `[MISSING_DATA]`, and the stock is flagged as `COMPLETE_WITH_GAPS`.
- **Two or more missing items**: The company is flagged as **`INSUFFICIENT DATA`**, pulled off the tier rankings, and marked as **`UNRANKED`**. We never guess or inflate a score when critical numbers are absent.

### The Impenetrable Fire-Wall
This screener is **100% separate from our daily paper-trading bot**:
- It does **NOT** place orders.
- It does **NOT** talk to the trading broker, the risk engine, or the daily scheduler.
- It fetches fresh, independent fundamental data (saved in `data/fundamentals/`) completely separate from the daily price charts (`data/raw/`).
- It is strictly an informational tool to help you study high-quality businesses for multi-year horizons.

---

## V2.1 Wave 1 — Opening the Black Box & The Drift Alarm

### What can the system do now that it couldn't before?
Machine learning models are notorious for being "black boxes": they give you a number, but you have no idea *why*, or whether the model is slowly becoming outdated as market conditions change.

V2.1 Wave 1 adds three practical tools to keep the system transparent, honest, and safe:

### 1. The "Why Did You Say That?" Inspector (SHAP Explainability)
Whenever the active model scores a stock (e.g. giving Apple a 42.3% chance or Visa a 63% chance), you can now open the hood and see exactly which indicators pushed the score up or dragged it down:
- **Green Factors (Pushed Up)**: Maybe the 20-day volatility was calm (+0.09) or the stock was outperforming the market (+0.06).
- **Red Factors (Dragged Down)**: Maybe MACD was rolling over (-0.10) or short-term momentum was fading (-0.08).
- **Exact Math**: The contributions add up precisely to the final score ($\text{Base Prior} + \sum \text{SHAP} = \text{Log-Odds}$). There are no guessing games.

### 2. The Drift Smoke Alarm (PSI & Distribution Monitoring)
Over time, markets evolve. A model trained on past years might see its predictions drift away from its original validation baseline.
- **Population Stability Index (PSI)**: Measures whether live probabilities still follow the healthy curve established during multi-year testing.
  - **Green (PSI < 0.10)**: Stable. The model is behaving normally.
  - **Yellow (0.10 to 0.25)**: Monitor. Market dynamics are slightly unusual.
  - **Red (PSI $\ge$ 0.25)**: Drift Alert! The predictions have meaningfully diverged.
- **Strict Human-in-the-Loop Rule**: The alarm **never** automatically retrains or replaces the model. It simply flags a warning for human review, keeping the human trader firmly in charge of model promotion.

### 3. Calibrated Confidence & The 30-Trade Honesty Rule
A raw number like "0.58" sounds like a solid prediction, but what does it actually mean in the real world?
- **Only 0.60 (Buy) and 0.45 (Exit) are real decisions**:
  - $P \ge 60\%$: Actionable buy bar (historical 55.8% win rate).
  - $45\% \le P < 60\%$: Dead-zone / neutral. **Zero buy orders permitted.** Even if a stock reaches 0.58 ("Leaning Buy"), the system explicitly labels it as **NOT A BUY SIGNAL**.
  - $P < 45\%$: Actionable exit bar (historical 41.5% win rate).
- **The Sample Size Honesty Stamp ($N \ge 30$)**:
  - Whenever empirical win rates are displayed, any category with fewer than 30 real test samples is stamped with **`LOW CONFIDENCE (N < 30)`**. We never present small-sample stats as proven edge.

---

## V2.2 Wave 1 — Portfolio Intelligence: Checking Your Eggs and Baskets

### What can the system do now that it couldn't before?
Before this update, the dashboard showed what stocks you owned and how much cash you had, but it couldn't tell you how well-balanced you were. If you owned three tech companies (like Apple, Microsoft, and Nvidia), you might feel diversified because you own three different logos — but if the technology sector takes a hit, all three fall together!

V2.2 Wave 1 adds **Portfolio Intelligence**: three transparent tools to help you understand your portfolio's balance before and after making moves:

### 1. The Sector Map ("Which baskets hold your eggs?")
The system now maps every single stock across 8 standard sectors of the economy:
- Technology (Apple, Microsoft, Nvidia)
- Communications (Google, Meta)
- Consumer Cyclical (Amazon, Tesla)
- Financials (JPMorgan, Visa)
- Healthcare (UnitedHealth)
- Industrials, Consumer Staples, and Energy

You can see at a glance what percentage of your portfolio sits in each sector, helping you spot when you are accidentally loading up too heavily in one corner of the market.

### 2. The 100-Point Diversification Scorecard
Rather than giving you a mystery number, the system grades your portfolio's diversification on an open 100-point rubric:
- **Sector Breadth (40 pts)**: Are you spread across different parts of the economy?
- **Sector Balance (40 pts)**: Are your sectors evenly weighted, or is 90% in one sector? (Uses the standard economic Herfindahl-Hirschman concentration formula).
- **Position Balance (20 pts)**: Are your individual holdings roughly similar in size?

*What about an all-cash portfolio?* When you are 100% in cash (like when starting out or waiting out a market downturn), your score is a clean **100/100 (Full Cash Preservation)** — because holding cash carries zero stock concentration risk.

### 3. The "What-If" Simulator ("Try before you buy")
Ever wonder: *"If I bought $2,833 of Visa right now, what would happen to my portfolio's balance?"*

The What-If Simulator lets you preview what happens to your weights, cash cushion, and diversification score **before** placing a trade:
- **Exact real sizing**: When adding a stock, it uses the exact real-world formula from our trading rules: $(\text{Total Equity} \times 85\%) / 3$ ($2,833.33 on our $10,000 account).
- **Strictly structural — no crystal balls**: The simulator **only** calculates portfolio math (weights, cash percentages, diversification). It strictly **never** pretends to guess future profits, losses, or prices.
- **Total isolation**: Running a simulation happens 100% in memory. It never touches the database, never alters your real cash, and never triggers an accidental trade.

---

## Streamlit Dashboard: Your Single Control Room

### Why keep things simple?
In earlier experiments, we briefly tested running a separate web API server (FastAPI) and a React website. But having two separate systems just added extra machinery and possible points of failure without adding any trading value.

We cleanly removed that extra layer and kept **Streamlit** as our single, direct control room. It talks directly to our filing cabinet (`repository.py`), loads instantly on your laptop, and gives you complete real-time visibility into your portfolio, predictions, backtests, and safety checks without any extra web-server baggage.

---

## The 15 Major Upgrades: Sections 1–4 Explained in Plain English

As the system matured, we rolled out 15 major improvements organized into 4 logical sections. Here is what each upgrade does, why it matters, and how it protects your trading capital:

---

### Section 1: Data & Universe Expansion (Items 1–4)

#### 1. Expanding the Stock Universe from 5 to 25 Stocks Across All 8 Sectors
- **Before**: The system only watched 3 to 5 big tech stocks (like Apple, Microsoft, and Nvidia). If tech had a slow month, the system either sat idle or was forced to pick between tech names.
- **Now**: The watchlist has expanded to 25 high-quality companies representing all 8 key sectors of the US economy (Technology, Communications, Consumer Cyclical, Financials, Healthcare, Industrials, Consumer Staples, and Energy).
- **The Safety Rule**: Even though the system now watches 25 stocks, the Risk Engine strictly enforces our rule of **holding at most 3 positions at any time**. Watching more stocks gives us better options; it never increases our risk.

#### 2. Smart, Dynamic Bad-Data Detection (No More Rigid 50% Rule)
- **Before**: The inspector used a blunt rule: if any stock jumped or dropped more than 50% overnight, reject it immediately. But during calm markets, a 20% jump is suspicious, while for a volatile growth stock in earnings season, a large swing might be genuine.
- **Now**: The system calculates a stock's normal personality using its rolling 30-day volatility and Average True Range (ATR). An overnight move is flagged as suspicious only if it exceeds 3 standard deviations from its normal range (with a minimum 15% floor). Calm stocks are held to tighter standards; volatile stocks aren't unfairly thrown out.

#### 3. New York Timezone-Aware Market Clock (Daylight Saving Safe)
- **Before**: The daily scheduler used a fixed UTC-5 offset. But New York switches between Eastern Standard Time (EST, UTC-5) in the winter and Eastern Daylight Time (EDT, UTC-4) in the summer. For half the year, the clock was an hour off!
- **Now**: The market clock uses official `zoneinfo` with `"America/New_York"`. It automatically adjusts for Daylight Saving Time switches, ensuring the pre-market routine always runs at exactly 9:00 AM Eastern (30 minutes before the opening bell).

#### 4. Fallback Data Provider Chain
- **Before**: If our primary free data connection (Yahoo Finance) had an outage or timed out, the daily pipeline failed and no trading decisions could be made.
- **Now**: The data fetcher has a "Plan B". If the primary provider fails, it automatically switches to a backup secondary provider, cleans the columns to match our standard format, and logs the failover event in the audit trail.

---

### Section 2: Risk & Safety Rules (Items 5–9)

#### 5. Weather-Adaptive Conviction Thresholds
- **Before**: The system required an exact 60% probability ($P \ge 0.60$) to buy, regardless of whether the market was calm or in the middle of a hurricane.
- **Now**: The bar adjusts with market volatility:
  - **Calm Seas (VIX < 15)**: The buy bar gently eases to 58% ($P \ge 0.58$), giving the model room to capture steady trends.
  - **Normal Markets (15 $\le$ VIX $\le$ 25)**: The standard 60% bar ($P \ge 0.60$) applies.
  - **Rough Seas (VIX > 25)**: The bar rises to a strict 65% ($P \ge 0.65$), demanding exceptional conviction before risking cash in a storm.

#### 6. Anti-Whipsaw 5-Day Minimum Hold
- **Before**: If the model scored a stock at 61% on Monday (buy) and it dipped to 44% on Tuesday, the system would sell immediately, incurring trading fees on both legs due to day-to-day noise.
- **Now**: Once a stock is purchased, "signal exit" (selling solely because the score dropped below 45%) is muted for the first 5 trading days. This lets the trade develop without getting whipsawed by short-term noise.
- **The Golden Rule**: Our safety stop-loss (-8%) and profit target (+15%) **always** remain active 100% of the time, even on Day 1.

#### 7. Tiered Defense for Held Stocks with Bad Data
- **Before**: If a stock we owned had missing or corrupt data on one day, the system issued an alert, but had no formal procedure for multi-day outages.
- **Now**: A three-day escalation protocol takes over:
  - **Day 1**: Hold position untouched and await clean data.
  - **Day 2**: Inspect the last known price. If within 2% of the -8% stop-loss, trigger an **URGENT_RISK_ALERT**.
  - **Day 3**: If data is still missing after 3 days, lock the position to prevent automated mistakes and flag for immediate human manual review.

#### 8. Macro Crash Circuit Breakers
- **Before**: The system checked individual stocks and the 200-day trend of the S&P 500, but a sudden flash crash (like COVID in March 2020) could unfold before a 200-day average caught up.
- **Now**: Emergency tripwires monitor rapid market drops:
  - If the S&P 500 (SPY) falls **more than 5% in 5 days**, or **more than 10% in 20 days**, a macro circuit breaker trips immediately.
  - All new buying is halted, preserving 100% of our dry powder until the panic settles.

#### 9. Model Drift Protection (PSI Sizing & Pause Protocol)
- **Before**: When the drift smoke alarm sounded (PSI rising), it warned us, but the trading engine kept placing standard-sized orders.
- **Now**: The risk engine acts automatically on drift levels:
  - **Moderate Drift ($0.10 \le \text{PSI} < 0.25$)**: Cuts new position sizes in half (allocating $1,416 instead of $2,833) to limit exposure.
  - **Severe Drift ($\text{PSI} \ge 0.25$)**: Pauses all new purchases completely until the model is audited.

---

### Section 3: Machine Learning & Training Improvements (Items 10–13)

#### 10. Survivorship-Bias Historical Data Ingestion
- **Before**: If you only train an AI on companies that exist today, the AI gets the false impression that stocks almost never fail! It never gets to learn from companies that went bankrupt or were forcibly acquired.
- **Now**: We ingested genuine, verified historical data for delisted and failed companies—including First Republic Bank (`FRCB`), Twitter (`TWTR`), and Celgene (`CELG`)—complete with permanent SHA-256 checksums. The model now learns what failure and distress look like.

#### 11. Extending Training Back to 2008 & Stress-Test Windows
- **Before**: Training history only went back to 2018, meaning the model had never experienced a true systemic banking collapse or credit freeze.
- **Now**: The historical data window stretches all the way back to **January 2008**. The evaluation report explicitly stress-tests models across both the **2008–2009 Global Financial Crisis** and the **2011 Eurozone Debt Crisis**, ensuring models are resilient across severe economic winter.

#### 12. Rolling Walk-Forward Cross-Validation
- **Before**: Testing was done with a single train/test cutoff date.
- **Now**: True **walk-forward validation**: the model trains on a 2-year rolling window, steps forward 6 months to test on unseen future data, and repeats this process across history. A mandatory 5-day embargo gap between training and testing ensures zero information leakage.

#### 13. XGBoost & Soft-Voting Ensemble Models
- **Before**: We had two candidate models: Logistic Regression (simple & linear) and Random Forest (tree-based).
- **Now**: We added a high-performance gradient booster (**XGBoost**) and a **Soft-Voting Ensemble** that combines the probabilities of all candidate models. The ensemble smooths out individual model quirks and makes more balanced, consensus-driven predictions.

---

### Section 4: Operational & Infrastructure (Items 14–15)

#### 14. Server-Grade Database Upgrade (PostgreSQL + SQLite Dual-Engine)
- **Before**: The filing cabinet was strictly tied to a local SQLite file (`trader.db`).
- **Now**: The database layer is upgraded to **SQLAlchemy ORM** with native support for **PostgreSQL**:
  - **Local Development**: Still defaults to zero-setup SQLite out of the box.
  - **Production Ready**: Simply set the `DATABASE_URL` environment variable to point to any PostgreSQL database (with automatic connection pooling and pre-ping health checks).
  - **Database Migrator (`db_migrate.py`)**: A handy utility that exports your entire trading history, snapshots, and audit logs to JSON, imports them into PostgreSQL, or transfers data live between databases in one command.

#### 15. Real-World Slippage Simulation Model
- **Before**: Simulated fills assumed you could buy or sell any number of shares at the exact market price with zero market impact.
- **Now**: The trade simulator incorporates **market impact slippage** based on how large your trade is compared to the stock's Average Daily Volume (ADV):
  - **Small Trade ($\le 1\%$ ADV)**: 0.05% slippage (5 basis points).
  - **Medium Trade ($1\%-5\%$ ADV)**: 0.15% slippage (15 basis points).
  - **Large Trade ($> 5\%$ ADV)**: 0.30% slippage (30 basis points).
- **Full Accountability**: Slippage pushes your buy price slightly higher and your sell price slightly lower. Every dollar lost to slippage is tracked cumulatively, recorded on the **Drawer 4 Daily Balance Sheet**, and displayed in the Phase 6 evaluation reports so you always know the true cost of trading.

---

### Section 5: Fully Automated Hands-Off Daily Operations

You no longer need to remember to open a terminal or type commands every evening. The system is equipped with an autopilot operational harness managed by **Windows Task Scheduler**.

#### 1. The 5 Automated Tasks
All automated routines are neatly grouped under the `AI-Stock-Trader` folder in Windows Task Scheduler:

1. **`AI-Stock-Trader-Daily` (Mon–Fri at 5:00 PM)**
   - **What it does**: Automatically wakes up 30 minutes after US markets close (4:30 PM ET), activates the Python virtual environment, executes the entire daily trading pipeline (`python -m src.pipeline.scheduler --force`), and streams all output into a dated log file (`logs/daily_run_YYYY-MM-DD.log`).
   - **Crash Guard**: If an unhandled crash or non-zero exit code occurs, it instantly tags the log with `PIPELINE FAILED` and captures the exact error trace so nothing fails silently.
   - **Housekeeping**: Automatically cleans out old log files older than 30 days so your hard drive never clutters.

2. **`AI-Stock-Trader-Morning-HealthCheck` (Mon–Fri at 9:00 AM)**
   - **What it does**: Runs `health_check.py` 30 minutes before the market opens. It verifies that last night's pipeline succeeded, checks database integrity, and prints an early morning green-light or red-alert status.

3. **`AI-Stock-Trader-Startup` (On Boot / Login)**
   - **What it does**: Runs `health_check.py` immediately whenever Windows boots up or you log in. If your computer was powered off yesterday, it immediately notifies you that a scheduled run was missed.

4. **`AI-Stock-Trader-Dashboard` (On Login, 60s delay)**
   - **What it does**: Launches the Streamlit dashboard (`start_dashboard.bat`) in the background and opens your browser directly to `http://localhost:8501`. If the dashboard is already running, it simply brings up the browser without launching duplicate processes.

5. **`AI-Stock-Trader-DailyStatus` (Mon–Fri at 6:00 PM)**
   - **What it does**: Runs `daily_status.py` one hour after the trading pipeline finishes. It reads the database and compiles a human-readable digest:
     - Portfolio value change today vs yesterday ($ and %)
     - Open positions and cash cushion
     - Pipeline completion status
     - Active safety alarms (Macro SPY Circuit Breaker, PSI Model Drift)
   - Outputs to both the console and a dated record (`logs/status_YYYY-MM-DD.log`).

#### 2. Crash Recovery & Missed-Run Resilience
- **Missed-Run Detection**: If your computer was powered off during the scheduled 5:00 PM run, the next run checks for yesterday's log file. If missing, it writes a clear warning `MISSED RUN DETECTED: YYYY-MM-DD` and proceeds safely. It never makes blind assumptions or executes dangerous retroactive backfills.
- **Log Sanitation**: Every daily run concludes with an automated purge of log files older than 30 days.

#### 3. One-Command Setup & Cleanup
- **Quick Setup**: Run `powershell -ExecutionPolicy Bypass -File .\setup_task_scheduler.ps1` from the project root. To enable Administrator highest privileges and boot triggers, add `-Elevate`.
- **Clean Uninstallation**: Run `powershell -ExecutionPolicy Bypass -File .\remove_task_scheduler.ps1` to cleanly unregister all tasks from Windows Task Scheduler and remove startup shortcuts.



---

## Section 5 — Making the System Smarter (Intelligence Upgrades)

Until now, the system made buy and sell decisions purely from price history and a trained machine learning model. Section 5 adds four new "brains" that the system consults before any trade decision is finalised.

---

### Intelligence Upgrade 1 — News Sentiment Analysis

**The idea in plain English**: Before buying a stock, the system reads the last 5 news headlines about that company and checks whether the news mood is positive, negative, or neutral.

**How it works**:
- Every day, before a buy decision is finalised, the system fetches the 5 most recent news headlines for each stock it is considering (from Yahoo Finance).
- It uses **VADER** (a well-tested academic sentiment scoring tool) to score each headline from -1.0 (terrible) to +1.0 (great) and averages them into one news mood score.
- Based on that mood score, the stock's ranking probability is adjusted:
  - Mood above +0.3 -> Good news -> probability score boosted by **+3%**
  - Mood between -0.3 and +0.3 -> Neutral news -> **no change**
  - Mood below -0.3 -> Bad news -> probability score penalised by **-5%**
  - Mood below -0.6 -> Severely bad news -> **BUY IS VETOED** and the system logs SENTIMENT_VETO
- If the news API fails or returns nothing, the system logs SENTIMENT_UNAVAILABLE and continues without blocking the trade.
- All sentiment scores are saved in the `sentiment_scores` database table.

**Practical example**: Apple's ML score is 0.71 but today's news scores -0.70 (major fraud scandal). The system vetoes the buy entirely.

---

### Intelligence Upgrade 2 — Earnings Calendar Awareness

**The idea in plain English**: Stocks often make wild moves on their earnings day. The system learns when every watchlist stock reports earnings and adjusts behaviour in the days leading up to it.

**How it works**:
- Based on how many days away the next earnings announcement is, a different rule applies:
  - Earnings is TODAY: Hold existing position. No new buys on announcement day.
  - 1-2 days away: EARNINGS BLACKOUT WINDOW, buying is blocked. Logs EARNINGS_BLACKOUT.
  - 3-5 days away: EARNINGS CAUTION WINDOW, may still buy but only **half the normal position size**. Logs EARNINGS_CAUTION.
  - More than 5 days away: No adjustment. Business as usual.
- The dashboard shows emoji badges next to each holding.
- All earnings dates are saved in the `earnings_calendar` database table.

**Practical example**: Microsoft reports earnings tomorrow. The system says Earnings blackout, no new buy. If earnings are 4 days away, it buys but only half the normal amount.

---

### Intelligence Upgrade 3 — Sector Rotation Intelligence

**The idea in plain English**: Some industries have the wind at their backs right now. The system tracks which of 8 broad sectors is performing best over the last 20 trading days and adjusts stock scores accordingly.

**How it works**:
- The system computes the 20-day return for every watchlist stock, groups them by sector (Technology, Communications, Consumer Cyclical, Financials, Healthcare, Industrials, Consumer Staples, Energy), and ranks all 8 sectors from 1 (best) to 8 (worst).
- Based on sector rank, each stock's probability score is adjusted:
  - Rank 1-2: **+4% boost**
  - Rank 3-5: **No adjustment**
  - Rank 6-8: **-4% penalty**
- The dashboard shows a Sector Rotation Strength panel with colour-coded signals.
- All rankings are saved in the `sector_rankings` table.

**Practical example**: Energy has been the worst sector for 20 days (rank 8). ExxonMobil's ML score is 0.62, drops to 0.58 after the -4% penalty, below the 0.60 buy threshold. The system holds off.

---

### Intelligence Upgrade 4 — Macro Economic Indicators

**The idea in plain English**: When interest rates are very high and inflation is raging, the system automatically becomes more cautious.

**How it works**:
- The system tracks four Federal Reserve (FRED) indicators: Fed Funds Rate, CPI YoY, Unemployment Rate, 10-Year Treasury Yield.
- Based on these, the economic environment is classified into one of three regimes:
  - FAVORABLE (low rates < 3% OR falling, AND inflation < 3%): Full position sizes. No buy-bar change.
  - NEUTRAL (moderate rates/inflation): All position sizes **reduced by 20%**.
  - RESTRICTIVE (rates > 5% AND inflation > 5%): Position sizes **reduced by 40%**. Minimum confidence to buy raised from 60% to **63%**.
- FRED data is only fetched on Mondays; Tue-Fri reuse the same week's reading. Falls back to DB cache or baseline defaults gracefully.
- The dashboard Tab 3 shows a Macro Economic Environment card colour-coded by regime.
- All indicators are saved in the `macro_indicators` table.
- To enable live data: Add FRED_API_KEY=your_key_here to .env. Free registration at https://fred.stlouisfed.org

---

### How All Four Intelligence Layers Work Together

After the ML model produces its base probability score, all four intelligence checks run in sequence. Every candidate's full breakdown is logged for full auditability:

    AAPL: Base=0.71 -> Sentiment+0.03 -> Sector+0.04 -> Macro-0.00 -> Earnings OK -> Final=0.78
    TSLA: Base=0.68 -> Sentiment-0.05 -> Sector-0.04 -> Macro+0.00 -> Earnings BLACKOUT -> VETO

**Graceful degradation**: If any external data source fails (news API down, FRED unreachable, yfinance error), the system logs a warning, skips that layer's adjustment, and continues the rest of the pipeline normally. No single API failure can crash or block the daily run.
