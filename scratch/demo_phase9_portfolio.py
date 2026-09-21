"""
scratch/demo_phase9_portfolio.py — Live Phase 9 Demonstration at $50 Scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, ".")

from config.settings import settings
from src.ranking.ranking import RankedOpportunity, RankingResult, TIER_BUY, TIER_NEUTRAL
from src.risk.risk_engine import evaluate_portfolio_risk
from src.portfolio.portfolio import allocate_portfolio

print("=" * 70)
print(f"PHASE 9 LIVE PORTFOLIO ALLOCATION DEMONSTRATION AT REAL ${settings.initial_capital if False else 50.00:.2f} SCALE")
print("=" * 70)

date = "2024-08-30"

# Current portfolio state at start of day:
# Starting capital scale: $50.00 total equity
# Sizing formula: (equity * 0.85) / 3 = ($49.99 * 0.85) / 3 = $14.16 per slot
# Cash reserve statutory floor: 15% ($7.50)
#
# Holdings:
# 1. AAPL: 0.0675 shares @ $210 cost basis ($14.17 cost). Price today: $229.00 -> +9.05% profit (HOLD)
# 2. JPM: 0.0590 shares @ $240 cost basis ($14.16 cost). Price today: $218.00 -> -9.17% loss (STOP-LOSS SELL)
# Cash: $21.67
# Total Equity: 21.67 + (0.0675 * 229.0) + (0.0590 * 218.0) = $49.99

positions = {
    "AAPL": {"quantity": 0.0675, "avg_cost": 210.0},
    "JPM": {"quantity": 0.0590, "avg_cost": 240.0},
}
prices = {
    "AAPL": 229.0,
    "JPM": 218.0,
    "MSFT": 417.23,
    "NVDA": 119.37,
}

# Opportunities from Ranking:
# Candidate 1: MSFT (P=0.68 >= 0.60) -> Qualified Buy Candidate
# Candidate 2: NVDA (P=0.54 < 0.60) -> Neutral / Hold Cash
buy_msft = RankedOpportunity("MSFT", date, 0.68, 1, TIER_BUY, 0.05, 1.02, is_buy_eligible=True)
neutral_nvda = RankedOpportunity("NVDA", date, 0.54, 2, TIER_NEUTRAL, -0.02, 0.98, is_buy_eligible=False)

ranking = RankingResult(
    date=date,
    regime_risk_on=True,
    total_evaluated=2,
    ranked_opportunities=[buy_msft, neutral_nvda],
    top_buy_candidates=[buy_msft],
    neutral_candidates=[neutral_nvda],
    exit_candidates=[],
)

# Step 1: Run Phase 8 Risk Engine
risk_res = evaluate_portfolio_risk(
    run_date=date,
    current_cash=21.67,
    current_positions=positions,
    current_prices=prices,
    ranking_result=ranking,
)

print("\n--- PHASE 8 RISK ENGINE OUTPUT ---")
print(risk_res.to_markdown_summary())

# Step 2: Run Phase 9 Portfolio Allocation
alloc_res = allocate_portfolio(
    risk_assessment=risk_res,
    current_cash=21.67,
    current_positions=positions,
    current_prices=prices,
)

print("\n--- PHASE 9 PORTFOLIO ALLOCATION OUTPUT ---")
print(alloc_res.to_markdown_summary())
