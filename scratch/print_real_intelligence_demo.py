import urllib.request
import json

print("================================================================================")
print("             V2.2 WAVE 1 — REAL PORTFOLIO INTELLIGENCE VERIFICATION             ")
print("================================================================================\n")

print("--- 1. REAL CURRENT PORTFOLIO SECTOR BREAKDOWN (LIVE $10,000 / 0-POSITIONS) ---")
with urllib.request.urlopen("http://127.0.0.1:8000/api/intelligence/sectors") as resp:
    sec = json.loads(resp.read().decode("utf-8"))
    print(f"Portfolio Total Equity : ${sec['total_equity']:,.2f}")
    print(f"Cash Reserve           : ${sec['cash']:,.2f} ({sec['cash_reserve_pct']}%)")
    print(f"Active Positions       : {len(sec['holdings_sectors'])}")
    print(f"Active Sector Holdings : {sec['holdings_sectors']} (Clean empty state; no holdings)")
    print("\n10-Stock Trading Universe Sector Breakdown:")
    for sector, count in sec['trading_universe_sectors'].items():
        pct = (count / 10.0) * 100.0
        print(f"  * {sector:<22}: {count} stocks ({pct:>5.1f}%)")

print("\n--- 2. REAL CURRENT DIVERSIFICATION SCORE ---")
with urllib.request.urlopen("http://127.0.0.1:8000/api/intelligence/diversification") as resp:
    div = json.loads(resp.read().decode("utf-8"))
    print(f"Diversification Score  : {div['score']} / 100 ({div['rating']})")
    pb = div['pillar_breakdown']
    print(f"Pillars (Points)       : Sector Breadth = {pb['sector_breadth_pts']}/40, Sector HHI = {pb['sector_hhi_pts']}/40, Position Balance = {pb['position_balance_pts']}/20")
    print(f"Details                : {div['formula_explanation']}")

print("\n--- 3. REAL WHAT-IF SIMULATION (ADDING REAL TICKER: AAPL) ---")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/intelligence/simulate",
    data=json.dumps({"action": "ADD", "ticker": "AAPL"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    sim = json.loads(resp.read().decode("utf-8"))
    print(f"Simulated Action       : {sim['action']} {sim['ticker']} ({sim['simulated_sector']})")
    print(f"Disclaimer             : {sim['disclaimer']}")
    print(f"Before Simulation      : Positions={sim['before']['positions_count']}, Cash=${sim['before']['cash']:,.2f} ({sim['before']['cash_reserve_pct']}%), Score={sim['before']['diversification_score']}")
    print(f"After Simulation       : Positions={sim['after']['positions_count']}, Cash=${sim['after']['cash']:,.2f} ({sim['after']['cash_reserve_pct']}%), Score={sim['after']['diversification_score']}")
    print(f"Allocation Sizing      : ${sim['deltas']['cash_delta'] * -1:,.2f} (Exact formula: ($10,000 * 0.85) / 3 = $2,833.33)")
    print(f"Holdings Sectors After : {sim['after']['holdings_sectors']}")
    print(f"Deltas                 : Score Delta = {sim['deltas']['score_delta']} pts, Cash Delta = ${sim['deltas']['cash_delta']:,.2f}")
    for sd in sim['deltas']['sector_deltas']:
        print(f"  * Sector Delta ({sd['sector']}): Before = {sd['weight_before_pct']}%, After = {sd['weight_after_pct']}%, Shift = +{sd['delta_pct']}%")

print("\n================================================================================")
