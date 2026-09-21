"""
src/screening/scorer.py — 5-Pillar Rule-Based Fundamental Scoring Engine.

Translates company financial data into an explainable 0–100 point scorecard
across 5 fundamental pillars (20 points each).

CRITICAL POLICY: MISSING DATA HANDLING:
1. Structural Exemptions:
   Financial institutions (Banks) do not report standard Debt/Equity, Current Ratio,
   or Free Cash Flow. They are evaluated on banking-specific metrics (P/B, ROE, ROA)
   without penalty.
2. Missing Data Rules:
   - If >= 2 pillars lack required metrics -> Status = INSUFFICIENT_DATA.
     The stock is excluded from Tier 1/2 rankings and marked UNRANKED.
   - If exactly 1 pillar lacks required metrics -> That pillar scores 0 points with
     explicit [MISSING_DATA] tag. The denominator remains 100 (no artificial inflation).
     Status = COMPLETE_WITH_GAPS.
   - If all required metrics present -> Status = COMPLETE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.screening.collector import FundamentalData
from src.screening.universe import ScreenerCompany, get_company_meta


@dataclass
class PillarScore:
    name: str
    score: int          # 0, 10, or 20
    max_score: int = 20
    passed: bool = False
    reason: str = ""
    is_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FundamentalScorecard:
    ticker: str
    company_name: str
    sector: str
    composite_score: int                # 0 to 100
    tier: str                           # Tier 1, Tier 2, Tier 3, or UNRANKED
    status: str                         # COMPLETE, COMPLETE_WITH_GAPS, INSUFFICIENT_DATA
    pillar_scores: Dict[str, PillarScore]
    missing_fields: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pillar_scores"] = {k: v.to_dict() for k, v in self.pillar_scores.items()}
        return d


def _score_valuation(data: FundamentalData, is_bank: bool) -> Tuple[PillarScore, List[str]]:
    missing: List[str] = []
    pe = data.trailing_pe
    fwd_pe = data.forward_pe
    peg = data.peg_ratio
    pb = data.price_to_book

    if pe is None and fwd_pe is None:
        missing.append("trailing_pe")
        return PillarScore(
            name="Valuation",
            score=0,
            passed=False,
            reason="Missing P/E multiple data.",
            is_missing=True,
        ), missing

    effective_pe = pe if pe is not None else fwd_pe

    if is_bank:
        if pb is None:
            missing.append("price_to_book")
        pb_val = pb if pb is not None else 999.0
        if effective_pe < 16.0 and pb_val < 2.0:
            return PillarScore("Valuation", 20, passed=True, reason=f"Attractive bank valuation: P/E {effective_pe:.1f}x < 16x and P/B {pb_val:.2f}x < 2.0x."), missing
        elif effective_pe < 20.0 or pb_val < 2.5:
            return PillarScore("Valuation", 10, passed=False, reason=f"Moderate bank valuation: P/E {effective_pe:.1f}x or P/B {pb_val:.2f}x."), missing
        else:
            return PillarScore("Valuation", 0, passed=False, reason=f"Elevated bank valuation: P/E {effective_pe:.1f}x, P/B {pb_val:.2f}x."), missing

    # Non-financials
    if pe is not None and pe <= 0:
        return PillarScore("Valuation", 0, passed=False, reason="Unprofitable (negative P/E multiple)."), missing

    fwd_discount = (fwd_pe is not None and pe is not None and fwd_pe < pe)
    peg_attractive = (peg is not None and peg > 0 and peg < 2.0)

    if effective_pe < 25.0 and (peg_attractive or fwd_discount):
        reason = f"Sensible valuation: P/E {effective_pe:.1f}x < 25x with supportive PEG ({peg if peg else 'N/A'}) or forward discount."
        return PillarScore("Valuation", 20, passed=True, reason=reason), missing
    elif effective_pe < 35.0 or (peg is not None and peg > 0 and peg < 2.5):
        reason = f"Moderate valuation: P/E {effective_pe:.1f}x, PEG {peg if peg else 'N/A'}."
        return PillarScore("Valuation", 10, passed=False, reason=reason), missing
    else:
        reason = f"Premium/expensive valuation: P/E {effective_pe:.1f}x > 35x."
        return PillarScore("Valuation", 0, passed=False, reason=reason), missing


def _score_profitability(data: FundamentalData, is_bank: bool) -> Tuple[PillarScore, List[str]]:
    missing: List[str] = []
    roe = data.return_on_equity
    roa = data.return_on_assets
    op_margin = data.operating_margins

    if roe is None:
        missing.append("return_on_equity")
        return PillarScore(
            name="Profitability",
            score=0,
            passed=False,
            reason="Missing Return on Equity (ROE).",
            is_missing=True,
        ), missing

    if is_bank:
        roa_val = roa if roa is not None else 0.0
        if roe >= 0.12 and roa_val >= 0.010:
            return PillarScore("Profitability", 20, passed=True, reason=f"Strong bank returns: ROE {roe*100:.1f}% >= 12% and ROA {roa_val*100:.2f}% >= 1.0%."), missing
        elif roe >= 0.09 or roa_val >= 0.007:
            return PillarScore("Profitability", 10, passed=False, reason=f"Adequate bank returns: ROE {roe*100:.1f}%, ROA {roa_val*100:.2f}%."), missing
        else:
            return PillarScore("Profitability", 0, passed=False, reason=f"Subpar bank returns: ROE {roe*100:.1f}% < 9%."), missing

    # Non-financials
    op_m = op_margin if op_margin is not None else 0.0
    if roe >= 0.15 and op_m >= 0.15:
        return PillarScore("Profitability", 20, passed=True, reason=f"High-moat profitability: ROE {roe*100:.1f}% >= 15% and Operating Margin {op_m*100:.1f}% >= 15%."), missing
    elif roe >= 0.10 or op_m >= 0.10:
        return PillarScore("Profitability", 10, passed=False, reason=f"Moderate profitability: ROE {roe*100:.1f}%, Operating Margin {op_m*100:.1f}%."), missing
    else:
        return PillarScore("Profitability", 0, passed=False, reason=f"Low profitability: ROE {roe*100:.1f}% < 10% and Operating Margin {op_m*100:.1f}% < 10%."), missing


def _score_solvency(data: FundamentalData, is_bank: bool) -> Tuple[PillarScore, List[str]]:
    missing: List[str] = []

    if is_bank:
        # For commercial banks, capital safety is tracked via ROA and asset quality
        roa = data.return_on_assets
        if roa is None:
            missing.append("return_on_assets")
            return PillarScore("Solvency", 10, passed=False, reason="Bank ROA metric unavailable.", is_missing=True), missing
        if roa >= 0.011:
            return PillarScore("Solvency", 20, passed=True, reason=f"Sound bank balance sheet: ROA {roa*100:.2f}% >= 1.1% indicates disciplined asset leverage."), missing
        elif roa >= 0.008:
            return PillarScore("Solvency", 10, passed=False, reason=f"Acceptable bank asset quality: ROA {roa*100:.2f}%."), missing
        else:
            return PillarScore("Solvency", 0, passed=False, reason=f"Thin bank asset buffer: ROA {roa*100:.2f}% < 0.8%."), missing

    de = data.debt_to_equity
    cr = data.current_ratio

    if de is None:
        missing.append("debt_to_equity")
    if cr is None:
        missing.append("current_ratio")

    if de is None and cr is None:
        return PillarScore("Solvency", 0, passed=False, reason="Missing debt and liquidity data.", is_missing=True), missing

    de_val = de if de is not None else 999.0
    cr_val = cr if cr is not None else 0.5

    # Note: yfinance debtToEquity is expressed as percentage (e.g. 78.4 = 78.4%, 150 = 1.5x D/E)
    if de_val <= 100.0 and cr_val >= 1.2:
        return PillarScore("Solvency", 20, passed=True, reason=f"Conservative balance sheet: D/E {de_val:.1f}% <= 100% and Current Ratio {cr_val:.2f} >= 1.2x."), missing
    elif de_val <= 175.0 and cr_val >= 0.9:
        return PillarScore("Solvency", 10, passed=False, reason=f"Manageable leverage: D/E {de_val:.1f}%, Current Ratio {cr_val:.2f}x."), missing
    else:
        return PillarScore("Solvency", 0, passed=False, reason=f"Elevated leverage / tight liquidity: D/E {de_val:.1f}%, Current Ratio {cr_val:.2f}x."), missing


def _score_cash_flow(data: FundamentalData, is_bank: bool) -> Tuple[PillarScore, List[str]]:
    missing: List[str] = []

    if is_bank:
        # Bank cash flows are evaluated via operating profitability and dividend continuity
        roe = data.return_on_equity
        div = data.dividend_yield
        if roe is not None and roe > 0.08:
            return PillarScore("Cash Flow", 20, passed=True, reason=f"Positive bank capital accumulation (ROE {roe*100:.1f}%)."), missing
        return PillarScore("Cash Flow", 10, passed=False, reason="Constrained bank earnings generation."), missing

    fcf = data.free_cash_flow
    ocf = data.operating_cash_flow

    if fcf is None and ocf is None:
        missing.append("free_cash_flow")
        return PillarScore("Cash Flow", 0, passed=False, reason="Missing cash flow metrics.", is_missing=True), missing

    if fcf is not None and fcf > 0:
        fcf_b = fcf / 1e9
        return PillarScore("Cash Flow", 20, passed=True, reason=f"Positive Free Cash Flow: ${fcf_b:.2f}B generating real cash surpluses."), missing
    elif ocf is not None and ocf > 0:
        return PillarScore("Cash Flow", 10, passed=False, reason="Operating cash flow positive, but FCF constrained by capital reinvestment."), missing
    else:
        return PillarScore("Cash Flow", 0, passed=False, reason="Negative cash generation (cash-burning business)."), missing


def _score_growth_stewardship(data: FundamentalData, is_bank: bool) -> Tuple[PillarScore, List[str]]:
    missing: List[str] = []
    rev_g = data.revenue_growth
    payout = data.payout_ratio

    if rev_g is None:
        missing.append("revenue_growth")
        return PillarScore("Growth", 10, passed=False, reason="Revenue growth data unavailable.", is_missing=True), missing

    payout_ok = (payout is None or (0.0 <= payout <= 0.70))

    if rev_g >= 0.05 and payout_ok:
        payout_txt = f"{payout*100:.1f}%" if payout is not None else "0% (retained)"
        return PillarScore("Growth", 20, passed=True, reason=f"Steady growth & capital discipline: Revenue Growth {rev_g*100:.1f}% >= 5%, Payout {payout_txt} <= 70%."), missing
    elif rev_g >= 0.0:
        return PillarScore("Growth", 10, passed=False, reason=f"Modest expansion: Revenue Growth {rev_g*100:.1f}% >= 0%."), missing
    else:
        return PillarScore("Growth", 0, passed=False, reason=f"Contracting top-line: Revenue Growth {rev_g*100:.1f}% < 0%."), missing


def score_fundamentals(data: FundamentalData) -> FundamentalScorecard:
    """
    Score a single company's fundamentals across the 5 pillars.

    Applies sector-specific adjustments and missing-data governance.
    """
    meta: ScreenerCompany = get_company_meta(data.ticker)
    is_bank = meta.is_financial

    p1, m1 = _score_valuation(data, is_bank)
    p2, m2 = _score_profitability(data, is_bank)
    p3, m3 = _score_solvency(data, is_bank)
    p4, m4 = _score_cash_flow(data, is_bank)
    p5, m5 = _score_growth_stewardship(data, is_bank)

    all_missing = sorted(set(m1 + m2 + m3 + m4 + m5))
    missing_pillars_count = sum(1 for p in [p1, p2, p3, p4, p5] if p.is_missing)

    pillar_map = {
        "Valuation": p1,
        "Profitability": p2,
        "Solvency": p3,
        "Cash Flow": p4,
        "Growth": p5,
    }

    composite_score = sum(p.score for p in pillar_map.values())

    # Missing Data Governance Rule:
    # If >= 2 pillars have missing data -> INSUFFICIENT_DATA and UNRANKED.
    # If 1 pillar has missing data -> COMPLETE_WITH_GAPS (scores 0 for that pillar, out of 100).
    # Otherwise -> COMPLETE.
    if missing_pillars_count >= 2:
        status = "INSUFFICIENT_DATA"
        tier = "UNRANKED (Insufficient Data)"
    else:
        status = "COMPLETE_WITH_GAPS" if missing_pillars_count == 1 else "COMPLETE"
        if composite_score >= 80:
            tier = "Tier 1 (High Quality)"
        elif composite_score >= 60:
            tier = "Tier 2 (Moderate)"
        else:
            tier = "Tier 3 (Elevated Risk)"

    # Formatted key metrics dictionary for clean UI rendering
    key_metrics = {
        "pe": f"{data.trailing_pe:.1f}x" if data.trailing_pe else ("N/A (unprofitable)" if data.forward_pe else "N/A"),
        "peg": f"{data.peg_ratio:.2f}" if data.peg_ratio else "N/A",
        "pb": f"{data.price_to_book:.2f}x" if data.price_to_book else "N/A",
        "roe": f"{data.return_on_equity*100:.1f}%" if data.return_on_equity is not None else "N/A",
        "op_margin": f"{data.operating_margins*100:.1f}%" if data.operating_margins is not None else "N/A",
        "de": f"{data.debt_to_equity:.1f}%" if data.debt_to_equity is not None else ("N/A (Bank)" if is_bank else "N/A"),
        "cr": f"{data.current_ratio:.2f}x" if data.current_ratio is not None else ("N/A (Bank)" if is_bank else "N/A"),
        "fcf_b": f"${data.free_cash_flow/1e9:.2f}B" if data.free_cash_flow is not None else ("N/A (Bank)" if is_bank else "N/A"),
        "rev_growth": f"{data.revenue_growth*100:+.1f}%" if data.revenue_growth is not None else "N/A",
        "div_yield": f"{data.dividend_yield*100:.2f}%" if data.dividend_yield else "0.00%",
        "payout": f"{data.payout_ratio*100:.1f}%" if data.payout_ratio else "N/A",
    }

    return FundamentalScorecard(
        ticker=data.ticker,
        company_name=meta.name,
        sector=meta.sector,
        composite_score=composite_score,
        tier=tier,
        status=status,
        pillar_scores=pillar_map,
        missing_fields=all_missing,
        key_metrics=key_metrics,
    )
