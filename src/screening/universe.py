"""
src/screening/universe.py — Curated Universe for Fundamental Screening.

Defines a balanced, cross-sector universe of 45 prominent US companies across
8 distinct economic sectors.

SURVIVORSHIP & SELECTION BIAS DISCLOSURE:
This universe is manually curated from prominent large-cap equities active today.
It does not model historical delistings, bankruptcies, or mergers, and is
designed exclusively as a transparent, forward-looking fundamental screen.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class ScreenerCompany(NamedTuple):
    ticker: str
    name: str
    sector: str
    is_financial: bool = False


# Curated 45-stock universe across 8 major sectors
SCREENER_UNIVERSE: List[ScreenerCompany] = [
    # ── Technology (7) ──────────────────────────────────────────────────────────
    ScreenerCompany("AAPL", "Apple Inc.", "Technology"),
    ScreenerCompany("MSFT", "Microsoft Corp.", "Technology"),
    ScreenerCompany("GOOGL", "Alphabet Inc.", "Technology"),
    ScreenerCompany("NVDA", "NVIDIA Corp.", "Technology"),
    ScreenerCompany("ADBE", "Adobe Inc.", "Technology"),
    ScreenerCompany("CRM", "Salesforce Inc.", "Technology"),
    ScreenerCompany("CSCO", "Cisco Systems Inc.", "Technology"),

    # ── Healthcare & Life Sciences (7) ──────────────────────────────────────────
    ScreenerCompany("JNJ", "Johnson & Johnson", "Healthcare"),
    ScreenerCompany("UNH", "UnitedHealth Group Inc.", "Healthcare"),
    ScreenerCompany("PFE", "Pfizer Inc.", "Healthcare"),
    ScreenerCompany("ABBV", "AbbVie Inc.", "Healthcare"),
    ScreenerCompany("LLY", "Eli Lilly and Co.", "Healthcare"),
    ScreenerCompany("MRK", "Merck & Co. Inc.", "Healthcare"),
    ScreenerCompany("TMO", "Thermo Fisher Scientific", "Healthcare"),

    # ── Financials (6) ──────────────────────────────────────────────────────────
    ScreenerCompany("JPM", "JPMorgan Chase & Co.", "Financials", is_financial=True),
    ScreenerCompany("BAC", "Bank of America Corp.", "Financials", is_financial=True),
    ScreenerCompany("V", "Visa Inc.", "Financials"),
    ScreenerCompany("MA", "Mastercard Inc.", "Financials"),
    ScreenerCompany("BRK-B", "Berkshire Hathaway Inc.", "Financials", is_financial=True),
    ScreenerCompany("GS", "Goldman Sachs Group Inc.", "Financials", is_financial=True),

    # ── Consumer Staples (5) ────────────────────────────────────────────────────
    ScreenerCompany("PG", "Procter & Gamble Co.", "Consumer Staples"),
    ScreenerCompany("KO", "Coca-Cola Co.", "Consumer Staples"),
    ScreenerCompany("PEP", "PepsiCo Inc.", "Consumer Staples"),
    ScreenerCompany("WMT", "Walmart Inc.", "Consumer Staples"),
    ScreenerCompany("COST", "Costco Wholesale Corp.", "Consumer Staples"),

    # ── Consumer Discretionary (5) ──────────────────────────────────────────────
    ScreenerCompany("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ScreenerCompany("HD", "Home Depot Inc.", "Consumer Discretionary"),
    ScreenerCompany("MCD", "McDonald's Corp.", "Consumer Discretionary"),
    ScreenerCompany("NKE", "NIKE Inc.", "Consumer Discretionary"),
    ScreenerCompany("TSLA", "Tesla Inc.", "Consumer Discretionary"),

    # ── Industrials & Materials (5) ─────────────────────────────────────────────
    ScreenerCompany("CAT", "Caterpillar Inc.", "Industrials"),
    ScreenerCompany("HON", "Honeywell International", "Industrials"),
    ScreenerCompany("UPS", "United Parcel Service", "Industrials"),
    ScreenerCompany("GE", "GE Aerospace", "Industrials"),
    ScreenerCompany("DE", "Deere & Company", "Industrials"),

    # ── Energy (5) ──────────────────────────────────────────────────────────────
    ScreenerCompany("XOM", "Exxon Mobil Corp.", "Energy"),
    ScreenerCompany("CVX", "Chevron Corp.", "Energy"),
    ScreenerCompany("COP", "ConocoPhillips", "Energy"),
    ScreenerCompany("SLB", "SLB (Schlumberger)", "Energy"),
    ScreenerCompany("EOG", "EOG Resources Inc.", "Energy"),

    # ── Utilities & Communications (5) ──────────────────────────────────────────
    ScreenerCompany("NEE", "NextEra Energy Inc.", "Utilities"),
    ScreenerCompany("SO", "Southern Company", "Utilities"),
    ScreenerCompany("DUK", "Duke Energy Corp.", "Utilities"),
    ScreenerCompany("VZ", "Verizon Communications", "Communications"),
    ScreenerCompany("CMCSA", "Comcast Corp.", "Communications"),
]

# Auxiliary companies in trading universe not part of 45-stock screener list
AUXILIARY_COMPANIES: List[ScreenerCompany] = [
    ScreenerCompany("META", "Meta Platforms Inc.", "Communications"),
]

TICKER_MAP: Dict[str, ScreenerCompany] = {c.ticker: c for c in SCREENER_UNIVERSE}
for _aux in AUXILIARY_COMPANIES:
    TICKER_MAP[_aux.ticker] = _aux


def get_screener_tickers() -> List[str]:
    """Returns list of all 45 curated tickers."""
    return [c.ticker for c in SCREENER_UNIVERSE]


def get_company_meta(ticker: str) -> ScreenerCompany:
    """Returns metadata for a given ticker or generic fallback."""
    clean = ticker.strip().upper()
    return TICKER_MAP.get(clean, ScreenerCompany(clean, clean, "Other"))
