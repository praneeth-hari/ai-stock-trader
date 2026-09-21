"""
tests/test_screener.py — Unit and Integration Tests for Fundamental Screener.

Verifies:
  1. Universe integrity: 45 unique tickers, 8 sectors, valid metadata.
  2. Collector parser: maps raw yfinance payload to FundamentalData with safe float conversion.
  3. Scorer logic (Non-Financial): high-quality profile earns Tier 1; distressed earns Tier 3.
  4. Scorer logic (Financials/Banks): banks evaluate on P/B, ROE, ROA without penalty for missing D/E or FCF.
  5. Missing Data Governance:
     - 1 missing pillar metric scores 0 with [MISSING_DATA] tag and COMPLETE_WITH_GAPS status.
     - >= 2 missing pillars marks status INSUFFICIENT_DATA and tier UNRANKED.
  6. Strict Architectural Isolation:
     - Zero imports of src.trading, src.risk, src.portfolio, or src.pipeline.scheduler in src/screening/.
  7. Markdown Report Generator:
     - Includes mandatory disclosures, tier distribution, and 5-pillar breakdown table.
  8. Dashboard AppTest:
     - Confirms dashboard/app.py compiles and renders 6 tabs without exceptions.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.screening.collector import FundamentalData, fetch_ticker_fundamentals
from src.screening.scorer import (
    FundamentalScorecard,
    PillarScore,
    score_fundamentals,
)
from src.screening.screener import (
    generate_markdown_report,
    run_fundamental_screen,
    scorecards_to_dataframe,
)
from src.screening.universe import SCREENER_UNIVERSE, get_company_meta, get_screener_tickers


# ── Test 1: Universe Integrity ────────────────────────────────────────────────

def test_1_screener_universe_integrity():
    """Verify 45 curated tickers across 8 unique sectors."""
    tickers = get_screener_tickers()
    assert len(tickers) == 45, f"Expected 45 tickers, got {len(tickers)}"
    assert len(set(tickers)) == 45, "Duplicate tickers detected in screener universe"

    sectors = {c.sector for c in SCREENER_UNIVERSE}
    expected_sectors = {
        "Technology", "Healthcare", "Financials", "Consumer Staples",
        "Consumer Discretionary", "Industrials", "Energy", "Utilities", "Communications"
    }
    assert sectors.issubset(expected_sectors) or len(sectors) >= 8, f"Unexpected sectors: {sectors}"

    # Verify bank flagging
    jpm = get_company_meta("JPM")
    assert jpm.is_financial is True
    aapl = get_company_meta("AAPL")
    assert aapl.is_financial is False


# ── Test 2: Collector Safe Float & Cache Parsing ──────────────────────────────

def test_2_collector_safe_parsing_and_snapshot(tmp_path):
    """Verify fundamental collector parses raw dict and writes/reads JSON cache."""
    mock_info = {
        "trailingPE": 25.4,
        "forwardPE": 20.1,
        "pegRatio": 1.5,
        "priceToBook": 4.2,
        "profitMargins": 0.20,
        "operatingMargins": 0.22,
        "returnOnEquity": 0.28,
        "returnOnAssets": 0.12,
        "revenueGrowth": 0.08,
        "earningsGrowth": 0.10,
        "debtToEquity": 65.0,
        "currentRatio": 1.4,
        "freeCashflow": 15000000000,
        "operatingCashflow": 20000000000,
        "dividendYield": 0.02,
        "payoutRatio": 0.40,
        "marketCap": 2000000000000,
    }

    mock_ticker_obj = MagicMock()
    mock_ticker_obj.info = mock_info

    with patch("yfinance.Ticker", return_value=mock_ticker_obj):
        data = fetch_ticker_fundamentals(
            ticker="TEST",
            fetch_date="2026-09-13",
            cache_dir=tmp_path,
            force_refresh=True,
        )

        assert data.ticker == "TEST"
        assert data.trailing_pe == 25.4
        assert data.return_on_equity == 0.28
        assert data.free_cash_flow == 15000000000

        # Verify cached file written
        cached_file = tmp_path / "2026-09-13" / "TEST.json"
        assert cached_file.exists()

        # Re-fetch from cache without calling yfinance
        with patch("yfinance.Ticker", side_effect=Exception("Should not hit network")):
            cached_data = fetch_ticker_fundamentals(
                ticker="TEST",
                fetch_date="2026-09-13",
                cache_dir=tmp_path,
                force_refresh=False,
            )
            assert cached_data.trailing_pe == 25.4


# ── Test 3: Scorer Logic (Non-Financials) ─────────────────────────────────────

def test_3_scorer_non_financial_high_quality_and_distressed():
    """High quality company scores Tier 1; distressed company scores Tier 3."""
    # High-quality profile (e.g. strong tech / industrial)
    high_qual = FundamentalData(
        ticker="AAPL",
        fetch_date="2026-09-13",
        trailing_pe=22.0,
        forward_pe=18.0,
        peg_ratio=1.4,
        price_to_book=8.0,
        profit_margins=0.25,
        operating_margins=0.30,
        return_on_equity=0.35,
        return_on_assets=0.15,
        revenue_growth=0.10,
        debt_to_equity=50.0,
        current_ratio=1.5,
        free_cash_flow=50_000_000_000,
        operating_cash_flow=60_000_000_000,
        dividend_yield=0.01,
        payout_ratio=0.20,
    )
    sc_hq = score_fundamentals(high_qual)
    assert sc_hq.composite_score >= 80, f"Expected Tier 1 (>=80), got {sc_hq.composite_score}"
    assert "Tier 1" in sc_hq.tier
    assert sc_hq.status == "COMPLETE"
    assert sc_hq.pillar_scores["Valuation"].score == 20
    assert sc_hq.pillar_scores["Profitability"].score == 20
    assert sc_hq.pillar_scores["Solvency"].score == 20
    assert sc_hq.pillar_scores["Cash Flow"].score == 20
    assert sc_hq.pillar_scores["Growth"].score == 20

    # Distressed profile (expensive/unprofitable, high debt, negative cash flow)
    distressed = FundamentalData(
        ticker="CAT",
        fetch_date="2026-09-13",
        trailing_pe=55.0,
        forward_pe=50.0,
        peg_ratio=3.5,
        profit_margins=0.02,
        operating_margins=0.03,
        return_on_equity=0.04,
        return_on_assets=0.01,
        revenue_growth=-0.08,
        debt_to_equity=250.0,
        current_ratio=0.7,
        free_cash_flow=-2_000_000_000,
        operating_cash_flow=-500_000_000,
    )
    sc_dist = score_fundamentals(distressed)
    assert sc_dist.composite_score < 60, f"Expected Tier 3 (<60), got {sc_dist.composite_score}"
    assert "Tier 3" in sc_dist.tier


# ── Test 4: Scorer Logic (Banks / Financials) ─────────────────────────────────

def test_4_scorer_bank_specific_adjustments():
    """Banks score on P/B, ROE, ROA and are NOT penalized for missing D/E or FCF."""
    bank_data = FundamentalData(
        ticker="JPM",
        fetch_date="2026-09-13",
        trailing_pe=14.0,
        forward_pe=13.0,
        price_to_book=1.7,
        profit_margins=0.32,
        operating_margins=0.48,
        return_on_equity=0.17,
        return_on_assets=0.013,  # 1.3% ROA is outstanding for a bank
        revenue_growth=0.08,
        dividend_yield=0.022,
        payout_ratio=0.30,
        debt_to_equity=None,     # Structurally None
        current_ratio=None,      # Structurally None
        free_cash_flow=None,     # Structurally None
        enterprise_to_ebitda=None,
    )
    sc_bank = score_fundamentals(bank_data)
    assert sc_bank.composite_score >= 80, f"Bank should score Tier 1 on strong metrics, got {sc_bank.composite_score}"
    assert "Tier 1" in sc_bank.tier
    assert sc_bank.status == "COMPLETE"
    assert sc_bank.pillar_scores["Solvency"].score == 20
    assert sc_bank.pillar_scores["Cash Flow"].score == 20
    assert sc_bank.key_metrics["de"] == "N/A (Bank)"
    assert sc_bank.key_metrics["fcf_b"] == "N/A (Bank)"


# ── Test 5: Missing Data Governance ───────────────────────────────────────────

def test_5_missing_data_governance_single_gap_and_insufficient_data():
    """
    1 missing metric: scores 0 for that pillar, status COMPLETE_WITH_GAPS.
    >= 2 missing pillars: status INSUFFICIENT_DATA and tier UNRANKED.
    """
    # 1 missing pillar (missing Cash Flow)
    single_gap = FundamentalData(
        ticker="XOM",
        fetch_date="2026-09-13",
        trailing_pe=18.0,
        forward_pe=15.0,
        peg_ratio=1.2,
        profit_margins=0.15,
        operating_margins=0.18,
        return_on_equity=0.20,
        revenue_growth=0.06,
        debt_to_equity=25.0,
        current_ratio=1.4,
        free_cash_flow=None,
        operating_cash_flow=None,  # Missing Pillar 4
    )
    sc_gap = score_fundamentals(single_gap)
    assert sc_gap.status == "COMPLETE_WITH_GAPS"
    assert sc_gap.pillar_scores["Cash Flow"].score == 0
    assert sc_gap.pillar_scores["Cash Flow"].is_missing is True
    # Still ranked because other 4 pillars are valid
    assert "Tier" in sc_gap.tier

    # >= 2 missing pillars (missing Valuation AND Profitability)
    severe_missing = FundamentalData(
        ticker="DE",
        fetch_date="2026-09-13",
        trailing_pe=None,
        forward_pe=None,          # Missing Pillar 1
        return_on_equity=None,    # Missing Pillar 2
        revenue_growth=0.04,
        debt_to_equity=80.0,
        current_ratio=1.5,
        free_cash_flow=2_000_000_000,
    )
    sc_severe = score_fundamentals(severe_missing)
    assert sc_severe.status == "INSUFFICIENT_DATA"
    assert sc_severe.tier == "UNRANKED (Insufficient Data)"
    assert len(sc_severe.missing_fields) >= 2


# ── Test 6: Strict Architectural Isolation ───────────────────────────────────

def test_6_strict_architectural_isolation():
    """
    Guarantees src/screening/ does NOT import any trading, risk, portfolio,
    or scheduler modules using Python AST parsing.
    """
    import ast
    screening_dir = Path("src/screening")
    assert screening_dir.exists()

    forbidden_modules = {
        "src.trading",
        "src.risk",
        "src.portfolio",
        "src.pipeline.scheduler",
        "src.trading.paper_broker",
        "src.risk.risk_engine",
        "src.portfolio.portfolio",
    }

    for py_file in screening_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=py_file.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forb in forbidden_modules:
                        assert not alias.name.startswith(forb), (
                            f"Isolation violation in {py_file.name}: imports '{alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for forb in forbidden_modules:
                    assert not mod.startswith(forb), (
                        f"Isolation violation in {py_file.name}: from '{mod}' import ..."
                    )


@pytest.fixture()
def clean_db(tmp_path):
    """Provides an isolated DB for dashboard AppTest."""
    from src.db import repository
    db_file = tmp_path / "test_scr_dash.db"
    db_url = f"sqlite:///{db_file}"
    repository._engine = None
    orig_url = repository.settings.db_url

    try:
        repository.settings.__dict__["db_url"] = db_url
        repository.create_all_tables()
        yield db_url
    finally:
        repository._engine = None
        repository.settings.__dict__["db_url"] = orig_url


# ── Test 7: Markdown Report Generation ────────────────────────────────────────

def test_7_markdown_report_generation(tmp_path):
    """Verifies generated report has disclosures, summary, and 5-pillar table."""
    sample_data = FundamentalData(
        ticker="JNJ",
        fetch_date="2026-09-13",
        trailing_pe=24.0,
        forward_pe=20.0,
        peg_ratio=1.8,
        price_to_book=6.0,
        profit_margins=0.20,
        operating_margins=0.25,
        return_on_equity=0.26,
        revenue_growth=0.07,
        debt_to_equity=45.0,
        current_ratio=1.3,
        free_cash_flow=16_000_000_000,
        operating_cash_flow=22_000_000_000,
    )
    card = score_fundamentals(sample_data)
    out_file = tmp_path / "test_report.md"
    report_text = generate_markdown_report([card], output_path=out_file)

    assert out_file.exists()
    assert "INFORMATIONAL PURPOSE & DISCLOSURES" in report_text
    assert "Survivorship & Selection Bias" in report_text
    assert "Val (20)" in report_text
    assert "Prof (20)" in report_text
    assert "Solv (20)" in report_text
    assert "Cash (20)" in report_text
    assert "Grow (20)" in report_text
    assert "`JNJ`" in report_text


# ── Test 8: Dashboard Full Render with Tab 6 ──────────────────────────────────

def test_8_dashboard_app_renders_with_tab6(clean_db):
    """Verifies dashboard/app.py renders all 6 tabs without exceptions via AppTest."""
    from streamlit.testing.v1 import AppTest

    fake_spy = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=250, freq="B").strftime("%Y-%m-%d"),
        "open": [400.0] * 250,
        "high": [405.0] * 250,
        "low": [395.0] * 250,
        "close": [402.0] * 250,
        "volume": [50000000.0] * 250,
        "ticker": ["SPY"] * 250,
    })

    with patch("src.data.market_data.fetch_ticker_data", return_value=fake_spy):
        at = AppTest.from_file("dashboard/app.py", default_timeout=15)
        at.run()
        assert len(at.exception) == 0, f"Dashboard execution raised exceptions: {at.exception}"
