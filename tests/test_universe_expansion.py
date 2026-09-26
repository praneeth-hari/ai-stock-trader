"""
tests/test_universe_expansion.py — Test Universe Expansion & Sector Coverage (Section 1).
"""

import pytest
from config.settings import settings
from src.screening.universe import TICKER_MAP, get_company_meta


def test_watchlist_expanded_size():
    """Verify ticker universe is expanded to 20-30 stocks while max positions remains 3."""
    tickers = settings.ticker_list
    assert 20 <= len(tickers) <= 30, f"Expected 20-30 tickers, found {len(tickers)}: {tickers}"
    # Risk engine position constraint remains 3, not constrained by universe size
    assert settings.max_positions == 3


def test_all_eight_sectors_represented():
    """Verify that all 8 required sectors have representation in the watchlist."""
    required_sectors = {
        "Technology",
        "Communications",
        "Consumer Cyclical",
        "Financials",
        "Healthcare",
        "Industrials",
        "Consumer Staples",
        "Energy",
    }

    found_sectors = set()
    for ticker in settings.ticker_list:
        meta = get_company_meta(ticker)
        assert meta is not None, f"Ticker {ticker} missing metadata in TICKER_MAP"
        sector = "Consumer Cyclical" if meta.sector == "Consumer Discretionary" else meta.sector
        assert sector in required_sectors, f"Ticker {ticker} has unexpected sector '{meta.sector}'"
        found_sectors.add(sector)

    missing_sectors = required_sectors - found_sectors
    assert not missing_sectors, f"Watchlist missing representation for sectors: {missing_sectors}"
    assert found_sectors == required_sectors


def test_each_sector_has_multiple_stocks():
    """Ensure healthy diversification across sectors (at least 2-3 stocks per sector)."""
    sector_counts = {}
    for ticker in settings.ticker_list:
        meta = get_company_meta(ticker)
        sector_counts[meta.sector] = sector_counts.get(meta.sector, 0) + 1

    for sector, count in sector_counts.items():
        assert count >= 2, f"Sector {sector} has only {count} stocks, expected at least 2 for diversification"


def test_india_tickers_count():
    from config.settings import settings
    assert len(settings.india_tickers) == 25


def test_india_tickers_have_ns_suffix():
    from config.settings import settings
    assert all(t.endswith(".NS") for t in settings.india_tickers)


def test_india_settings_exist():
    from config.settings import settings
    assert settings.india_initial_capital == 10000.0
    assert settings.india_benchmark == "^NSEI"
    assert settings.india_vix_symbol == "^INDIAVIX"
    assert settings.india_currency == "INR"
    assert settings.india_pipeline_hour_ist == 16
