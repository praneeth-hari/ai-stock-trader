"""
tests/test_correlation_filter.py — Unit tests for Section 8b: Correlation Filter.
"""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.db import repository
from src.ranking.ranking import RankedOpportunity, RankingResult
from src.risk.correlation import (
    calculate_returns_correlation,
    evaluate_candidate_correlation,
    compute_and_save_correlation_matrix,
    CorrelationEvaluationResult,
)
from src.risk.risk_engine import (
    evaluate_portfolio_risk,
    RiskVetoReason,
)


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """Ensure tests run against a fresh isolated SQLite database."""
    db_file = tmp_path / "test_correlation.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setattr(settings, "db_url", db_url)
    import src.db.repository as repo
    monkeypatch.setattr(repo, "_engine", None)
    repository.create_all_tables()
    yield
    monkeypatch.setattr(repo, "_engine", None)


def _generate_price_history(
    base_price: float,
    daily_returns: np.ndarray,
    start_date: str = "2026-01-01",
) -> pd.DataFrame:
    """Helper to construct OHLCV DataFrame from a sequence of daily returns."""
    num_days = len(daily_returns)
    date_range = pd.bdate_range(start=start_date, periods=num_days)
    prices = [base_price]
    for r in daily_returns:
        prices.append(prices[-1] * (1.0 + r))
    prices = prices[1:]

    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in date_range],
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000.0] * num_days,
    })
    return df


class TestSettingsDefaults:
    def test_correlation_settings_defaults(self):
        assert settings.correlation_lookback_days == 60
        assert settings.correlation_block_threshold == 0.85
        assert settings.correlation_warn_threshold == 0.70
        assert settings.correlation_different_sector_threshold == 0.90


class TestCorrelationCalculation:
    def test_calculate_returns_correlation_perfect(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 65)
        df_a = _generate_price_history(100.0, returns)
        df_b = _generate_price_history(50.0, returns)

        r = calculate_returns_correlation(df_a, df_b, lookback_days=60)
        assert r is not None
        assert r == pytest.approx(1.0, abs=0.01)

    def test_calculate_returns_correlation_insufficient_data(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 30)  # Only 30 days
        df_a = _generate_price_history(100.0, returns)
        df_b = _generate_price_history(50.0, returns)

        r = calculate_returns_correlation(df_a, df_b, lookback_days=60)
        assert r is None


class TestCorrelationEvaluationRules:
    def test_no_positions_held_allows_buy(self):
        res = evaluate_candidate_correlation("AAPL", [])
        assert res.allowed is True
        assert res.warning is False
        assert "CORRELATION_OK: low correlation with held stocks" in res.log_message

    def test_low_correlation_allows_buy(self, caplog):
        import logging
        np.random.seed(100)
        ret_a = np.random.normal(0.001, 0.015, 65)
        ret_b = np.random.normal(-0.001, 0.015, 65)  # Independent/uncorrelated
        df_aapl = _generate_price_history(150.0, ret_a)
        df_jpm = _generate_price_history(120.0, ret_b)

        histories = {"AAPL": df_aapl, "JPM": df_jpm}
        with caplog.at_level(logging.INFO):
            res = evaluate_candidate_correlation("AAPL", ["JPM"], price_histories=histories)

        assert res.allowed is True
        assert res.warning is False
        assert "CORRELATION_OK: low correlation with held stocks" in caplog.text

    def test_moderate_correlation_warns_and_allows(self, caplog):
        np.random.seed(10)
        base_ret = np.random.normal(0.001, 0.015, 65)
        noise = np.random.normal(0.0, 0.008, 65)
        # Moderate correlation around 0.75-0.80
        ret_msft = base_ret + noise

        df_aapl = _generate_price_history(150.0, base_ret)
        df_msft = _generate_price_history(250.0, ret_msft)

        histories = {"AAPL": df_aapl, "MSFT": df_msft}
        res = evaluate_candidate_correlation("AAPL", ["MSFT"], price_histories=histories)

        assert res.allowed is True
        assert res.warning is True
        assert "CORRELATION_WARNING: moderately correlated with MSFT" in caplog.text
        assert "proceeding" in caplog.text

    def test_high_correlation_blocks_buy_same_sector(self, caplog):
        np.random.seed(42)
        base_ret = np.random.normal(0.001, 0.02, 65)
        # Both AAPL and NVDA are Technology sector
        df_aapl = _generate_price_history(150.0, base_ret)
        df_nvda = _generate_price_history(400.0, base_ret)

        histories = {"NVDA": df_nvda, "AAPL": df_aapl}
        res = evaluate_candidate_correlation("NVDA", ["AAPL"], price_histories=histories)

        assert res.allowed is False
        assert res.veto_reason == "CORRELATION_VETO"
        assert "CORRELATION_VETO: too correlated with AAPL" in caplog.text
        assert "skipping" in caplog.text

    def test_different_sector_threshold_leniency(self, caplog):
        np.random.seed(42)
        # Generate returns that give correlation ~ 0.87 (between 0.85 and 0.90)
        base_ret = np.random.normal(0.001, 0.015, 65)
        noise = np.random.normal(0.0, 0.008, 65)
        ret_diff = base_ret * 0.7 + noise * 0.7

        # AAPL (Technology) vs XOM (Energy) -> different sectors!
        df_aapl = _generate_price_history(150.0, base_ret)
        df_xom = _generate_price_history(80.0, ret_diff)

        histories = {"AAPL": df_aapl, "XOM": df_xom}
        res = evaluate_candidate_correlation("AAPL", ["XOM"], price_histories=histories)

        # Correlation between 0.85 and 0.90 (different sector threshold).
        # Should be ALLOWED with warning!
        assert res.allowed is True
        assert res.warning is True
        assert "CORRELATION_WARNING: moderately correlated with XOM" in caplog.text

    def test_insufficient_data_skips_check_and_allows(self, caplog):
        np.random.seed(42)
        ret_short = np.random.normal(0.001, 0.015, 30)  # Only 30 days
        df_aapl = _generate_price_history(150.0, ret_short)
        df_msft = _generate_price_history(250.0, ret_short)

        histories = {"AAPL": df_aapl, "MSFT": df_msft}
        res = evaluate_candidate_correlation("AAPL", ["MSFT"], price_histories=histories)

        assert res.allowed is True
        assert "CORRELATION_INSUFFICIENT_DATA: less than 60 days price history for AAPL — skipping check" in caplog.text


class TestRiskEngineIntegration:
    def test_risk_engine_correlation_veto(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 65)
        df_aapl = _generate_price_history(150.0, returns)
        df_nvda = _generate_price_history(400.0, returns)

        price_histories = {"AAPL": df_aapl, "NVDA": df_nvda}

        opp = RankedOpportunity(
            ticker="NVDA",
            date="2026-04-01",
            probability=0.85,
            rank=1,
            conviction_tier="BUY_CANDIDATE",
            is_buy_eligible=True,
        )

        ranking_result = RankingResult(
            date="2026-04-01",
            regime_risk_on=True,
            total_evaluated=1,
            ranked_opportunities=[opp],
            top_buy_candidates=[opp],
        )

        current_positions = {
            "AAPL": {"quantity": 10, "avg_cost": 150.0, "current_price": 155.0, "holding_days": 5}
        }
        current_prices = {"AAPL": 155.0, "NVDA": 400.0}

        res = evaluate_portfolio_risk(
            run_date="2026-04-01",
            current_cash=5000.0,
            current_positions=current_positions,
            current_prices=current_prices,
            ranking_result=ranking_result,
            price_histories=price_histories,
        )

        assert len(res.buy_orders) == 0
        vetoed = [v for v in res.vetoed_orders if v.veto_reason == RiskVetoReason.CORRELATION_VETO]
        assert len(vetoed) == 1
        assert vetoed[0].ticker == "NVDA"
        assert "CORRELATION_VETO: too correlated with AAPL" in vetoed[0].details


class TestDatabaseAndDashboard:
    def test_save_and_get_correlation_matrix(self):
        records = [
            {"date": "2026-04-01", "ticker_a": "AAPL", "ticker_b": "MSFT", "correlation": 0.82, "lookback_days": 60},
            {"date": "2026-04-01", "ticker_a": "AAPL", "ticker_b": "NVDA", "correlation": 0.88, "lookback_days": 60},
        ]
        inserted = repository.save_correlation_matrix(records)
        assert inserted == 2

        fetched = repository.get_correlation_matrix("2026-04-01")
        assert len(fetched) == 2
        assert fetched[0]["ticker_a"] == "AAPL"

    def test_compute_and_save_correlation_matrix(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.015, 65)
        df_aapl = _generate_price_history(150.0, ret)
        df_msft = _generate_price_history(250.0, ret)

        histories = {"AAPL": df_aapl, "MSFT": df_msft}
        corr_df = compute_and_save_correlation_matrix(
            date_str="2026-04-01",
            tickers=["AAPL", "MSFT"],
            price_histories=histories,
        )
        assert corr_df.shape == (2, 2)
        assert corr_df.loc["AAPL", "MSFT"] == pytest.approx(1.0, abs=0.01)

        db_rows = repository.get_correlation_matrix("2026-04-01")
        assert len(db_rows) >= 2
