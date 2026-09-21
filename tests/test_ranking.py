"""
tests/test_ranking.py — Phase 7 Stock Ranking Unit & Integration Tests.

TEST SUITE OBJECTIVES:
  1. Ordering: highest probability ranks at the top.
  2. Tie-Breaking: relative strength (rel_strength_21) and price_to_ma50 resolve ties.
  3. Conviction Tiers: verify buy_bar (0.60) and signal_exit (0.45) boundary conditions.
  4. Sizing Separation (CLAUDE.md Rule 2): verify top_buy_candidates is NOT pre-capped at 3.
  5. Cash-First / Zero-Candidate: when no stock achieves P >= 0.60, buy_candidates is empty.
  6. Real Active-Model Integration: end-to-end inference and ranking using promoted active_model.joblib.
  7. Market Regime Check: SPY >= 200d MA sets regime_risk_on = True; SPY < 200d MA sets False.
"""

from typing import List
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS
from src.ml.evaluate import load_active_model
from src.ml.train import TrainedModel
from src.ranking.ranking import (
    RankedOpportunity,
    RankingResult,
    check_market_regime,
    rank_candidates,
    TIER_BUY,
    TIER_NEUTRAL,
    TIER_EXIT,
)


class MockEstimator:
    """Mock estimator returning predetermined probabilities for unit testing."""
    def predict_proba(self, X):
        p = X["roc_5"].values
        return np.column_stack([1.0 - p, p])


def make_mock_model() -> TrainedModel:
    return TrainedModel(
        model_name="mock_test_model",
        model_type="baseline",
        estimator=MockEstimator(),
        features=FEATURE_COLUMNS,
        metadata={"mock": True},
    )


def make_dummy_features(
    tickers: List[str],
    date: str = "2024-06-14",
    prob_map: dict = None,
    rel_strengths: dict = None,
    prices_to_ma50: dict = None,
) -> pd.DataFrame:
    rows = []
    prob_map = prob_map or {}
    rel_strengths = rel_strengths or {}
    prices_to_ma50 = prices_to_ma50 or {}
    for t in tickers:
        row = {"date": date, "ticker": t}
        for col in FEATURE_COLUMNS:
            row[col] = 0.0
        row["roc_5"] = prob_map.get(t, 0.50)  # Encodes mock probability in a valid feature
        row["rel_strength_21"] = rel_strengths.get(t, 0.0)
        row["price_to_ma50"] = prices_to_ma50.get(t, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


class TestRankingOrder:
    def test_1_ranking_order_by_probability(self):
        """Verify candidates are strictly sorted by probability descending."""
        tickers = ["AAPL", "MSFT", "NVDA", "JPM", "UNH"]
        prob_map = {
            "AAPL": 0.72,
            "MSFT": 0.64,
            "NVDA": 0.55,
            "JPM": 0.48,
            "UNH": 0.38,
        }
        model = make_mock_model()
        df = make_dummy_features(tickers, prob_map=prob_map)

        result = rank_candidates(df, model=model)

        assert result.total_evaluated == 5
        ordered_tickers = [o.ticker for o in result.ranked_opportunities]
        assert ordered_tickers == ["AAPL", "MSFT", "NVDA", "JPM", "UNH"]
        assert [o.rank for o in result.ranked_opportunities] == [1, 2, 3, 4, 5]

    def test_2_tie_breaker_relative_strength_and_ma50(self):
        """When probabilities are identical, relative strength and price_to_ma50 break ties."""
        tickers = ["STOCK_A", "STOCK_B", "STOCK_C"]
        # All three have identical 0.65 probability
        prob_map = {"STOCK_A": 0.65, "STOCK_B": 0.65, "STOCK_C": 0.65}
        # STOCK_B has highest relative strength
        # STOCK_A and STOCK_C have same RS, but STOCK_C has higher MA-50 buffer
        rel_strengths = {"STOCK_A": 0.02, "STOCK_B": 0.08, "STOCK_C": 0.02}
        prices_to_ma50 = {"STOCK_A": 0.01, "STOCK_B": 0.03, "STOCK_C": 0.05}

        model = make_mock_model()
        df = make_dummy_features(
            tickers,
            prob_map=prob_map,
            rel_strengths=rel_strengths,
            prices_to_ma50=prices_to_ma50,
        )

        result = rank_candidates(df, model=model)
        ordered_tickers = [o.ticker for o in result.ranked_opportunities]

        # STOCK_B has highest RS (+8%) -> Rank 1
        # STOCK_C and STOCK_A have same RS (+2%), but STOCK_C has higher MA50 (+5% vs +1%) -> Rank 2
        # STOCK_A -> Rank 3
        assert ordered_tickers == ["STOCK_B", "STOCK_C", "STOCK_A"]


class TestConvictionTiers:
    def test_3_conviction_tier_boundaries(self):
        """Verify strict adherence to Section 1.4 entry/exit boundaries."""
        tickers = ["BUY_EXACT", "NEUTRAL_HIGH", "NEUTRAL_LOW", "EXIT_EXACT", "DEEP_EXIT"]
        prob_map = {
            "BUY_EXACT": 0.60,      # P >= 0.60 -> BUY_CANDIDATE
            "NEUTRAL_HIGH": 0.5999, # 0.45 <= P < 0.60 -> NEUTRAL_HOLD
            "NEUTRAL_LOW": 0.45,    # 0.45 <= P < 0.60 -> NEUTRAL_HOLD
            "EXIT_EXACT": 0.4499,   # P < 0.45 -> EXIT_CANDIDATE
            "DEEP_EXIT": 0.30,      # P < 0.45 -> EXIT_CANDIDATE
        }
        model = make_mock_model()
        df = make_dummy_features(tickers, prob_map=prob_map)

        result = rank_candidates(df, model=model)

        tier_by_ticker = {o.ticker: o.conviction_tier for o in result.ranked_opportunities}
        assert tier_by_ticker["BUY_EXACT"] == TIER_BUY
        assert tier_by_ticker["NEUTRAL_HIGH"] == TIER_NEUTRAL
        assert tier_by_ticker["NEUTRAL_LOW"] == TIER_NEUTRAL
        assert tier_by_ticker["EXIT_EXACT"] == TIER_EXIT
        assert tier_by_ticker["DEEP_EXIT"] == TIER_EXIT

        # Check flags
        buy_exact_opp = next(o for o in result.ranked_opportunities if o.ticker == "BUY_EXACT")
        assert buy_exact_opp.is_buy_eligible is True
        assert buy_exact_opp.is_exit_signal is False

        exit_opp = next(o for o in result.ranked_opportunities if o.ticker == "EXIT_EXACT")
        assert exit_opp.is_buy_eligible is False
        assert exit_opp.is_exit_signal is True


class TestRiskEngineSeparationAndNoPreCapping:
    def test_4_uncapped_buy_candidates_no_precapping_at_3(self):
        """
        CRITICAL ARCHITECTURAL TEST (CLAUDE.md Rule 2):
        top_buy_candidates in RankingResult must NOT be pre-capped at 3.
        It must contain ALL qualifying candidates (P >= 0.60) so Risk Engine
        retains full visibility for vetoes and portfolio allocation.
        """
        tickers = ["TICKER_1", "TICKER_2", "TICKER_3", "TICKER_4", "TICKER_5", "TICKER_6"]
        # All 6 tickers qualify above the 0.60 buy bar
        prob_map = {
            "TICKER_1": 0.75,
            "TICKER_2": 0.72,
            "TICKER_3": 0.70,
            "TICKER_4": 0.68,
            "TICKER_5": 0.65,
            "TICKER_6": 0.61,
        }
        model = make_mock_model()
        df = make_dummy_features(tickers, prob_map=prob_map)

        result = rank_candidates(df, model=model)

        # Ensure top_buy_candidates has ALL 6, NOT capped at 3
        assert len(result.top_buy_candidates) == 6, (
            f"Expected 6 buy candidates, but got {len(result.top_buy_candidates)}. "
            "Ranking must NOT pre-cap at 3!"
        )
        assert len(result.buy_candidates) == 6
        assert [o.ticker for o in result.top_buy_candidates] == tickers

    def test_5_zero_candidate_cash_recommendation(self):
        """When no stock crosses P >= 0.60, top_buy_candidates must be empty."""
        tickers = ["AAPL", "MSFT", "JPM"]
        prob_map = {
            "AAPL": 0.55,
            "MSFT": 0.48,
            "JPM": 0.42,
        }
        model = make_mock_model()
        df = make_dummy_features(tickers, prob_map=prob_map)

        result = rank_candidates(df, model=model)

        assert len(result.top_buy_candidates) == 0
        assert len(result.buy_candidates) == 0
        assert len(result.neutral_candidates) == 2  # AAPL (0.55), MSFT (0.48)
        assert len(result.exit_candidates) == 1     # JPM (0.42)


class TestRealActiveModelIntegration:
    def test_6_active_model_end_to_end_ranking(self):
        """Verify real candidate/active model executes inference and ranking successfully."""
        from pathlib import Path
        active_path = Path(settings.data_models_dir) / "active_model.joblib"
        if active_path.exists():
            active = load_active_model()
            assert active is not None
            model_to_test = None  # None triggers rank_candidates to load_active_model automatically
        else:
            # When active_model is reset awaiting human re-promotion, load_active_model must raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                load_active_model()
            # Test inference with candidate model directly
            candidate_files = list(Path(settings.data_models_dir).glob("*.joblib"))
            assert len(candidate_files) > 0, "Expected at least one trained candidate model in data/models/"
            from src.ml.train import load_model
            model_to_test = load_model(candidate_files[0])

        tickers = ["AAPL", "MSFT", "NVDA", "JPM"]
        # Build synthetic features adhering strictly to FEATURE_COLUMNS
        df = make_dummy_features(tickers, date="2024-06-28")

        # Rank with model (or default active model if promoted)
        result = rank_candidates(df, model=model_to_test)

        assert isinstance(result, RankingResult)
        assert result.total_evaluated == 4
        assert len(result.ranked_opportunities) == 4
        assert result.date == "2024-06-28"

        # Check all probabilities are valid in [0.0, 1.0]
        for opp in result.ranked_opportunities:
            assert 0.0 <= opp.probability <= 1.0
            assert opp.rank >= 1

        # Check markdown rendering
        md = result.to_markdown_table()
        assert "Universe Ranking for 2024-06-28" in md
        assert "AAPL" in md
        assert "MSFT" in md


class TestRegimeCheck:
    def test_7_regime_filter_evaluation(self):
        """Verify regime detection when SPY >= 200d MA vs SPY < 200d MA."""
        # Risk-ON: close above 200d MA
        closes_bull = [100.0] * 250
        closes_bull[-1] = 110.0  # Above MA (100.04)
        spy_bull = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=250).strftime("%Y-%m-%d"), "close": closes_bull})
        assert check_market_regime(spy_df=spy_bull) is True

        # Risk-OFF: close below 200d MA
        closes_bear = [100.0] * 250
        closes_bear[-1] = 85.0   # Below MA (99.94)
        spy_bear = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=250).strftime("%Y-%m-%d"), "close": closes_bear})
        assert check_market_regime(spy_df=spy_bear) is False
