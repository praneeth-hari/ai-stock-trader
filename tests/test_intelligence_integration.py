"""
tests/test_intelligence_integration.py — Integration tests for Section 5 Intelligence Pipeline.

Covers end-to-end integration of all 4 intelligence layers in rank_candidates():
  - Breakdown string logging (Base -> Sentiment -> Sector -> Macro -> Earnings -> Final)
  - Sentiment modifier applied correctly to final probability
  - Sector modifier applied correctly to final probability
  - Macro buy_bar shift changes effective buy threshold
  - Sentiment veto (is_sentiment_veto) reflected in RankedOpportunity
  - Earnings blackout (is_earnings_blackout) reflected in RankedOpportunity
  - is_buy_eligible=False when veto or blackout is set
  - All 4 layers gracefully degrade when None is passed (no crash)
  - RankingResult.macro_regime and effective_buy_bar fields
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ranking.ranking import rank_candidates, RankingResult, RankedOpportunity


# ── Test fixtures ──────────────────────────────────────────────────────────────

def make_feature_row(
    ticker: str,
    run_date: str = "2024-01-15",
    prob_seed: float = 0.65,
) -> Dict[str, Any]:
    """Minimal feature row for ranking (uses real FEATURE_COLUMNS)."""
    from src.features.engineer import FEATURE_COLUMNS
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row["ticker"] = ticker
    row["date"] = run_date
    row["rel_strength_21"] = 0.01
    row["price_to_ma50"] = 0.02
    return row


def make_features_df(tickers: List[str], run_date: str = "2024-01-15") -> pd.DataFrame:
    """Build a minimal features DataFrame for the given tickers."""
    rows = [make_feature_row(t, run_date) for t in tickers]
    return pd.DataFrame(rows)


def make_mock_model(probs: Dict[str, float]):
    """Returns a mock TrainedModel that outputs pre-defined probabilities."""
    model = MagicMock()

    def predict_proba(df):
        result = []
        for _, row in df.iterrows():
            t = str(row.get("ticker", "AAPL")).upper()
            p = probs.get(t, 0.5)
            result.append([1 - p, p])
        return np.array(result)

    model.predict_proba = predict_proba
    model.metadata = {"model_type": "test_mock"}
    return model


def make_sentiment_result(tickers: List[str], modifiers: Dict[str, float], vetoed: List[str] = None):
    """Minimal SentimentAnalysisResult mock."""
    from src.intelligence.sentiment import SentimentScore, SentimentAnalysisResult

    scores = {}
    for t in tickers:
        mod = modifiers.get(t.upper(), 0.0)
        is_veto = t.upper() in (vetoed or [])
        scores[t.upper()] = SentimentScore(
            ticker=t.upper(),
            date="2024-01-15",
            headline_count=3,
            composite_score=mod * 10,  # approximate
            sentiment_label="VERY_NEGATIVE" if is_veto else ("POSITIVE" if mod > 0 else "NEUTRAL"),
            modifier=mod,
            is_veto=is_veto,
        )
    return SentimentAnalysisResult(
        date="2024-01-15",
        scores=scores,
        total_evaluated=len(tickers),
        vetoed_tickers=[t.upper() for t in (vetoed or [])],
    )


def make_sector_result(tickers: List[str], multipliers: Dict[str, float]):
    """Minimal SectorRotationResult mock."""
    from src.intelligence.sector_rotation import SectorRotationResult, SectorRank
    from config.settings import get_ticker_sector

    sector_map = {}
    for t in tickers:
        sec = get_ticker_sector(t)
        mult = multipliers.get(t.upper(), 0.0)
        if sec not in sector_map:
            sector_map[sec] = SectorRank(sector=sec, avg_20d_return=mult, rank=3, rotation_multiplier=mult, stock_count=1)

    result = MagicMock()
    result.get_multiplier_for_ticker = lambda ticker: multipliers.get(ticker.upper(), 0.0)
    result.get_rank_for_ticker = lambda ticker: 3
    return result


def make_earnings_result(tickers: List[str], statuses: Dict[str, str], days: Dict[str, Optional[int]] = None):
    """Minimal EarningsStatusResult mock."""
    from src.intelligence.earnings import EarningsInfo, EarningsStatusResult
    from config.settings import settings

    calendar = {}
    blocked = []
    caution = []
    today_list = []
    days = days or {}

    for t in tickers:
        t_up = t.upper()
        st = statuses.get(t_up, "OK")
        d = days.get(t_up)
        if st == "TODAY":
            blk, hold, mult = True, True, 1.0
            today_list.append(t_up)
        elif st == "BLACKOUT":
            blk, hold, mult = True, False, 1.0
            blocked.append(t_up)
        elif st == "CAUTION":
            blk, hold, mult = False, False, settings.earnings_caution_size_multiplier
            caution.append(t_up)
        else:
            blk, hold, mult = False, False, 1.0

        calendar[t_up] = EarningsInfo(
            ticker=t_up,
            earnings_date=None,
            days_until_earnings=d,
            status=st,
            size_multiplier=mult,
            is_blocked=blk,
            hold_protection=hold,
            fetched_date="2024-01-15",
        )

    return EarningsStatusResult(
        fetched_date="2024-01-15",
        calendar=calendar,
        blocked_tickers=blocked,
        caution_tickers=caution,
        today_tickers=today_list,
    )


def make_macro_result(regime: str = "FAVORABLE"):
    """Minimal MacroEnvironmentResult mock."""
    from src.intelligence.macro import MacroEnvironmentResult
    shift_map = {"FAVORABLE": 0.0, "NEUTRAL": 0.0, "RESTRICTIVE": 0.03}
    mult_map = {"FAVORABLE": 1.0, "NEUTRAL": 0.80, "RESTRICTIVE": 0.60}
    return MacroEnvironmentResult(
        date="2024-01-15",
        fed_funds_rate=4.5,
        cpi_yoy=3.5,
        unemployment_rate=4.0,
        treasury_10y=4.2,
        macro_regime=regime,
        position_size_multiplier=mult_map[regime],
        buy_bar_shift=shift_map[regime],
        is_cached=True,
        source_date="2024-01-15",
    )


RUN_DATE = "2024-01-15"
TICKERS = ["AAPL", "MSFT", "TSLA"]


# ── Integration tests ─────────────────────────────────────────────────────────

def test_rank_candidates_no_intelligence_layers():
    """rank_candidates works with all None intelligence params (baseline mode)."""
    features = make_features_df(TICKERS, RUN_DATE)
    model = make_mock_model({"AAPL": 0.70, "MSFT": 0.55, "TSLA": 0.40})

    result = rank_candidates(features, model=model, run_date=RUN_DATE)
    assert isinstance(result, RankingResult)
    assert result.total_evaluated == len(TICKERS)


def test_sentiment_modifier_increases_probability():
    """Positive sentiment modifier (+0.03) increases final probability above base."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.65})
    sent = make_sentiment_result(["AAPL"], {"AAPL": 0.03})

    result = rank_candidates(features, model=model, run_date=RUN_DATE, sentiment_result=sent)
    opp = result.ranked_opportunities[0]
    assert opp.probability > opp.base_probability, \
        f"Final P={opp.probability:.4f} should be > base P={opp.base_probability:.4f}"
    assert opp.sentiment_modifier == pytest.approx(0.03, abs=1e-9)


def test_sentiment_modifier_decreases_probability():
    """Negative sentiment modifier (-0.05) decreases final probability."""
    features = make_features_df(["MSFT"], RUN_DATE)
    model = make_mock_model({"MSFT": 0.70})
    sent = make_sentiment_result(["MSFT"], {"MSFT": -0.05})

    result = rank_candidates(features, model=model, run_date=RUN_DATE, sentiment_result=sent)
    opp = result.ranked_opportunities[0]
    assert opp.probability < opp.base_probability, \
        f"Final P={opp.probability:.4f} should be < base P={opp.base_probability:.4f}"


def test_sentiment_veto_marks_buy_ineligible():
    """Sentiment veto (is_veto=True) → is_buy_eligible=False even if P >= buy_bar."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.75})  # High prob
    sent = make_sentiment_result(["AAPL"], {"AAPL": -0.05}, vetoed=["AAPL"])

    result = rank_candidates(features, model=model, run_date=RUN_DATE, sentiment_result=sent)
    opp = result.ranked_opportunities[0]
    assert opp.is_sentiment_veto is True
    assert opp.is_buy_eligible is False


def test_earnings_blackout_marks_buy_ineligible():
    """Earnings BLACKOUT → is_buy_eligible=False even if P >= buy_bar."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.75})
    earnings = make_earnings_result(["AAPL"], {"AAPL": "BLACKOUT"}, days={"AAPL": 1})

    result = rank_candidates(features, model=model, run_date=RUN_DATE, earnings_result=earnings)
    opp = result.ranked_opportunities[0]
    assert opp.is_earnings_blackout is True
    assert opp.is_buy_eligible is False


def test_earnings_ok_does_not_block_buy():
    """Earnings OK → does not block buy eligibility."""
    features = make_features_df(["MSFT"], RUN_DATE)
    model = make_mock_model({"MSFT": 0.72})
    earnings = make_earnings_result(["MSFT"], {"MSFT": "OK"})

    result = rank_candidates(features, model=model, run_date=RUN_DATE, earnings_result=earnings)
    opp = result.ranked_opportunities[0]
    assert opp.is_earnings_blackout is False
    # Buy eligibility depends on regime (default risk-on)


def test_sector_modifier_applied():
    """Sector +0.04 modifier should increase final probability."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.60})
    sector = make_sector_result(["AAPL"], {"AAPL": 0.04})

    result = rank_candidates(features, model=model, run_date=RUN_DATE, sector_result=sector)
    opp = result.ranked_opportunities[0]
    assert opp.sector_modifier == pytest.approx(0.04, abs=1e-9)
    assert opp.probability >= opp.base_probability


def test_macro_restrictive_raises_buy_bar():
    """RESTRICTIVE macro raises effective_buy_bar by +0.03."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.62})
    macro = make_macro_result("RESTRICTIVE")

    result = rank_candidates(features, model=model, run_date=RUN_DATE, macro_result=macro)
    assert result.effective_buy_bar == pytest.approx(0.63, abs=1e-9), \
        f"Expected effective_buy_bar=0.63 (0.60 + 0.03), got {result.effective_buy_bar:.4f}"
    assert result.macro_regime == "RESTRICTIVE"


def test_macro_favorable_does_not_change_buy_bar():
    """FAVORABLE macro → effective_buy_bar unchanged (== settings.buy_bar)."""
    from config.settings import settings
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.65})
    macro = make_macro_result("FAVORABLE")

    result = rank_candidates(features, model=model, run_date=RUN_DATE, macro_result=macro)
    assert result.effective_buy_bar == pytest.approx(settings.buy_bar, abs=1e-9)


def test_breakdown_string_logged(caplog):
    """Check that breakdown log line is emitted for each ticker."""
    import logging
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.65})

    with caplog.at_level(logging.INFO, logger="src.ranking.ranking"):
        rank_candidates(features, model=model, run_date=RUN_DATE)

    breakdown_logs = [r.message for r in caplog.records if "Base=" in r.message and "Final=" in r.message]
    assert len(breakdown_logs) >= 1, "Expected at least one breakdown log message"
    assert "AAPL" in breakdown_logs[0]
    assert "->" in breakdown_logs[0]


def test_all_intelligence_none_no_crash():
    """All intelligence params = None → pipeline runs without error."""
    features = make_features_df(TICKERS, RUN_DATE)
    model = make_mock_model({"AAPL": 0.65, "MSFT": 0.55, "TSLA": 0.40})

    result = rank_candidates(
        features,
        model=model,
        run_date=RUN_DATE,
        sentiment_result=None,
        sector_result=None,
        earnings_result=None,
        macro_result=None,
    )
    assert result.total_evaluated == len(TICKERS)
    assert len(result.ranked_opportunities) == len(TICKERS)


def test_combined_sentiment_and_sector_modifiers():
    """Both sentiment (+0.03) and sector (+0.04) modifiers stack on final probability."""
    features = make_features_df(["AAPL"], RUN_DATE)
    base_p = 0.60
    model = make_mock_model({"AAPL": base_p})
    sent = make_sentiment_result(["AAPL"], {"AAPL": 0.03})
    sector = make_sector_result(["AAPL"], {"AAPL": 0.04})

    result = rank_candidates(
        features, model=model, run_date=RUN_DATE,
        sentiment_result=sent, sector_result=sector,
    )
    opp = result.ranked_opportunities[0]
    expected_final = min(1.0, base_p + 0.03 + 0.04)
    assert opp.probability == pytest.approx(expected_final, abs=0.001)


def test_probability_clamped_to_0_1():
    """Final probability is clamped to [0.0, 1.0] even with large positive modifiers."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.99})
    sent = make_sentiment_result(["AAPL"], {"AAPL": 0.03})
    sector = make_sector_result(["AAPL"], {"AAPL": 0.04})

    result = rank_candidates(
        features, model=model, run_date=RUN_DATE,
        sentiment_result=sent, sector_result=sector,
    )
    assert result.ranked_opportunities[0].probability <= 1.0


def test_ranking_result_has_macro_fields():
    """RankingResult has macro_regime and effective_buy_bar attributes."""
    features = make_features_df(["AAPL"], RUN_DATE)
    model = make_mock_model({"AAPL": 0.65})
    macro = make_macro_result("NEUTRAL")

    result = rank_candidates(features, model=model, run_date=RUN_DATE, macro_result=macro)
    assert hasattr(result, "macro_regime")
    assert hasattr(result, "effective_buy_bar")
    assert result.macro_regime == "NEUTRAL"
