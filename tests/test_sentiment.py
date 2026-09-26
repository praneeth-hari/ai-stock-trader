"""
tests/test_sentiment.py — Unit tests for Section 5 Item 1: News Sentiment Analysis.

Covers:
  - VADER sentiment scoring of individual headlines (score_headlines)
  - Positive modifier (+0.03 when composite > +0.3) via injected scores
  - Negative modifier (-0.05 when composite < -0.3)
  - Sentiment VETO flag (composite < -0.6)
  - Neutral band (no modifier when -0.3 <= composite <= 0.3)
  - Offline / no-headlines fallback (UNAVAILABLE label, 0.0 modifier)
  - analyze_universe_sentiment integration (no persistence)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.intelligence.sentiment import (
    evaluate_ticker_sentiment,
    score_headlines,
    analyze_universe_sentiment,
    SentimentScore,
)


# ── Fixtures: headlines known to produce specific VADER scores ─────────────────
# VADER compound range: simple strongly positive words score ~0.5-0.9
# We use extreme adjectives to guarantee crossing thresholds.

VERY_POSITIVE_HEADLINES = [
    "outstanding fantastic incredible wonderful brilliant excellent",
    "amazing superb glorious perfect tremendous phenomenal",
    "outstanding brilliant magnificent extraordinary wonderful",
    "superb fantastic joyful awesome magnificent perfect",
    "excellent incredible amazing brilliant outstanding triumph",
]

VERY_NEGATIVE_HEADLINES = [
    "catastrophic terrible horrible failure awful devastating",
    "disastrous dreadful horrific crash collapse terrible",
    "horrible catastrophe failure disaster terrible awful",
    "devastating terrible nightmare collapse disaster failure",
    "awful horrific terrible disaster catastrophic failure",
]

NEGATIVE_HEADLINES = [
    "disappointing loss miss failure worse decline down",
    "bad loss miss disappointing poor decline fall",
    "disappointing bad miss worse fall drop",
]

NEUTRAL_HEADLINES = [
    "Apple announces product event next month",
    "Apple changes executive roles in reorganization",
    "Apple reports quarterly results next week",
]


# ── score_headlines tests ─────────────────────────────────────────────────────

def test_score_headlines_positive():
    """Extreme positive adjectives should score > 0.3."""
    score = score_headlines(VERY_POSITIVE_HEADLINES)
    assert score > 0.3, f"Expected strongly positive score > 0.3, got {score}"


def test_score_headlines_very_negative():
    """Extreme negative adjectives should score < -0.3."""
    score = score_headlines(VERY_NEGATIVE_HEADLINES)
    assert score < -0.3, f"Expected negative score < -0.3, got {score}"


def test_score_headlines_empty_returns_zero():
    assert score_headlines([]) == 0.0


def test_score_headlines_returns_float():
    score = score_headlines(["good"])
    assert isinstance(score, float)


# ── evaluate_ticker_sentiment: test via injected composite scores ──────────────
# Rather than relying on VADER scoring specific financial phrases,
# we inject a pre-computed composite score directly into evaluate_ticker_sentiment
# by patching score_headlines. This isolates the classification logic cleanly.

def _evaluate_with_fixed_score(ticker: str, score: float) -> SentimentScore:
    """Helper: evaluate_ticker_sentiment with a known composite score."""
    with patch("src.intelligence.sentiment.score_headlines", return_value=score):
        return evaluate_ticker_sentiment(
            ticker, "2024-01-15",
            headlines=["headline1", "headline2", "headline3"],  # non-empty list
        )


def test_sentiment_positive_modifier():
    """Composite > 0.3 → modifier = +0.03, label = POSITIVE, is_veto = False."""
    result = _evaluate_with_fixed_score("AAPL", 0.45)
    assert result.modifier == pytest.approx(0.03, abs=1e-9)
    assert result.sentiment_label == "POSITIVE"
    assert result.is_veto is False


def test_sentiment_negative_modifier():
    """Composite -0.5 (< -0.3 but > -0.6) → modifier = -0.05, label = NEGATIVE."""
    result = _evaluate_with_fixed_score("MSFT", -0.50)
    assert result.modifier == pytest.approx(-0.05, abs=1e-9)
    assert result.sentiment_label == "NEGATIVE"
    assert result.is_veto is False


def test_sentiment_very_negative_veto():
    """Composite < -0.6 → modifier = -0.05, label = VERY_NEGATIVE, is_veto = True."""
    result = _evaluate_with_fixed_score("TSLA", -0.75)
    assert result.modifier == pytest.approx(-0.05, abs=1e-9)
    assert result.sentiment_label == "VERY_NEGATIVE"
    assert result.is_veto is True


def test_sentiment_neutral_band_positive_edge():
    """Composite = 0.25 (within neutral band) → modifier = 0.0, NEUTRAL."""
    result = _evaluate_with_fixed_score("GOOGL", 0.25)
    assert result.modifier == 0.0
    assert result.sentiment_label == "NEUTRAL"
    assert result.is_veto is False


def test_sentiment_neutral_band_negative_edge():
    """Composite = -0.20 (within neutral band) → modifier = 0.0, NEUTRAL."""
    result = _evaluate_with_fixed_score("NVDA", -0.20)
    assert result.modifier == 0.0
    assert result.sentiment_label == "NEUTRAL"
    assert result.is_veto is False


def test_sentiment_exactly_at_positive_threshold():
    """Composite exactly at 0.3 is NOT positive (must be > 0.3 to boost)."""
    result = _evaluate_with_fixed_score("META", 0.30)
    assert result.modifier == 0.0  # 0.30 is the threshold; > 0.30 would be POSITIVE


def test_sentiment_unavailable_no_headlines():
    """Empty headline list → UNAVAILABLE label, 0.0 modifier, is_veto False."""
    result = evaluate_ticker_sentiment("META", "2024-01-15", headlines=[])
    assert result.sentiment_label == "UNAVAILABLE"
    assert result.modifier == 0.0
    assert result.composite_score == 0.0
    assert result.is_veto is False


def test_sentiment_to_dict():
    """SentimentScore.to_dict() returns expected keys."""
    with patch("src.intelligence.sentiment.score_headlines", return_value=0.45):
        result = evaluate_ticker_sentiment("NVDA", "2024-01-15", headlines=["great news"])
    d = result.to_dict()
    assert "ticker" in d
    assert "composite_score" in d
    assert "sentiment_label" in d
    assert "modifier" in d
    assert "is_veto" in d


# ── analyze_universe_sentiment tests ─────────────────────────────────────────

def test_analyze_universe_no_persist():
    """analyze_universe_sentiment processes multiple tickers without hitting DB."""
    with patch("src.intelligence.sentiment.score_headlines") as mock_score:
        # Different scores per call - positive, very_negative, neutral
        mock_score.side_effect = [0.45, -0.75, 0.10]
        result = analyze_universe_sentiment(
            tickers=["AAPL", "TSLA", "MSFT"],
            date_str="2024-01-15",
            injected_headlines={
                "AAPL": ["good news"],
                "TSLA": ["bad news"],
                "MSFT": ["neutral news"],
            },
            persist=False,
        )
    assert result.total_evaluated == 3
    assert "AAPL" in result.scores
    assert "TSLA" in result.scores
    assert "MSFT" in result.scores


def test_analyze_universe_veto_list():
    """TSLA with composite < -0.6 should appear in vetoed_tickers."""
    with patch("src.intelligence.sentiment.score_headlines") as mock_score:
        mock_score.side_effect = [0.45, -0.75]
        result = analyze_universe_sentiment(
            tickers=["AAPL", "TSLA"],
            date_str="2024-01-15",
            injected_headlines={
                "AAPL": ["great"],
                "TSLA": ["terrible"],
            },
            persist=False,
        )
    assert "TSLA" in result.vetoed_tickers
    assert "AAPL" not in result.vetoed_tickers


def test_analyze_universe_get_modifier():
    """get_modifier() returns correct modifier for a ticker."""
    with patch("src.intelligence.sentiment.score_headlines", return_value=0.45):
        result = analyze_universe_sentiment(
            tickers=["AAPL"],
            date_str="2024-01-15",
            injected_headlines={"AAPL": ["great"]},
            persist=False,
        )
    mod = result.get_modifier("AAPL")
    assert mod == pytest.approx(0.03, abs=1e-9)


def test_analyze_universe_is_vetoed():
    """is_vetoed() correctly identifies veto status."""
    with patch("src.intelligence.sentiment.score_headlines", return_value=-0.75):
        result = analyze_universe_sentiment(
            tickers=["TSLA"],
            date_str="2024-01-15",
            injected_headlines={"TSLA": ["terrible"]},
            persist=False,
        )
    assert result.is_vetoed("TSLA") is True
    assert result.is_vetoed("AAPL") is False


def test_analyze_universe_missing_ticker_returns_zero():
    """get_modifier() for ticker not in result returns 0.0."""
    with patch("src.intelligence.sentiment.score_headlines", return_value=0.10):
        result = analyze_universe_sentiment(
            tickers=["AAPL"],
            date_str="2024-01-15",
            injected_headlines={"AAPL": ["ok"]},
            persist=False,
        )
    assert result.get_modifier("UNKNOWN_TICKER") == 0.0


def test_sentiment_network_failure_graceful():
    """If yfinance raises exception during headline fetch, returns UNAVAILABLE gracefully."""
    with patch("src.intelligence.sentiment.yf.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("Network error")
        result = evaluate_ticker_sentiment("AAPL", "2024-01-15")
    assert result.sentiment_label == "UNAVAILABLE"
    assert result.modifier == 0.0
    assert result.is_veto is False


def test_fingpt_sentiment_bullish():
    from src.intelligence.sentiment import get_fingpt_sentiment
    result = get_fingpt_sentiment("Company beats earnings by 20%")
    assert isinstance(result, dict)
    assert "score" in result
    assert "label" in result
    assert "source" in result
    assert result["label"] in ["BULLISH", "BEARISH", "NEUTRAL"]


def test_fingpt_sentiment_fallback_no_api():
    from src.intelligence.sentiment import get_fingpt_sentiment
    import unittest.mock as mock
    with mock.patch("google.genai.Client") as m:
        m.side_effect = Exception("API unavailable")
        result = get_fingpt_sentiment("Market crashes today")
        assert result["source"] == "vader_fallback"
        assert result["label"] in ["BULLISH", "BEARISH", "NEUTRAL"]


def test_historical_replay_no_live_news_leakage():
    """
    Historical replay dates must NOT call live yfinance news APIs.
    If no point-in-time headlines exist, must return neutral UNAVAILABLE without hitting live network.
    If point-in-time DB cache exists, must consume only that historical record.
    """
    historical_date = "2023-05-15"

    # 1. No cached record -> strictly returns neutral without calling fetch_headlines_for_ticker
    with patch("src.intelligence.sentiment.repository.get_latest_sentiment_scores", return_value={}):
        with patch("src.intelligence.sentiment.fetch_headlines_for_ticker") as mock_fetch:
            res = evaluate_ticker_sentiment("AAPL", historical_date, headlines=None)
            mock_fetch.assert_not_called()

    assert res.sentiment_label == "UNAVAILABLE"
    assert res.modifier == 0.0
    assert res.composite_score == 0.0
    assert res.is_veto is False
    assert res.data_source == "HistoricalUnavailable"

    # 2. Point-in-time cached record -> uses historical record without calling live API
    cached_record = {
        "AAPL": {
            "headline_count": 3,
            "composite_score": 0.45,
            "sentiment_label": "POSITIVE",
            "modifier": 0.03,
            "is_veto": False,
        }
    }
    with patch("src.intelligence.sentiment.repository.get_latest_sentiment_scores", return_value=cached_record):
        with patch("src.intelligence.sentiment.fetch_headlines_for_ticker") as mock_fetch:
            res_cached = evaluate_ticker_sentiment("AAPL", historical_date, headlines=None)
            mock_fetch.assert_not_called()

    assert res_cached.sentiment_label == "POSITIVE"
    assert res_cached.modifier == 0.03
    assert res_cached.composite_score == 0.45
    assert res_cached.data_source == "DBCacheHistorical"
