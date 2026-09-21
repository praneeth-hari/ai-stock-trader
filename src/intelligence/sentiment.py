"""
src/intelligence/sentiment.py — News Sentiment Analysis via VADER (Section 5, Item 1).

Scoring rules:
  - Last 5 headlines scored per ticker using VADER compound polarity.
  - Sentiment > +0.3:  +3% score boost (+0.03)
  - Sentiment -0.3 to +0.3: Neutral (no change)
  - Sentiment < -0.3:  -5% score penalty (-0.05)
  - Sentiment < -0.6:  VETO BUY entirely -> "SENTIMENT_VETO: {ticker} — negative news detected"
  - API down/missing: Log "SENTIMENT_UNAVAILABLE", proceed with neutral 0.0 modifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import settings
from src.db import repository

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Individual stock sentiment assessment."""
    ticker: str
    date: str
    headline_count: int
    composite_score: float
    sentiment_label: str       # POSITIVE / NEUTRAL / NEGATIVE / VERY_NEGATIVE / UNAVAILABLE
    modifier: float            # +0.03, 0.0, -0.05
    is_veto: bool              # True if score < -0.6
    data_source: str = "YahooFinance"
    headlines: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "headline_count": self.headline_count,
            "composite_score": round(self.composite_score, 4),
            "sentiment_label": self.sentiment_label,
            "modifier": round(self.modifier, 4),
            "is_veto": self.is_veto,
            "data_source": self.data_source,
        }


@dataclass
class SentimentAnalysisResult:
    """Aggregated sentiment analysis across universe."""
    date: str
    scores: Dict[str, SentimentScore]
    total_evaluated: int
    vetoed_tickers: List[str]

    def get_modifier(self, ticker: str) -> float:
        score = self.scores.get(ticker.upper())
        return score.modifier if score else 0.0

    def is_vetoed(self, ticker: str) -> bool:
        score = self.scores.get(ticker.upper())
        return score.is_veto if score else False


_analyzer: Optional[SentimentIntensityAnalyzer] = None


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def fetch_headlines_for_ticker(ticker: str, limit: int = 5) -> List[str]:
    """
    Fetches the latest news headlines for a ticker using yfinance.
    Gracefully returns empty list on network or parsing error.
    """
    headlines: List[str] = []
    try:
        t = yf.Ticker(ticker)
        news_items = getattr(t, "news", None)
        if news_items and isinstance(news_items, list):
            for item in news_items:
                title = None
                if isinstance(item, dict):
                    title = item.get("title")
                    if not title and "content" in item and isinstance(item["content"], dict):
                        title = item["content"].get("title")
                if title and isinstance(title, str) and title.strip():
                    headlines.append(title.strip())
                if len(headlines) >= limit:
                    break
    except Exception as exc:
        logger.warning("Could not fetch news headlines for %s: %s", ticker, exc)
    return headlines


def score_headlines(headlines: List[str]) -> float:
    """
    Computes the composite compound VADER sentiment score across headlines.
    Returns float in range [-1.0, 1.0]. Returns 0.0 if no headlines.
    """
    if not headlines:
        return 0.0
    analyzer = get_sentiment_analyzer()
    compounds: List[float] = []
    for h in headlines:
        try:
            scores = analyzer.polarity_scores(h)
            compounds.append(scores.get("compound", 0.0))
        except Exception:
            continue
    if not compounds:
        return 0.0
    return sum(compounds) / len(compounds)


def evaluate_ticker_sentiment(
    ticker: str,
    date_str: str,
    headlines: Optional[List[str]] = None,
) -> SentimentScore:
    """
    Scores and classifies news sentiment for a single ticker.
    """
    ticker_clean = ticker.upper().strip()
    data_source = "YahooFinance"
    
    if headlines is None:
        headlines = fetch_headlines_for_ticker(ticker_clean, limit=5)

    if not headlines:
        logger.warning("SENTIMENT_UNAVAILABLE: No headlines retrieved for %s. Using neutral score.", ticker_clean)
        return SentimentScore(
            ticker=ticker_clean,
            date=date_str,
            headline_count=0,
            composite_score=0.0,
            sentiment_label="UNAVAILABLE",
            modifier=0.0,
            is_veto=False,
            data_source=data_source,
            headlines=[],
        )

    composite = score_headlines(headlines)
    
    # Classify against settings thresholds
    if composite > settings.sentiment_positive_threshold:
        label = "POSITIVE"
        modifier = settings.sentiment_boost_pct  # +0.03
        is_veto = False
    elif composite < settings.sentiment_veto_threshold:
        label = "VERY_NEGATIVE"
        modifier = -settings.sentiment_penalty_pct  # -0.05
        is_veto = True
        veto_msg = f"SENTIMENT_VETO: {ticker_clean} — negative news detected (Score: {composite:.2f})"
        logger.warning(veto_msg)
        repository.log_event("WARNING", "sentiment", veto_msg, {"ticker": ticker_clean, "score": composite})
    elif composite < settings.sentiment_negative_threshold:
        label = "NEGATIVE"
        modifier = -settings.sentiment_penalty_pct  # -0.05
        is_veto = False
    else:
        label = "NEUTRAL"
        modifier = 0.0
        is_veto = False

    return SentimentScore(
        ticker=ticker_clean,
        date=date_str,
        headline_count=len(headlines),
        composite_score=composite,
        sentiment_label=label,
        modifier=modifier,
        is_veto=is_veto,
        data_source=data_source,
        headlines=headlines,
    )


def analyze_universe_sentiment(
    tickers: List[str],
    date_str: str,
    injected_headlines: Optional[Dict[str, List[str]]] = None,
    persist: bool = True,
) -> SentimentAnalysisResult:
    """
    Analyzes news sentiment for a list of tickers, classifies modifiers,
    and optionally persists records to the database.
    """
    scores_map: Dict[str, SentimentScore] = {}
    vetoed: List[str] = []
    db_records: List[Dict[str, Any]] = []

    for t in tickers:
        h_list = injected_headlines.get(t) if injected_headlines else None
        res = evaluate_ticker_sentiment(t, date_str=date_str, headlines=h_list)
        scores_map[t.upper()] = res
        if res.is_veto:
            vetoed.append(t.upper())
        db_records.append(res.to_dict())

    if persist and db_records:
        try:
            repository.save_sentiment_scores(db_records)
            logger.info("Saved %d sentiment scores to database for date %s", len(db_records), date_str)
        except Exception as exc:
            logger.warning("Could not persist sentiment scores to database: %s", exc)

    return SentimentAnalysisResult(
        date=date_str,
        scores=scores_map,
        total_evaluated=len(tickers),
        vetoed_tickers=vetoed,
    )
