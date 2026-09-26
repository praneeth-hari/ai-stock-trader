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

from datetime import date, datetime, timezone
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
    FinGPT upgrade: uses finance-specific sentiment for low confidence scores.
    """
    if not headlines:
        return 0.0
    analyzer = get_sentiment_analyzer()
    compounds: List[float] = []
    for h in headlines:
        try:
            vader_score = analyzer.polarity_scores(h)
            compound = vader_score.get("compound", 0.0)
            # FinGPT upgrade: use finance-specific sentiment for low confidence scores
            if abs(compound) < 0.2:
                fingpt_result = get_fingpt_sentiment(h)
                final_score = fingpt_result["score"]
            else:
                final_score = compound
            compounds.append(final_score)
        except Exception:
            continue
    if not compounds:
        return 0.0
    return sum(compounds) / len(compounds)


def get_vader_sentiment(text: str) -> dict:
    """Returns VADER sentiment scores for given text."""
    analyzer = get_sentiment_analyzer()
    return analyzer.polarity_scores(text)


def get_fingpt_sentiment(text: str) -> dict:
    """
    FinGPT-inspired sentiment analysis using a finance-specific prompt.
    Uses Gemini API (already configured) with a FinGPT-style finance prompt.
    Falls back to VADER if Gemini is unavailable.
    Returns dict with keys: score (float -1 to 1), label (str), source (str)
    """
    import re

    FINGPT_PROMPT = """You are a financial sentiment analyzer trained like FinGPT.
Analyze the following financial text and respond with ONLY one word:
BULLISH, BEARISH, or NEUTRAL.

Text: {text}

Response:"""

    try:
        from google import genai
        from config.settings import settings
        if not settings.gemini_api_key:
            raise ValueError("No Gemini API key")
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=getattr(settings, "gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash",
            contents=FINGPT_PROMPT.format(text=text[:500]),
        )
        label = response.text.strip().upper()
        label = re.sub(r"[^A-Z]", "", label)
        if label == "BULLISH":
            return {"score": 0.6, "label": "BULLISH", "source": "fingpt"}
        elif label == "BEARISH":
            return {"score": -0.6, "label": "BEARISH", "source": "fingpt"}
        else:
            return {"score": 0.0, "label": "NEUTRAL", "source": "fingpt"}
    except Exception:
        vader = get_vader_sentiment(text)
        compound = vader.get("compound", 0.0)
        return {
            "score": compound,
            "label": "BULLISH" if compound > 0.05
                     else "BEARISH" if compound < -0.05
                     else "NEUTRAL",
            "source": "vader_fallback",
        }


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
    
    # Check if target date is in the past (historical replay / backtest)
    try:
        target_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        target_date = date.today()
    today_date = min(date.today(), datetime.now(timezone.utc).date())
    is_historical = target_date < today_date

    if headlines is None:
        if is_historical:
            # Historical replay: Do NOT call live API for past dates (avoids look-ahead leakage).
            # 1. Attempt to load historical point-in-time scores from database
            cached_scores = repository.get_latest_sentiment_scores(as_of_date=date_str[:10])
            if ticker_clean in cached_scores:
                rec = cached_scores[ticker_clean]
                return SentimentScore(
                    ticker=ticker_clean,
                    date=date_str,
                    headline_count=int(rec.get("headline_count", 0)),
                    composite_score=float(rec.get("composite_score", 0.0)),
                    sentiment_label=str(rec.get("sentiment_label", "NEUTRAL")),
                    modifier=float(rec.get("modifier", 0.0)),
                    is_veto=bool(rec.get("is_veto", False)),
                    data_source="DBCacheHistorical",
                    headlines=[],
                )
            # 2. If point-in-time historical headlines unavailable: fail-safe neutral
            logger.info("SENTIMENT_HISTORICAL_UNAVAILABLE: No historical news for %s on %s. Using neutral score.", ticker_clean, date_str)
            return SentimentScore(
                ticker=ticker_clean,
                date=date_str,
                headline_count=0,
                composite_score=0.0,
                sentiment_label="UNAVAILABLE",
                modifier=0.0,
                is_veto=False,
                data_source="HistoricalUnavailable",
                headlines=[],
            )
        else:
            # Live trading mode: fetch live headlines from Yahoo Finance
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
        if res.data_source != "HistoricalUnavailable":
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
