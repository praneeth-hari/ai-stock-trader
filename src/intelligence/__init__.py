"""
src/intelligence — Advanced Intelligence Upgrades (Section 5).

Provides four market-awareness layers:
  1. News Sentiment Analysis (VADER scoring & vetoes)
  2. Earnings Calendar Awareness (blackout windows & hold protection)
  3. Sector Rotation Intelligence (20-day momentum ranking & modifiers)
  4. Macro Economic Indicators (FRED regime scoring & sizing shifts)
"""

from src.intelligence.sentiment import analyze_universe_sentiment, SentimentAnalysisResult
from src.intelligence.earnings import get_universe_earnings_calendar, EarningsStatusResult
from src.intelligence.sector_rotation import calculate_sector_rotation, SectorRotationResult
from src.intelligence.macro import get_macro_environment, MacroEnvironmentResult

__all__ = [
    "analyze_universe_sentiment",
    "SentimentAnalysisResult",
    "get_universe_earnings_calendar",
    "EarningsStatusResult",
    "calculate_sector_rotation",
    "SectorRotationResult",
    "get_macro_environment",
    "MacroEnvironmentResult",
]
