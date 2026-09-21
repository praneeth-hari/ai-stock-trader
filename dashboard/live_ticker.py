"""
dashboard/live_ticker.py — Real-time price updates for currently held portfolio stocks (Section 6 Item 3).

BEHAVIOR:
- During market hours (9:30 AM - 4:00 PM ET Monday-Friday): refreshes live quotes every 60s.
- Outside market hours: displays last known closing prices with 'CLOSED' badge.
- Only queries yfinance for CURRENTLY HELD stocks (open positions). Never queries inactive stocks.
- Gracefully handles network failures/delays by serving cached quotes with 'STALE' badge.
- Caches quotes in session state to prevent UI flicker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yfinance as yf

from src.pipeline.scheduler import is_nyse_holiday

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")


def is_us_market_open(now_dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Determines if the US stock market (NYSE/NASDAQ) is currently open for regular trading.

    Hours: 09:30 AM to 04:00 PM US Eastern Time, Monday through Friday (excluding holidays).
    Returns (is_open: bool, status_reason: str).
    """
    if now_dt is None:
        ny_dt = datetime.now(NY_TZ)
    elif now_dt.tzinfo is None:
        ny_dt = now_dt.replace(tzinfo=NY_TZ)
    else:
        ny_dt = now_dt.astimezone(NY_TZ)

    target_date = ny_dt.date()

    # Weekend check
    if target_date.weekday() == 5:
        return False, "Market Closed (Saturday)"
    if target_date.weekday() == 6:
        return False, "Market Closed (Sunday)"

    # Market holiday check
    if is_nyse_holiday(target_date):
        return False, "Market Closed (NYSE Holiday)"

    market_open = ny_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ny_dt.replace(hour=16, minute=0, second=0, microsecond=0)

    if ny_dt < market_open:
        return False, "Market Closed (Pre-Market)"
    if ny_dt >= market_close:
        return False, "Market Closed (After-Hours)"

    return True, "Market Open"


def fetch_single_ticker_live_price(
    ticker: str,
    fallback_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fetches the live quote and previous close for a single ticker via yfinance.

    Returns:
        {
            "ticker": str,
            "current_price": float,
            "prev_close": float,
            "change_dollar": float,
            "change_pct": float,
            "is_up": bool,
            "is_stale": bool,
            "timestamp": str,
        }
    """
    tkr_upper = ticker.upper()
    now_str = datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S ET")

    try:
        t = yf.Ticker(tkr_upper)
        fi = getattr(t, "fast_info", None)

        last_price = getattr(fi, "last_price", None) if fi else None
        prev_close = getattr(fi, "previous_close", None) if fi else None

        # If fast_info fields are missing, attempt history pull
        if last_price is None or prev_close is None:
            hist = t.history(period="5d")
            if not hist.empty and "Close" in hist.columns:
                last_price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_price

        if last_price is not None and last_price > 0:
            price_val = float(last_price)
            prev_val = float(prev_close) if prev_close is not None and prev_close > 0 else price_val
            chg_dollar = round(price_val - prev_val, 2)
            chg_pct = round((chg_dollar / prev_val) * 100.0, 2) if prev_val > 0 else 0.0

            return {
                "ticker": tkr_upper,
                "current_price": round(price_val, 2),
                "prev_close": round(prev_val, 2),
                "change_dollar": chg_dollar,
                "change_pct": chg_pct,
                "is_up": chg_dollar >= 0,
                "is_stale": False,
                "timestamp": now_str,
            }
    except Exception as exc:
        logger.warning("Live price fetch failed for %s: %s", tkr_upper, exc)

    # Fallback to provided price or last known state
    fallback = float(fallback_price) if fallback_price is not None and fallback_price > 0 else 100.0
    return {
        "ticker": tkr_upper,
        "current_price": round(fallback, 2),
        "prev_close": round(fallback, 2),
        "change_dollar": 0.0,
        "change_pct": 0.0,
        "is_up": True,
        "is_stale": True,
        "timestamp": now_str,
    }


def get_live_quotes_for_held_stocks(
    positions: List[Dict[str, Any]],
    session_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches live quotes strictly for stocks currently held in open positions.
    Does NOT query any tickers outside the current holdings list.

    Args:
        positions: List of position dicts from get_portfolio_summary()["positions"].
        session_cache: Optional dict of cached quotes from st.session_state.

    Returns:
        List of quote dictionaries with unrealized P&L calculated against entry price.
    """
    if not positions:
        return []

    cache = session_cache or {}
    quotes: List[Dict[str, Any]] = []

    for pos in positions:
        tkr = str(pos.get("ticker", "")).upper()
        if not tkr:
            continue

        shares = float(pos.get("shares", pos.get("quantity", 0.0)))
        entry_price = float(pos.get("entry_price", pos.get("price", pos.get("avg_cost", 0.0))))
        last_known = float(pos.get("current_price", entry_price))

        # Check cache if available as fallback
        cached_entry = cache.get(tkr, {})
        fallback = cached_entry.get("current_price", last_known)

        quote = fetch_single_ticker_live_price(ticker=tkr, fallback_price=fallback)

        curr_price = float(quote["current_price"])
        unrealized_pnl = round((curr_price - entry_price) * shares, 2)
        unrealized_pnl_pct = round(((curr_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 0.0

        quote["shares"] = round(shares, 4)
        quote["entry_price"] = round(entry_price, 2)
        quote["unrealized_pnl"] = unrealized_pnl
        quote["unrealized_pnl_pct"] = unrealized_pnl_pct

        quotes.append(quote)

    return quotes
