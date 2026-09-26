"""
src/intelligence/earnings.py — Earnings Calendar Awareness (Section 5, Item 2).

Earnings volatility rules:
  - 1-2 days until earnings:  BLOCK new buys (EARNINGS_BLACKOUT)
  - 3-5 days until earnings:  REDUCE position size by 50% (EARNINGS_CAUTION)
  - 0 days (earnings today):  HOLD existing position (override exit panic)
  - API down or unavailable: Log warning, proceed normally (graceful degradation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
import yfinance as yf

from config.settings import settings
from src.db import repository

logger = logging.getLogger(__name__)


@dataclass
class EarningsInfo:
    """Earnings status for a single ticker."""
    ticker: str
    earnings_date: Optional[str]        # 'YYYY-MM-DD'
    days_until_earnings: Optional[int]
    status: str                         # OK / BLACKOUT / CAUTION / TODAY / UNKNOWN
    size_multiplier: float              # 1.0 or 0.5
    is_blocked: bool                    # True if 1 <= days <= 2
    hold_protection: bool               # True if days == 0 (today)
    fetched_date: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "earnings_date": self.earnings_date or "N/A",
            "days_until_earnings": self.days_until_earnings,
            "status": self.status,
            "size_multiplier": self.size_multiplier,
            "is_blocked": self.is_blocked,
            "hold_protection": self.hold_protection,
            "fetched_date": self.fetched_date,
        }

    @property
    def badge_text(self) -> str:
        if self.days_until_earnings is not None:
            if self.days_until_earnings == 0:
                return "⚠️ Earnings Today"
            elif 1 <= self.days_until_earnings <= 5:
                return f"⚠️ Earnings in {self.days_until_earnings}d"
        return ""


@dataclass
class EarningsStatusResult:
    """Aggregated earnings calendar state for universe."""
    fetched_date: str
    calendar: Dict[str, EarningsInfo]
    blocked_tickers: List[str]
    caution_tickers: List[str]
    today_tickers: List[str]

    def get_info(self, ticker: str) -> Optional[EarningsInfo]:
        return self.calendar.get(ticker.upper())

    def is_blocked(self, ticker: str) -> bool:
        info = self.get_info(ticker)
        return info.is_blocked if info else False

    def get_size_multiplier(self, ticker: str) -> float:
        info = self.get_info(ticker)
        return info.size_multiplier if info else 1.0

    def has_hold_protection(self, ticker: str) -> bool:
        info = self.get_info(ticker)
        return info.hold_protection if info else False


def parse_earnings_date_from_calendar(cal_data: Any) -> Optional[date]:
    """
    Extracts the earliest future/upcoming earnings date from yfinance calendar payload.
    """
    if not cal_data:
        return None

    candidate_dates: List[date] = []
    
    # If dict with 'Earnings Date'
    if isinstance(cal_data, dict):
        ed_val = cal_data.get("Earnings Date")
        if isinstance(ed_val, list):
            for d in ed_val:
                if isinstance(d, date):
                    candidate_dates.append(d)
                elif isinstance(d, datetime):
                    candidate_dates.append(d.date())
                elif isinstance(d, str):
                    try:
                        candidate_dates.append(datetime.strptime(d[:10], "%Y-%m-%d").date())
                    except Exception:
                        pass
        elif isinstance(ed_val, (date, datetime)):
            candidate_dates.append(ed_val.date() if isinstance(ed_val, datetime) else ed_val)

    # If pandas DataFrame
    elif hasattr(cal_data, "loc") or hasattr(cal_data, "columns"):
        try:
            if "Earnings Date" in cal_data.index:
                val = cal_data.loc["Earnings Date"]
                if hasattr(val, "values"):
                    for item in val.values:
                        if isinstance(item, (date, datetime)):
                            candidate_dates.append(item.date() if isinstance(item, datetime) else item)
        except Exception:
            pass

    if candidate_dates:
        return min(candidate_dates)
    return None


def fetch_earnings_for_ticker(
    ticker: str,
    as_of_date: date,
) -> EarningsInfo:
    """
    Queries yfinance for the earnings date and classifies status relative to as_of_date.
    """
    ticker_clean = ticker.upper().strip()
    fetched_str = as_of_date.strftime("%Y-%m-%d")

    today_date = min(date.today(), datetime.now(timezone.utc).date())
    is_historical = as_of_date < today_date

    earnings_dt: Optional[date] = None

    if is_historical:
        # Historical replay: Do NOT call live Yahoo Finance API (returns today's future calendar).
        # Check if point-in-time earnings calendar is in DB cache:
        cached_cal = repository.get_latest_earnings_calendar(as_of_date=fetched_str)
        if ticker_clean in cached_cal:
            ed_str_val = cached_cal[ticker_clean].get("earnings_date")
            if ed_str_val and ed_str_val != "N/A":
                try:
                    earnings_dt = datetime.strptime(ed_str_val[:10], "%Y-%m-%d").date()
                except Exception:
                    earnings_dt = None

        if earnings_dt is None:
            # Point-in-time earnings unavailable: use fail-safe neutral
            logger.info("EARNINGS_HISTORICAL_UNAVAILABLE: No point-in-time earnings for %s on %s. Using neutral UNKNOWN.", ticker_clean, fetched_str)
            return EarningsInfo(
                ticker=ticker_clean,
                earnings_date=None,
                days_until_earnings=None,
                status="UNKNOWN",
                size_multiplier=1.0,
                is_blocked=False,
                hold_protection=False,
                fetched_date=fetched_str,
            )
    else:
        # Live trading mode: query live yfinance calendar
        try:
            t = yf.Ticker(ticker_clean)
            cal = getattr(t, "calendar", None)
            earnings_dt = parse_earnings_date_from_calendar(cal)
        except Exception as exc:
            logger.warning("Could not fetch earnings calendar for %s: %s", ticker_clean, exc)

    if earnings_dt is None:
        return EarningsInfo(
            ticker=ticker_clean,
            earnings_date=None,
            days_until_earnings=None,
            status="UNKNOWN",
            size_multiplier=1.0,
            is_blocked=False,
            hold_protection=False,
            fetched_date=fetched_str,
        )

    days_until = (earnings_dt - as_of_date).days
    ed_str = earnings_dt.strftime("%Y-%m-%d")

    if days_until < 0:
        return EarningsInfo(
            ticker=ticker_clean,
            earnings_date=ed_str,
            days_until_earnings=days_until,
            status="OK",
            size_multiplier=1.0,
            is_blocked=False,
            hold_protection=False,
            fetched_date=fetched_str,
        )

    # Evaluate rules
    if days_until == 0:
        status = "TODAY"
        size_mult = 1.0
        is_blocked = True  # don't buy into earnings announcement day
        hold_protect = True
        logger.info("EARNINGS_TODAY: %s reports today. Holding position if open.", ticker_clean)
    elif 1 <= days_until <= settings.earnings_blackout_days:
        status = "BLACKOUT"
        size_mult = 1.0
        is_blocked = True
        hold_protect = False
        msg = f"EARNINGS_BLACKOUT: {ticker_clean} reports in {days_until} days ({ed_str})"
        logger.warning(msg)
        repository.log_event("WARNING", "earnings", msg, {"ticker": ticker_clean, "days": days_until, "date": ed_str})
    elif settings.earnings_caution_min_days <= days_until <= settings.earnings_caution_max_days:
        status = "CAUTION"
        size_mult = settings.earnings_caution_size_multiplier  # 0.50
        is_blocked = False
        hold_protect = False
        msg = f"EARNINGS_CAUTION: {ticker_clean} reports in {days_until} days ({ed_str}) — position size reduced by 50%"
        logger.info(msg)
        repository.log_event("INFO", "earnings", msg, {"ticker": ticker_clean, "days": days_until, "date": ed_str})
    else:
        status = "OK"
        size_mult = 1.0
        is_blocked = False
        hold_protect = False

    return EarningsInfo(
        ticker=ticker_clean,
        earnings_date=ed_str,
        days_until_earnings=days_until,
        status=status,
        size_multiplier=size_mult,
        is_blocked=is_blocked,
        hold_protection=hold_protect,
        fetched_date=fetched_str,
    )


def get_universe_earnings_calendar(
    tickers: List[str],
    as_of_date_str: str,
    injected_dates: Optional[Dict[str, str]] = None,
    persist: bool = True,
) -> EarningsStatusResult:
    """
    Gathers earnings calendar for the entire watchlist and optionally persists to DB.
    """
    try:
        as_of_dt = datetime.strptime(as_of_date_str[:10], "%Y-%m-%d").date()
    except Exception:
        as_of_dt = date.today()

    today_date = min(date.today(), datetime.now(timezone.utc).date())
    is_historical = as_of_dt < today_date

    calendar_map: Dict[str, EarningsInfo] = {}
    blocked: List[str] = []
    caution: List[str] = []
    today_list: List[str] = []
    db_records: List[Dict[str, Any]] = []

    for t in tickers:
        t_clean = t.upper().strip()
        if injected_dates and t_clean in injected_dates:
            ed_str = injected_dates[t_clean]
            if ed_str:
                ed_dt = datetime.strptime(ed_str[:10], "%Y-%m-%d").date()
                days = (ed_dt - as_of_dt).days
                if days == 0:
                    st, mult, blk, hold = "TODAY", 1.0, True, True
                elif 1 <= days <= settings.earnings_blackout_days:
                    st, mult, blk, hold = "BLACKOUT", 1.0, True, False
                elif settings.earnings_caution_min_days <= days <= settings.earnings_caution_max_days:
                    st, mult, blk, hold = "CAUTION", settings.earnings_caution_size_multiplier, False, False
                else:
                    st, mult, blk, hold = "OK", 1.0, False, False
                info = EarningsInfo(
                    ticker=t_clean,
                    earnings_date=ed_str,
                    days_until_earnings=days,
                    status=st,
                    size_multiplier=mult,
                    is_blocked=blk,
                    hold_protection=hold,
                    fetched_date=as_of_date_str[:10],
                )
            else:
                info = EarningsInfo(
                    ticker=t_clean,
                    earnings_date=None,
                    days_until_earnings=None,
                    status="UNKNOWN",
                    size_multiplier=1.0,
                    is_blocked=False,
                    hold_protection=False,
                    fetched_date=as_of_date_str[:10],
                )
        else:
            info = fetch_earnings_for_ticker(t_clean, as_of_date=as_of_dt)

        calendar_map[t_clean] = info
        if info.is_blocked:
            blocked.append(t_clean)
        if info.status == "CAUTION":
            caution.append(t_clean)
        if info.status == "TODAY":
            today_list.append(t_clean)

        if info.status != "UNKNOWN" or not is_historical:
            db_records.append({
                "ticker": info.ticker,
                "earnings_date": info.earnings_date or "N/A",
                "days_until_earnings": info.days_until_earnings,
                "fetched_date": info.fetched_date,
            })

    if persist and db_records:
        try:
            repository.save_earnings_calendar(db_records)
            logger.info("Saved %d earnings calendar records for %s", len(db_records), as_of_date_str)
        except Exception as exc:
            logger.warning("Could not persist earnings calendar: %s", exc)

    return EarningsStatusResult(
        fetched_date=as_of_date_str[:10],
        calendar=calendar_map,
        blocked_tickers=blocked,
        caution_tickers=caution,
        today_tickers=today_list,
    )
