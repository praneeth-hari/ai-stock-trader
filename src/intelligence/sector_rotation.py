"""
src/intelligence/sector_rotation.py — Sector Rotation Intelligence (Section 5, Item 3).

Sector Momentum Rules:
  - 20-day trailing return computed per stock in universe.
  - Sector return = average 20-day return of constituent stocks in that sector.
  - All 8 sectors ranked from 1 (best) to 8 (worst).
  - Rank 1-2:  +4% boost to stock model scores (+0.04)
  - Rank 3-5:  Neutral (no change, 0.00)
  - Rank 6-8:  -4% penalty to stock model scores (-0.04)
  - Persisted to 'sector_rankings' and audit logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from config.settings import TICKER_SECTOR_MAP, get_ticker_sector, settings
from src.db import repository

logger = logging.getLogger(__name__)

ALL_SECTORS: List[str] = [
    "Technology",
    "Communications",
    "Consumer Cyclical",
    "Financials",
    "Healthcare",
    "Industrials",
    "Consumer Staples",
    "Energy",
]


@dataclass
class SectorRank:
    """Performance and modifier rank for an individual sector."""
    sector: str
    avg_20d_return: float
    rank: int                     # 1 to 8
    rotation_multiplier: float    # +0.04, 0.0, -0.04
    stock_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector,
            "avg_20d_return": round(self.avg_20d_return, 4),
            "rank": self.rank,
            "rotation_multiplier": round(self.rotation_multiplier, 4),
            "stock_count": self.stock_count,
        }


@dataclass
class SectorRotationResult:
    """Aggregated sector rotation ranking state."""
    date: str
    rankings: List[SectorRank]
    sector_map: Dict[str, SectorRank]

    def get_multiplier_for_ticker(self, ticker: str) -> float:
        sector = get_ticker_sector(ticker)
        s_rank = self.sector_map.get(sector)
        return s_rank.rotation_multiplier if s_rank else 0.0

    def get_rank_for_ticker(self, ticker: str) -> Optional[int]:
        sector = get_ticker_sector(ticker)
        s_rank = self.sector_map.get(sector)
        return s_rank.rank if s_rank else None


def calculate_stock_20d_returns(
    universe_dfs: Dict[str, pd.DataFrame],
    as_of_date: str,
    lookback: int = 20,
) -> Dict[str, float]:
    """
    Computes the 20-day return for each ticker using available daily close history.
    """
    stock_returns: Dict[str, float] = {}
    for ticker, df in universe_dfs.items():
        if df.empty or "close" not in df.columns or "date" not in df.columns:
            continue
        df_sub = df[df["date"] <= as_of_date].sort_values("date")
        if len(df_sub) >= lookback:
            p_now = float(df_sub.iloc[-1]["close"])
            p_past = float(df_sub.iloc[-lookback]["close"])
            if p_past > 0:
                ret = (p_now - p_past) / p_past
                stock_returns[ticker.upper()] = ret
        elif len(df_sub) >= 2:
            p_now = float(df_sub.iloc[-1]["close"])
            p_past = float(df_sub.iloc[0]["close"])
            if p_past > 0:
                ret = (p_now - p_past) / p_past
                stock_returns[ticker.upper()] = ret
    return stock_returns


def calculate_sector_rotation(
    universe_dfs: Dict[str, pd.DataFrame],
    as_of_date: str,
    lookback: int = 20,
    persist: bool = True,
) -> SectorRotationResult:
    """
    Calculates sector performance, orders 1 through 8, assigns rotation modifiers,
    persists results to the database and event audit trail.
    """
    stock_returns = calculate_stock_20d_returns(universe_dfs, as_of_date, lookback=lookback)

    # Group returns by sector
    sector_returns: Dict[str, List[float]] = {s: [] for s in ALL_SECTORS}
    for ticker, ret in stock_returns.items():
        sec = get_ticker_sector(ticker)
        if sec in sector_returns:
            sector_returns[sec].append(ret)

    # Compute average return per sector
    sector_avg: List[Dict[str, Any]] = []
    for sec in ALL_SECTORS:
        rets = sector_returns[sec]
        avg_ret = sum(rets) / len(rets) if rets else 0.0
        sector_avg.append({
            "sector": sec,
            "avg_return": avg_ret,
            "count": len(rets),
        })

    # Sort descending by 20d return
    sector_avg_sorted = sorted(sector_avg, key=lambda x: x["avg_return"], reverse=True)

    ranks_list: List[SectorRank] = []
    ranks_dict: Dict[str, SectorRank] = {}
    db_records: List[Dict[str, Any]] = []

    for idx, s_data in enumerate(sector_avg_sorted):
        rank = idx + 1
        sec_name = s_data["sector"]
        ret = s_data["avg_return"]
        cnt = s_data["count"]

        # Assign multiplier
        if rank <= 2:
            multiplier = settings.sector_boost_pct      # +0.04
        elif rank >= 6:
            multiplier = -settings.sector_penalty_pct   # -0.04
        else:
            multiplier = 0.0                            # Neutral

        s_rank = SectorRank(
            sector=sec_name,
            avg_20d_return=ret,
            rank=rank,
            rotation_multiplier=multiplier,
            stock_count=cnt,
        )
        ranks_list.append(s_rank)
        ranks_dict[sec_name] = s_rank

        db_records.append({
            "date": as_of_date[:10],
            "sector": sec_name,
            "avg_20d_return": ret,
            "rank": rank,
            "rotation_multiplier": multiplier,
        })

    if persist and db_records:
        try:
            repository.save_sector_rankings(db_records)
            logger.info("Saved sector rankings for %s (%d sectors)", as_of_date, len(db_records))
            
            top_sectors = [r.sector for r in ranks_list[:2]]
            bottom_sectors = [r.sector for r in ranks_list[-3:]]
            repository.log_event(
                "INFO", "sector_rotation",
                f"Sector rotation ranked for {as_of_date}. Top: {top_sectors} (+4%), Bottom: {bottom_sectors} (-4%)",
                {"top": top_sectors, "bottom": bottom_sectors},
            )
        except Exception as exc:
            logger.warning("Could not persist sector rankings: %s", exc)

    return SectorRotationResult(
        date=as_of_date[:10],
        rankings=ranks_list,
        sector_map=ranks_dict,
    )
