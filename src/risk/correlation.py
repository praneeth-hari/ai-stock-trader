"""
src/risk/correlation.py — Section 8b Correlation Filter & Matrix Manager.

Calculates 60-day rolling returns correlation between stocks, saves correlation
matrix records to the database, and enforces correlation veto / warning rules
for new position entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.settings import get_ticker_sector, settings
from src.db import repository

logger = logging.getLogger(__name__)


@dataclass
class CorrelationEvaluationResult:
    allowed: bool
    warning: bool
    veto_reason: Optional[str]
    max_correlation: float
    max_correlated_ticker: Optional[str]
    log_message: str
    details: str


def calculate_returns_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    lookback_days: int = 60,
) -> Optional[float]:
    """
    Calculates Pearson correlation between 60-day daily price returns of two tickers.

    Returns None if fewer than `lookback_days` overlapping daily return data points are available.
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return None

    # Ensure close column exists
    if "close" not in df_a.columns or "close" not in df_b.columns:
        return None

    # Sort by date and clean
    df_a_clean = df_a.copy()
    df_b_clean = df_b.copy()

    if "date" in df_a_clean.columns:
        df_a_clean = df_a_clean.sort_values("date")
    if "date" in df_b_clean.columns:
        df_b_clean = df_b_clean.sort_values("date")

    df_a_clean = df_a_clean.dropna(subset=["close"])
    df_b_clean = df_b_clean.dropna(subset=["close"])

    if len(df_a_clean) < 2 or len(df_b_clean) < 2:
        return None

    # Calculate daily returns
    df_a_clean["return"] = df_a_clean["close"].pct_change()
    df_b_clean["return"] = df_b_clean["close"].pct_change()

    # Merge on date if present, or index
    if "date" in df_a_clean.columns and "date" in df_b_clean.columns:
        merged = pd.merge(
            df_a_clean[["date", "return"]].dropna(subset=["return"]),
            df_b_clean[["date", "return"]].dropna(subset=["return"]),
            on="date",
            suffixes=("_a", "_b"),
        )
    else:
        merged = pd.concat([df_a_clean["return"], df_b_clean["return"]], axis=1, join="inner").dropna()
        merged.columns = ["return_a", "return_b"]

    if len(merged) < lookback_days:
        return None

    # Take the last `lookback_days` trading days
    tail_df = merged.tail(lookback_days)
    ret_a = tail_df["return_a"]
    ret_b = tail_df["return_b"]

    std_a = ret_a.std()
    std_b = ret_b.std()
    if std_a == 0 or std_b == 0 or np.isnan(std_a) or np.isnan(std_b):
        return 0.0

    corr = float(ret_a.corr(ret_b))
    if np.isnan(corr):
        return 0.0

    return round(corr, 4)


def evaluate_candidate_correlation(
    candidate_ticker: str,
    held_tickers: List[str],
    price_histories: Optional[Dict[str, pd.DataFrame]] = None,
) -> CorrelationEvaluationResult:
    """
    Evaluates a candidate buy ticker against currently held tickers using correlation rules.

    - Correlation < 0.70: ALLOW buy -> "CORRELATION_OK: low correlation with held stocks"
    - Correlation 0.70 - 0.85: WARN but allow -> "CORRELATION_WARNING: moderately correlated with {ticker} (r={value:.2f}) — proceeding"
    - Correlation > 0.85: BLOCK buy -> "CORRELATION_VETO: too correlated with {ticker} (r={value:.2f}) — skipping"
    - Different sector: block threshold relaxed to 0.90 (settings.correlation_different_sector_threshold).
    - Less than 60 days price history: skip check, log warning, allow buy.
    """
    cand_upper = candidate_ticker.strip().upper()
    held_clean = [h.strip().upper() for h in held_tickers if h.strip().upper() != cand_upper]

    if not held_clean:
        msg = "CORRELATION_OK: low correlation with held stocks"
        logger.info(msg)
        return CorrelationEvaluationResult(
            allowed=True,
            warning=False,
            veto_reason=None,
            max_correlation=0.0,
            max_correlated_ticker=None,
            log_message=msg,
            details=msg,
        )

    cand_df = None
    if price_histories and cand_upper in price_histories:
        cand_df = price_histories[cand_upper]
    else:
        try:
            cand_df = repository.get_market_data(cand_upper)
        except Exception:
            cand_df = None

    cand_sector = get_ticker_sector(cand_upper)

    max_corr = -1.0
    max_corr_ticker: Optional[str] = None
    veto_ticker: Optional[str] = None
    veto_corr: Optional[float] = None

    valid_comparisons = 0

    for held in held_clean:
        held_df = None
        if price_histories and held in price_histories:
            held_df = price_histories[held]
        else:
            try:
                held_df = repository.get_market_data(held)
            except Exception:
                held_df = None

        r = calculate_returns_correlation(
            cand_df, held_df, lookback_days=settings.correlation_lookback_days
        )

        if r is None:
            logger.warning(
                "CORRELATION_INSUFFICIENT_DATA: less than 60 days price history for %s — skipping check",
                cand_upper,
            )
            continue

        valid_comparisons += 1
        held_sector = get_ticker_sector(held)

        eff_block_thresh = (
            settings.correlation_different_sector_threshold
            if cand_sector != held_sector
            else settings.correlation_block_threshold
        )

        if r > max_corr:
            max_corr = r
            max_corr_ticker = held

        if r > eff_block_thresh:
            veto_ticker = held
            veto_corr = r
            # Highest correlation veto trumps earlier ones if found
            break

    if veto_ticker is not None and veto_corr is not None:
        msg = f"CORRELATION_VETO: too correlated with {veto_ticker} (r={veto_corr:.2f}) — skipping"
        logger.warning("CORRELATION_VETO: too correlated with %s (r=%.2f) — skipping", veto_ticker, veto_corr)
        return CorrelationEvaluationResult(
            allowed=False,
            warning=False,
            veto_reason="CORRELATION_VETO",
            max_correlation=veto_corr,
            max_correlated_ticker=veto_ticker,
            log_message=msg,
            details=msg,
        )

    if valid_comparisons > 0 and max_corr_ticker is not None and max_corr >= settings.correlation_warn_threshold:
        msg = f"CORRELATION_WARNING: moderately correlated with {max_corr_ticker} (r={max_corr:.2f}) — proceeding"
        logger.warning(
            "CORRELATION_WARNING: moderately correlated with %s (r=%.2f) — proceeding",
            max_corr_ticker, max_corr,
        )
        return CorrelationEvaluationResult(
            allowed=True,
            warning=True,
            veto_reason=None,
            max_correlation=max_corr,
            max_correlated_ticker=max_corr_ticker,
            log_message=msg,
            details=msg,
        )

    msg = "CORRELATION_OK: low correlation with held stocks"
    logger.info(msg)
    return CorrelationEvaluationResult(
        allowed=True,
        warning=False,
        veto_reason=None,
        max_correlation=max_corr if valid_comparisons > 0 else 0.0,
        max_correlated_ticker=max_corr_ticker,
        log_message=msg,
        details=msg,
    )


def compute_and_save_correlation_matrix(
    date_str: str,
    tickers: Optional[List[str]] = None,
    price_histories: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Computes 60-day pairwise correlation matrix for watchlist tickers and saves to DB.

    Returns square DataFrame with tickers on both axes.
    """
    if tickers is None:
        tickers = settings.ticker_list

    clean_tickers = sorted(list(set(t.strip().upper() for t in tickers if t.strip())))
    n = len(clean_tickers)
    corr_matrix = pd.DataFrame(1.0, index=clean_tickers, columns=clean_tickers)

    db_records: List[Dict[str, Any]] = []

    # Cache market data
    dfs: Dict[str, Optional[pd.DataFrame]] = {}
    for t in clean_tickers:
        if price_histories and t in price_histories:
            dfs[t] = price_histories[t]
        else:
            try:
                dfs[t] = repository.get_market_data(t)
            except Exception:
                dfs[t] = None

    for i in range(n):
        t_a = clean_tickers[i]
        for j in range(i, n):
            t_b = clean_tickers[j]
            if t_a == t_b:
                r = 1.0
            else:
                r_calc = calculate_returns_correlation(
                    dfs[t_a], dfs[t_b], lookback_days=settings.correlation_lookback_days
                )
                r = r_calc if r_calc is not None else 0.0

            corr_matrix.loc[t_a, t_b] = r
            corr_matrix.loc[t_b, t_a] = r

            db_records.append({
                "date": date_str,
                "ticker_a": t_a,
                "ticker_b": t_b,
                "correlation": r,
                "lookback_days": settings.correlation_lookback_days,
            })
            if t_a != t_b:
                db_records.append({
                    "date": date_str,
                    "ticker_a": t_b,
                    "ticker_b": t_a,
                    "correlation": r,
                    "lookback_days": settings.correlation_lookback_days,
                })

    try:
        repository.save_correlation_matrix(db_records)
    except Exception as exc:
        logger.warning("Could not save correlation matrix to DB: %s", exc)

    return corr_matrix
