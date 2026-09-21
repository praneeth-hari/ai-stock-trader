"""
src/ml/dataset.py — Phase 4 ML Dataset Construction (the label & training set).

PURPOSE
-------
Construct a clean, chronologically-ordered ML training dataset by combining:
  1. Phase 3 Feature vectors (computed strictly using data up through date T).
  2. Forward-looking binary labels (computed per ticker using future trading closes).

LABEL SPECIFICATION (§1.3 & PROJECT_PLAN.md Phase 4)
----------------------------------------------------
  - Label: 1 if the stock rises > +win_threshold over the next prediction_horizon
    trading days, else 0.
  - Default horizon: settings.prediction_horizon (5 trading days).
  - Default threshold: settings.win_threshold (0.01 = +1.0%).
  - Formula:
      forward_return = (close[T + horizon] / close[T]) - 1.0
      label = 1 if forward_return > win_threshold else 0
  - Strictly greater than (>): exactly +1.0% return is NOT a win (label = 0).
  - Strictly per-ticker: close.shift(-horizon) is applied exclusively within each
    ticker's own date-sorted series via groupby('ticker'). Cross-ticker data
    concatenation NEVER leaks another ticker's close into a ticker's future window.

SAFETY & LEAKAGE BOUNDARIES
---------------------------
  1. Features are computed completely blind to labels.
  2. Labels intentionally use future data (T + horizon), but are ONLY stored as
     the target column ('label'), NEVER as an input feature.
  3. Tail truncation: The last `prediction_horizon` trading rows of each ticker
     have no known future close. Their labels are NaN, and these rows are strictly
     dropped from the training dataset. They are NEVER filled with 0, 1, or mean.
  4. Chronological ordering: The output dataset is sorted by (date ASC, ticker ASC).
     Zero random shuffling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config.settings import settings
from src.features.engineer import (
    FEATURE_COLUMNS,
    compute_features,
    drop_warmup_rows,
)

logger = logging.getLogger(__name__)

LABEL_COLUMN: str = "label"
FORWARD_RETURN_COLUMN: str = "forward_return"
FUTURE_CLOSE_COLUMN: str = "future_close"


def compute_labels(
    df: pd.DataFrame,
    prediction_horizon: Optional[int] = None,
    win_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute forward-looking binary labels with strict per-ticker isolation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['date', 'ticker', 'close'].
    prediction_horizon : Optional[int]
        Number of forward trading days to evaluate. Defaults to
        settings.prediction_horizon (5).
    win_threshold : Optional[float]
        Fractional return required for label=1. Defaults to
        settings.win_threshold (0.01 = +1%).

    Returns
    -------
    pd.DataFrame
        DataFrame with ['date', 'ticker', 'future_close', 'forward_return', 'label'].
        Rows for the last `prediction_horizon` bars of each ticker have NaN for
        future_close, forward_return, and label.
    """
    required_cols = {"date", "ticker", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"compute_labels missing required columns: {sorted(missing)}")

    horizon: int = (
        int(prediction_horizon)
        if prediction_horizon is not None
        else int(settings.prediction_horizon)
    )
    threshold: float = (
        float(win_threshold)
        if win_threshold is not None
        else float(settings.win_threshold)
    )

    if horizon <= 0:
        raise ValueError(f"prediction_horizon must be positive, got {horizon}")
    if threshold <= 0:
        raise ValueError(f"win_threshold must be positive, got {threshold}")

    if df.empty:
        return pd.DataFrame(
            columns=["date", "ticker", FUTURE_CLOSE_COLUMN, FORWARD_RETURN_COLUMN, LABEL_COLUMN]
        )

    # Work on a copy with standardized sort: ticker, date
    work = df[["date", "ticker", "close"]].copy()
    work = work.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # STRICT PER-TICKER ISOLATION:
    # groupby('ticker')['close'].shift(-horizon) shifts within each ticker's group.
    # The last `horizon` rows for each ticker will produce NaN.
    work[FUTURE_CLOSE_COLUMN] = work.groupby("ticker")["close"].shift(-horizon)

    # Forward return: (future_close - close) / close (numerically exact for boundary conditions)
    work[FORWARD_RETURN_COLUMN] = (work[FUTURE_CLOSE_COLUMN] - work["close"]) / work["close"]

    # Binary label: 1 if return > threshold, 0 if return <= threshold, NaN if future unknown
    has_future = work[FUTURE_CLOSE_COLUMN].notna()
    work[LABEL_COLUMN] = np.nan
    work.loc[has_future, LABEL_COLUMN] = (
        work.loc[has_future, FORWARD_RETURN_COLUMN] > threshold
    ).astype(float)

    logger.debug(
        "compute_labels: horizon=%d, threshold=%.4f, input_rows=%d, labelled_rows=%d, unlabelled_tail=%d",
        horizon,
        threshold,
        len(work),
        int(has_future.sum()),
        len(work) - int(has_future.sum()),
    )

    return work[["date", "ticker", FUTURE_CLOSE_COLUMN, FORWARD_RETURN_COLUMN, LABEL_COLUMN]]


def check_label_balance(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    warn_low: float = 0.40,
    warn_high: float = 0.60,
) -> Dict[str, float]:
    """
    Compute class balance statistics and emit structured logs.

    Emits an INFO log with class percentages, and a WARNING log if the
    positive class ratio falls outside [warn_low, warn_high].

    Returns
    -------
    Dict[str, float]
        Dictionary with total, positive, negative, and positive_ratio.
    """
    valid_labels = df[label_col].dropna()
    total = len(valid_labels)
    if total == 0:
        logger.warning("check_label_balance: No valid labels found in DataFrame.")
        return {"total": 0, "positive": 0, "negative": 0, "positive_ratio": 0.0}

    positive = int((valid_labels == 1).sum())
    negative = int((valid_labels == 0).sum())
    pos_ratio = positive / total

    logger.info(
        "Dataset label balance: %d rows (%d positive [%.1f%%], %d negative [%.1f%%]).",
        total,
        positive,
        pos_ratio * 100.0,
        negative,
        (1.0 - pos_ratio) * 100.0,
    )

    if pos_ratio < warn_low or pos_ratio > warn_high:
        logger.warning(
            "Class imbalance detected: positive class ratio %.1f%% is outside the "
            "[%.1f%%, %.1f%%] balanced range. Consider adjusting win_threshold or evaluating sample bias.",
            pos_ratio * 100.0,
            warn_low * 100.0,
            warn_high * 100.0,
        )

    return {
        "total": float(total),
        "positive": float(positive),
        "negative": float(negative),
        "positive_ratio": float(pos_ratio),
    }


def load_survivorship_data(
    survivorship_dir: Optional[Union[str, Path]] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load genuine historical OHLCV data for delisted/acquired stocks (Item 10)
    to mitigate survivorship bias in ML training.

    Default files: FRCB.csv, TWTR.csv, CELG.csv in data/raw/survivorship/.
    """
    base_dir = Path(survivorship_dir) if survivorship_dir is not None else Path("data/raw/survivorship")
    if not base_dir.exists():
        logger.warning("Survivorship directory not found: %s", base_dir)
        return pd.DataFrame()

    target_tickers = [t.upper() for t in tickers] if tickers else ["FRCB", "TWTR", "CELG"]
    frames = []
    for ticker in target_tickers:
        csv_path = base_dir / f"{ticker}.csv"
        if not csv_path.exists():
            logger.warning("Survivorship CSV missing for %s at %s", ticker, csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
            col_map = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=col_map)
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            req = ["date", "open", "high", "low", "close", "volume", "ticker"]
            missing = [c for c in req if c not in df.columns]
            if missing:
                logger.warning("Survivorship file %s missing columns: %s", csv_path, missing)
                continue
            df = df[req].dropna()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values(by="date").reset_index(drop=True)
            frames.append(df)
            logger.info("Loaded %d rows of survivorship data for %s from %s", len(df), ticker, csv_path)
        except Exception as e:
            logger.error("Error loading survivorship data from %s: %s", csv_path, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    return combined


def build_dataset(
    ohlcv_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    drop_warmup: bool = True,
    prediction_horizon: Optional[int] = None,
    win_threshold: Optional[float] = None,
    include_auxiliary: bool = False,
    include_survivorship: bool = False,
    survivorship_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct the full ML dataset: features joined with forward labels.

    Steps:
      1. Optionally merge survivorship-bias mitigation data (Item 10: FRCB, TWTR, CELG).
      2. Compute Phase 3 features (blind to labels, strict backward lookback).
      3. Compute Phase 4 forward labels (strictly per-ticker, forward lookahead).
      4. Join features and labels on ['date', 'ticker'].
      5. Drop unlabelled tail rows (where label is NaN).
      6. If drop_warmup=True, drop rows where any feature is NaN (warm-up bars).
      7. Cast label to integer (0 or 1).
      8. Sort chronologically: date ASC, ticker ASC. No shuffling.
      9. Check and log class balance.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV market data for one or more tickers.
    spy_df : Optional[pd.DataFrame]
        SPY benchmark data for relative strength features.
    drop_warmup : bool
        If True, drops rows where features are NaN due to rolling window warm-up.
    prediction_horizon : Optional[int]
        Forward trading days for label. Defaults to settings.prediction_horizon (5).
    win_threshold : Optional[float]
        Return threshold for positive label. Defaults to settings.win_threshold (0.01).
    include_auxiliary : bool
        If True, retains 'future_close' and 'forward_return' in the output.
        Default is False (only 'label' is kept) to prevent target leakage into model.
    include_survivorship : bool
        If True, automatically ingests delisted historical stocks (FRCB, TWTR, CELG)
        to guard models against survivorship bias (Item 10).
    survivorship_df : Optional[pd.DataFrame]
        Optional custom DataFrame of delisted tickers. If None and include_survivorship=True,
        loads from data/raw/survivorship/.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted dataset with features and target label.
    """
    input_ohlcv = ohlcv_df.copy() if not ohlcv_df.empty else pd.DataFrame()
    if include_survivorship:
        surv = survivorship_df if survivorship_df is not None else load_survivorship_data()
        if not surv.empty:
            if input_ohlcv.empty:
                input_ohlcv = surv
            else:
                input_ohlcv = pd.concat([input_ohlcv, surv], ignore_index=True)
            input_ohlcv = input_ohlcv.sort_values(by=["date", "ticker"]).reset_index(drop=True)

    if input_ohlcv.empty:
        cols = ["date", "ticker"] + FEATURE_COLUMNS + [LABEL_COLUMN]
        if include_auxiliary:
            cols += [FUTURE_CLOSE_COLUMN, FORWARD_RETURN_COLUMN]
        return pd.DataFrame(columns=cols)

    # 1. Feature Engineering (Phase 3)
    features_df = compute_features(input_ohlcv, spy_df=spy_df)

    # 2. Label Engineering (Phase 4)
    labels_df = compute_labels(
        input_ohlcv,
        prediction_horizon=prediction_horizon,
        win_threshold=win_threshold,
    )

    # 3. Join on date and ticker
    merged = pd.merge(
        features_df,
        labels_df,
        on=["date", "ticker"],
        how="inner",
    )

    # 4. Drop unlabelled tail rows (last N rows per ticker where future is unknown)
    valid_mask = merged[LABEL_COLUMN].notna()
    tail_dropped = int((~valid_mask).sum())
    merged = merged[valid_mask].copy()

    # 5. Drop feature warmup rows if requested
    warmup_dropped = 0
    if drop_warmup:
        before_len = len(merged)
        merged = drop_warmup_rows(merged)
        warmup_dropped = before_len - len(merged)

    # 6. Cast label to int
    merged[LABEL_COLUMN] = merged[LABEL_COLUMN].astype(int)

    # 7. Strict chronological ordering (date ASC, ticker ASC)
    merged = merged.sort_values(by=["date", "ticker"], ascending=[True, True]).reset_index(drop=True)

    # 8. Filter columns: strip auxiliary forward values unless explicitly requested
    if not include_auxiliary:
        drop_cols = [c for c in [FUTURE_CLOSE_COLUMN, FORWARD_RETURN_COLUMN] if c in merged.columns]
        merged = merged.drop(columns=drop_cols)

    # 9. Audit log and class balance check
    logger.info(
        "build_dataset complete: %d rows, %d tickers. Dropped %d unlabelled tail rows, %d warmup rows.",
        len(merged),
        merged["ticker"].nunique() if not merged.empty else 0,
        tail_dropped,
        warmup_dropped,
    )
    if not merged.empty:
        check_label_balance(merged, label_col=LABEL_COLUMN)

    return merged


def build_ml_dataset() -> pd.DataFrame:
    """
    Load market data from DB for all tickers in settings.ticker_list and build the ML dataset.
    """
    from config.settings import settings
    from src.db import repository

    dfs = []
    for ticker in settings.ticker_list:
        try:
            df = repository.get_market_data(ticker)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    all_ohlcv = pd.concat(dfs, ignore_index=True)
    spy_df = None
    try:
        spy_df = repository.get_market_data(settings.benchmark)
    except Exception:
        pass

    return build_dataset(all_ohlcv, spy_df=spy_df)
