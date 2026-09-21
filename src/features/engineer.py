"""
src/features/engineer.py — Phase 3 Feature Engineering (zero-leakage).

LEAKAGE GUARANTEE
-----------------
For a feature row labeled date=D, ALL features use only data from dates ≤ D.
No future data (D+1, D+2, …) is ever incorporated.

The daily pipeline consumes the feature row for date D on day D+1 (the
next trading day). All features are therefore naturally one period behind
the decision point — this is the correct architecture; no extra .shift()
is applied to the features.

FEATURE CATALOGUE (15 scale-invariant features)
-----------------------------------------------
  Family A — Trend (4, dimensionless % distance & cross):
    price_to_ma20, price_to_ma50, price_to_ma200, ma5_above_ma20
    (Note: raw dollar MAs ma_5..ma_200 are excluded to ensure scale invariance across share prices)

  Family B — Momentum (2, dimensionless % returns):
    roc_5, roc_21

  Family C — RSI (1, dimensionless 0-100 oscillator):
    rsi_14  (Wilder's smoothing: alpha=1/14, adjust=False, recursive EMA)

  Family D — MACD (3, normalized by close price, dimensionless):
    macd_line, macd_signal, macd_hist
    (all EMAs use adjust=False; divided by close to prevent nominal price distortion)

  Family E — Volatility (2, dimensionless std & normalized ATR):
    vol_20  (rolling std of log returns, 20-day)
    atr_14  (Wilder's 14-period ATR, normalized by close)

  Family F — Volume (2, dimensionless ratios):
    vol_ratio_5_20  (5-day mean volume / 20-day mean volume)
    vol_spike       (today's volume / 20-day mean volume)

  Family G — Relative Strength vs Benchmark (1, dimensionless % diff):
    rel_strength_21  (ticker roc_21 − SPY roc_21, joined by date string)

EMA CONVENTION
--------------
All exponential moving averages use adjust=False (recursive/streaming EMA),
consistent with Bloomberg, FactSet, TradingView, and most trading platforms.

  pandas default adjust=True uses a bias-corrected sum that weights the
  first observation heavily — this diverges meaningfully from the recursive
  formula for the first ~3 × span observations and is NOT used here.

RSI — Wilder's smoothing:
  alpha = 1/14; ewm(alpha=1/14, adjust=False) is mathematically identical to:
    avg_gain_t = (avg_gain_{t-1} × 13 + gain_t) / 14
  min_periods=14 ensures no RSI value is emitted until 14 gain/loss values exist.

ATR — Wilder's 14-period EMA of True Range:
  Same alpha=1/14, adjust=False convention. min_periods=14.

NaN POLICY
----------
Rows with insufficient look-back history return NaN — no expanding-window
approximation. This is the correct choice: an expanding-window MA-200 at
day 5 is semantically a 5-day MA wearing a 200-day label; mixing these in
training would corrupt the feature distribution.

  Maximum warm-up: 200 rows (price_to_ma200). fetch_ticker_data automatically pulls
  ~305 calendar days of extra history before start_date to cover this.

PUBLIC API
----------
  compute_features(df, spy_df=None) → DataFrame with date/ticker + 15 feature cols
  drop_warmup_rows(feature_df)      → drop rows where any FEATURE_COLUMNS is NaN
  has_all_features(feature_row)     → True if no FEATURE_COLUMNS value is NaN
  FEATURE_COLUMNS                   → ordered list of all 15 scale-invariant feature names
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Feature column registry ────────────────────────────────────────────────────

FEATURE_COLUMNS: List[str] = [
    # Family A: Trend (Dimensionless % distances & cross)
    "price_to_ma20", "price_to_ma50", "price_to_ma200",
    "ma5_above_ma20",
    # Family B: Momentum (Dimensionless % returns)
    "roc_5", "roc_21",
    # Family C: RSI (Dimensionless 0-100)
    "rsi_14",
    # Family D: MACD (Normalized by close price, dimensionless)
    "macd_line", "macd_signal", "macd_hist",
    # Family E: Volatility (Dimensionless std & normalized ATR)
    "vol_20", "atr_14",
    # Family F: Volume (Dimensionless ratios)
    "vol_ratio_5_20", "vol_spike",
    # Family G: Relative Strength vs Benchmark (Dimensionless % diff)
    "rel_strength_21",
]

# ── Private per-ticker computation ────────────────────────────────────────────

def _compute_ticker_features(
    df: pd.DataFrame,
    spy_roc21: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute all 15 scale-invariant features for a SINGLE ticker's DataFrame.

    Args:
        df:         Sorted (ascending date), reset-indexed OHLCV DataFrame
                    for exactly one ticker. Must not contain NaN in OHLCV cols.
        spy_roc21:  Optional pd.Series indexed by date string ('YYYY-MM-DD'),
                    containing SPY's 21-day rate-of-change for each date.
                    Joined by date label — NEVER by row position.

    Returns:
        DataFrame with columns ['date', 'ticker'] + FEATURE_COLUMNS.
        NaN appears where look-back history is insufficient (see module docs).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    feat = pd.DataFrame({"date": df["date"].values, "ticker": df["ticker"].values})

    # ── Family A: Trend ────────────────────────────────────────────────────────

    ma_5 = close.rolling(5, min_periods=5).mean()
    ma_20 = close.rolling(20, min_periods=20).mean()
    ma_50 = close.rolling(50, min_periods=50).mean()
    ma_200 = close.rolling(200, min_periods=200).mean()

    # Price relative to MAs (signed % distance: positive = above MA)
    feat["price_to_ma20"] = (close.values / ma_20.values) - 1
    feat["price_to_ma50"] = (close.values / ma_50.values) - 1
    feat["price_to_ma200"] = (close.values / ma_200.values) - 1

    # Golden/death cross indicator: 1.0 if ma_5 > ma_20, 0.0 if not, NaN if either is NaN
    feat["ma5_above_ma20"] = np.where(
        ~np.isnan(ma_5.values) & ~np.isnan(ma_20.values),
        (ma_5.values > ma_20.values).astype(float),
        np.nan,
    )

    # ── Family B: Momentum ─────────────────────────────────────────────────────
    # pct_change(n) = close / close.shift(n) - 1; NaN for first n rows.
    feat["roc_5"] = close.pct_change(5).values
    feat["roc_21"] = close.pct_change(21).values

    # ── Family C: RSI — Wilder's smoothing ────────────────────────────────────
    # alpha=1/14 with adjust=False is Wilder's exact recursive formula:
    #   avg_gain_t = (avg_gain_{t-1} × 13 + gain_t) / 14
    # min_periods=14 ensures NaN until 14 gain/loss values are available.
    # numpy float arithmetic: x/0 → inf → RSI=100 (all gains);  0/0 → nan (flat)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain.values / avg_loss.values     # inf when avg_loss=0 (all gains → RSI=100)
        feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ── Family D: MACD — recursive EMA (adjust=False), normalized by close ────
    # Normalized by close price to ensure cross-sectional scale invariance (% of price).
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = (ema_12 - ema_26) / close
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    feat["macd_line"] = macd_line.values
    feat["macd_signal"] = macd_signal.values
    feat["macd_hist"] = (macd_line - macd_signal).values

    # ── Family E: Volatility ───────────────────────────────────────────────────

    # 20-day rolling std of log returns (annualised not needed — raw std is feature)
    log_ret = np.log(close / close.shift(1))
    feat["vol_20"] = log_ret.rolling(20, min_periods=20).std().values

    # ATR: Wilder's 14-period EMA of True Range, normalized by close.
    # True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    # At row 0 prev_close is NaN; max(axis=1, skipna=True) correctly uses only H-L.
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feat["atr_14"] = (atr / close).values    # normalized: dimensionless

    # ── Family F: Volume ───────────────────────────────────────────────────────
    vol_ma5 = volume.rolling(5, min_periods=5).mean()
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    feat["vol_ratio_5_20"] = (vol_ma5 / vol_ma20).values   # < 1 → volume fading
    feat["vol_spike"] = (volume / vol_ma20).values          # > 1 → unusual activity

    # ── Family G: Relative Strength vs Benchmark ──────────────────────────────
    # CRITICAL: join by date label, NEVER by row position.
    # Joining by position would silently cross-reference different calendar dates.
    if spy_roc21 is not None:
        ticker_roc21 = pd.Series(feat["roc_21"].values, index=feat["date"].values)
        spy_aligned = feat["date"].map(spy_roc21)           # NaN for dates absent from SPY
        feat["rel_strength_21"] = (ticker_roc21.values - spy_aligned.values)
    else:
        feat["rel_strength_21"] = np.nan

    return feat[["date", "ticker"] + FEATURE_COLUMNS].copy()


# ── Public API ────────────────────────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all 15 scale-invariant features for a universe of tickers.

    Each ticker is processed independently — no cross-ticker contamination.
    The SPY roc_21 mapping is computed once and joined by date label.

    Args:
        df:      Validated OHLCV DataFrame (output of Phase 2 validation gate).
                 Must have columns: date, ticker, open, high, low, close, volume.
                 May contain multiple tickers (grouped internally).
        spy_df:  Optional normalized SPY DataFrame (same schema). Used to compute
                 the rel_strength_21 feature. If None, rel_strength_21 = NaN.

    Returns:
        DataFrame with columns ['date', 'ticker'] + FEATURE_COLUMNS.
        Row ordering matches the input after per-ticker date sorting.
        NaN values appear where look-back history is insufficient.
        Rows are NOT dropped — call drop_warmup_rows() for training.

    LEAKAGE CONTRACT:
        Features for date D use only data from dates ≤ D.
        This is enforced by right-aligned rolling/ewm operations and by
        joining SPY data by date label (not position).
        The Future Blackout, Append-Recompute, and Label Independence tests
        in tests/test_features.py are the machine-verifiable proof of this.
    """
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_features: input DataFrame missing columns {missing}")

    if df.empty:
        return pd.DataFrame(columns=["date", "ticker"] + FEATURE_COLUMNS)

    # Pre-compute SPY roc_21 mapping: {date_str → roc_21_value}
    spy_roc21: Optional[pd.Series] = None
    if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
        spy_sorted = spy_df.sort_values("date").reset_index(drop=True)
        spy_roc21 = spy_sorted.set_index("date")["close"].pct_change(21)
        logger.debug("SPY roc_21 computed for %d dates.", len(spy_roc21))

    result_frames = []
    for ticker, ticker_df in df.groupby("ticker"):
        sorted_df = ticker_df.sort_values("date").reset_index(drop=True)
        feat_df = _compute_ticker_features(sorted_df, spy_roc21=spy_roc21)
        result_frames.append(feat_df)
        logger.debug(
            "Features computed for %s: %d rows, %d valid (non-NaN across all features).",
            ticker,
            len(feat_df),
            has_all_features_count(feat_df),
        )

    if not result_frames:
        return pd.DataFrame(columns=["date", "ticker"] + FEATURE_COLUMNS)

    result = pd.concat(result_frames, ignore_index=True)
    logger.info(
        "compute_features: %d tickers, %d total rows, %d fully-valid rows.",
        df["ticker"].nunique(),
        len(result),
        has_all_features_count(result),
    )
    return result


def drop_warmup_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where ANY feature column contains NaN.

    Use this after compute_features() to produce the training dataset.
    For live inference, use has_all_features() per row instead.

    Returns a new DataFrame with NaN rows dropped and index reset.
    """
    if feature_df.empty:
        return feature_df.copy()
    present_feat_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    mask = feature_df[present_feat_cols].notna().all(axis=1)
    return feature_df[mask].reset_index(drop=True)


def has_all_features(feature_row: pd.Series) -> bool:
    """
    Return True if a single feature row has no NaN in any FEATURE_COLUMNS value.

    Use this at inference time to gate whether a ticker can receive a prediction.
    If False, log the reason and skip the ticker for that day.
    """
    for col in FEATURE_COLUMNS:
        val = feature_row.get(col, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
    return True


def has_all_features_count(feature_df: pd.DataFrame) -> int:
    """Return count of rows where all FEATURE_COLUMNS are non-NaN."""
    if feature_df.empty:
        return 0
    present = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    return int(feature_df[present].notna().all(axis=1).sum())
