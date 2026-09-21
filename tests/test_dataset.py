"""
tests/test_dataset.py — Phase 4 ML Dataset & Label Verification.

Tests:
  TestLabelIsolation       — Per-ticker label shift isolation (no cross-ticker contamination)
  TestConfigIntegrity      — Prediction horizon and win threshold read from settings
  TestLabelContamination   — Features in dataset match standalone compute_features bit-for-bit
  TestFutureCloseMutation  — Future close mutation affects labels but zero past features
  TestTailTruncation       — Last horizon rows per ticker are dropped, never filled
  TestMonotonicOrdering    — Chronological ordering is preserved, zero shuffling
  TestClassBalance         — Label balance statistics and imbalance warning
  TestKnownValues          — Analytical boundary conditions (0.99%, 1.00%, 1.01%)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS, compute_features
from src.ml.dataset import (
    FORWARD_RETURN_COLUMN,
    FUTURE_CLOSE_COLUMN,
    LABEL_COLUMN,
    build_dataset,
    check_label_balance,
    compute_labels,
)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_ohlcv(
    prices: List[float],
    dates: Optional[List[str]] = None,
    ticker: str = "TEST",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame from close prices."""
    n = len(prices)
    if dates is None:
        dates = pd.bdate_range(start="2020-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": [p * 0.995 for p in prices],
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.990 for p in prices],
        "close": prices,
        "volume": [volume] * n,
    })


def _arithmetic_prices(n: int, start: float = 100.0, step: float = 1.0) -> List[float]:
    """Return n prices in arithmetic progression."""
    return [start + i * step for i in range(n)]


# ── 1. Per-Ticker Isolation Test ──────────────────────────────────────────────

class TestLabelIsolation:
    """
    Verify per-ticker isolation for the label shift.
    close.shift(-5) must be computed within each ticker's own date-sorted series,
    NEVER across concatenated multi-ticker data.
    """

    def test_multi_ticker_label_isolation(self):
        """
        Mirroring Phase 3's test_5_multi_ticker_isolation:
        Two tickers with completely different price levels (e.g. AAPL at $100, MSFT at $500).
        Confirm that each ticker's labels when processed jointly are bit-for-bit
        identical to each ticker's labels when processed solo.
        """
        dates = pd.bdate_range("2020-01-02", periods=30).strftime("%Y-%m-%d").tolist()

        # AAPL: flat $100 -> all returns 0.0 -> all valid labels are 0
        prices_aapl = [100.0] * 30
        # MSFT: starts at $500 and increases by $10 per day -> all valid returns > 1% -> all labels 1
        prices_msft = [500.0 + i * 10.0 for i in range(30)]

        df_aapl = _make_ohlcv(prices_aapl, dates=dates, ticker="AAPL")
        df_msft = _make_ohlcv(prices_msft, dates=dates, ticker="MSFT")

        # Solo runs
        labels_aapl_solo = compute_labels(df_aapl)
        labels_msft_solo = compute_labels(df_msft)

        # Joint run
        df_combined = pd.concat([df_aapl, df_msft], ignore_index=True)
        labels_combined = compute_labels(df_combined)

        labels_aapl_joint = (
            labels_combined[labels_combined["ticker"] == "AAPL"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        labels_msft_joint = (
            labels_combined[labels_combined["ticker"] == "MSFT"]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # 1. AAPL solo vs joint must match identically
        pd.testing.assert_frame_equal(labels_aapl_solo, labels_aapl_joint)

        # 2. MSFT solo vs joint must match identically
        pd.testing.assert_frame_equal(labels_msft_solo, labels_msft_joint)

    def test_concatenation_boundary_leakage_prevented(self):
        """
        Adversarial test:
        If shift(-5) were run naively across concatenated rows without groupby,
        the last rows of Ticker A would look into the first rows of Ticker B.
        Ticker A is at $100. Ticker B is at $500.
        A naive shift would see a return of (500 - 100)/100 = +400% -> label 1!
        With groupby('ticker'):
        Ticker A's last 5 rows MUST have future_close = NaN and label = NaN.
        """
        dates = pd.bdate_range("2020-01-02", periods=10).strftime("%Y-%m-%d").tolist()
        df_a = _make_ohlcv([100.0] * 10, dates=dates, ticker="TICK_A")
        df_b = _make_ohlcv([500.0] * 10, dates=dates, ticker="TICK_B")

        df_concat = pd.concat([df_a, df_b], ignore_index=True)
        labels = compute_labels(df_concat, prediction_horizon=5)

        # Last 5 rows of TICK_A (rows 5..9) must be NaN
        a_tail = labels[labels["ticker"] == "TICK_A"].iloc[5:]
        assert a_tail["future_close"].isna().all(), (
            "CRITICAL BUG: TICK_A tail rows leaked TICK_B prices as future closes! "
            f"Got future_close: {a_tail['future_close'].tolist()}"
        )
        assert a_tail["label"].isna().all(), "TICK_A tail rows must have NaN label"


# ── 2. Config Integrity Test ──────────────────────────────────────────────────

class TestConfigIntegrity:
    """Verify parameters are read from settings without hardcoding."""

    def test_default_reads_from_settings(self, monkeypatch):
        dates = pd.bdate_range("2020-01-02", periods=20).strftime("%Y-%m-%d").tolist()
        df = _make_ohlcv([100.0 + i for i in range(20)], dates=dates)

        # Default run
        labels_default = compute_labels(df)
        # Default horizon = 5 -> last 5 rows should be NaN
        assert labels_default[LABEL_COLUMN].iloc[-5:].isna().all()
        assert labels_default[LABEL_COLUMN].iloc[:-5].notna().all()

        # Monkeypatch settings.prediction_horizon_days to 3
        monkeypatch.setattr(settings, "prediction_horizon_days", 3)
        assert settings.prediction_horizon == 3

        labels_patched = compute_labels(df)
        # Now exactly last 3 rows should be NaN
        assert labels_patched[LABEL_COLUMN].iloc[-3:].isna().all()
        assert labels_patched[LABEL_COLUMN].iloc[:-3].notna().all()

    def test_explicit_arguments_override_settings(self):
        dates = pd.bdate_range("2020-01-02", periods=20).strftime("%Y-%m-%d").tolist()
        # Flat prices: returns are 0.0
        df = _make_ohlcv([100.0] * 20, dates=dates)

        # With negative win_threshold (e.g. -0.05), a 0.0 return is a "win"
        # but function requires win_threshold > 0, so test custom 0.0001
        labels = compute_labels(df, prediction_horizon=2, win_threshold=0.005)
        # Exactly 2 rows should be NaN tail
        assert labels[LABEL_COLUMN].iloc[-2:].isna().all()
        assert labels[LABEL_COLUMN].iloc[:-2].notna().all()


# ── 3. Label Contamination Test ───────────────────────────────────────────────

class TestLabelContamination:
    """
    Verify that features in the dataset match standalone compute_features output
    bit-for-bit, proving labels never contaminate feature columns.
    """

    def test_features_in_dataset_match_standalone_features(self):
        prices = _arithmetic_prices(250, start=100.0, step=0.5)
        df = _make_ohlcv(prices, ticker="AAPL")

        # Standalone features
        standalone_feat = compute_features(df)

        # Dataset features
        dataset = build_dataset(df, drop_warmup=False)

        # Check overlapping (date, ticker) rows
        dataset_indexed = dataset.set_index(["date", "ticker"])
        standalone_indexed = standalone_feat.set_index(["date", "ticker"])

        common_index = dataset_indexed.index.intersection(standalone_indexed.index)
        assert len(common_index) > 0

        for col in FEATURE_COLUMNS:
            ds_vals = dataset_indexed.loc[common_index, col].values
            sa_vals = standalone_indexed.loc[common_index, col].values

            nan_match = np.array_equal(np.isnan(ds_vals), np.isnan(sa_vals))
            assert nan_match, f"NaN pattern differs in '{col}' between dataset and standalone features"

            mask = ~np.isnan(ds_vals)
            if mask.any():
                diff = np.abs(ds_vals[mask] - sa_vals[mask]).max()
                assert diff < 1e-10, (
                    f"Feature '{col}' changed by {diff:.2e} in dataset vs standalone! "
                    "Label computation contaminated feature space."
                )

        # Ensure future-looking columns are NOT in the final training dataset by default
        assert FUTURE_CLOSE_COLUMN not in dataset.columns
        assert FORWARD_RETURN_COLUMN not in dataset.columns
        assert LABEL_COLUMN in dataset.columns


# ── 4. Future-Close Mutation Test ─────────────────────────────────────────────

class TestFutureCloseMutation:
    """
    Verify that mutating future close values changes labels, but leaves past
    and same-day features completely untouched.
    """

    def test_future_close_mutation_leaves_past_features_identical(self):
        n = 230  # Enough for warmup (200) + training rows
        prices_base = _arithmetic_prices(n, start=100.0, step=0.5)
        df_base = _make_ohlcv(prices_base, ticker="AAPL")
        spy_df = _make_ohlcv(prices_base, ticker="SPY")

        ds_base = build_dataset(df_base, spy_df=spy_df, drop_warmup=True)

        # Mutate the last 10 rows' close prices drastically (+100%)
        df_mutated = df_base.copy()
        df_mutated.loc[n - 10:, "close"] = df_mutated.loc[n - 10:, "close"] * 2.0
        df_mutated.loc[n - 10:, "high"] = df_mutated.loc[n - 10:, "close"] * 1.005
        df_mutated.loc[n - 10:, "low"] = df_mutated.loc[n - 10:, "close"] * 0.990

        ds_mutated = build_dataset(df_mutated, spy_df=spy_df, drop_warmup=True)

        # Check rows prior to the mutation window (rows 0 to n-16)
        # Because mutation starts at n-10, rows prior to n-10 are unmutated in OHLCV.
        # Features at row D (where D < n-10) depend ONLY on prices <= D.
        # Therefore, for D < n-10, features must be 100% identical!
        safe_dates = df_base["date"].iloc[: n - 10].tolist()

        ds_base_safe = ds_base[ds_base["date"].isin(safe_dates)].set_index("date")
        ds_mut_safe = ds_mutated[ds_mutated["date"].isin(safe_dates)].set_index("date")

        common_dates = ds_base_safe.index.intersection(ds_mut_safe.index)
        assert len(common_dates) > 0

        for col in FEATURE_COLUMNS:
            base_vals = ds_base_safe.loc[common_dates, col].values
            mut_vals = ds_mut_safe.loc[common_dates, col].values
            mask = ~np.isnan(base_vals)
            if mask.any():
                diff = np.abs(base_vals[mask] - mut_vals[mask]).max()
                assert diff < 1e-10, (
                    f"FUTURE LEAKAGE: Feature '{col}' changed by {diff:.2e} when future "
                    "prices were mutated!"
                )


# ── 5. Tail Truncation Test ───────────────────────────────────────────────────

class TestTailTruncation:
    """Verify that unlabelled tail rows are dropped, never filled."""

    def test_tail_rows_dropped(self):
        n = 220
        horizon = 5
        prices = _arithmetic_prices(n, start=100.0)
        df = _make_ohlcv(prices, ticker="AAPL")

        # Labels raw
        raw_labels = compute_labels(df, prediction_horizon=horizon)
        assert raw_labels[LABEL_COLUMN].iloc[-horizon:].isna().all()

        # Build dataset
        dataset = build_dataset(df, drop_warmup=False, prediction_horizon=horizon)
        # Dataset must NOT contain the last 5 dates
        tail_dates = set(df["date"].iloc[-horizon:])
        dataset_dates = set(dataset["date"])
        assert tail_dates.isdisjoint(dataset_dates), (
            f"Tail dates {tail_dates} found in dataset! Unlabelled rows were not dropped."
        )

        # Label column must contain only 0 and 1, no NaNs
        assert dataset[LABEL_COLUMN].isin([0, 1]).all()
        assert dataset[LABEL_COLUMN].dtype == int or np.issubdtype(dataset[LABEL_COLUMN].dtype, np.integer)


# ── 6. Monotonic Ordering Test ────────────────────────────────────────────────

class TestMonotonicOrdering:
    """Verify chronological ordering is strictly preserved."""

    def test_shuffled_input_produces_monotonic_output(self):
        dates = pd.bdate_range("2020-01-02", periods=30).strftime("%Y-%m-%d").tolist()
        df_a = _make_ohlcv([100.0] * 30, dates=dates, ticker="AAPL")
        df_b = _make_ohlcv([200.0] * 30, dates=dates, ticker="MSFT")

        combined = pd.concat([df_a, df_b], ignore_index=True)
        # Deliberately scramble order
        scrambled = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

        dataset = build_dataset(scrambled, drop_warmup=False)

        assert dataset["date"].is_monotonic_increasing, (
            "Dataset date column is not monotonically increasing! Strict chronological order violated."
        )


# ── 7. Class Balance Logging Test ─────────────────────────────────────────────

class TestClassBalance:
    """Verify label balance calculation and warning emission."""

    def test_balanced_distribution(self, caplog):
        # 50 rows of 1, 50 rows of 0
        df = pd.DataFrame({
            "label": [1] * 50 + [0] * 50,
        })
        with caplog.at_level(logging.INFO):
            stats = check_label_balance(df)

        assert stats["total"] == 100
        assert stats["positive"] == 50
        assert stats["negative"] == 50
        assert np.isclose(stats["positive_ratio"], 0.50)
        assert not any("Class imbalance detected" in rec.message for rec in caplog.records)

    def test_imbalance_triggers_warning(self, caplog):
        # 90 rows of 1, 10 rows of 0 (90% positive > 60% threshold)
        df = pd.DataFrame({
            "label": [1] * 90 + [0] * 10,
        })
        with caplog.at_level(logging.WARNING):
            stats = check_label_balance(df)

        assert np.isclose(stats["positive_ratio"], 0.90)
        warning_records = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "Class imbalance detected" in warning_records[0].message


# ── 8. Known Values & Boundary Test ───────────────────────────────────────────

class TestKnownValues:
    """Analytical verification of return threshold formula."""

    def test_threshold_strict_greater_than(self):
        """
        Verify the '>' operator (strictly greater than):
        threshold = 0.01 (+1.00%)
        close[T] = 100.0
          close[T+5] = 100.99 -> +0.99% -> label 0
          close[T+5] = 101.00 -> +1.00% -> label 0 (strictly greater required)
          close[T+5] = 101.01 -> +1.01% -> label 1
          close[T+5] = 95.00  -> -5.00% -> label 0
        """
        dates = pd.bdate_range("2020-01-02", periods=9).strftime("%Y-%m-%d").tolist()
        # Horizon = 5:
        # Row 0: close = 100.0, row 5 close = 100.99 -> 0
        # Row 1: close = 100.0, row 6 close = 101.00 -> 0
        # Row 2: close = 100.0, row 7 close = 101.01 -> 1
        # Row 3: close = 100.0, row 8 close = 95.00  -> 0
        # Rows 4..8: future unknown -> NaN
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.99, 101.00, 101.01, 95.00]
        df = _make_ohlcv(prices, dates=dates)

        labels = compute_labels(df, prediction_horizon=5, win_threshold=0.01)

        assert labels[LABEL_COLUMN].iloc[0] == 0.0
        assert labels[LABEL_COLUMN].iloc[1] == 0.0
        assert labels[LABEL_COLUMN].iloc[2] == 1.0
        assert labels[LABEL_COLUMN].iloc[3] == 0.0
        assert labels[LABEL_COLUMN].iloc[4:].isna().all()
