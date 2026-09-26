"""
tests/test_train.py — Phase 5 Model Training Verification.

Tests:
  TestChronologicalGap       — Chronological train/test split with 5-day embargo gap
  TestFeatureIntegrity       — Feature column requirements and NaN rejection
  TestProbabilityOutput      — Predict_proba shape, range [0, 1], and row-sum 1.0
  TestModelPersistence       — Serialization, reload, bit-for-bit prediction parity
  TestScalerPreservation     — StandardScaler fit on train only, mean_/scale_ bit-for-bit identical after reload
  TestDualModelMetrics       — Both primary (HistGBM) and baseline (LogReg) train and evaluate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS
from src.ml.dataset import LABEL_COLUMN, build_dataset
from src.ml.train import (
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_PRIMARY,
    TrainedModel,
    load_model,
    save_model,
    split_chronological,
    train_model,
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


def _make_synthetic_dataset(n_dates: int = 350) -> pd.DataFrame:
    """
    Generate a full valid ML dataset with all 19 FEATURE_COLUMNS and LABEL_COLUMN.
    Uses realistic trending + oscillating prices with warmup to ensure both
    classes are present in train and test partitions.
    """
    dates = pd.bdate_range("2020-01-02", periods=n_dates).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=n_dates)
    prices = 100.0 * np.cumprod(1.0 + returns)

    df_ticker = _make_ohlcv(prices.tolist(), dates=dates, ticker="AAPL")
    df_spy = _make_ohlcv((prices * 1.5).tolist(), dates=dates, ticker="SPY")

    # Build dataset with warmup dropped
    ds = build_dataset(df_ticker, spy_df=df_spy, drop_warmup=True)
    return ds


# ── 1. Chronological Gap Test ─────────────────────────────────────────────────

class TestChronologicalGap:
    """Verify chronological split mechanics and strict 5-day forward-label embargo gap."""

    def test_embargo_gap_purges_forward_window(self):
        dates = pd.bdate_range("2020-01-02", periods=100).strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "date": dates,
            "ticker": ["AAPL"] * 100,
            "label": [1] * 50 + [0] * 50,
        })
        for col in FEATURE_COLUMNS:
            df[col] = 1.0

        gap = 5
        train_df, test_df, info = split_chronological(df, test_ratio=0.20, gap_days=gap)

        train_dates = set(train_df["date"])
        test_dates = set(test_df["date"])
        purged_dates = set(info["purged_gap_dates"])

        # 1. Zero overlap between partitions
        assert train_dates.isdisjoint(test_dates), "Train and test dates must not overlap"
        assert train_dates.isdisjoint(purged_dates), "Train and purged gap dates must not overlap"
        assert test_dates.isdisjoint(purged_dates), "Test and purged gap dates must not overlap"

        # 2. Exactly gap_days unique dates purged
        assert len(purged_dates) == gap
        assert info["gap_days_count"] == gap

        # 3. Maximum train date strictly precedes minimum test date by > gap dates
        max_train_date = max(train_dates)
        min_test_date = min(test_dates)
        assert max_train_date < min(purged_dates)
        assert max(purged_dates) < min_test_date

        # 4. Total rows conserved
        assert len(train_df) + len(test_df) + info["purged_rows"] == len(df)

    def test_insufficient_dates_raises_error(self):
        # Only 5 dates cannot support 20% test + 5 days gap
        df = pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "ticker": ["AAPL"] * 3,
            "label": [1, 0, 1],
        })
        with pytest.raises(ValueError, match="Not enough unique trading dates"):
            split_chronological(df, test_ratio=0.20, gap_days=5)


# ── 2. Feature Integrity & NaN Rejection ──────────────────────────────────────

class TestFeatureIntegrity:
    """Verify input requirements and strict rejection of NaN values."""

    def test_missing_feature_columns_rejected(self):
        df = pd.DataFrame({"date": ["2020-01-02"], "ticker": ["AAPL"], "label": [1]})
        with pytest.raises(ValueError, match="missing required FEATURE_COLUMNS"):
            train_model(df, model_type=MODEL_TYPE_PRIMARY)

    def test_missing_label_column_rejected(self):
        dates = pd.bdate_range("2020-01-02", periods=250).strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({"date": dates, "ticker": ["AAPL"] * 250})
        for col in FEATURE_COLUMNS:
            df[col] = 1.0
        with pytest.raises(ValueError, match="missing target label column"):
            train_model(df, model_type=MODEL_TYPE_PRIMARY)

    def test_nan_in_features_raises_error_at_inference(self):
        ds = _make_synthetic_dataset(260)
        model = train_model(ds, model_type=MODEL_TYPE_PRIMARY)

        # Inject NaN into a feature row
        bad_df = ds.copy().iloc[:5]
        bad_df.loc[0, "rsi_14"] = np.nan

        with pytest.raises(ValueError, match="Features contain NaN values"):
            model.predict_proba(bad_df)


# ── 3. Probability Output Verification ────────────────────────────────────────

class TestProbabilityOutput:
    """Verify predict_proba format, row-sum consistency, and bounded probabilities."""

    def test_predict_proba_format_and_bounds(self):
        ds = _make_synthetic_dataset(260)
        model = train_model(ds, model_type=MODEL_TYPE_PRIMARY)

        probs = model.predict_proba(ds)

        # 1. Shape: N rows, 2 classes [P(0), P(1)]
        assert probs.ndim == 2
        assert probs.shape == (len(ds), 2)

        # 2. Bounded in [0.0, 1.0]
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

        # 3. Sum of P(0) + P(1) == 1.000 for every row
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

        # 4. Binary predictions match argmax
        preds = model.predict(ds)
        assert set(preds).issubset({0, 1})
        expected_preds = np.argmax(probs, axis=1)
        assert np.array_equal(preds, expected_preds)


# ── 4. Model Persistence & Reload Parity ──────────────────────────────────────

class TestModelPersistence:
    """Verify serialization, metadata JSON sidecar, and bit-for-bit reload parity."""

    def test_save_and_reload_parity(self, tmp_path):
        ds = _make_synthetic_dataset(260)
        model = train_model(ds, model_type=MODEL_TYPE_PRIMARY)

        probs_before = model.predict_proba(ds)

        # Save to temporary directory
        model_path, meta_path = save_model(model, models_dir=tmp_path, filename_prefix="test_model_v1")
        assert model_path.exists()
        assert meta_path.exists()

        # Check metadata content
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model_name"] == "hist_gradient_boosting_v1"
        assert meta["features_count"] == len(FEATURE_COLUMNS)
        assert meta["split_info"]["gap_days_count"] == 5
        assert meta["promotion_status"] == "candidate (pending Phase 6 human review)"

        # Reload model
        reloaded_model = load_model(model_path)
        assert reloaded_model.model_name == model.model_name
        assert reloaded_model.features == FEATURE_COLUMNS

        # Bit-for-bit prediction parity
        probs_after = reloaded_model.predict_proba(ds)
        assert np.array_equal(probs_before, probs_after), (
            "Reloaded model predict_proba does not match pre-save predictions bit-for-bit!"
        )


# ── 5. Scaler Fit-Once & Bit-for-Bit Preservation ─────────────────────────────

class TestScalerPreservation:
    """
    Verify that StandardScaler in the baseline pipeline is:
      1. Fit ONCE on training data only.
      2. Preserved bit-for-bit through save_model and load_model.
      3. Never refit or modified when predicting on test or live data.
    """

    def test_scaler_mean_scale_preserved_bit_for_bit(self, tmp_path):
        ds = _make_synthetic_dataset(260)
        baseline_model = train_model(ds, model_type=MODEL_TYPE_BASELINE)

        # Extract fitted scaler from the pipeline
        pipeline = baseline_model.estimator
        scaler = pipeline.named_steps["scaler"]
        orig_mean = scaler.mean_.copy()
        orig_scale = scaler.scale_.copy()

        assert orig_mean is not None
        assert orig_scale is not None
        assert len(orig_mean) == len(FEATURE_COLUMNS)

        # Save and reload
        model_path, _ = save_model(baseline_model, models_dir=tmp_path, filename_prefix="baseline_v1")
        reloaded_model = load_model(model_path)

        reloaded_scaler = reloaded_model.estimator.named_steps["scaler"]
        reloaded_mean = reloaded_scaler.mean_
        reloaded_scale = reloaded_scaler.scale_

        # 1. Bit-for-bit identical mean_ and scale_ after reload
        assert np.array_equal(orig_mean, reloaded_mean), (
            "StandardScaler mean_ changed after joblib save/reload!"
        )
        assert np.array_equal(orig_scale, reloaded_scale), (
            "StandardScaler scale_ changed after joblib save/reload!"
        )

        # 2. Predicting on new/test data must NOT alter scaler state
        reloaded_model.predict_proba(ds)
        assert np.array_equal(reloaded_scaler.mean_, orig_mean), (
            "StandardScaler mean_ was mutated during inference! Must never refit on test data."
        )
        assert np.array_equal(reloaded_scaler.scale_, orig_scale), (
            "StandardScaler scale_ was mutated during inference! Must never refit on test data."
        )


# ── 6. Dual-Model Training & Preliminary Metrics ──────────────────────────────

class TestDualModelMetrics:
    """Verify both primary and baseline models train, predict, and compute metrics."""

    def test_both_models_produce_valid_metrics(self):
        ds = _make_synthetic_dataset(260)

        # Train primary
        primary = train_model(ds, model_type=MODEL_TYPE_PRIMARY)
        assert primary.model_type == MODEL_TYPE_PRIMARY
        p_metrics = primary.metadata["preliminary_test_metrics"]
        assert 0.0 <= p_metrics["accuracy"] <= 1.0
        assert 0.0 <= p_metrics["roc_auc"] <= 1.0
        assert 0.0 <= p_metrics["brier_score"] <= 1.0

        # Train baseline
        baseline = train_model(ds, model_type=MODEL_TYPE_BASELINE)
        assert baseline.model_type == MODEL_TYPE_BASELINE
        b_metrics = baseline.metadata["preliminary_test_metrics"]
        assert 0.0 <= b_metrics["accuracy"] <= 1.0
        assert 0.0 <= b_metrics["roc_auc"] <= 1.0
        assert 0.0 <= b_metrics["brier_score"] <= 1.0

        # Confirm neither model is automatically promoted
        assert "candidate" in primary.metadata["promotion_status"]
        assert "candidate" in baseline.metadata["promotion_status"]


# ── Performance-Based Ensemble Weighting Tests ───────────────────────────────────

def test_get_performance_weights_proportional():
    from src.ml.train import get_performance_weights
    scores = {"hgb": 0.60, "lr": 0.40}
    weights = get_performance_weights(scores)
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert weights["hgb"] > weights["lr"]


def test_get_performance_weights_equal_fallback():
    from src.ml.train import get_performance_weights
    weights = get_performance_weights({})
    assert weights == {}


def test_weighted_ensemble_probability():
    from src.ml.train import get_weighted_ensemble_probability
    probs = {"hgb": 0.70, "lr": 0.50}
    weights = {"hgb": 0.75, "lr": 0.25}
    result = get_weighted_ensemble_probability(probs, weights)
    assert 0.50 < result < 0.70
    assert isinstance(result, float)
