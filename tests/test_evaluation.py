"""
tests/test_evaluation.py — Phase 6 Model Evaluation & Walk-Forward Engine Verification.

Tests:
  TestWindowBaseRateAndEdge     — Precision@0.60 vs Window Base Rate & Edge calculation
  TestTooGoodAlarm              — §1.5 Anti-self-deception alarm on suspicious metrics
  TestWalkForwardEmbargo        — Embargo gap enforcement across walk-forward boundaries
  TestModelPromotionWorkflow    — Explicit human promotion utility & active_model loading
  TestWalkForwardExecution      — Multi-window walk-forward execution and Markdown table output
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
from src.ml.evaluate import (
    ALARM_ACCURACY_THRESHOLD,
    ALARM_ROC_AUC_THRESHOLD,
    ACTIVE_METADATA_FILENAME,
    ACTIVE_MODEL_FILENAME,
    EvaluationReport,
    WindowMetrics,
    define_standard_windows,
    evaluate_predictions,
    load_active_model,
    promote_model,
    run_walk_forward_evaluation,
)
from src.ml.train import (
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_PRIMARY,
    TrainedModel,
    save_model,
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


def _make_multi_year_dataset(n_dates: int = 500) -> pd.DataFrame:
    """
    Generate a synthetic multi-year dataset spanning multiple regimes.
    """
    dates = pd.bdate_range("2020-01-02", periods=n_dates).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    # Random walk with positive drift
    returns = rng.normal(0.0008, 0.012, size=n_dates)
    prices = 100.0 * np.cumprod(1.0 + returns)

    df_ticker = _make_ohlcv(prices.tolist(), dates=dates, ticker="AAPL")
    df_spy = _make_ohlcv((prices * 1.2).tolist(), dates=dates, ticker="SPY")

    ds = build_dataset(df_ticker, spy_df=df_spy, drop_warmup=True)
    return ds


# ── 1. Base Rate & Edge Over Base Rate Tests ──────────────────────────────────

class TestWindowBaseRateAndEdge:
    """
    Verify Precision@0.60 calculation against the window's own empirical base rate.
    Edge = Precision@0.60 - base_rate (reported in percentage points).
    """

    def test_precision_and_edge_calculation(self):
        # 100 rows, 55 positive (base_rate = 0.55 = 55.0%)
        y_true = np.array([1] * 55 + [0] * 45)

        # 20 rows with P >= 0.60, of which 14 are positive (Precision = 14/20 = 70.0%)
        # 80 rows with P < 0.60
        y_prob = np.zeros(100)
        # 14 true positives with P >= 0.60
        y_prob[:14] = 0.65
        # 6 false positives with P >= 0.60 (indices 55..60 are class 0)
        y_prob[55:61] = 0.65
        # Remaining rows have P < 0.60
        y_prob[14:55] = 0.40
        y_prob[61:] = 0.30

        metrics = evaluate_predictions(y_true, y_prob, buy_bar=0.60)

        assert np.isclose(metrics["base_rate"], 0.55)
        assert metrics["count_at_buy_bar"] == 20
        assert np.isclose(metrics["precision_at_buy_bar"], 0.70)

        # Edge = 0.70 - 0.55 = +0.15 (+15.0 pts)
        assert np.isclose(metrics["edge_over_base_rate"], 0.15)

    def test_zero_conviction_rows_handled_gracefully(self):
        # When no predictions reach buy_bar (all P < 0.60)
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.45, 0.40, 0.52, 0.35, 0.48])

        metrics = evaluate_predictions(y_true, y_prob, buy_bar=0.60)

        assert metrics["count_at_buy_bar"] == 0
        assert metrics["precision_at_buy_bar"] is None
        assert metrics["edge_over_base_rate"] is None


# ── 2. "Too Good to Be True" Alarm Tests (§1.5) ───────────────────────────────

class TestTooGoodAlarm:
    """
    Verify §1.5 anti-self-deception rule:
    Metrics > 65% AUC or > 62% accuracy trigger a loud LEAKAGE SUSPICION ALARM.
    """

    def test_alarm_triggers_on_unrealistic_auc(self):
        # Perfect separation (AUC = 1.0)
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.array([0.90] * 50 + [0.10] * 50)

        metrics = evaluate_predictions(y_true, y_prob)

        assert metrics["roc_auc"] > ALARM_ROC_AUC_THRESHOLD
        assert metrics["too_good_alarm"] is True

    def test_alarm_silent_on_realistic_financial_edge(self):
        # Typical realistic edge: AUC ~ 0.55, Acc ~ 53%
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], size=200, p=[0.5, 0.5])
        # Realistic noisy probabilities (weak edge ~ 54% AUC)
        noise = rng.normal(0, 1.5, size=200)
        logits = 0.08 * (2 * y_true - 1) + noise
        y_prob = 1.0 / (1.0 + np.exp(-logits))

        metrics = evaluate_predictions(y_true, y_prob)

        assert metrics["roc_auc"] < ALARM_ROC_AUC_THRESHOLD
        assert metrics["accuracy"] < ALARM_ACCURACY_THRESHOLD
        assert metrics["too_good_alarm"] is False


# ── 3. Embargo Gap Enforcement in Walk-Forward ────────────────────────────────

class TestWalkForwardEmbargo:
    """
    Verify that walk-forward window generation strictly respects the 5-day embargo gap.
    """

    def test_invalid_embargo_gap_raises_error(self):
        ds = _make_multi_year_dataset(450)
        unique_dates = sorted(ds["date"].unique())

        # Create window where test starts only 2 days after train ends (violating 5-day gap)
        invalid_window = [{
            "name": "Invalid Window",
            "train_start": unique_dates[0],
            "train_end": unique_dates[50],
            "test_start": unique_dates[52],  # Only 1 day gap!
            "test_end": unique_dates[100],
            "regime_tag": "Test",
        }]

        with pytest.raises(ValueError, match="violates the 5-day embargo gap"):
            run_walk_forward_evaluation(ds, windows=invalid_window, gap_days=5)


# ── 4. Human-in-the-Loop Model Promotion Workflow ─────────────────────────────

class TestModelPromotionWorkflow:
    """
    Verify the explicit human promotion mechanism:
      1. Candidate model is copied to active_model.joblib.
      2. Metadata JSON is updated with 'promoted_active' and promotion reason.
      3. load_active_model() loads the promoted model for inference.
    """

    def test_promote_model_and_load_active(self, tmp_path):
        ds = _make_multi_year_dataset(350)
        candidate = train_model(ds, model_type=MODEL_TYPE_PRIMARY)

        # Save candidate in tmp_path
        cand_joblib, cand_meta = save_model(candidate, models_dir=tmp_path, filename_prefix="candidate_hgb")

        # Explicit human promotion
        reason = "Primary model beat baseline in 2022 bear market test with +4.2 pts edge."
        active_joblib, active_meta = promote_model(
            cand_joblib,
            reason=reason,
            author="lead_trader",
            models_dir=tmp_path,
        )

        assert active_joblib.name == ACTIVE_MODEL_FILENAME
        assert active_meta.name == ACTIVE_METADATA_FILENAME
        assert active_joblib.exists()
        assert active_meta.exists()

        # Check metadata
        with open(active_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["promotion_status"] == "promoted_active"
        assert meta["promoted_by"] == "lead_trader"
        assert meta["promotion_reason"] == reason
        assert meta["source_candidate_file"] == cand_joblib.name

        # Load active model
        active_model = load_active_model(models_dir=tmp_path)
        assert active_model.model_name == candidate.model_name
        assert active_model.features == FEATURE_COLUMNS

        # Inference parity
        probs_cand = candidate.predict_proba(ds.head(10))
        probs_active = active_model.predict_proba(ds.head(10))
        assert np.array_equal(probs_cand, probs_active)

    def test_load_active_model_missing_raises_error(self, tmp_path):
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No active model found"):
            load_active_model(models_dir=empty_dir)


# ── 5. Full Walk-Forward Execution & Markdown Report Output ───────────────────

class TestWalkForwardExecution:
    """
    Verify that run_walk_forward_evaluation runs successfully across dynamic or
    standard windows, reporting Precision@0.60 and Edge side by side.
    """

    def test_walk_forward_evaluation_produces_report(self):
        ds = _make_multi_year_dataset(380)

        report = run_walk_forward_evaluation(
            ds,
            model_types=(MODEL_TYPE_PRIMARY, MODEL_TYPE_BASELINE),
            buy_bar=0.60,
            gap_days=5,
        )

        assert isinstance(report, EvaluationReport)
        assert len(report.window_results) > 0

        # Check that table contains required headers
        md_table = report.to_markdown_table()
        assert "Base Rate" in md_table
        assert "Prec@0.60" in md_table
        assert "Edge over Base Rate" in md_table
        assert "Sample Confidence" in md_table
        assert "Alarm" in md_table

        # Summary contains both models and sample-size-weighted metrics
        for m in (MODEL_TYPE_PRIMARY, MODEL_TYPE_BASELINE):
            assert m in report.summary_by_model
            s = report.summary_by_model[m]
            assert "total_opportunities_fired" in s
            assert "sample_weighted_edge_pts" in s
            assert "unweighted_flat_edge_pts" in s
            assert "high_confidence_windows" in s
            assert "low_confidence_windows" in s
            assert s["total_opportunities_fired"] >= 0

    def test_low_sample_flagging_under_floor(self):
        """Verify that windows with N < 30 signals are flagged as LOW CONFIDENCE."""
        ds = _make_multi_year_dataset(380)
        report = run_walk_forward_evaluation(
            ds,
            model_types=(MODEL_TYPE_BASELINE,),
            buy_bar=0.60,
            gap_days=5,
        )
        for w in report.window_results:
            if w.count_at_buy_bar < 30:
                assert w.is_low_sample is True
                assert "LOW CONFIDENCE" in w.sample_confidence
            else:
                assert w.is_low_sample is False
                assert "HIGH" in w.sample_confidence
