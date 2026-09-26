"""
tests/test_ml_section3.py — Test Section 3 ML & Training Improvements (Items 10, 11, 12, 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS
from src.ml.dataset import LABEL_COLUMN, build_dataset, load_survivorship_data
from src.ml.evaluate import define_standard_windows
from src.ml.train import (
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_ENSEMBLE,
    MODEL_TYPE_PRIMARY,
    MODEL_TYPE_XGBOOST,
    TrainedModel,
    generate_walk_forward_folds,
    train_model,
    train_walk_forward,
)


def _make_dummy_dataset(n_days: int = 550, ticker: str = "AAPL") -> pd.DataFrame:
    """Helper to generate a valid ML dataset with realistic features and labels."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    
    data = {
        "date": dates,
        "ticker": [ticker] * n_days,
        LABEL_COLUMN: np.random.choice([0, 1], size=n_days, p=[0.55, 0.45]),
    }
    for feat in FEATURE_COLUMNS:
        data[feat] = np.random.randn(n_days)

    return pd.DataFrame(data)


# ── Item 10: Survivorship Bias Ingestion Integration ───────────────────────────

def test_load_survivorship_data_real_files():
    """Verify loading real historical CSV data for FRCB, TWTR, and CELG."""
    surv_df = load_survivorship_data()
    assert not surv_df.empty
    assert set(surv_df["ticker"].unique()) == {"FRCB", "TWTR", "CELG"}
    
    # Check expected columns
    for col in ["date", "open", "high", "low", "close", "volume", "ticker"]:
        assert col in surv_df.columns

    # Verify chronological monotonicity per ticker
    for t, group in surv_df.groupby("ticker"):
        dates = group["date"].tolist()
        assert dates == sorted(dates), f"Dates for {t} must be sorted ascending"


def test_build_dataset_with_survivorship_integration():
    """Verify that build_dataset can incorporate survivorship data without leakage."""
    # Synthetic live ticker OHLCV
    dates = pd.date_range("2022-01-03", periods=120, freq="B").strftime("%Y-%m-%d").tolist()
    p = 100.0
    prices = []
    for _ in range(120):
        p *= 1.001
        prices.append(p)

    ohlcv = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [x * 1.01 for x in prices],
        "low": [x * 0.99 for x in prices],
        "close": prices,
        "volume": [1_000_000.0] * 120,
        "ticker": "AAPL",
    })

    # Small mock survivorship subset
    mock_surv = pd.DataFrame({
        "date": dates,
        "open": [50.0] * 120,
        "high": [51.0] * 120,
        "low": [49.0] * 120,
        "close": [50.0] * 120,
        "volume": [500_000.0] * 120,
        "ticker": "FRCB",
    })

    ds = build_dataset(ohlcv, drop_warmup=False, include_survivorship=True, survivorship_df=mock_surv)
    assert not ds.empty
    assert "FRCB" in ds["ticker"].values
    assert "AAPL" in ds["ticker"].values
    assert LABEL_COLUMN in ds.columns


# ── Item 11: Extended Training History & Evaluation Windows ────────────────────

def test_settings_historical_start_date():
    """Verify settings.training_history_start_date is January 2008."""
    assert settings.training_history_start_date == "2008-01-01"


def test_define_standard_windows_includes_2008_and_2011_stress():
    """Verify 2008-2009 GFC and 2011 Eurozone stress windows are defined when dates cover 2008-2024."""
    long_dates = pd.date_range("2008-01-02", "2024-09-01", freq="B").strftime("%Y-%m-%d").tolist()
    windows = define_standard_windows(long_dates)
    
    window_names = [w["name"] for w in windows]
    assert any("2008-2009 GFC" in name for name in window_names)
    assert any("2011 Eurozone" in name for name in window_names)

    # Check GFC window boundaries
    gfc = next(w for w in windows if "2008-2009 GFC" in w["name"])
    assert gfc["test_start"] == "2008-09-09"  # 5 trading-day embargo after 2008-08-29 (Labor Day 2008-09-01)
    assert gfc["test_end"] == "2009-06-30"
    assert "GFC Crisis" in gfc["regime_tag"]

    # Check 2011 window boundaries
    ez = next(w for w in windows if "2011 Eurozone" in w["name"])
    assert ez["test_start"] == "2011-07-11"
    assert ez["test_end"] == "2011-12-30"


# ── Item 12: Walk-Forward Validation (2-Year Rolling, 6-Month Steps) ───────────

def test_generate_walk_forward_folds_rolling_and_embargo():
    """Verify rolling walk-forward fold generation with 2-year window, 6-month steps, and embargo gap."""
    # 3.5 years of dates (~880 trading days)
    ds = _make_dummy_dataset(n_days=880)
    folds = generate_walk_forward_folds(ds, train_years=2.0, test_months=6.0, gap_days=5, rolling=True)
    
    assert len(folds) >= 2
    for fold in folds:
        # Check train and test rows
        assert fold.train_rows > 0
        assert fold.test_rows > 0
        assert fold.train_end < fold.test_start
        
        # Verify strict embargo gap: dates between train_end and test_start must be >= gap_days
        train_end_dt = pd.to_datetime(fold.train_end)
        test_start_dt = pd.to_datetime(fold.test_start)
        assert (test_start_dt - train_end_dt).days >= 5


def test_train_walk_forward_execution():
    """Verify train_walk_forward executes across folds and aggregates metrics."""
    ds = _make_dummy_dataset(n_days=600)
    result = train_walk_forward(
        ds,
        model_type=MODEL_TYPE_PRIMARY,
        train_years=1.5,
        test_months=4.0,
        gap_days=5,
        random_state=42,
    )

    assert result["total_folds"] >= 1
    assert "mean_accuracy" in result
    assert "mean_roc_auc" in result
    assert "mean_brier_score" in result
    assert 0.0 <= result["mean_accuracy"] <= 1.0
    assert 0.0 <= result["mean_roc_auc"] <= 1.0
    assert isinstance(result["latest_model"], TrainedModel)


# ── Item 13: XGBoost & Simple Ensemble Models ──────────────────────────────────

def test_train_xgboost_candidate_model():
    """Verify training, prediction, and probability output of XGBoost candidate."""
    ds = _make_dummy_dataset(n_days=300)
    model = train_model(ds, model_type=MODEL_TYPE_XGBOOST, test_ratio=0.20, gap_days=5, random_state=42)

    assert model.model_type == MODEL_TYPE_XGBOOST
    assert model.model_name == "xgboost_classifier_v1"
    
    # Inference test
    preds = model.predict(ds.head(20))
    probs = model.predict_proba(ds.head(20))
    
    assert len(preds) == 20
    assert probs.shape == (20, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_train_voting_ensemble_model():
    """Verify soft-voting ensemble across Primary (HistGBM), Baseline (LogReg), and XGBoost."""
    ds = _make_dummy_dataset(n_days=300)
    model = train_model(ds, model_type=MODEL_TYPE_ENSEMBLE, test_ratio=0.20, gap_days=5, random_state=42)

    assert model.model_type == MODEL_TYPE_ENSEMBLE
    assert model.model_name == "voting_ensemble_v1"
    assert "VotingClassifier" in model.metadata["hyperparameters"]["ensemble_type"]
    assert len(model.metadata["hyperparameters"]["sub_models"]) == 3

    # Inference test
    preds = model.predict(ds.head(25))
    probs = model.predict_proba(ds.head(25))

    assert len(preds) == 25
    assert probs.shape == (25, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)
