"""
tests/test_hyperparameter_tuning.py — Unit tests for Section 9 Item 1: Hyperparameter Auto-Tuning.

Tests cover:
  - Settings defaults
  - tune_primary: runs Optuna, returns valid result dict, saves JSON
  - tune_xgboost: runs Optuna, returns valid result dict, saves JSON
  - tune_all: runs both, returns summary dict
  - load_best_params: loads saved JSON correctly
  - _save_best_params / _print_tuning_summary helpers
  - CLI arg parsing
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.ml.tune import (
    BEST_PARAMS_PRIMARY_FILENAME,
    BEST_PARAMS_XGBOOST_FILENAME,
    TUNE_MODEL_ALL,
    TUNE_MODEL_PRIMARY,
    TUNE_MODEL_XGBOOST,
    _save_best_params,
    load_best_params,
    tune_primary,
    tune_xgboost,
    tune_all,
    _print_tuning_summary,
)
from src.ml.train import generate_walk_forward_folds, WalkForwardFold
from src.ml.dataset import LABEL_COLUMN
from src.features.engineer import FEATURE_COLUMNS


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(n_dates: int = 80, n_tickers: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal synthetic ML dataset large enough for walk-forward folds.
    """
    rng = np.random.default_rng(seed)
    base_dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y-%m-%d").tolist()
    tickers = [f"TST{i}" for i in range(n_tickers)]

    records = []
    for dt in base_dates:
        for tk in tickers:
            row = {col: float(rng.normal(0, 1)) for col in FEATURE_COLUMNS}
            row["date"] = dt
            row["ticker"] = tk
            row[LABEL_COLUMN] = int(rng.integers(0, 2))
            records.append(row)

    df = pd.DataFrame(records)
    return df


@pytest.fixture()
def synthetic_dataset():
    return _make_synthetic_dataset(n_dates=80, n_tickers=3)


@pytest.fixture()
def tmp_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


# ── Settings ───────────────────────────────────────────────────────────────────

class TestSettingsDefaults:
    def test_auto_tune_on_retrain_default_false(self):
        assert settings.auto_tune_on_retrain is False

    def test_optuna_trials_default(self):
        assert settings.optuna_trials == 50

    def test_optuna_timeout_default(self):
        assert settings.optuna_timeout_seconds == 600

    def test_random_seed_default(self):
        assert settings.random_seed == 42


# ── _save_best_params helper ──────────────────────────────────────────────────

class TestSaveBestParams:
    def test_saves_valid_json(self, tmp_models_dir):
        result = {
            "model_type": "primary",
            "best_roc_auc": 0.587,
            "n_trials": 10,
            "tuned_at": datetime.utcnow().isoformat(),
            "params": {"max_iter": 200, "max_depth": 5},
        }
        path = _save_best_params(result, BEST_PARAMS_PRIMARY_FILENAME, tmp_models_dir)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model_type"] == "primary"
        assert loaded["best_roc_auc"] == pytest.approx(0.587)
        assert loaded["params"]["max_iter"] == 200

    def test_creates_missing_directory(self, tmp_path):
        new_dir = tmp_path / "new_subdir" / "models"
        _save_best_params({"a": 1}, "test.json", new_dir)
        assert (new_dir / "test.json").exists()


# ── load_best_params ───────────────────────────────────────────────────────────

class TestLoadBestParams:
    def test_returns_none_when_missing(self, tmp_models_dir):
        result = load_best_params(TUNE_MODEL_PRIMARY, tmp_models_dir)
        assert result is None

    def test_loads_primary_params(self, tmp_models_dir):
        data = {"model_type": "primary", "best_roc_auc": 0.59, "params": {"max_iter": 300}}
        (tmp_models_dir / BEST_PARAMS_PRIMARY_FILENAME).write_text(json.dumps(data))
        loaded = load_best_params(TUNE_MODEL_PRIMARY, tmp_models_dir)
        assert loaded is not None
        assert loaded["params"]["max_iter"] == 300

    def test_loads_xgboost_params(self, tmp_models_dir):
        data = {"model_type": "xgboost", "best_roc_auc": 0.57, "params": {"n_estimators": 150}}
        (tmp_models_dir / BEST_PARAMS_XGBOOST_FILENAME).write_text(json.dumps(data))
        loaded = load_best_params(TUNE_MODEL_XGBOOST, tmp_models_dir)
        assert loaded is not None
        assert loaded["params"]["n_estimators"] == 150

    def test_raises_on_unknown_model_type(self, tmp_models_dir):
        with pytest.raises(ValueError, match="Unknown model_type"):
            load_best_params("baseline", tmp_models_dir)


# ── tune_primary ───────────────────────────────────────────────────────────────

class TestTunePrimary:
    def test_returns_valid_result_dict(self, synthetic_dataset, tmp_models_dir):
        result = tune_primary(
            dataset_df=synthetic_dataset,
            n_trials=3,
            timeout=60,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        assert result["model_type"] == TUNE_MODEL_PRIMARY
        assert 0.0 <= result["best_roc_auc"] <= 1.0
        assert result["n_trials"] >= 1
        assert "params" in result
        assert "tuned_at" in result

    def test_saves_json_file(self, synthetic_dataset, tmp_models_dir):
        tune_primary(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        json_path = tmp_models_dir / BEST_PARAMS_PRIMARY_FILENAME
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "params" in data
        assert "best_roc_auc" in data

    def test_saved_params_contain_expected_keys(self, synthetic_dataset, tmp_models_dir):
        tune_primary(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        data = json.loads((tmp_models_dir / BEST_PARAMS_PRIMARY_FILENAME).read_text())
        for key in ("max_iter", "max_depth", "learning_rate", "min_samples_leaf", "l2_regularization"):
            assert key in data["params"], f"Missing expected param key: {key}"

    def test_result_is_deterministic_with_same_seed(self, synthetic_dataset, tmp_models_dir):
        r1 = tune_primary(
            dataset_df=synthetic_dataset, n_trials=3, timeout=30,
            random_seed=42, models_dir=tmp_models_dir,
        )
        r2 = tune_primary(
            dataset_df=synthetic_dataset, n_trials=3, timeout=30,
            random_seed=42, models_dir=tmp_models_dir,
        )
        # With same seed and same trials count, best params should match
        assert r1["best_roc_auc"] == pytest.approx(r2["best_roc_auc"], abs=1e-4)


# ── tune_xgboost ───────────────────────────────────────────────────────────────

class TestTuneXGBoost:
    def test_returns_valid_result_dict(self, synthetic_dataset, tmp_models_dir):
        result = tune_xgboost(
            dataset_df=synthetic_dataset,
            n_trials=3,
            timeout=60,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        assert result["model_type"] == TUNE_MODEL_XGBOOST
        assert 0.0 <= result["best_roc_auc"] <= 1.0
        assert result["n_trials"] >= 1
        assert "params" in result

    def test_saves_json_file(self, synthetic_dataset, tmp_models_dir):
        tune_xgboost(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        json_path = tmp_models_dir / BEST_PARAMS_XGBOOST_FILENAME
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "params" in data

    def test_saved_params_contain_expected_keys(self, synthetic_dataset, tmp_models_dir):
        tune_xgboost(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        data = json.loads((tmp_models_dir / BEST_PARAMS_XGBOOST_FILENAME).read_text())
        for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
            assert key in data["params"], f"Missing expected param key: {key}"


# ── tune_all ───────────────────────────────────────────────────────────────────

class TestTuneAll:
    def test_returns_both_models(self, synthetic_dataset, tmp_models_dir):
        results = tune_all(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        assert TUNE_MODEL_PRIMARY in results
        assert TUNE_MODEL_XGBOOST in results
        assert results[TUNE_MODEL_PRIMARY]["model_type"] == TUNE_MODEL_PRIMARY
        assert results[TUNE_MODEL_XGBOOST]["model_type"] == TUNE_MODEL_XGBOOST

    def test_both_json_files_created(self, synthetic_dataset, tmp_models_dir):
        tune_all(
            dataset_df=synthetic_dataset,
            n_trials=2,
            timeout=30,
            random_seed=42,
            models_dir=tmp_models_dir,
        )
        assert (tmp_models_dir / BEST_PARAMS_PRIMARY_FILENAME).exists()
        assert (tmp_models_dir / BEST_PARAMS_XGBOOST_FILENAME).exists()


# ── _print_tuning_summary ─────────────────────────────────────────────────────

class TestPrintTuningSummary:
    def test_prints_without_error(self, capsys):
        _print_tuning_summary(0.587, 0.571)
        captured = capsys.readouterr()
        assert "TUNING COMPLETE" in captured.out
        assert "0.587" in captured.out
        assert "0.571" in captured.out
        assert "Primary best ROC-AUC" in captured.out
        assert "XGBoost best ROC-AUC" in captured.out


# ── Saved JSON format ─────────────────────────────────────────────────────────

class TestJsonFormat:
    def test_primary_json_has_required_fields(self, synthetic_dataset, tmp_models_dir):
        tune_primary(
            dataset_df=synthetic_dataset, n_trials=2, timeout=30,
            random_seed=42, models_dir=tmp_models_dir,
        )
        data = json.loads((tmp_models_dir / BEST_PARAMS_PRIMARY_FILENAME).read_text())
        assert data["model_type"] == "primary"
        assert isinstance(data["best_roc_auc"], float)
        assert isinstance(data["n_trials"], int)
        assert isinstance(data["tuned_at"], str)
        assert isinstance(data["params"], dict)

    def test_xgboost_json_has_required_fields(self, synthetic_dataset, tmp_models_dir):
        tune_xgboost(
            dataset_df=synthetic_dataset, n_trials=2, timeout=30,
            random_seed=42, models_dir=tmp_models_dir,
        )
        data = json.loads((tmp_models_dir / BEST_PARAMS_XGBOOST_FILENAME).read_text())
        assert data["model_type"] == "xgboost"
        assert isinstance(data["best_roc_auc"], float)
        assert isinstance(data["n_trials"], int)
        assert isinstance(data["tuned_at"], str)
        assert isinstance(data["params"], dict)
