"""
tests/test_feature_importance.py — Unit tests for Section 9 Item 2: Feature Importance Dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import settings
from src.db import repository
from src.features.engineer import FEATURE_COLUMNS
from src.ml.importance import (
    _FI_JSON_TEMPLATE,
    compute_global_shap,
    extract_feature_importances,
    load_importances_json,
    save_importances_from_trained_model,
)
from src.ml.train import TrainedModel
from dashboard.data_loader import (
    get_feature_importance_data,
    get_feature_importance_history_data,
)
from dashboard.charts import (
    build_feature_importance_bar_chart,
    build_feature_importance_history_chart,
    build_shap_summary_chart,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Fixture providing a temporary SQLite database for testing repository functions."""
    db_file = tmp_path / "test_trader.db"
    db_url = f"sqlite:///{db_file}"

    orig_url = repository.settings.db_url
    repository.settings.__dict__["db_url"] = db_url
    repository._engine = None
    repository.create_all_tables()

    yield db_url

    repository._engine = None
    repository.settings.__dict__["db_url"] = orig_url


@pytest.fixture
def mock_feature_data():
    np.random.seed(42)
    n_samples = 100
    n_features = len(FEATURE_COLUMNS)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=FEATURE_COLUMNS,
    )
    y = np.random.randint(0, 2, size=n_samples)
    return X, y


def test_extract_tree_importances(mock_feature_data):
    X, y = mock_feature_data
    clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    clf.fit(X, y)

    importances = extract_feature_importances("xgboost", clf, FEATURE_COLUMNS)
    assert isinstance(importances, dict)
    assert len(importances) == len(FEATURE_COLUMNS)
    total = sum(importances.values())
    assert pytest.approx(total, abs=1e-4) == 1.0
    scores = list(importances.values())
    assert scores == sorted(scores, reverse=True)


def test_extract_baseline_importances(mock_feature_data):
    X, y = mock_feature_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=100)),
    ])
    pipe.fit(X, y)

    importances = extract_feature_importances("baseline", pipe, FEATURE_COLUMNS)
    assert isinstance(importances, dict)
    assert len(importances) == len(FEATURE_COLUMNS)
    total = sum(importances.values())
    assert pytest.approx(total, abs=1e-4) == 1.0


def test_extract_ensemble_importances(mock_feature_data):
    X, y = mock_feature_data
    est1 = ("xgb", XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss"))
    est2 = ("lr", Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=100)),
    ]))
    voting = VotingClassifier(estimators=[est1, est2], voting="soft")
    voting.fit(X, y)

    importances = extract_feature_importances("ensemble", voting, FEATURE_COLUMNS)
    assert isinstance(importances, dict)
    assert len(importances) == len(FEATURE_COLUMNS)
    total = sum(importances.values())
    assert pytest.approx(total, abs=1e-4) == 1.0


def test_save_importances_from_trained_model(tmp_path, mock_feature_data):
    X, y = mock_feature_data
    clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    clf.fit(X, y)

    trained = TrainedModel(
        model_name="xgboost_v1",
        model_type="xgboost",
        estimator=clf,
        features=FEATURE_COLUMNS,
        metadata={"model_type": "xgboost"},
    )

    date_str = "2026-09-19"
    importances = save_importances_from_trained_model(
        trained_model=trained,
        models_dir=tmp_path,
        date_str=date_str,
    )

    assert isinstance(importances, dict)
    assert len(importances) == len(FEATURE_COLUMNS)

    # Check JSON file exists and content is correct
    json_file = tmp_path / _FI_JSON_TEMPLATE.format(model_type="xgboost")
    assert json_file.exists()

    payload = load_importances_json(model_type="xgboost", models_dir=tmp_path)
    assert payload is not None
    assert payload["model_type"] == "xgboost"
    assert payload["date"] == date_str
    assert "importances" in payload
    assert len(payload["importances"]) == len(FEATURE_COLUMNS)


def test_db_repository_feature_importance_crud(tmp_db):
    date_str = "2026-09-19"
    model_type = "primary"
    sample_importances = {feat: round(1.0 / len(FEATURE_COLUMNS), 4) for feat in FEATURE_COLUMNS}

    saved_count = repository.save_feature_importances(
        date_str=date_str,
        model_type=model_type,
        importances=sample_importances,
    )
    assert saved_count == len(FEATURE_COLUMNS)

    # Query back latest importances
    retrieved = repository.get_feature_importances(model_type=model_type)
    assert len(retrieved) == len(FEATURE_COLUMNS)
    assert retrieved[0]["model_type"] == model_type
    assert retrieved[0]["date"] == date_str

    # Query history
    history = repository.get_feature_importance_history(model_type=model_type, top_n_features=3, last_n_retrains=5)
    assert len(history) > 0


def test_dashboard_data_loaders(tmp_db):
    date_str = "2026-09-19"
    sample_importances = {feat: round(1.0 / len(FEATURE_COLUMNS), 4) for feat in FEATURE_COLUMNS}
    repository.save_feature_importances(date_str=date_str, model_type="primary", importances=sample_importances)

    fi_data = get_feature_importance_data(model_type="primary")
    assert len(fi_data) == len(FEATURE_COLUMNS)

    history_data = get_feature_importance_history_data(model_type="primary", top_n=5, last_n=5)
    assert len(history_data) > 0


def test_dashboard_charts_builders():
    sample_rows = [
        {"feature_name": "RSI_14", "importance_score": 0.25, "date": "2026-09-19", "model_type": "primary"},
        {"feature_name": "momentum_20d", "importance_score": 0.15, "date": "2026-09-19", "model_type": "primary"},
    ]

    chart_bar = build_feature_importance_bar_chart(sample_rows)
    assert chart_bar is not None

    chart_hist = build_feature_importance_history_chart(sample_rows)
    assert chart_hist is not None

    shap_rows = [
        {"feature_name": "RSI_14", "mean_abs_shap": 0.12, "mean_signed_shap": 0.08, "direction": "UP"},
        {"feature_name": "momentum_20d", "mean_abs_shap": 0.09, "mean_signed_shap": -0.05, "direction": "DOWN"},
    ]
    chart_shap = build_shap_summary_chart(shap_rows)
    assert chart_shap is not None


def test_compute_global_shap(mock_feature_data):
    X, y = mock_feature_data
    clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    clf.fit(X, y)

    trained = TrainedModel(
        model_name="xgboost_v1",
        model_type="xgboost",
        estimator=clf,
        features=FEATURE_COLUMNS,
        metadata={"model_type": "xgboost"},
    )

    try:
        import shap
        shap_res = compute_global_shap(trained, X, sample_size=30)
        assert isinstance(shap_res, list)
        assert len(shap_res) == len(FEATURE_COLUMNS)
        item = shap_res[0]
        assert "feature_name" in item
        assert "mean_abs_shap" in item
        assert "mean_signed_shap" in item
        assert item["direction"] in ("UP", "DOWN")
    except ImportError:
        pytest.skip("shap library not installed")
