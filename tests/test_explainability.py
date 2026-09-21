"""
tests/test_explainability.py — Unit and Mathematical Additivity Tests for SHAP Explainability (V2.1 Wave 1).

Tests:
1. SHAP linear pipeline attribution computation.
2. SHAP sum-to-output test: sum(shap_values) + base_value == log_odds (exact additivity).
3. Sigmoid probability consistency: 1 / (1 + exp(-log_odds)) == probability.
4. Top positive and negative feature sorting and attribution breakdown.
5. Explainability with active system model.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineer import FEATURE_COLUMNS
from src.ml.evaluate import load_active_model
from src.ml.explainability import compute_shap_attribution, explain_prediction


def test_shap_linear_pipeline_exact_additivity():
    """
    CRITICAL TEST: SHAP values must sum exactly to the model's log-odds output:
    sum(shap_values) + base_value == log_odds.
    """
    # Create synthetic dataset with standard feature columns
    np.random.seed(42)
    X_train = np.random.randn(100, len(FEATURE_COLUMNS))
    y_train = (X_train[:, 0] * 1.5 + X_train[:, 1] * -1.0 + np.random.randn(100) > 0).astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # Test row
    row_vals = np.random.randn(1, len(FEATURE_COLUMNS))
    row_df = pd.DataFrame(row_vals, columns=FEATURE_COLUMNS)

    attr = compute_shap_attribution(
        estimator=pipeline,
        features_row=row_df,
        feature_names=FEATURE_COLUMNS,
    )

    shap_vals = attr["shap_values"]
    base_val = attr["base_value"]
    log_odds = attr["log_odds"]
    prob = attr["probability"]

    # 1. Sum-to-output verification (additivity theorem)
    reconstructed_log_odds = float(np.sum(shap_vals) + base_val)
    np.testing.assert_almost_equal(
        reconstructed_log_odds,
        log_odds,
        decimal=4,
        err_msg=f"SHAP sum-to-output violated! {reconstructed_log_odds} != {log_odds}",
    )

    # 2. Probability consistency via sigmoid
    expected_prob = 1.0 / (1.0 + np.exp(-log_odds))
    np.testing.assert_almost_equal(prob, expected_prob, decimal=4)


def test_explain_prediction_structure_and_sorting():
    """
    Verifies that explain_prediction correctly segregates and sorts
    top positive and negative contributors.
    """
    model = load_active_model()
    fake_features = {col: float(np.sin(i)) for i, col in enumerate(FEATURE_COLUMNS)}

    explanation = explain_prediction(
        ticker="AAPL",
        feature_data=fake_features,
        model=model,
        top_k=3,
    )

    assert explanation["ticker"] == "AAPL"
    assert 0.0 <= explanation["probability"] <= 1.0
    assert "base_value" in explanation
    assert "log_odds" in explanation
    assert "sum_shap_plus_base" in explanation

    # Verify mathematical sum
    np.testing.assert_almost_equal(
        explanation["sum_shap_plus_base"],
        explanation["log_odds"],
        decimal=3,
    )

    # Check top positive are positive and sorted descending
    pos = explanation["top_positive_features"]
    assert len(pos) <= 3
    for p in pos:
        assert p["shap_value"] > 0
        assert p["impact"] == "positive"
    if len(pos) >= 2:
        assert pos[0]["shap_value"] >= pos[1]["shap_value"]

    # Check top negative are negative and sorted ascending
    neg = explanation["top_negative_features"]
    assert len(neg) <= 3
    for n in neg:
        assert n["shap_value"] < 0
        assert n["impact"] == "negative"
    if len(neg) >= 2:
        assert neg[0]["shap_value"] <= neg[1]["shap_value"]
