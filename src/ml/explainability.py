"""
src/ml/explainability.py — SHAP-based Model Explainability (V2.1 Wave 1).

PURPOSE:
--------
Calculates Shapley additive feature attributions (SHAP) for model predictions:
1. Explains why the promoted active model generated a specific probability for a ticker.
2. Works seamlessly with both:
   - Baseline pipeline: Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])
   - Primary model: HistGradientBoostingClassifier (tree ensemble)
3. Deconstructs prediction into:
   - Base value (model expected log-odds / prior)
   - Feature contributions (positive pushes score UP, negative pulls score DOWN)
   - Top-K positive and negative contributors
   - Verification that contributions sum to raw model decision output.

STRICT BOUNDARY:
----------------
Purely observational / explainability. Zero changes to trading rules or Risk Engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS
from src.ml.evaluate import load_active_model
from src.ml.train import TrainedModel

logger = logging.getLogger(__name__)


def compute_shap_attribution(
    estimator: Any,
    features_row: pd.DataFrame,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes exact SHAP attributions for a single feature vector.

    Parameters:
    -----------
    estimator : Pipeline or Classifier
        The fitted model estimator.
    features_row : pd.DataFrame
        Single-row DataFrame with the feature columns.
    feature_names : List[str]
        List of feature column names.
    background_data : Optional[np.ndarray]
        Background data for SHAP explainer. If None, uses zeros (zero-centered scaled space).

    Returns:
    --------
    Dict containing base_value, shap_values, log_odds, and probability.
    """
    # Ensure finite and non-NaN values
    clean_vals = features_row[feature_names].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    if isinstance(estimator, Pipeline):
        scaler = estimator.named_steps.get("scaler")
        classifier = estimator.named_steps.get("classifier")

        # Transform features through scaler
        if scaler is not None:
            scaled_vals = scaler.transform(clean_vals)
            scaled_bg = background_data if background_data is not None else np.zeros((1, len(feature_names)))
        else:
            scaled_vals = clean_vals.values
            scaled_bg = background_data if background_data is not None else np.zeros((1, len(feature_names)))

        # Use LinearExplainer for linear classifier
        explainer = shap.LinearExplainer(classifier, scaled_bg)
        shap_res = explainer(scaled_vals)

        shap_values = shap_res.values[0]
        base_value = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__len__") else float(shap_res.base_values)
        log_odds = float(classifier.decision_function(scaled_vals)[0])

        # Compute probability from log_odds via sigmoid
        prob = float(1.0 / (1.0 + np.exp(-log_odds)))

    else:
        # Tree-based model (e.g. HistGradientBoostingClassifier)
        bg = background_data if background_data is not None else np.zeros((1, len(feature_names)))
        explainer = shap.TreeExplainer(estimator, bg)
        shap_res = explainer(clean_vals.values)

        shap_values = shap_res.values[0]
        base_value = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__len__") else float(shap_res.base_values)

        if hasattr(estimator, "decision_function"):
            log_odds = float(estimator.decision_function(clean_vals.values)[0])
            prob = float(1.0 / (1.0 + np.exp(-log_odds)))
        else:
            raw_probs = estimator.predict_proba(clean_vals.values)[0]
            prob = float(raw_probs[1]) if len(raw_probs) > 1 else float(raw_probs[0])
            eps = 1e-6
            clipped_p = np.clip(prob, eps, 1.0 - eps)
            log_odds = float(np.log(clipped_p / (1.0 - clipped_p)))

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "log_odds": log_odds,
        "probability": prob,
    }


def explain_prediction(
    ticker: str,
    feature_data: Union[pd.Series, pd.DataFrame, Dict[str, float]],
    model: Optional[TrainedModel] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Explains the model prediction for a specific ticker and feature set.

    Returns structured dictionary with:
    - ticker
    - probability
    - base_value
    - sum_shap_plus_base
    - log_odds
    - top_positive_features (pushed score UP)
    - top_negative_features (pulled score DOWN)
    - all_contributions (dict of {feature: shap_value})
    - raw_feature_values (dict of {feature: raw_value})
    """
    trained_model = model if model is not None else load_active_model()
    estimator = trained_model.estimator
    feature_cols = trained_model.features or FEATURE_COLUMNS

    # Convert feature_data to single-row DataFrame
    if isinstance(feature_data, dict):
        row_df = pd.DataFrame([feature_data])
    elif isinstance(feature_data, pd.Series):
        row_df = pd.DataFrame([feature_data.to_dict()])
    else:
        row_df = feature_data.head(1).copy()

    # Ensure all required feature columns exist and contain finite numeric values
    for c in feature_cols:
        if c not in row_df.columns:
            row_df[c] = 0.0

    row_df[feature_cols] = row_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    raw_features = {c: float(row_df[c].iloc[0]) for c in feature_cols}

    # Compute SHAP
    attr = compute_shap_attribution(estimator, row_df, feature_cols)
    shap_vals = attr["shap_values"]
    base_val = attr["base_value"]
    log_odds = attr["log_odds"]
    prob = attr["probability"]

    # Map contributions
    contributions = []
    for f_name, s_val in zip(feature_cols, shap_vals):
        val_float = float(s_val)
        raw_val = raw_features.get(f_name, 0.0)
        contributions.append({
            "feature": f_name,
            "shap_value": round(val_float, 4),
            "raw_value": round(raw_val, 4),
            "impact": "positive" if val_float > 0 else ("negative" if val_float < 0 else "neutral"),
        })

    # Sort positive (descending) and negative (ascending)
    positives = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
    negatives = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])

    sum_plus_base = round(float(np.sum(shap_vals)) + base_val, 4)

    return {
        "ticker": ticker.upper(),
        "model_name": trained_model.model_name,
        "model_type": trained_model.model_type,
        "probability": round(prob, 4),
        "log_odds": round(log_odds, 4),
        "base_value": round(base_val, 4),
        "sum_shap_plus_base": sum_plus_base,
        "top_positive_features": positives[:top_k],
        "top_negative_features": negatives[:top_k],
        "all_contributions": {c["feature"]: c["shap_value"] for c in contributions},
        "raw_feature_values": raw_features,
    }
