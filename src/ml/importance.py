"""
src/ml/importance.py — Section 9 Item 2: Feature Importance Extraction.

PURPOSE
-------
Extract, save, and serve feature importance scores for every model type:

  - Primary (HistGradientBoostingClassifier): model.feature_importances_
  - XGBoost:                                  model.feature_importances_
  - Baseline (LogisticRegression pipeline):   abs(coef_[0]) normalised to sum=1
  - Ensemble (VotingClassifier):              average importances across sub-models

After every save_model() call, call save_importances_from_trained_model() to:
  1. Persist a JSON file: models/feature_importance_{model_type}.json
  2. Write per-feature rows into the feature_importance DB table

SHAP GLOBAL ANALYSIS (on-demand)
---------------------------------
compute_global_shap(trained_model, background_df) — slow, called from dashboard
Returns per-feature mean absolute SHAP value + mean signed SHAP for direction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.engineer import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# JSON filename template: feature_importance_primary.json
_FI_JSON_TEMPLATE = "feature_importance_{model_type}.json"


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _extract_tree_importances(estimator: Any, feature_names: List[str]) -> np.ndarray:
    """Return raw feature_importances_ from a tree-based model."""
    if hasattr(estimator, "feature_importances_") and estimator.feature_importances_ is not None:
        importances = np.array(estimator.feature_importances_, dtype=float)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
    if importances.sum() == 0:
        return np.ones(len(feature_names)) / len(feature_names)
    return importances / importances.sum()  # normalise to sum=1


def _extract_lr_importances(pipeline: Any, feature_names: List[str]) -> np.ndarray:
    """
    Extract abs(coef) from a Pipeline([scaler, LogisticRegression]).
    Normalises to sum=1.
    """
    clf = pipeline.named_steps.get("classifier") if isinstance(pipeline, Pipeline) else pipeline
    coef = np.abs(clf.coef_[0])
    total = coef.sum()
    if total == 0:
        return np.ones(len(feature_names)) / len(feature_names)
    return coef / total


def _extract_ensemble_importances(
    voting_clf: Any,
    feature_names: List[str],
) -> np.ndarray:
    """
    Average feature importances across VotingClassifier sub-estimators.
    """
    sub_importances: List[np.ndarray] = []
    estimators = getattr(voting_clf, "estimators_", []) or [est for _, est in getattr(voting_clf, "estimators", [])]
    for est in estimators:
        if isinstance(est, Pipeline):
            imp = _extract_lr_importances(est, feature_names)
        elif hasattr(est, "feature_importances_") and est.feature_importances_ is not None:
            imp = _extract_tree_importances(est, feature_names)
        else:
            imp = np.ones(len(feature_names)) / len(feature_names)
        sub_importances.append(imp)

    if not sub_importances:
        return np.ones(len(feature_names)) / len(feature_names)

    avg = np.mean(sub_importances, axis=0)
    total = avg.sum()
    return avg / total if total > 0 else avg


def extract_feature_importances(
    model_type: str,
    estimator: Any,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Extract normalised feature importances from any supported model type.

    Returns {feature_name: importance_score} where scores sum to 1.0.

    Parameters
    ----------
    model_type : str
        One of 'primary', 'xgboost', 'baseline', 'ensemble'.
    estimator : fitted sklearn/xgb estimator
        The actual model object (not TrainedModel wrapper).
    feature_names : Optional[List[str]]
        Feature column list. Defaults to FEATURE_COLUMNS.

    Returns
    -------
    dict {feature_name: float} sorted by importance descending.
    """
    names = feature_names or FEATURE_COLUMNS

    if model_type in ("primary", "xgboost"):
        raw = _extract_tree_importances(estimator, names)
    elif model_type == "baseline":
        raw = _extract_lr_importances(estimator, names)
    elif model_type == "ensemble":
        raw = _extract_ensemble_importances(estimator, names)
    else:
        logger.warning("Unknown model_type %r — returning uniform importances.", model_type)
        raw = np.ones(len(names)) / len(names)

    # Zip and sort descending
    pairs = sorted(zip(names, raw.tolist()), key=lambda x: x[1], reverse=True)
    return {feat: round(float(score), 6) for feat, score in pairs}


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_importances_from_trained_model(
    trained_model: Any,           # TrainedModel (avoids circular import)
    models_dir: Optional[Path] = None,
    date_str: Optional[str] = None,
) -> Dict[str, float]:
    """
    Extract importances from a TrainedModel, persist JSON and DB rows.

    Called automatically by save_model() in train.py after every retrain.

    Returns the {feature_name: score} dict.
    """
    from config.settings import settings
    from src.db import repository

    mdir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    mdir.mkdir(parents=True, exist_ok=True)
    today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    importances = extract_feature_importances(
        model_type=trained_model.model_type,
        estimator=trained_model.estimator,
        feature_names=trained_model.features,
    )

    # 1. Save JSON sidecar
    json_path = mdir / _FI_JSON_TEMPLATE.format(model_type=trained_model.model_type)
    payload = {
        "model_type": trained_model.model_type,
        "model_name": trained_model.model_name,
        "date": today,
        "importances": importances,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 2. Persist to DB
    try:
        repository.create_all_tables()
        repository.save_feature_importances(
            date_str=today,
            model_type=trained_model.model_type,
            importances=importances,
        )
    except Exception as exc:
        logger.warning("Could not persist feature importances to DB: %s", exc)

    logger.info(
        "Feature importances saved: model=%s  top=%s=%.4f  json=%s",
        trained_model.model_type,
        next(iter(importances)), list(importances.values())[0],
        json_path.name,
    )
    return importances


def load_importances_json(
    model_type: str,
    models_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the last-saved feature importance JSON for a given model type.
    Returns None if the file doesn't exist.
    """
    from config.settings import settings

    mdir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    path = mdir / _FI_JSON_TEMPLATE.format(model_type=model_type)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── SHAP global analysis (on-demand, slow) ─────────────────────────────────────

def compute_global_shap(
    trained_model: Any,
    background_df: pd.DataFrame,
    sample_size: int = 200,
) -> List[Dict[str, Any]]:
    """
    Compute global SHAP importance and direction for each feature.

    Returns list of dicts sorted by mean_abs_shap descending:
      {feature_name, mean_abs_shap, mean_signed_shap, direction}

    where direction is 'UP' (positive mean SHAP) or 'DOWN' (negative mean SHAP).

    Parameters
    ----------
    trained_model : TrainedModel
        The active or candidate model to explain.
    background_df : pd.DataFrame
        A representative sample of feature rows for background distribution.
    sample_size : int
        Max number of background rows to use (SHAP is O(n^2)).
    """
    import shap

    estimator = trained_model.estimator
    feat_cols = trained_model.features or FEATURE_COLUMNS

    # Subsample background
    bg = background_df[feat_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    if len(bg) > sample_size:
        bg = bg.sample(sample_size, random_state=42)

    if isinstance(estimator, Pipeline):
        scaler = estimator.named_steps.get("scaler")
        classifier = estimator.named_steps.get("classifier")
        bg_scaled = scaler.transform(bg) if scaler is not None else bg.values
        explainer = shap.LinearExplainer(classifier, bg_scaled)
        sample_scaled = scaler.transform(bg) if scaler is not None else bg.values
        shap_result = explainer(sample_scaled)
    else:
        bg_vals = bg.values
        explainer = shap.TreeExplainer(estimator, bg_vals)
        shap_result = explainer(bg_vals)

    shap_matrix = shap_result.values  # shape (N, n_features)
    if shap_matrix.ndim == 3:
        # Multi-class: use class 1 slice
        shap_matrix = shap_matrix[:, :, 1]

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    mean_signed = shap_matrix.mean(axis=0)

    results = []
    for i, feat in enumerate(feat_cols):
        results.append({
            "feature_name": feat,
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "mean_signed_shap": round(float(mean_signed[i]), 5),
            "direction": "UP" if mean_signed[i] >= 0 else "DOWN",
        })

    results.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    return results
