"""
src/ml/tune.py — Section 9 Item 1: Hyperparameter Auto-Tuning with Optuna.

PURPOSE
-------
Automatically optimize hyperparameters for the two tunable models:
  1. Primary Model: HistGradientBoostingClassifier
  2. XGBoost Classifier

LogisticRegression (baseline) stays fixed — it is the simple benchmark.

HOW IT WORKS
------------
- Run Optuna with up to `settings.optuna_trials` trials per model.
- Each trial trains on walk-forward folds and scores on weighted ROC-AUC.
- Best parameters saved to models/best_params_primary.json and
  models/best_params_xgboost.json.
- Retrain final model using best params and save it.
- Log best params and score to the audit trail.

CLI USAGE
---------
  python -m src.ml.tune --model primary
  python -m src.ml.tune --model xgboost
  python -m src.ml.tune --model all

SETTINGS
--------
  auto_tune_on_retrain: bool = False  (opt-in per retrain run)
  optuna_trials: int = 50
  optuna_timeout_seconds: int = 600   (10 minutes max per model)
  random_seed: int = 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TUNE_MODEL_PRIMARY = "primary"
TUNE_MODEL_XGBOOST = "xgboost"
TUNE_MODEL_ALL = "all"
VALID_TUNE_TARGETS = {TUNE_MODEL_PRIMARY, TUNE_MODEL_XGBOOST, TUNE_MODEL_ALL}

BEST_PARAMS_PRIMARY_FILENAME = "best_params_primary.json"
BEST_PARAMS_XGBOOST_FILENAME = "best_params_xgboost.json"


# ── Optuna objective functions ─────────────────────────────────────────────────

def _score_folds_primary(
    folds: list,
    params: Dict[str, Any],
    feature_columns: List[str],
    label_column: str,
    random_state: int,
) -> float:
    """Train HistGradientBoostingClassifier with given params on each fold. Return weighted mean ROC-AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    auc_scores: List[float] = []
    weights: List[int] = []

    for fold in folds:
        X_tr = fold.train_df[feature_columns].copy()
        y_tr = fold.train_df[label_column].astype(int).values
        X_te = fold.test_df[feature_columns].copy()
        y_te = fold.test_df[label_column].astype(int).values

        if X_tr.isna().any().any() or X_te.isna().any().any():
            continue
        if len(np.unique(y_te)) < 2:
            continue

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]

        try:
            auc = float(roc_auc_score(y_te, y_prob))
        except (ValueError, ZeroDivisionError):
            auc = 0.5

        if not np.isnan(auc):
            auc_scores.append(auc)
            weights.append(len(y_te))

    if not auc_scores:
        return 0.5

    return float(np.average(auc_scores, weights=weights))


def _score_folds_xgboost(
    folds: list,
    params: Dict[str, Any],
    feature_columns: List[str],
    label_column: str,
    random_state: int,
) -> float:
    """Train XGBClassifier with given params on each fold. Return weighted mean ROC-AUC."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    auc_scores: List[float] = []
    weights: List[int] = []

    for fold in folds:
        X_tr = fold.train_df[feature_columns].copy()
        y_tr = fold.train_df[label_column].astype(int).values
        X_te = fold.test_df[feature_columns].copy()
        y_te = fold.test_df[label_column].astype(int).values

        if X_tr.isna().any().any() or X_te.isna().any().any():
            continue
        if len(np.unique(y_te)) < 2:
            continue

        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        scale_pos_weight = float(n_neg / max(1, n_pos))

        clf = xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
            **params,
        )
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]

        try:
            auc = float(roc_auc_score(y_te, y_prob))
        except (ValueError, ZeroDivisionError):
            auc = 0.5

        if not np.isnan(auc):
            auc_scores.append(auc)
            weights.append(len(y_te))

    if not auc_scores:
        return 0.5

    return float(np.average(auc_scores, weights=weights))


# ── Core tuning functions ──────────────────────────────────────────────────────

def tune_primary(
    dataset_df: pd.DataFrame,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    random_seed: Optional[int] = None,
    models_dir: Optional[Path] = None,
    folds: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run Optuna TPE search over HistGradientBoostingClassifier hyperparameters.

    Returns dict with keys: model_type, best_roc_auc, n_trials, tuned_at, params.
    """
    import optuna

    from config.settings import settings
    from src.features.engineer import FEATURE_COLUMNS
    from src.ml.dataset import LABEL_COLUMN
    from src.ml.train import generate_walk_forward_folds

    seed = random_seed if random_seed is not None else settings.random_seed
    n_t = n_trials if n_trials is not None else settings.optuna_trials
    t_out = timeout if timeout is not None else settings.optuna_timeout_seconds
    mdir = models_dir or settings.data_models_dir

    if folds is None:
        folds = generate_walk_forward_folds(dataset_df)
    if not folds:
        raise ValueError("No walk-forward folds generated for primary tuning.")

    # Silence Optuna's verbose output — only show trial/score logs at WARNING level
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        }
        score = _score_folds_primary(folds, params, FEATURE_COLUMNS, LABEL_COLUMN, seed)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info("OPTUNA TUNING: primary model — %d trials, timeout=%ds", n_t, t_out)
    study.optimize(objective, n_trials=n_t, timeout=t_out, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    result = {
        "model_type": TUNE_MODEL_PRIMARY,
        "best_roc_auc": round(best_auc, 4),
        "n_trials": completed,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "params": best_params,
    }

    _save_best_params(result, BEST_PARAMS_PRIMARY_FILENAME, mdir)
    logger.info(
        "TUNING COMPLETE — Primary best ROC-AUC: %.4f  params=%s",
        best_auc, best_params,
    )
    return result


def tune_xgboost(
    dataset_df: pd.DataFrame,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    random_seed: Optional[int] = None,
    models_dir: Optional[Path] = None,
    folds: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run Optuna TPE search over XGBClassifier hyperparameters.

    Returns dict with keys: model_type, best_roc_auc, n_trials, tuned_at, params.
    """
    import optuna

    from config.settings import settings
    from src.features.engineer import FEATURE_COLUMNS
    from src.ml.dataset import LABEL_COLUMN
    from src.ml.train import generate_walk_forward_folds

    seed = random_seed if random_seed is not None else settings.random_seed
    n_t = n_trials if n_trials is not None else settings.optuna_trials
    t_out = timeout if timeout is not None else settings.optuna_timeout_seconds
    mdir = models_dir or settings.data_models_dir

    if folds is None:
        folds = generate_walk_forward_folds(dataset_df)
    if not folds:
        raise ValueError("No walk-forward folds generated for xgboost tuning.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        score = _score_folds_xgboost(folds, params, FEATURE_COLUMNS, LABEL_COLUMN, seed)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info("OPTUNA TUNING: xgboost model — %d trials, timeout=%ds", n_t, t_out)
    study.optimize(objective, n_trials=n_t, timeout=t_out, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    result = {
        "model_type": TUNE_MODEL_XGBOOST,
        "best_roc_auc": round(best_auc, 4),
        "n_trials": completed,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "params": best_params,
    }

    _save_best_params(result, BEST_PARAMS_XGBOOST_FILENAME, mdir)
    logger.info(
        "TUNING COMPLETE — XGBoost best ROC-AUC: %.4f  params=%s",
        best_auc, best_params,
    )
    return result


def tune_all(
    dataset_df: pd.DataFrame,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    random_seed: Optional[int] = None,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Tune both primary and xgboost models, reusing the same walk-forward folds.
    Returns summary dict with results for both models.
    """
    from config.settings import settings
    from src.ml.train import generate_walk_forward_folds

    folds = generate_walk_forward_folds(dataset_df)
    if not folds:
        raise ValueError("No walk-forward folds generated for tuning.")

    primary_result = tune_primary(
        dataset_df=dataset_df,
        n_trials=n_trials,
        timeout=timeout,
        random_seed=random_seed,
        models_dir=models_dir,
        folds=folds,
    )
    xgboost_result = tune_xgboost(
        dataset_df=dataset_df,
        n_trials=n_trials,
        timeout=timeout,
        random_seed=random_seed,
        models_dir=models_dir,
        folds=folds,
    )

    summary = {
        "primary": primary_result,
        "xgboost": xgboost_result,
    }

    _print_tuning_summary(primary_result["best_roc_auc"], xgboost_result["best_roc_auc"])
    return summary


# ── Load saved best params ─────────────────────────────────────────────────────

def load_best_params(model_type: str, models_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load saved best hyperparameters JSON for a given model type.

    Returns the params dict, or None if the file doesn't exist.
    """
    from config.settings import settings

    mdir = models_dir or settings.data_models_dir
    mdir = Path(mdir)

    if model_type == TUNE_MODEL_PRIMARY:
        path = mdir / BEST_PARAMS_PRIMARY_FILENAME
    elif model_type == TUNE_MODEL_XGBOOST:
        path = mdir / BEST_PARAMS_XGBOOST_FILENAME
    else:
        raise ValueError(f"Unknown model_type for loading best params: {model_type!r}")

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save_best_params(result: Dict[str, Any], filename: str, models_dir: Any) -> Path:
    """Save best params JSON to models directory."""
    mdir = Path(models_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    out_path = mdir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Best params saved to %s", out_path)
    return out_path


def _print_tuning_summary(primary_auc: float, xgboost_auc: float) -> None:
    """Print human-readable tuning summary to stdout."""
    from config.settings import settings
    mdir = settings.data_models_dir
    print(
        f"\nTUNING COMPLETE\n"
        f"Primary best ROC-AUC: {primary_auc:.3f}\n"
        f"XGBoost best ROC-AUC: {xgboost_auc:.3f}\n"
        f"Best params saved to {mdir}/"
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """CLI: python -m src.ml.tune --model [primary|xgboost|all]"""
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Hyperparameter auto-tuning with Optuna.")
    parser.add_argument(
        "--model",
        choices=sorted(VALID_TUNE_TARGETS),
        default="all",
        help="Which model to tune: primary, xgboost, or all (default: all).",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout (seconds) per model.")
    args = parser.parse_args()

    # Load dataset from DB
    from config.settings import settings
    from src.db import repository
    from src.ml.dataset import build_ml_dataset

    repository.create_all_tables()
    dataset_df = build_ml_dataset()
    if dataset_df.empty:
        print("ERROR: No dataset available. Run the data pipeline first.")
        sys.exit(1)

    target = args.model.lower()
    if target == TUNE_MODEL_ALL:
        results = tune_all(
            dataset_df=dataset_df,
            n_trials=args.trials,
            timeout=args.timeout,
        )
    elif target == TUNE_MODEL_PRIMARY:
        result = tune_primary(
            dataset_df=dataset_df,
            n_trials=args.trials,
            timeout=args.timeout,
        )
        _print_tuning_summary(result["best_roc_auc"], float("nan"))
    elif target == TUNE_MODEL_XGBOOST:
        result = tune_xgboost(
            dataset_df=dataset_df,
            n_trials=args.trials,
            timeout=args.timeout,
        )
        _print_tuning_summary(float("nan"), result["best_roc_auc"])
    else:
        parser.error(f"Unknown model target: {target!r}")


if __name__ == "__main__":
    main()
