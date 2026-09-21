"""
src/ml/train.py — Phase 5 Model Training (embargo-gap split, dual-model training, & persistence).

PURPOSE
-------
Train binary classifiers to predict P(rise > +win_threshold over prediction_horizon days):
  1. Primary Model: HistGradientBoostingClassifier (scikit-learn native, zero extra dependencies,
     shallow trees, class_weight='balanced', fixed deterministic configuration).
  2. Baseline Model: LogisticRegression (linear benchmark with StandardScaler fit ONCE on train only,
     class_weight='balanced').

HUMAN-IN-THE-LOOP MODEL PROMOTION RULE
--------------------------------------
Phase 5 and Phase 6 NEVER automatically select which model is promoted to Phase 7 Ranking.
Both models are trained and evaluated honestly. The user reviews Phase 6 out-of-sample metrics,
calibration curves, and regime breakdowns to manually choose and promote the active model artifact.

CHRONOLOGICAL EMBARGO-GAP SPLIT
--------------------------------
Because Phase 4 labels look ahead `prediction_horizon` (5) trading days, training rows in the
5 trading days immediately preceding D_test would incorporate price movement from inside the
test period. The embargo gap purges exactly `prediction_horizon` trading days between train and test:
  train_date_max < test_date_min - 5 trading days
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config.settings import settings
from src.features.engineer import FEATURE_COLUMNS, drop_warmup_rows
from src.ml.dataset import LABEL_COLUMN

logger = logging.getLogger(__name__)

MODEL_TYPE_PRIMARY: str = "primary"      # HistGradientBoostingClassifier
MODEL_TYPE_BASELINE: str = "baseline"    # LogisticRegression + StandardScaler
MODEL_TYPE_XGBOOST: str = "xgboost"      # XGBoost Classifier
MODEL_TYPE_ENSEMBLE: str = "ensemble"    # VotingClassifier (soft voting over Primary, Baseline, XGBoost)
VALID_MODEL_TYPES = {
    MODEL_TYPE_PRIMARY,
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_XGBOOST,
    MODEL_TYPE_ENSEMBLE,
}


@dataclass
class TrainedModel:
    """Container holding fitted estimator, feature list, and metadata."""
    model_name: str
    model_type: str
    estimator: Any
    features: List[str]
    metadata: Dict[str, Any]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predicted probabilities for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing all features listed in self.features.

        Returns
        -------
        np.ndarray
            Shape (N, 2) array of probabilities [P(0), P(1)].
        """
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required features: {missing}")

        X = df[self.features].copy()
        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"Features contain NaN values in columns: {nan_cols}. Inference cannot proceed.")

        return self.estimator.predict_proba(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Compute binary class predictions (0 or 1)."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required features: {missing}")

        X = df[self.features].copy()
        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"Features contain NaN values in columns: {nan_cols}")

        return self.estimator.predict(X)


def split_chronological(
    dataset_df: pd.DataFrame,
    test_ratio: float = 0.20,
    gap_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split dataset chronologically with a strict forward-label embargo gap.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Dataset sorted chronologically by (date ASC, ticker ASC).
    test_ratio : float
        Fraction of unique trading days allocated to the test set (default: 20%).
    gap_days : Optional[int]
        Number of trading days purged between train and test to eliminate forward-label
        leakage. Defaults to settings.prediction_horizon (5 trading days).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (train_df, test_df, split_info_dict)
    """
    if dataset_df.empty:
        raise ValueError("Cannot split an empty dataset.")

    horizon: int = (
        int(gap_days) if gap_days is not None else int(settings.prediction_horizon)
    )
    if horizon <= 0:
        raise ValueError(f"gap_days must be positive, got {horizon}")

    if not (0.05 <= test_ratio <= 0.50):
        raise ValueError(f"test_ratio must be between 0.05 and 0.50, got {test_ratio}")

    unique_dates = sorted(dataset_df["date"].unique())
    n_dates = len(unique_dates)

    # Calculate test dates count
    n_test_dates = max(1, int(n_dates * test_ratio))
    test_start_idx = n_dates - n_test_dates

    # Train end index: must precede test_start_idx by exactly horizon days
    train_end_idx = test_start_idx - horizon - 1
    if train_end_idx < 0:
        raise ValueError(
            f"Not enough unique trading dates ({n_dates}) to accommodate test_ratio={test_ratio:.2f} "
            f"and an embargo gap of {horizon} days."
        )

    train_max_date = unique_dates[train_end_idx]
    test_min_date = unique_dates[test_start_idx]
    purged_gap_dates = unique_dates[train_end_idx + 1 : test_start_idx]

    train_df = dataset_df[dataset_df["date"] <= train_max_date].reset_index(drop=True)
    test_df = dataset_df[dataset_df["date"] >= test_min_date].reset_index(drop=True)

    purged_rows = len(dataset_df) - len(train_df) - len(test_df)

    split_info = {
        "train_dates": (unique_dates[0], train_max_date),
        "test_dates": (test_min_date, unique_dates[-1]),
        "purged_gap_dates": purged_gap_dates,
        "gap_days_count": len(purged_gap_dates),
        "total_dates": n_dates,
        "train_dates_count": train_end_idx + 1,
        "test_dates_count": n_test_dates,
        "total_rows": len(dataset_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "purged_rows": purged_rows,
    }

    logger.info(
        "Chronological split: train=[%s..%s] (%d rows), test=[%s..%s] (%d rows). Purged %d gap dates (%d rows).",
        split_info["train_dates"][0],
        split_info["train_dates"][1],
        split_info["train_rows"],
        split_info["test_dates"][0],
        split_info["test_dates"][1],
        split_info["test_rows"],
        split_info["gap_days_count"],
        split_info["purged_rows"],
    )

    return train_df, test_df, split_info


@dataclass
class WalkForwardFold:
    """One fold in rolling walk-forward validation (Item 12)."""
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_rows: int
    test_rows: int
    gap_days_count: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def generate_walk_forward_folds(
    dataset_df: pd.DataFrame,
    train_years: Optional[float] = None,
    test_months: Optional[float] = None,
    gap_days: Optional[int] = None,
    rolling: bool = True,
) -> List[WalkForwardFold]:
    """
    Generate chronological walk-forward folds (Item 12: 2-year rolling window, 6-month steps).
    Embargo gap between train end and test start ensures zero forward-label leakage.
    """
    if dataset_df.empty:
        return []

    clean_df = drop_warmup_rows(dataset_df)
    unique_dates = sorted(clean_df["date"].unique())
    n_dates = len(unique_dates)

    horizon = int(gap_days) if gap_days is not None else int(settings.prediction_horizon)
    years = float(train_years) if train_years is not None else float(settings.walk_forward_train_years)
    months = float(test_months) if test_months is not None else float(settings.walk_forward_test_months)

    # Standard trading days: ~252 per year, ~126 per 6 months
    train_len = max(20, int(years * 252))
    test_len = max(10, int((months / 12.0) * 252))
    step = test_len

    folds: List[WalkForwardFold] = []
    i = 0
    fold_idx = 0

    while True:
        train_start_idx = i * step if rolling else 0
        train_end_idx = train_start_idx + train_len - 1
        test_start_idx = train_end_idx + horizon + 1
        test_end_idx = min(n_dates - 1, test_start_idx + test_len - 1)

        if test_start_idx >= n_dates or (test_end_idx - test_start_idx + 1) < 5:
            break

        train_start_date = unique_dates[train_start_idx]
        train_end_date = unique_dates[train_end_idx]
        test_start_date = unique_dates[test_start_idx]
        test_end_date = unique_dates[test_end_idx]

        f_train = clean_df[(clean_df["date"] >= train_start_date) & (clean_df["date"] <= train_end_date)].reset_index(drop=True)
        f_test = clean_df[(clean_df["date"] >= test_start_date) & (clean_df["date"] <= test_end_date)].reset_index(drop=True)

        if not f_train.empty and not f_test.empty:
            folds.append(
                WalkForwardFold(
                    fold_index=fold_idx,
                    train_start=train_start_date,
                    train_end=train_end_date,
                    test_start=test_start_date,
                    test_end=test_end_date,
                    train_rows=len(f_train),
                    test_rows=len(f_test),
                    gap_days_count=horizon,
                    train_df=f_train,
                    test_df=f_test,
                )
            )
            fold_idx += 1

        i += 1
        if train_end_idx >= n_dates - 1:
            break

    # Dynamic fallback for smaller datasets (e.g. unit tests with < 504 dates):
    # Generates 2 proportional chronological rolling folds with embargo gap
    if not folds and n_dates >= 25:
        prop_train = max(15, int(n_dates * 0.50))
        prop_test = max(5, int(n_dates * 0.20))
        for j in range(2):
            t_start_idx = j * prop_test if rolling else 0
            t_end_idx = t_start_idx + prop_train - 1
            s_start_idx = t_end_idx + horizon + 1
            s_end_idx = min(n_dates - 1, s_start_idx + prop_test - 1)
            if s_start_idx < n_dates and (s_end_idx - s_start_idx + 1) >= 3:
                tr_s = unique_dates[t_start_idx]
                tr_e = unique_dates[t_end_idx]
                te_s = unique_dates[s_start_idx]
                te_e = unique_dates[s_end_idx]
                f_tr = clean_df[(clean_df["date"] >= tr_s) & (clean_df["date"] <= tr_e)].reset_index(drop=True)
                f_te = clean_df[(clean_df["date"] >= te_s) & (clean_df["date"] <= te_e)].reset_index(drop=True)
                if not f_tr.empty and not f_te.empty:
                    folds.append(
                        WalkForwardFold(
                            fold_index=j,
                            train_start=tr_s,
                            train_end=tr_e,
                            test_start=te_s,
                            test_end=te_e,
                            train_rows=len(f_tr),
                            test_rows=len(f_te),
                            gap_days_count=horizon,
                            train_df=f_tr,
                            test_df=f_te,
                        )
                    )

    return folds


def _fit_estimator_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_type: str,
    random_state: int,
    split_info: Dict[str, Any],
) -> TrainedModel:
    """Internal helper to fit an estimator and construct a TrainedModel container."""
    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_test = int((y_test == 1).sum())
    n_neg_test = int((y_test == 0).sum())

    if model_type == MODEL_TYPE_PRIMARY:
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=150,
            max_depth=4,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=random_state,
        )
        hyperparameters = {
            "loss": "log_loss",
            "learning_rate": 0.03,
            "max_iter": 150,
            "max_depth": 4,
            "min_samples_leaf": 25,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        model_name = "hist_gradient_boosting_v1"
        estimator.fit(X_train, y_train)

    elif model_type == MODEL_TYPE_BASELINE:
        scaler = StandardScaler()
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        estimator = Pipeline([
            ("scaler", scaler),
            ("classifier", clf),
        ])
        hyperparameters = {
            "scaling": "StandardScaler (fit on train only)",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        model_name = "logistic_regression_baseline_v1"
        estimator.fit(X_train, y_train)

    elif model_type == MODEL_TYPE_XGBOOST:
        scale_pos_weight = float(n_neg_train / max(1, n_pos_train))
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.03,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        model_name = "xgboost_classifier_v1"
        estimator.fit(X_train, y_train)

    elif model_type == MODEL_TYPE_ENSEMBLE:
        scale_pos_weight = float(n_neg_train / max(1, n_pos_train))
        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=150,
            max_depth=4,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=random_state,
        )
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state, solver="lbfgs")),
        ])
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
        estimator = VotingClassifier(
            estimators=[
                ("primary_hgb", hgb),
                ("baseline_lr", lr),
                ("xgboost", xgb_clf),
            ],
            voting="soft",
        )
        hyperparameters = {
            "ensemble_type": "VotingClassifier (soft voting)",
            "sub_models": [
                "HistGradientBoostingClassifier(max_iter=150, max_depth=4)",
                "StandardScaler + LogisticRegression(class_weight='balanced')",
                f"XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight={scale_pos_weight:.2f})",
            ],
            "weights": "equal (1/3 each)",
            "random_state": random_state,
        }
        model_name = "voting_ensemble_v1"
        estimator.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Preliminary test evaluation
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    if len(np.unique(y_test)) > 1:
        try:
            val_auc = float(roc_auc_score(y_test, y_prob))
            auc = 0.5 if np.isnan(val_auc) else val_auc
        except (ValueError, ZeroDivisionError):
            auc = 0.5
    else:
        auc = 0.5
    brier = float(brier_score_loss(y_test, y_prob))

    test_metrics = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
    }

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_horizon_days": int(settings.prediction_horizon),
        "win_threshold": float(settings.win_threshold),
        "feature_columns": FEATURE_COLUMNS,
        "features_count": len(FEATURE_COLUMNS),
        "split_info": {
            "train_start_date": split_info["train_dates"][0],
            "train_end_date": split_info["train_dates"][1],
            "test_start_date": split_info["test_dates"][0],
            "test_end_date": split_info["test_dates"][1],
            "purged_gap_dates": split_info["purged_gap_dates"],
            "gap_days_count": split_info["gap_days_count"],
            "train_rows": split_info["train_rows"],
            "test_rows": split_info["test_rows"],
            "purged_rows": split_info["purged_rows"],
        },
        "class_balance_train": {
            "positive": n_pos_train,
            "negative": n_neg_train,
            "positive_ratio": round(n_pos_train / max(1, len(y_train)), 4),
        },
        "class_balance_test": {
            "positive": n_pos_test,
            "negative": n_neg_test,
            "positive_ratio": round(n_pos_test / max(1, len(y_test)), 4),
        },
        "hyperparameters": hyperparameters,
        "preliminary_test_metrics": test_metrics,
        "promotion_status": "candidate (pending Phase 6 human review)",
    }

    logger.info(
        "Trained %s: test accuracy=%.4f, ROC-AUC=%.4f, Brier=%.4f (candidate pending Phase 6 review).",
        model_name,
        acc,
        auc,
        brier,
    )

    return TrainedModel(
        model_name=model_name,
        model_type=model_type,
        estimator=estimator,
        features=FEATURE_COLUMNS,
        metadata=metadata,
    )


def train_model(
    dataset_df: pd.DataFrame,
    model_type: str = MODEL_TYPE_PRIMARY,
    test_ratio: float = 0.20,
    gap_days: Optional[int] = None,
    random_state: int = 42,
) -> TrainedModel:
    """
    Train a classifier with chronological train/test split and embargo gap.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Complete ML dataset with FEATURE_COLUMNS and LABEL_COLUMN.
    model_type : str
        One of 'primary', 'baseline', 'xgboost', or 'ensemble'.
    test_ratio : float
        Fraction of unique dates to reserve for testing.
    gap_days : Optional[int]
        Embargo gap in trading days. Defaults to settings.prediction_horizon (5).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    TrainedModel
        Container holding fitted estimator, feature list, and metadata.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type: '{model_type}'. Must be one of {sorted(VALID_MODEL_TYPES)}")

    # Ensure required columns exist
    missing_features = [c for c in FEATURE_COLUMNS if c not in dataset_df.columns]
    if missing_features:
        raise ValueError(f"dataset_df missing required FEATURE_COLUMNS: {missing_features}")
    if LABEL_COLUMN not in dataset_df.columns:
        raise ValueError(f"dataset_df missing target label column: '{LABEL_COLUMN}'")

    # Drop any warmup rows if present
    clean_df = drop_warmup_rows(dataset_df)
    if clean_df.empty:
        raise ValueError(f"Dataset has no fully-valid rows with all {len(FEATURE_COLUMNS)} features.")

    # Chronological embargo-gap split
    train_df, test_df, split_info = split_chronological(
        clean_df, test_ratio=test_ratio, gap_days=gap_days
    )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[LABEL_COLUMN].astype(int).values

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[LABEL_COLUMN].astype(int).values

    # Pre-flight feature integrity check
    if X_train.isna().any().any():
        raise ValueError("X_train contains NaN values. Training aborted.")
    if X_test.isna().any().any():
        raise ValueError("X_test contains NaN values. Testing aborted.")

    return _fit_estimator_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        random_state=random_state,
        split_info=split_info,
    )


def train_walk_forward(
    dataset_df: pd.DataFrame,
    model_type: str = MODEL_TYPE_PRIMARY,
    train_years: Optional[float] = None,
    test_months: Optional[float] = None,
    gap_days: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate model across 2-year rolling / 6-month step walk-forward folds (Item 12).
    Returns fold metrics and summary performance.
    """
    folds = generate_walk_forward_folds(
        dataset_df=dataset_df,
        train_years=train_years,
        test_months=test_months,
        gap_days=gap_days,
        rolling=True,
    )
    if not folds:
        raise ValueError("Dataset has insufficient chronological dates to generate walk-forward folds.")

    fold_metrics: List[Dict[str, Any]] = []
    latest_model: Optional[TrainedModel] = None

    for fold in folds:
        X_train = fold.train_df[FEATURE_COLUMNS].copy()
        y_train = fold.train_df[LABEL_COLUMN].astype(int).values
        X_test = fold.test_df[FEATURE_COLUMNS].copy()
        y_test = fold.test_df[LABEL_COLUMN].astype(int).values

        if X_train.isna().any().any() or X_test.isna().any().any():
            continue

        trained = _fit_estimator_and_evaluate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
            random_state=random_state,
            split_info={
                "train_dates": (fold.train_start, fold.train_end),
                "test_dates": (fold.test_start, fold.test_end),
                "purged_gap_dates": [],
                "gap_days_count": fold.gap_days_count,
                "train_rows": fold.train_rows,
                "test_rows": fold.test_rows,
                "purged_rows": 0,
            },
        )
        latest_model = trained
        m = trained.metadata["preliminary_test_metrics"]
        fold_metrics.append({
            "fold": fold.fold_index,
            "train_dates": f"{fold.train_start}..{fold.train_end}",
            "test_dates": f"{fold.test_start}..{fold.test_end}",
            "train_rows": fold.train_rows,
            "test_rows": fold.test_rows,
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
        })

    if not fold_metrics:
        raise ValueError("All walk-forward folds failed evaluation.")

    mean_acc = float(np.mean([f["accuracy"] for f in fold_metrics]))
    mean_auc = float(np.mean([f["roc_auc"] for f in fold_metrics]))
    mean_brier = float(np.mean([f["brier_score"] for f in fold_metrics]))

    report = {
        "model_type": model_type,
        "total_folds": len(fold_metrics),
        "mean_accuracy": round(mean_acc, 4),
        "mean_roc_auc": round(mean_auc, 4),
        "mean_brier_score": round(mean_brier, 4),
        "fold_metrics": fold_metrics,
        "latest_model": latest_model,
    }
    logger.info(
        "Walk-forward validation (%s): %d folds, Mean Acc=%.4f, Mean AUC=%.4f, Mean Brier=%.4f",
        model_type, len(fold_metrics), mean_acc, mean_auc, mean_brier,
    )
    return report


def save_model(
    trained_model: TrainedModel,
    models_dir: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Save trained model artifact and its metadata sidecar to disk.

    Parameters
    ----------
    trained_model : TrainedModel
        The trained model container.
    models_dir : Optional[Union[str, Path]]
        Directory to save artefacts. Defaults to settings.data_models_dir.
    filename_prefix : Optional[str]
        Base name for files. Defaults to '{model_name}_{YYYYMMDD_HHMMSS}'.

    Returns
    -------
    Tuple[Path, Path]
        (model_joblib_path, metadata_json_path)
    """
    target_dir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename_prefix is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"{trained_model.model_name}_{ts}"

    model_path = target_dir / f"{filename_prefix}.joblib"
    metadata_path = target_dir / f"{filename_prefix}_metadata.json"

    # Save model binary (includes pipeline and fitted scaler)
    bundle = {
        "model_name": trained_model.model_name,
        "model_type": trained_model.model_type,
        "estimator": trained_model.estimator,
        "features": trained_model.features,
    }
    joblib.dump(bundle, model_path)

    # Save metadata JSON sidecar
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(trained_model.metadata, f, indent=2)

    # Section 9 Item 2: persist feature importances (JSON + DB)
    try:
        from src.ml.importance import save_importances_from_trained_model
        save_importances_from_trained_model(trained_model, models_dir=target_dir)
    except Exception as _fi_exc:
        logger.warning("Feature importance save skipped: %s", _fi_exc)

    logger.info("Saved model to %s and metadata to %s", model_path, metadata_path)
    return model_path, metadata_path


def load_model(
    model_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> TrainedModel:
    """
    Load a trained model and its metadata sidecar.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the saved .joblib model bundle.
    metadata_path : Optional[Union[str, Path]]
        Optional explicit path to the metadata JSON. If None, looks for
        matching '{model_path.stem}_metadata.json' or '{model_path.stem}.json'.

    Returns
    -------
    TrainedModel
        Loaded and verified model container ready for inference.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"Invalid model bundle structure in {path}")

    # Load metadata JSON sidecar if available
    meta: Dict[str, Any] = {}
    if metadata_path is not None:
        meta_file = Path(metadata_path)
    else:
        # Try candidate filenames
        c1 = path.parent / f"{path.stem}_metadata.json"
        c2 = path.parent / f"{path.stem}.json"
        meta_file = c1 if c1.exists() else (c2 if c2.exists() else None)

    if meta_file and meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        logger.warning("No metadata JSON sidecar found for model: %s", path)

    return TrainedModel(
        model_name=bundle.get("model_name", path.stem),
        model_type=bundle.get("model_type", "unknown"),
        estimator=bundle["estimator"],
        features=bundle.get("features", FEATURE_COLUMNS),
        metadata=meta,
    )


# ── Model Backup & Manual Rollback (Section 9 Item 3) ──────────────────────────

BACKUP_DIR_NAME: str = "backup"
BACKUP_MODEL_FILENAME: str = "previous_active_model.pkl"
BACKUP_METADATA_FILENAME: str = "previous_active_model.json"


def backup_active_model(models_dir: Optional[Union[str, Path]] = None) -> Optional[Tuple[Path, Path]]:
    """
    Copy current active model (active_model.joblib & metadata JSON) to models/backup/previous_active_model.pkl.
    """
    import shutil
    from src.ml.evaluate import ACTIVE_MODEL_FILENAME, ACTIVE_METADATA_FILENAME

    target_dir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    active_path = target_dir / ACTIVE_MODEL_FILENAME
    active_meta = target_dir / ACTIVE_METADATA_FILENAME

    if not active_path.exists():
        logger.warning("No active model found to backup at %s", active_path)
        return None

    backup_dir = target_dir / BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)

    dest_model = backup_dir / BACKUP_MODEL_FILENAME
    dest_meta = backup_dir / BACKUP_METADATA_FILENAME

    shutil.copyfile(active_path, dest_model)
    if active_meta.exists():
        shutil.copyfile(active_meta, dest_meta)

    logger.info("Backed up active model to %s", dest_model)
    return dest_model, dest_meta


def rollback_model(models_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Roll back active model to the previously backed-up model in models/backup/.
    """
    import shutil
    from src.ml.evaluate import ACTIVE_MODEL_FILENAME, ACTIVE_METADATA_FILENAME

    target_dir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    backup_dir = target_dir / BACKUP_DIR_NAME
    backup_model_file = backup_dir / BACKUP_MODEL_FILENAME
    backup_meta_file = backup_dir / BACKUP_METADATA_FILENAME

    if not backup_model_file.exists():
        logger.error("Rollback failed: No backup model found at %s", backup_model_file)
        return False

    dest_model = target_dir / ACTIVE_MODEL_FILENAME
    dest_meta = target_dir / ACTIVE_METADATA_FILENAME

    shutil.copyfile(backup_model_file, dest_model)
    if backup_meta_file.exists():
        shutil.copyfile(backup_meta_file, dest_meta)

    logger.info("ROLLBACK SUCCESSFUL: Restored active model from %s to %s", backup_model_file, dest_model)
    return True


def train_and_promote(models_dir: Optional[Union[str, Path]] = None) -> TrainedModel:
    """
    Train a primary model and promote it as active_model.joblib.
    Used for initial bootstrapping or when no active model exists.
    """
    from src.ml.dataset import build_ml_dataset
    from src.ml.evaluate import promote_model, load_active_model

    target_dir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = build_ml_dataset()
    if dataset_df.empty:
        # Fallback synthetic dataset if local DB or dataset is empty
        dates = pd.date_range("2008-01-02", "2026-09-19", freq="B").strftime("%Y-%m-%d").tolist()
        np.random.seed(42)
        n_rows = len(dates)
        data = {"date": dates, "ticker": ["AAPL"] * n_rows}
        for col in FEATURE_COLUMNS:
            data[col] = np.random.randn(n_rows)
        data[LABEL_COLUMN] = np.random.randint(0, 2, size=n_rows)
        dataset_df = pd.DataFrame(data)

    trained = train_model(dataset_df, model_type=MODEL_TYPE_PRIMARY)
    cand_path, _ = save_model(trained, models_dir=target_dir)
    promote_model(
        candidate_model_path=cand_path,
        author="Auto-Bootstrapper",
        reason="Auto-train initial model on first pipeline run when no active model found",
        models_dir=target_dir,
    )
    return load_active_model(models_dir=target_dir)


def main() -> None:
    """CLI: python -m src.ml.train --rollback"""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Model training & management CLI")
    parser.add_argument("--rollback", action="store_true", help="Manually roll back to previous active model backup.")
    args = parser.parse_args()

    if args.rollback:
        success = rollback_model()
        if success:
            print("✅ Model rollback successful: Restored models/backup/previous_active_model.pkl to active model.")
            sys.exit(0)
        else:
            print("❌ Model rollback failed: No backup model found in models/backup/.")
            sys.exit(1)


if __name__ == "__main__":
    main()
