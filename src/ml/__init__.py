"""
src/ml/__init__.py — Machine Learning package exports.
"""

from src.ml.dataset import (
    FORWARD_RETURN_COLUMN,
    FUTURE_CLOSE_COLUMN,
    LABEL_COLUMN,
    build_dataset,
    check_label_balance,
    compute_labels,
    load_survivorship_data,
)
from src.ml.evaluate import (
    ALARM_ACCURACY_THRESHOLD,
    ALARM_ROC_AUC_THRESHOLD,
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
    MODEL_TYPE_ENSEMBLE,
    MODEL_TYPE_PRIMARY,
    MODEL_TYPE_XGBOOST,
    TrainedModel,
    WalkForwardFold,
    generate_walk_forward_folds,
    load_model,
    save_model,
    split_chronological,
    train_model,
    train_walk_forward,
)

__all__ = [
    "LABEL_COLUMN",
    "FORWARD_RETURN_COLUMN",
    "FUTURE_CLOSE_COLUMN",
    "compute_labels",
    "build_dataset",
    "check_label_balance",
    "load_survivorship_data",
    "TrainedModel",
    "WalkForwardFold",
    "split_chronological",
    "generate_walk_forward_folds",
    "train_model",
    "train_walk_forward",
    "save_model",
    "load_model",
    "MODEL_TYPE_PRIMARY",
    "MODEL_TYPE_BASELINE",
    "MODEL_TYPE_XGBOOST",
    "MODEL_TYPE_ENSEMBLE",
    "EvaluationReport",
    "WindowMetrics",
    "evaluate_predictions",
    "run_walk_forward_evaluation",
    "promote_model",
    "load_active_model",
    "define_standard_windows",
    "ALARM_ROC_AUC_THRESHOLD",
    "ALARM_ACCURACY_THRESHOLD",
]
