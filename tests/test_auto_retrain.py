"""
tests/test_auto_retrain.py — Unit tests for Section 9 Item 3: Automatic Retraining Schedule.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from src.db import repository
from src.ml.evaluate import ACTIVE_METADATA_FILENAME, ACTIVE_MODEL_FILENAME
from src.ml.retrain import is_first_saturday, run_automatic_retrain
from src.ml.train import (
    BACKUP_DIR_NAME,
    BACKUP_MODEL_FILENAME,
    TrainedModel,
    backup_active_model,
    rollback_model,
)


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Fixture providing an isolated models and logs directory for testing."""
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy active_model.joblib and active_model_metadata.json
    active_path = models_dir / ACTIVE_MODEL_FILENAME
    active_meta = models_dir / ACTIVE_METADATA_FILENAME

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])

    bundle = {
        "model_name": "dummy_active_v1",
        "model_type": "primary",
        "estimator": clf,
        "features": ["f1", "f2"],
    }
    import joblib
    joblib.dump(bundle, active_path)

    meta = {
        "model_name": "dummy_active_v1",
        "model_type": "primary",
        "preliminary_test_metrics": {"roc_auc": 0.571},
    }
    with open(active_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return models_dir, logs_dir


def test_settings_auto_retrain_defaults():
    assert settings.auto_retrain_enabled is True
    assert settings.auto_retrain_day == "saturday"
    assert settings.auto_retrain_hour == 10
    assert pytest.approx(settings.auto_promote_min_improvement, abs=1e-4) == 0.01
    assert settings.auto_retrain_keep_backup is True


def test_is_first_saturday():
    # 2026-09-05 is the first Saturday of September 2026
    dt_first_sat = datetime(2026, 9, 5, 10, 0, 0)
    assert is_first_saturday(dt_first_sat) is True

    # 2026-09-12 is the second Saturday
    dt_second_sat = datetime(2026, 9, 12, 10, 0, 0)
    assert is_first_saturday(dt_second_sat) is False

    # 2026-09-18 is a Friday
    dt_fri = datetime(2026, 9, 18, 10, 0, 0)
    assert is_first_saturday(dt_fri) is False


def test_backup_and_rollback_model(tmp_models_dir):
    models_dir, _ = tmp_models_dir

    # Backup active model
    backup_res = backup_active_model(models_dir=models_dir)
    assert backup_res is not None
    backup_pkl, backup_json = backup_res

    assert backup_pkl.exists()
    assert backup_pkl.name == BACKUP_MODEL_FILENAME

    # Modify active model file to simulate overwrite
    active_path = models_dir / ACTIVE_MODEL_FILENAME
    with open(active_path, "w", encoding="utf-8") as f:
        f.write("OVERWRITTEN_CANDIDATE_MODEL")

    # Rollback model
    success = rollback_model(models_dir=models_dir)
    assert success is True

    # Active model should now be restored
    import joblib
    restored_bundle = joblib.load(active_path)
    assert isinstance(restored_bundle, dict)
    assert restored_bundle.get("model_name") == "dummy_active_v1"


def test_rollback_fails_when_no_backup(tmp_path):
    empty_models_dir = tmp_path / "empty_models"
    empty_models_dir.mkdir()

    success = rollback_model(models_dir=empty_models_dir)
    assert success is False


def test_run_automatic_retrain_skipped_not_scheduled(tmp_models_dir):
    models_dir, logs_dir = tmp_models_dir

    # Run on a Friday without force
    res = run_automatic_retrain(
        run_date="2026-09-18",
        force=False,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )
    assert res["status"] == "SKIPPED_NOT_SCHEDULED"


@patch("src.ml.retrain.notify")
@patch("src.ml.retrain.send_alert")
def test_run_automatic_retrain_execution_flow(mock_send_alert, mock_notify, tmp_models_dir):
    models_dir, logs_dir = tmp_models_dir

    # Run forced automatic retrain with 1 trial
    res = run_automatic_retrain(
        run_date="2026-09-05",
        force=True,
        optuna_trials=1,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )

    assert res["status"] in ("PROMOTED", "RETAINED")
    assert "report_file" in res
    assert Path(res["report_file"]).exists()

    report_text = Path(res["report_file"]).read_text(encoding="utf-8")
    assert "AI STOCK TRADER — MONTHLY RETRAIN REPORT" in report_text
    assert "TUNING RESULTS:" in report_text
    assert "EVALUATION RESULTS:" in report_text
    assert "PROMOTION DECISION:" in report_text

    # Notifications should have been invoked
    assert mock_notify.called or mock_send_alert.called
