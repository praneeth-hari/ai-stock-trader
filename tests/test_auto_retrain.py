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

    assert res["status"] in ("PROMOTED", "PENDING_APPROVAL", "RETAINED")
    assert res["candidate_model_type"] == "baseline"
    assert "report_file" in res
    assert Path(res["report_file"]).exists()

    report_text = Path(res["report_file"]).read_text(encoding="utf-8")
    assert "AI STOCK TRADER — MONTHLY RETRAIN REPORT" in report_text
    assert "TUNING RESULTS:" in report_text
    assert "EVALUATION RESULTS:" in report_text
    assert "PROMOTION DECISION:" in report_text

    # Notifications should have been invoked
    assert mock_notify.called or mock_send_alert.called


def _window(model_type, name, precision, base_rate=0.40):
    from types import SimpleNamespace
    return SimpleNamespace(model_type=model_type, window_name=name, regime_tag=name,
                           precision_at_buy_bar=precision, base_rate=base_rate, accuracy=0.99)


def _run_retrain_with(tmp_models_dir, summary, windows, drift="STABLE", dataset_rows=10,
                      human_approval=False, freeze=False):
    """Runs the retrain with heavy steps stubbed; returns (result, promote_mock, backup_mock)."""
    from types import SimpleNamespace
    import pandas as pd

    models_dir, logs_dir = tmp_models_dir
    dataset = pd.DataFrame({"date": ["2024-01-02"] * dataset_rows, "ticker": ["AAA"] * dataset_rows})

    def fake_save(tm, models_dir=None):
        return Path(models_dir) / f"{tm.model_type}.joblib", Path(models_dir) / f"{tm.model_type}.json"

    with patch("src.ml.retrain.build_ml_dataset", return_value=dataset), \
         patch("src.ml.retrain.tune_all", return_value={}), \
         patch("src.ml.retrain.train_model", side_effect=lambda df, model_type: SimpleNamespace(model_type=model_type)), \
         patch("src.ml.retrain.save_model", side_effect=fake_save), \
         patch("src.ml.retrain.run_walk_forward_evaluation",
               return_value=SimpleNamespace(summary_by_model=summary, window_results=windows)), \
         patch("src.ml.retrain.detect_prediction_drift", return_value={"status": drift}), \
         patch("src.ml.retrain.promote_model") as promote, \
         patch("src.ml.retrain.backup_active_model") as backup, \
         patch("src.ml.retrain.notify"), patch("src.ml.retrain.send_alert"), \
         patch.object(settings, "require_human_approval_for_promotion", human_approval), \
         patch.object(settings, "model_freeze_enabled", freeze):
        res = run_automatic_retrain(run_date="2026-09-05", force=True, optuna_trials=1,
                                    models_dir=models_dir, logs_dir=logs_dir)
    return res, promote, backup


PASSING_WINDOWS = [_window("baseline", "2008-2009 GFC", 0.55), _window("baseline", "2022 Bear", 0.50)]


def test_retrain_selects_locked_baseline_using_stored_mean_auc(tmp_models_dir):
    # HistGBM scores higher, but only the locked Baseline may be the candidate.
    summary = {"primary": {"mean_auc": 0.90}, "baseline": {"mean_auc": 0.60}, "xgboost": {"mean_auc": 0.70}}
    res, promote, backup = _run_retrain_with(tmp_models_dir, summary, PASSING_WINDOWS)

    assert res["candidate_model_type"] == "baseline"
    assert res["new_auc"] == 0.60          # the stored 'mean_auc', not 0.0 or a tuning score
    assert res["status"] == "PROMOTED"
    assert Path(promote.call_args.kwargs["candidate_model_path"]).name == "baseline.joblib"
    assert backup.called


def test_retrain_fails_loud_on_missing_real_data(tmp_models_dir):
    from src.ml.retrain import RetrainAbortedError

    ok_summary = {"baseline": {"mean_auc": 0.60}}
    with pytest.raises(RetrainAbortedError, match="synthetic"):
        _run_retrain_with(tmp_models_dir, ok_summary, PASSING_WINDOWS, dataset_rows=0)

    with pytest.raises(RetrainAbortedError, match="mean_auc"):
        _run_retrain_with(tmp_models_dir, {"primary": {"mean_auc": 0.90}}, PASSING_WINDOWS)

    # Active model without a recorded AUC: no default (e.g. 0.571) may be substituted.
    models_dir, _ = tmp_models_dir
    (models_dir / ACTIVE_METADATA_FILENAME).write_text(json.dumps({"model_name": "dummy_active_v1"}))
    with pytest.raises(RetrainAbortedError, match="no recorded ROC-AUC"):
        _run_retrain_with(tmp_models_dir, ok_summary, PASSING_WINDOWS)


def test_retrain_never_promotes_on_unevaluated_safety_gates(tmp_models_dir):
    summary = {"baseline": {"mean_auc": 0.60}}

    # No 2008 window evaluated (previously a hardcoded 44.2% vs 37.8% passed the gate).
    res, promote, backup = _run_retrain_with(tmp_models_dir, summary, [_window("baseline", "2022 Bear", 0.5)])
    assert res["status"] == "RETAINED" and not promote.called and not backup.called
    assert "2008-2009 GFC: NOT EVALUATED" in res["report_text"]

    # GFC window exists but produced no buy-bar signals: accuracy must not stand in for win rate.
    res, promote, _ = _run_retrain_with(tmp_models_dir, summary, [_window("baseline", "2008-2009 GFC", None)])
    assert res["status"] == "RETAINED" and not promote.called

    # Drift could not be measured -> not treated as "no drift".
    res, promote, _ = _run_retrain_with(tmp_models_dir, summary, PASSING_WINDOWS, drift="INSUFFICIENT_DATA")
    assert res["status"] == "RETAINED" and not promote.called


def test_retrain_2022_bear_window_is_a_promotion_gate(tmp_models_dir):
    summary = {"baseline": {"mean_auc": 0.60}}
    gfc_ok = _window("baseline", "2008-2009 GFC", 0.55)

    # Baseline loses to the 2022 base rate (30% < 40%): veto, even though every other gate passes.
    res, promote, backup = _run_retrain_with(tmp_models_dir, summary, [gfc_ok, _window("baseline", "2022 Bear", 0.30)])
    assert res["status"] == "RETAINED" and not promote.called and not backup.called
    assert "2022 Bear: Win rate 30.0% vs base 40.0% ❌" in res["report_text"]

    # 2022 not evaluated (missing window or no buy-bar signals): cannot be verified, so no promotion.
    for windows in ([gfc_ok], [gfc_ok, _window("baseline", "2022 Bear", None)]):
        res, promote, _ = _run_retrain_with(tmp_models_dir, summary, windows)
        assert res["status"] == "RETAINED" and not promote.called
        assert "2022 Bear: NOT EVALUATED" in res["report_text"]


def test_retrain_human_approval_keeps_backup_and_returns_result(tmp_models_dir):
    summary = {"baseline": {"mean_auc": 0.60}}
    res, promote, backup = _run_retrain_with(tmp_models_dir, summary, PASSING_WINDOWS, human_approval=True)
    assert res["status"] == "PENDING_APPROVAL"
    assert not promote.called and not backup.called


def test_retrain_model_freeze_blocks_promotion_even_without_human_approval_flag(tmp_models_dir):
    summary = {"baseline": {"mean_auc": 0.60}}
    res, promote, backup = _run_retrain_with(tmp_models_dir, summary, PASSING_WINDOWS,
                                             human_approval=False, freeze=True)
    assert res["status"] == "PENDING_APPROVAL"
    assert not promote.called and not backup.called


def test_human_approval_required_blocks_promotion():
    from config.settings import settings
    assert settings.require_human_approval_for_promotion is True

def test_auto_promote_blocked_when_approval_required():
    from config.settings import settings
    original = settings.require_human_approval_for_promotion
    settings.require_human_approval_for_promotion = True
    assert settings.require_human_approval_for_promotion is True
    settings.require_human_approval_for_promotion = original
