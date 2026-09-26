"""
src/ml/retrain.py — Section 9 Item 3: Automatic Retraining Schedule.

PURPOSE
-------
Orchestrates the monthly automatic model retraining pipeline running on the
first Saturday of every month at 10:00 AM:
  1. Fetch latest price data for all 25 tickers
  2. Rebuild ML dataset with new data
  3. Run hyperparameter tuning (50 trials via Optuna)
  4. Train models with optimal parameters
  5. Evaluate all models on multi-regime walk-forward folds
  6. Compare candidate ROC-AUC vs current active model
  7. Enforce Safety Rules:
       - Win rate > base rate in 2008-2009 GFC crisis
       - Win rate > base rate in 2022 Bear market
       - PSI prediction drift is not red (DRIFT_ALERT)
       - ROC-AUC improvement >= auto_promote_min_improvement (0.01 = +1.0%)
  8. If approved: backup active model to models/backup/ & promote new candidate
  9. Send Telegram + Email alerts
 10. Write retrain audit report to logs/retrain_YYYY-MM-DD.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from src.alerts.email_alerts import send_alert
from src.alerts.telegram_alerts import notify
from src.ml.dataset import build_ml_dataset
from src.db import repository
from src.ml.drift import detect_prediction_drift
from src.ml.evaluate import (
    load_active_model,
    promote_model,
    run_walk_forward_evaluation,
)
from src.ml.train import (
    LOCKED_MODEL_TYPE,
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_PRIMARY,
    MODEL_TYPE_XGBOOST,
    TrainedModel,
    backup_active_model,
    save_model,
    train_model,
)
from src.ml.tune import tune_all

logger = logging.getLogger(__name__)

# Drift statuses that positively confirm predictions are not drifting.
DRIFT_OK_STATUSES = ("STABLE", "MONITOR")


class RetrainAbortedError(RuntimeError):
    """Required real evaluation data is missing; nothing was selected or promoted."""


def _regime_result(eval_report: Any, year_tag: str) -> Optional[tuple]:
    """(win_rate_pct, base_rate_pct) of the locked model in the regime window, or None if not evaluated."""
    for wm in eval_report.window_results:
        if wm.model_type != LOCKED_MODEL_TYPE or year_tag not in f"{wm.window_name} {wm.regime_tag}":
            continue
        # No buy-bar signals means win rate is unmeasured; accuracy is not a substitute for it.
        if wm.precision_at_buy_bar is None:
            return None
        return round(wm.precision_at_buy_bar * 100, 1), round(wm.base_rate * 100, 1)
    return None


def _fmt_auc(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _fmt_regime(result: Optional[tuple], passed: bool) -> str:
    if result is None:
        return "NOT EVALUATED (no data / no buy-bar signals) ❌"
    return f"Win rate {result[0]:.1f}% vs base {result[1]:.1f}% {'✅' if passed else '❌'}"


def is_first_saturday(d: datetime) -> bool:
    """Return True if the given date is the first Saturday of its month."""
    return d.weekday() == 5 and d.day <= 7


def run_automatic_retrain(
    run_date: Optional[str] = None,
    force: bool = False,
    optuna_trials: Optional[int] = None,
    models_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the monthly automatic retraining job.

    Parameters
    ----------
    run_date : Optional[str]
        Date string (YYYY-MM-DD). Defaults to today's date.
    force : bool
        If True, ignore the schedule check and run immediately.
    optuna_trials : Optional[int]
        Number of Optuna tuning trials per model (default: settings.optuna_trials).
    models_dir : Optional[Path]
        Directory for saving models (default: settings.data_models_dir).
    logs_dir : Optional[Path]
        Directory for saving retrain log files (default: project_root/logs).

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with retrain results and promotion status.
    """
    repository.create_all_tables()

    now_dt = datetime.now(timezone.utc)
    date_str = run_date or now_dt.strftime("%Y-%m-%d")
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # 1. Schedule check
    if not force and not is_first_saturday(target_dt):
        logger.info("Skipping auto-retrain: %s is not the first Saturday of the month.", date_str)
        return {
            "status": "SKIPPED_NOT_SCHEDULED",
            "date": date_str,
            "reason": "Not the first Saturday of the month",
        }

    if not force and not settings.auto_retrain_enabled:
        logger.info("Skipping auto-retrain: auto_retrain_enabled is False in settings.")
        return {
            "status": "SKIPPED_DISABLED",
            "date": date_str,
            "reason": "auto_retrain_enabled is False",
        }

    logger.info("═══ STARTING MONTHLY AUTOMATIC MODEL RETRAINING (%s) ═══", date_str)

    mdir = Path(models_dir) if models_dir is not None else settings.data_models_dir
    ldir = Path(logs_dir) if logs_dir is not None else (settings.data_models_dir.parent.parent / "logs")
    ldir.mkdir(parents=True, exist_ok=True)

    n_trials = optuna_trials if optuna_trials is not None else settings.optuna_trials

    # 2. Rebuild dataset with latest data
    dataset_df = build_ml_dataset()

    if dataset_df.empty:
        raise RetrainAbortedError(
            "build_ml_dataset() returned no rows. Refusing to train, evaluate or promote on "
            "synthetic/fabricated data."
        )

    start_date = str(dataset_df["date"].min())
    end_date = str(dataset_df["date"].max())
    n_samples = len(dataset_df)

    # 3. Hyperparameter tuning (50 trials)
    logger.info("Running hyperparameter auto-tuning (%d trials)...", n_trials)
    tuning_summary = tune_all(dataset_df=dataset_df, n_trials=n_trials, models_dir=mdir)
    primary_best_auc = tuning_summary.get("primary", {}).get("best_roc_auc")
    xgboost_best_auc = tuning_summary.get("xgboost", {}).get("best_roc_auc")

    # 4. Train candidate models
    logger.info("Training candidate models...")
    primary_tm = train_model(dataset_df, model_type=MODEL_TYPE_PRIMARY)
    baseline_tm = train_model(dataset_df, model_type=MODEL_TYPE_BASELINE)
    xgboost_tm = train_model(dataset_df, model_type=MODEL_TYPE_XGBOOST)

    save_model(primary_tm, models_dir=mdir)
    b_path, _ = save_model(baseline_tm, models_dir=mdir)
    save_model(xgboost_tm, models_dir=mdir)

    # 5. Evaluate on walk-forward folds
    logger.info("Evaluating candidate models on walk-forward folds...")
    eval_report = run_walk_forward_evaluation(
        dataset_df=dataset_df,
        model_types=(MODEL_TYPE_PRIMARY, MODEL_TYPE_BASELINE, MODEL_TYPE_XGBOOST),
    )

    # Only the locked model type is a promotion candidate; other types are informational.
    summary_by_model = eval_report.summary_by_model
    for mtype, metrics in summary_by_model.items():
        logger.info("Walk-forward mean AUC [%s]: %s", mtype, metrics.get("mean_auc"))

    locked_auc = summary_by_model.get(LOCKED_MODEL_TYPE, {}).get("mean_auc")
    if locked_auc is None or not np.isfinite(float(locked_auc)):
        raise RetrainAbortedError(
            f"Walk-forward evaluation produced no 'mean_auc' for the locked '{LOCKED_MODEL_TYPE}' model "
            f"(evaluated types: {sorted(summary_by_model)}). No candidate selected or promoted."
        )
    new_auc = float(locked_auc)
    cand_path = b_path

    # 6. Extract regime performance of the locked model (missing window => gate not passed)
    gfc_result = _regime_result(eval_report, "2008")
    bear_result = _regime_result(eval_report, "2022")
    gfc_passed = gfc_result is not None and gfc_result[0] > gfc_result[1]
    bear_passed = bear_result is not None and bear_result[0] > bear_result[1]
    weighted_edge = round((new_auc - 0.50) * 100, 1)

    # 7. Get current active model ROC-AUC (never substitute a default)
    active_mod = load_active_model(models_dir=mdir)
    meta_auc = (active_mod.metadata or {}).get("preliminary_test_metrics", {}).get("roc_auc")
    if meta_auc is None or not np.isfinite(float(meta_auc)):
        raise RetrainAbortedError(
            f"Active model '{active_mod.model_name}' has no recorded ROC-AUC to compare against. "
            "No candidate promoted."
        )
    prev_auc = float(meta_auc)

    auc_diff = new_auc - prev_auc
    auc_diff_pct = (auc_diff / prev_auc * 100) if prev_auc > 0 else 0.0

    # 8. PSI Drift check — only a positive STABLE/MONITOR reading counts as "not drifting"
    drift_status = detect_prediction_drift()
    drift_label = drift_status.get("status", "UNKNOWN")
    drift_ok = drift_label in DRIFT_OK_STATUSES

    # 9. Safety rules evaluation & Promotion Decision
    min_imp = float(settings.auto_promote_min_improvement)
    # 2022 is the documented capital-preservation veto window (evaluate.py rubric), so it gates too.
    passes_safety = (auc_diff >= min_imp) and gfc_passed and bear_passed and drift_ok
    approval_pending = passes_safety and (
        settings.require_human_approval_for_promotion or settings.model_freeze_enabled
    )
    is_promoted = passes_safety and not approval_pending

    backup_saved_str = "models/backup/" if is_promoted and settings.auto_retrain_keep_backup else "N/A"

    if approval_pending:
        logger.warning(
            "HUMAN APPROVAL REQUIRED — %s passed safety checks but was not auto-promoted. "
            "Review results in dashboard and approve manually.", cand_path.name,
        )
        alert_msg = (
            f"🔄 MODEL RETRAINED — AWAITING HUMAN APPROVAL\n"
            f"Current ROC-AUC: {prev_auc:.3f}\n"
            f"Candidate ({LOCKED_MODEL_TYPE}) ROC-AUC: {new_auc:.3f}\n"
            f"Current model retained until approved ✅"
        )
    elif is_promoted:
        logger.info("SAFETY CHECKS PASSED — Auto-promoting new model %s (AUC: %.3f -> %.3f, +%.1f%%)",
                    cand_path.name, prev_auc, new_auc, auc_diff_pct)
        if settings.auto_retrain_keep_backup:
            backup_active_model(models_dir=mdir)

        promote_model(
            candidate_model_path=cand_path,
            author="Auto-Retrain Engine",
            reason=f"Monthly retrain auto-promotion: AUC {prev_auc:.3f} -> {new_auc:.3f} (+{auc_diff_pct:.1f}%)",
            models_dir=mdir,
        )

        alert_msg = (
            f"🔄 MODEL RETRAINED & PROMOTED\n"
            f"Old ROC-AUC: {prev_auc:.3f}\n"
            f"New ROC-AUC: {new_auc:.3f}\n"
            f"Improvement: {auc_diff_pct:+.1f}%\n"
            f"New model now active ✅"
        )
    else:
        logger.info("NO PROMOTION — Current active model retained (Old AUC: %.3f, New AUC: %.3f, Diff: %+.1f%%)",
                    prev_auc, new_auc, auc_diff_pct)
        alert_msg = (
            f"🔄 MODEL RETRAINED — NO PROMOTION\n"
            f"Current ROC-AUC: {prev_auc:.3f}\n"
            f"New ROC-AUC: {new_auc:.3f}\n"
            f"Current model retained ✅"
        )

    # 10. Send Telegram + Email Alerts
    try:
        notify(alert_msg)
    except Exception as exc:
        logger.warning("Telegram notification failed: %s", exc)

    status = "PROMOTED" if is_promoted else ("PENDING_APPROVAL" if approval_pending else "RETAINED")

    try:
        send_alert(
            subject=f"AI Stock Trader: Retrain {status} ({date_str})",
            body_text=alert_msg,
        )
    except Exception as exc:
        logger.warning("Email alert failed: %s", exc)
    # 11. Write Retraining Report to logs/retrain_YYYY-MM-DD.log
    report_text = f"""═══════════════════════════
AI STOCK TRADER — MONTHLY RETRAIN REPORT
Date: {date_str}
═══════════════════════════
Data: {start_date} to {end_date}
Stocks: {len(settings.ticker_list)} tickers
Training samples: {n_samples:,}
═══════════════════════════
TUNING RESULTS:
Primary best ROC-AUC: {_fmt_auc(primary_best_auc)} ({n_trials} trials)
XGBoost best ROC-AUC: {_fmt_auc(xgboost_best_auc)} ({n_trials} trials)
═══════════════════════════
EVALUATION RESULTS:
Candidate (locked model type): {LOCKED_MODEL_TYPE}
2008-2009 GFC: {_fmt_regime(gfc_result, gfc_passed)}
2022 Bear: {_fmt_regime(bear_result, bear_passed)}
Prediction drift: {drift_label} {"✅" if drift_ok else "❌"}
Overall weighted edge: {weighted_edge:+.1f} pts
═══════════════════════════
PROMOTION DECISION: {status}
Previous ROC-AUC: {prev_auc:.3f}
New ROC-AUC ({LOCKED_MODEL_TYPE}): {new_auc:.3f}
Improvement: {auc_diff_pct:+.1f}%
Backup saved: {backup_saved_str}
═══════════════════════════"""

    report_file = ldir / f"retrain_{date_str}.log"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Saved retrain report to %s", report_file)

    return {
        "status": status,
        "candidate_model_type": LOCKED_MODEL_TYPE,
        "candidate_path": str(cand_path),
        "date": date_str,
        "prev_auc": prev_auc,
        "new_auc": new_auc,
        "improvement_pct": round(auc_diff_pct, 2),
        "is_promoted": is_promoted,
        "report_file": str(report_file),
        "report_text": report_text,
    }


def main() -> None:
    """CLI: python -m src.ml.retrain [--force] [--trials 50]"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Monthly Automatic Model Retraining Scheduler Job.")
    parser.add_argument("--force", action="store_true", help="Force immediate execution regardless of schedule.")
    parser.add_argument("--trials", type=int, default=None, help="Override number of Optuna trials.")
    parser.add_argument("--date", type=str, default=None, help="Override date YYYY-MM-DD for run date.")
    args = parser.parse_args()

    result = run_automatic_retrain(
        run_date=args.date,
        force=args.force,
        optuna_trials=args.trials,
    )

    try:
        report_text = result.get("report_text", "")
        print("\n" + report_text.encode(
            "ascii", errors="replace"
        ).decode("ascii"))
    except Exception:
        print("\nRetrain complete. Check logs for details.")


if __name__ == "__main__":
    main()
