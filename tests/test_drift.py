"""
tests/test_drift.py — Statistical Correctness & Strict Isolation Tests for Model Drift (V2.1 Wave 1).

Tests:
1. PSI metric calculation on identical vs. shifted distributions.
2. KS-test statistic and p-value calculation.
3. Insufficient data handling (< 10 samples).
4. CRITICAL ISOLATION TEST: Drift detection NEVER modifies, overwrites, or promotes
   any model file, with both AST static analysis and binary hash invariance checks.
"""

import ast
import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.ml.drift import (
    PSI_DRIFT_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    calculate_ks_test,
    calculate_psi,
    detect_prediction_drift,
)


def test_psi_identical_distributions_is_stable():
    """Identical distributions must have PSI close to 0 (< 0.10, STABLE)."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.5, 0.1, 500)
    live = rng.normal(0.5, 0.1, 500)

    psi = calculate_psi(baseline, live)
    assert psi < PSI_STABLE_THRESHOLD
    assert psi >= 0.0


def test_psi_shifted_distributions_triggers_drift():
    """Heavily shifted distribution must have PSI >= 0.25 (DRIFT_ALERT)."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.3, 0.05, 500)
    live = rng.normal(0.7, 0.05, 500)

    psi = calculate_psi(baseline, live)
    assert psi >= PSI_DRIFT_THRESHOLD


def test_ks_test_detects_distribution_difference():
    """KS test p-value should be high for identical, and ~0 for different distributions."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.5, 0.1, 200)
    s2 = rng.normal(0.5, 0.1, 200)
    _, pval_same = calculate_ks_test(s1, s2)
    assert pval_same > 0.05

    s3 = rng.normal(0.8, 0.05, 200)
    stat_diff, pval_diff = calculate_ks_test(s1, s3)
    assert pval_diff < 0.001
    assert stat_diff > 0.5


def test_insufficient_data_handling():
    """If live samples < 10, returns INSUFFICIENT_DATA status gracefully."""
    few_preds = [0.52, 0.49, 0.51]
    res = detect_prediction_drift(live_probabilities=few_preds)
    assert res["status"] == "INSUFFICIENT_DATA"
    assert res["drift_detected"] is False
    assert res["sample_size_live"] == 3


def test_drift_never_modifies_model_isolation():
    """
    CRITICAL ARCHITECTURAL ISOLATION TEST:
    Verifies that drift detection is strictly observational and NEVER:
    1. Writes to or creates .joblib or model files.
    2. Imports or calls promote_model or joblib.dump.
    3. Mutates active_model.joblib or active_model_metadata.json (verified via SHA256 hashes).
    """
    drift_file = Path("src/ml/drift.py")
    assert drift_file.exists()

    # 1. AST Analysis of src/ml/drift.py
    with open(drift_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(drift_file))

    for node in ast.walk(tree):
        # Disallow imports of promote_model
        if isinstance(node, ast.ImportFrom):
            if node.module and "promote" in node.module:
                pytest.fail(f"Illegal import in drift.py: {node.module}")
            for alias in node.names:
                if alias.name in ("promote_model", "dump", "train_model"):
                    pytest.fail(f"Illegal import of '{alias.name}' in drift.py")

        # Disallow joblib.dump or shutil.copy / write operations
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("dump", "save_model"):
                    pytest.fail(f"Illegal call to '{node.func.attr}' found in drift.py")
                if node.func.attr in ("copy", "copyfile", "move") and getattr(node.func.value, "id", "") == "shutil":
                    pytest.fail("Illegal shutil copy/move operation found in drift.py")

    # 2. SHA256 Invariance of Active Model Artifacts before and after drift computation
    active_joblib = Path("data/models/active_model.joblib")
    active_json = Path("data/models/active_model_metadata.json")

    def file_hash(p: Path) -> str:
        if not p.exists():
            return ""
        return hashlib.sha256(p.read_bytes()).hexdigest()

    h_joblib_before = file_hash(active_joblib)
    h_json_before = file_hash(active_json)

    # Run drift detection multiple times across different scenarios
    _ = detect_prediction_drift(days=30)
    _ = detect_prediction_drift(live_probabilities=[0.75] * 50)  # Extreme drift simulation
    _ = detect_prediction_drift(live_probabilities=[0.50] * 50)  # Stable simulation

    h_joblib_after = file_hash(active_joblib)
    h_json_after = file_hash(active_json)

    assert h_joblib_before == h_joblib_after, "CRITICAL: active_model.joblib was modified during drift detection!"
    assert h_json_before == h_json_after, "CRITICAL: active_model_metadata.json was modified during drift detection!"
