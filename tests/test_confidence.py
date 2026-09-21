"""
tests/test_confidence.py — Tests for Calibrated Confidence & Decision Rule Thresholds (V2.1 Wave 1).

Tests:
1. Only 0.60 (buy bar) and 0.45 (signal exit) are actionable decision boundaries.
2. [0.55, 0.60) tier is strictly informational commentary, not an actionable buy signal.
3. MIN_CONVICTION_SAMPLE_SIZE = 30 honesty check: any empirical win rate with N < 30
   is explicitly tagged "LOW CONFIDENCE".
"""

import pytest

from config.settings import settings
from src.ml.evaluate import (
    MIN_CONVICTION_SAMPLE_SIZE,
    format_calibrated_confidence,
    get_calibration_table,
)


def test_actionable_decision_thresholds_strictly_locked():
    """
    CRITICAL RULE: Only 0.60 (buy bar) and 0.45 (signal exit) are actionable.
    No intermediate tier (such as 0.55) is allowed to be actionable.
    """
    table = get_calibration_table()

    # 1. Bucket [0.60, 1.00]
    b_buy = table[0]
    assert b_buy["is_actionable_buy"] is True
    assert b_buy["is_actionable_exit"] is False

    # 2. Bucket [0.55, 0.60)
    b_sub = table[1]
    assert b_sub["is_actionable_buy"] is False
    assert b_sub["is_actionable_exit"] is False
    assert "NOT A BUY SIGNAL" in b_sub["decision_rule"]

    # 3. Bucket [0.45, 0.55)
    b_neutral = table[2]
    assert b_neutral["is_actionable_buy"] is False
    assert b_neutral["is_actionable_exit"] is False

    # 4. Bucket [0.00, 0.45)
    b_exit = table[3]
    assert b_exit["is_actionable_buy"] is False
    assert b_exit["is_actionable_exit"] is True


def test_min_sample_size_honesty_check():
    """
    Any empirical win-rate shown with sample size N < 30 must be tagged LOW CONFIDENCE
    per MIN_CONVICTION_SAMPLE_SIZE = 30.
    """
    table = get_calibration_table()

    for bucket in table:
        n = bucket["sample_size"]
        conf_tag = bucket["sample_confidence"]
        if n < MIN_CONVICTION_SAMPLE_SIZE:
            assert "LOW CONFIDENCE" in conf_tag, f"Bucket {bucket['probability_range']} with N={n} missing LOW CONFIDENCE tag!"
        else:
            assert "HIGH" in conf_tag


def test_format_calibrated_confidence_mapping():
    """Verifies that format_calibrated_confidence maps predictions accurately."""
    # Conviction Buy
    c_high = format_calibrated_confidence(0.68)
    assert c_high["is_actionable_buy"] is True
    assert c_high["is_actionable_exit"] is False
    assert c_high["sample_size"] == 44

    # Leaning / Sub-tier (0.57) — strictly not a buy signal
    c_mid_high = format_calibrated_confidence(0.57)
    assert c_mid_high["is_actionable_buy"] is False
    assert c_mid_high["is_actionable_exit"] is False
    assert "NOT A BUY SIGNAL" in c_mid_high["decision_rule"]
    assert "LOW CONFIDENCE" in c_mid_high["sample_confidence"]

    # Dead zone neutral (0.50)
    c_neutral = format_calibrated_confidence(0.50)
    assert c_neutral["is_actionable_buy"] is False
    assert c_neutral["is_actionable_exit"] is False

    # Exit candidate (0.38)
    c_exit = format_calibrated_confidence(0.38)
    assert c_exit["is_actionable_buy"] is False
    assert c_exit["is_actionable_exit"] is True
