"""
tests/test_macro.py — Unit tests for Section 5 Item 4: Macro Economic Indicators.

Covers:
  - classify_macro_regime (FAVORABLE / NEUTRAL / RESTRICTIVE)
  - Regime boundary conditions
  - MacroEnvironmentResult.to_dict() structure
  - get_macro_environment baseline fallback (no API key, no DB cache)
  - Correct size multiplier per regime (FAVORABLE=1.0, NEUTRAL=0.80, RESTRICTIVE=0.60)
  - Correct buy_bar_shift per regime (FAVORABLE=0.0, NEUTRAL=0.0, RESTRICTIVE=0.03)
  - Weekly cache cadence (Monday vs Tue-Fri) logic
  - FRED API graceful failure (falls back to baseline defaults)
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch, MagicMock
import pytest

from src.intelligence.macro import (
    classify_macro_regime,
    get_macro_environment,
    MacroEnvironmentResult,
    BASELINE_MACRO,
)


# ── classify_macro_regime tests ───────────────────────────────────────────────

def test_regime_favorable_low_rates_low_inflation():
    """Fed < 3%, CPI < 3% → FAVORABLE."""
    assert classify_macro_regime(fed_funds=2.5, cpi_yoy=2.0) == "FAVORABLE"


def test_regime_favorable_rates_falling():
    """Rates falling flag + CPI < 3% → FAVORABLE even if rate >= 3%."""
    assert classify_macro_regime(fed_funds=3.2, cpi_yoy=2.8, rates_falling=True) == "FAVORABLE"


def test_regime_neutral_mid_rates_mid_inflation():
    """Fed 3-5%, CPI 3-5% → NEUTRAL."""
    assert classify_macro_regime(fed_funds=4.0, cpi_yoy=3.5) == "NEUTRAL"


def test_regime_neutral_high_rates_low_inflation():
    """Fed > 5% but CPI <= 5% → not RESTRICTIVE → NEUTRAL."""
    assert classify_macro_regime(fed_funds=5.5, cpi_yoy=4.0) == "NEUTRAL"


def test_regime_restrictive_both_high():
    """Fed > 5% AND CPI > 5% → RESTRICTIVE."""
    assert classify_macro_regime(fed_funds=5.5, cpi_yoy=6.0) == "RESTRICTIVE"


def test_regime_restrictive_boundary():
    """Exactly at 5.01% fed and 5.01% CPI → RESTRICTIVE."""
    assert classify_macro_regime(fed_funds=5.01, cpi_yoy=5.01) == "RESTRICTIVE"


def test_regime_not_restrictive_if_only_rates_high():
    """Fed > 5% but CPI <= 5% → NOT restrictive."""
    assert classify_macro_regime(fed_funds=6.0, cpi_yoy=4.5) != "RESTRICTIVE"


def test_regime_not_restrictive_if_only_cpi_high():
    """CPI > 5% but Fed <= 5% → NOT restrictive."""
    assert classify_macro_regime(fed_funds=4.0, cpi_yoy=6.0) != "RESTRICTIVE"


# ── MacroEnvironmentResult.to_dict() tests ────────────────────────────────────

def test_macro_result_to_dict_keys():
    result = MacroEnvironmentResult(
        date="2024-01-15",
        fed_funds_rate=4.5,
        cpi_yoy=3.2,
        unemployment_rate=3.8,
        treasury_10y=4.1,
        macro_regime="NEUTRAL",
        position_size_multiplier=0.80,
        buy_bar_shift=0.0,
        is_cached=True,
        source_date="2024-01-15",
    )
    d = result.to_dict()
    expected_keys = [
        "date", "fed_funds_rate", "cpi_yoy", "unemployment_rate",
        "treasury_10y", "macro_regime", "position_size_multiplier",
        "buy_bar_shift", "is_cached", "source_date",
    ]
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"


# ── get_macro_environment tests ───────────────────────────────────────────────

def test_get_macro_environment_baseline_fallback_no_api_key():
    """
    When FRED_API_KEY is empty and no DB cache exists,
    get_macro_environment should return baseline defaults (no crash).
    """
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            result = get_macro_environment("2024-01-15", persist=False)

    assert isinstance(result, MacroEnvironmentResult)
    assert result.fed_funds_rate == BASELINE_MACRO["fed_funds_rate"]
    assert result.cpi_yoy == BASELINE_MACRO["cpi_yoy"]
    assert result.is_cached is True


def test_get_macro_environment_neutral_sizing():
    """NEUTRAL regime → position_size_multiplier == 0.80."""
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.BASELINE_MACRO", {
                "fed_funds_rate": 4.0,
                "cpi_yoy": 3.5,
                "unemployment_rate": 4.0,
                "treasury_10y": 4.0,
                "macro_regime": "NEUTRAL",
            }):
                result = get_macro_environment("2024-01-15", persist=False)

    # With fed=4.0 and cpi=3.5, regime should be NEUTRAL
    assert result.macro_regime == "NEUTRAL"
    assert result.position_size_multiplier == pytest.approx(0.80, abs=1e-9)
    assert result.buy_bar_shift == pytest.approx(0.0, abs=1e-9)


def test_get_macro_environment_restrictive_sizing():
    """RESTRICTIVE regime → multiplier == 0.60 and buy_bar_shift == 0.03."""
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.BASELINE_MACRO", {
                "fed_funds_rate": 6.0,
                "cpi_yoy": 7.0,
                "unemployment_rate": 3.5,
                "treasury_10y": 5.5,
                "macro_regime": "RESTRICTIVE",
            }):
                result = get_macro_environment("2024-01-15", persist=False)

    assert result.macro_regime == "RESTRICTIVE"
    assert result.position_size_multiplier == pytest.approx(0.60, abs=1e-9)
    assert result.buy_bar_shift == pytest.approx(0.03, abs=1e-9)


def test_get_macro_environment_favorable_sizing():
    """FAVORABLE regime → multiplier == 1.0 and buy_bar_shift == 0.0."""
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.BASELINE_MACRO", {
                "fed_funds_rate": 2.0,
                "cpi_yoy": 1.5,
                "unemployment_rate": 3.5,
                "treasury_10y": 2.5,
                "macro_regime": "FAVORABLE",
            }):
                result = get_macro_environment("2024-01-15", persist=False)

    assert result.macro_regime == "FAVORABLE"
    assert result.position_size_multiplier == pytest.approx(1.0, abs=1e-9)
    assert result.buy_bar_shift == pytest.approx(0.0, abs=1e-9)


def test_get_macro_environment_uses_db_cache():
    """When DB has a cached record, it should be used without calling FRED."""
    cached_data = {
        "date": "2024-01-10",
        "fed_funds_rate": 5.25,
        "cpi_yoy": 3.1,
        "unemployment_rate": 3.9,
        "treasury_10y": 4.4,
        "macro_regime": "NEUTRAL",
        "fetched_date": "2024-01-10",
    }
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=cached_data):
            with patch("src.intelligence.macro.fetch_fred_series_latest", side_effect=AssertionError("Should not call FRED")):
                result = get_macro_environment("2024-01-15", persist=False)

    assert result.fed_funds_rate == pytest.approx(5.25, abs=1e-9)
    assert result.is_cached is True


def test_get_macro_environment_fred_exception_falls_to_baseline():
    """FRED API network failure → falls back to baseline defaults cleanly."""
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = "FAKE_KEY_TO_TRIGGER_FETCH"
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.fetch_fred_series_latest", return_value=None):
                with patch("src.intelligence.macro.fetch_fred_cpi_yoy", return_value=None):
                    # Monday forces fetch attempt
                    result = get_macro_environment("2024-01-08", force_fetch=True, persist=False)

    assert result is not None
    assert isinstance(result.macro_regime, str)
    assert result.is_cached is True


def test_get_macro_environment_no_crash_on_monday():
    """Monday with no API key → gracefully uses baseline without error."""
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            result = get_macro_environment("2024-01-08", persist=False)  # Monday

    assert isinstance(result, MacroEnvironmentResult)
    assert result.macro_regime in ("FAVORABLE", "NEUTRAL", "RESTRICTIVE")


def test_historical_replay_no_live_macro_leakage():
    """
    Historical replay dates must NOT call live FRED APIs even on Mondays.
    If no point-in-time record exists, must return baseline defaults without hitting live network.
    If point-in-time DB cache exists, must consume only that historical record.
    If force_fetch is requested, observation_end must strictly bound observations on or before simulated date.
    """
    historical_monday = "2023-05-15"  # Monday

    # 1. Historical Monday with FRED API key but no DB cache -> must NOT call FRED without force_fetch
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = "MOCK_KEY"
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.fetch_fred_series_latest") as mock_fetch:
                result = get_macro_environment(historical_monday, force_fetch=False, persist=False)
                mock_fetch.assert_not_called()

    assert result.macro_regime == "NEUTRAL"
    assert result.fed_funds_rate == BASELINE_MACRO["fed_funds_rate"]
    assert result.cpi_yoy == BASELINE_MACRO["cpi_yoy"]
    assert result.is_cached is True
    assert result.source_date == "BASELINE_DEFAULT"

    # 2. When force_fetch=True, observation_end must be passed to bound observations
    with patch("src.intelligence.macro.settings") as mock_settings:
        mock_settings.fred_api_key = "MOCK_KEY"
        mock_settings.macro_neutral_size_multiplier = 0.80
        mock_settings.macro_restrictive_size_multiplier = 0.60
        mock_settings.macro_restrictive_buy_bar_shift = 0.03
        with patch("src.intelligence.macro.repository.get_latest_macro_indicators", return_value=None):
            with patch("src.intelligence.macro.fetch_fred_series_latest", return_value=4.5) as mock_fetch:
                with patch("src.intelligence.macro.fetch_fred_cpi_yoy", return_value=2.8):
                    get_macro_environment(historical_monday, force_fetch=True, persist=False)
                    # Verify observation_end was passed to strictly prevent leakage
                    for call in mock_fetch.call_args_list:
                        assert call.kwargs.get("observation_end") == historical_monday
