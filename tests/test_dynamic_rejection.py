"""
tests/test_dynamic_rejection.py — Test Dynamic Price Spike & Volume Rejection Gates (Section 1).
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.data.validation import check_price_spikes, validate_ticker_data


def _create_sample_ohlcv(n_bars: int = 50, base_price: float = 100.0, base_vol: float = 1_000_000.0) -> pd.DataFrame:
    """Helper to generate standard OHLCV dataframe with mild volatility (~1%)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B").strftime("%Y-%m-%d").tolist()
    closes = [base_price]
    for _ in range(n_bars - 1):
        ret = np.random.normal(0, 0.01)
        closes.append(round(closes[-1] * (1 + ret), 2))

    df = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [base_vol] * n_bars,
        "ticker": "TEST",
    })
    return df


def test_normal_volatility_passes():
    """Bars with typical 1-2% daily variations must pass cleanly."""
    df = _create_sample_ohlcv(n_bars=30)
    errors = check_price_spikes(df, "TEST")
    assert len(errors) == 0


def test_10x_volatility_spike_rejected():
    """
    If a stock has average daily move of ~0.8%, 10x volatility would be ~8%,
    capped at the 20% floor. A jump of 25% exceeds the floor and must be rejected.
    """
    df = _create_sample_ohlcv(n_bars=30, base_price=100.0)
    # Inject a 25% overnight spike on the last day
    df.loc[len(df) - 1, "close"] = 125.0
    df.loc[len(df) - 1, "high"] = 126.0

    errors = check_price_spikes(df, "TEST")
    assert len(errors) > 0
    assert any("dynamic price spike" in e for e in errors)


def test_suspicious_jump_with_normal_volume_rejected():
    """
    A price jump >= 25% without accompanying volume spike (volume <= 1.5x 20d avg)
    must be flagged as suspicious bad data / bad print.
    """
    df = _create_sample_ohlcv(n_bars=30, base_price=100.0, base_vol=1_000_000.0)
    last_idx = len(df) - 1
    # 26% jump with flat volume (1.0x avg volume <= 1.5x threshold)
    df.loc[last_idx, "close"] = 126.0
    df.loc[last_idx, "high"] = 127.0
    df.loc[last_idx, "volume"] = 1_000_000.0

    val_res = validate_ticker_data(df, ticker="TEST")
    assert val_res.is_valid is False
    assert any("suspicious price jump" in err and "normal volume" in err for err in val_res.errors)


def test_legitimate_catalyst_with_high_volume_evaluated_by_dynamic_threshold():
    """
    A price jump with massive volume surge (e.g. 5x volume on buyout/earnings)
    is NOT rejected by the normal-volume gate, though it still adheres to the dynamic spike ceiling.
    """
    df = _create_sample_ohlcv(n_bars=30, base_price=100.0, base_vol=1_000_000.0)
    # Adjust volatility so 10x threshold is high enough (~30%)
    for i in range(1, len(df) - 1):
        df.loc[i, "close"] = df.loc[i - 1, "close"] * (1.025 if i % 2 == 0 else 0.975)
    
    last_idx = len(df) - 1
    # 22% move with 5x volume surge
    df.loc[last_idx, "close"] = df.loc[last_idx - 1, "close"] * 1.22
    df.loc[last_idx, "volume"] = 5_000_000.0

    errors = check_price_spikes(df, "TEST")
    # Secondary normal-volume check should not trigger since volume surged 5x
    assert not any("normal volume" in e for e in errors)
