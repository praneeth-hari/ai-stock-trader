"""
tests/test_features.py — Phase 3 feature engineering verification.

Test structure:
  TestKnownValues  — exact numerical assertions computed by hand
  TestLeakage      — five adversarial leakage proofs
  TestNaNHandling  — NaN policy, drop_warmup_rows, has_all_features

LEAKAGE TESTS EXPLAINED:
  1. Future Blackout   — appending one extreme future row must not change any past feature
  2. Append-Recompute  — adding N rows must not retroactively alter previously-computed values
  3. Label Independence— modifying close values beyond the feature date must not alter features
  4. Benchmark Join    — SPY roc_21 must join by date string, not row position
  5. Multi-ticker Isolation — two tickers' features must not contaminate each other
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    FEATURE_COLUMNS,
    compute_features,
    drop_warmup_rows,
    has_all_features,
)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_ohlcv(
    prices: List[float],
    dates: Optional[List[str]] = None,
    ticker: str = "TEST",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from a close-price list.
    High = close * 1.005, Low = close * 0.990, Open = close * 0.995.
    """
    n = len(prices)
    if dates is None:
        dates = pd.bdate_range(start="2020-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": [p * 0.995 for p in prices],
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.990 for p in prices],
        "close": prices,
        "volume": [volume] * n,
    })


def _arithmetic_prices(n: int, start: float = 100.0, step: float = 1.0) -> List[float]:
    """Return n prices in arithmetic progression: [start, start+step, ...]."""
    return [start + i * step for i in range(n)]


# ── Known-value tests ─────────────────────────────────────────────────────────

class TestKnownValues:
    """
    Exact numerical assertions computed by hand.
    Uses an arithmetic price series (100, 101, 102, …) where the correct
    feature values are derivable analytically without running any trading library.
    """

    @pytest.fixture
    def arith_df(self):
        """
        30 rows of prices 100..129, constant volume 1,000,000.
        Enough for: ma_5 (needs 5), ma_20 (needs 20), roc_5 (needs 6),
        roc_21 (needs 22), rsi_14 (needs 15), vol_spike (needs 20).
        Not enough for ma_200 — those rows will be NaN (verified separately).
        """
        return _make_ohlcv(_arithmetic_prices(30, start=100.0))

    def test_price_to_ma20_exact_value(self, arith_df):
        """
        price_to_ma20 at row 19 (0-indexed):
        close[19] = 119.0, ma_20 = mean(100..119) = 109.5.
        price_to_ma20 = (119.0 / 109.5) - 1.
        Rows 0-18 must be NaN.
        """
        feat = compute_features(arith_df)
        assert feat["price_to_ma20"].iloc[:20].isna().all(), "price_to_ma20 rows 0-19 must be NaN"
        expected = (119.0 / 109.5) - 1.0
        assert abs(feat["price_to_ma20"].iloc[20] - expected) < 1e-9, (
            f"price_to_ma20 at row 20: expected {expected}, got {feat['price_to_ma20'].iloc[20]}"
        )

    def test_roc_5_exact_value(self, arith_df):
        """
        roc_5 at row 5: (105 - 100) / 100 = 0.05 exactly.
        roc_5 is NaN for rows 0-4.
        """
        feat = compute_features(arith_df)
        assert feat["roc_5"].iloc[:6].isna().all(), "roc_5 rows 0-5 must be NaN"
        assert abs(feat["roc_5"].iloc[6] - 0.05) < 1e-9, (
            f"roc_5 at row 6: expected 0.05, got {feat['roc_5'].iloc[6]}"
        )

    def test_roc_21_exact_value(self, arith_df):
        """
        roc_21 at row 21: (121 - 100) / 100 = 0.21 exactly.
        roc_21 is NaN for rows 0-20.
        """
        feat = compute_features(arith_df)
        assert feat["roc_21"].iloc[:22].isna().all(), "roc_21 rows 0-21 must be NaN"
        assert abs(feat["roc_21"].iloc[22] - 0.21) < 1e-9, (
            f"roc_21 at row 22: expected 0.21, got {feat['roc_21'].iloc[22]}"
        )

    def test_vol_spike_constant_volume(self, arith_df):
        """
        With constant volume = 1,000,000 and 1-day lag:
        vol_spike = volume_lag1 / mean(volume_lag1, 20) = 1.0.
        vol_spike is NaN for rows 0-19 (1-day lag + 19 more for 20-period rolling window).
        """
        feat = compute_features(arith_df)
        assert feat["vol_spike"].iloc[:20].isna().all(), "vol_spike rows 0-19 must be NaN"
        # Rows 20-29: all volumes equal → vol_spike = 1.0 exactly
        valid_spikes = feat["vol_spike"].iloc[20:]
        assert (valid_spikes.notna()).all()
        assert (abs(valid_spikes - 1.0) < 1e-9).all(), (
            f"vol_spike expected 1.0 with constant volume, got: {valid_spikes.tolist()}"
        )

    def test_vol_ratio_5_20_constant_volume(self, arith_df):
        """
        With constant volume: vol_ratio_5_20 = mean(vol,5) / mean(vol,20) = 1.0.
        """
        feat = compute_features(arith_df)
        valid = feat["vol_ratio_5_20"].dropna()
        assert len(valid) > 0
        assert (abs(valid - 1.0) < 1e-9).all(), (
            f"vol_ratio_5_20 expected 1.0 with constant volume, got non-1.0 values"
        )

    def test_rsi_all_gains_converges_to_100(self):
        """
        Monotonically rising prices → all changes are gains → avg_loss → 0
        → RSI = 100 − 100/(1+inf) = 100.
        With Wilder's EWM, this holds from row 14 onward (min_periods=14 satisfied).
        """
        prices = _arithmetic_prices(40, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        rsi_valid = feat["rsi_14"].dropna()
        assert len(rsi_valid) > 0, "RSI should have valid values for 40-row series"
        # All valid RSI values should be ≈ 100 (monotonic price increase)
        assert (abs(rsi_valid - 100.0) < 1e-6).all(), (
            f"RSI with all gains must be 100.0, got: {rsi_valid.tolist()}"
        )

    def test_macd_positive_in_sustained_uptrend(self):
        """
        In a sustained uptrend: EMA-12 > EMA-26 → macd_line > 0.
        First valid macd_signal at row 33 (26 for ema_26 + 9 - 1 for signal, 0-indexed row 33).
        """
        prices = _arithmetic_prices(60, start=100.0)
        feat = compute_features(_make_ohlcv(prices))

        # macd_line NaN for rows 0-25, valid from row 26
        assert feat["macd_line"].iloc[:26].isna().all(), "macd_line rows 0-25 must be NaN"
        assert feat["macd_line"].iloc[26:].notna().all(), "macd_line from row 26 must be valid"

        # All valid macd_line values should be positive (EMA-12 > EMA-26 in uptrend)
        valid_macd = feat["macd_line"].iloc[26:]
        assert (valid_macd > 0).all(), f"macd_line must be >0 in uptrend, got negatives"

        # macd_signal: NaN until row 33 (0-indexed), valid from row 34 onward
        # (ema_26 valid at row 26; need 9 more macd_line values → valid at row 26+8=34)
        assert feat["macd_signal"].iloc[:34].isna().all(), "macd_signal rows 0-33 must be NaN"
        assert feat["macd_signal"].iloc[34:].notna().all()

    def test_price_to_ma200_nan_until_200_rows(self):
        """
        price_to_ma200 requires ma_200. With only 210 rows, first valid at row 199.
        """
        prices = _arithmetic_prices(210, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        assert feat["price_to_ma200"].iloc[:200].isna().all(), (
            "price_to_ma200 rows 0-199 must be NaN (insufficient ma_200 history)"
        )
        assert feat["price_to_ma200"].iloc[200:].notna().all(), (
            "price_to_ma200 from row 200 must be valid (200-day MA available)"
        )

    def test_ma5_above_ma20_nan_when_either_nan(self, arith_df):
        """
        ma5_above_ma20 must be NaN when either ma_5 or ma_20 is NaN.
        For an arithmetic series 100..129: ma_5 valid from row 4, ma_20 from row 19.
        So ma5_above_ma20 should be NaN for rows 0-18 and {0.0, 1.0} from row 19.
        In a rising series, ma_5 > ma_20 → value = 1.0.
        """
        feat = compute_features(arith_df)
        assert feat["ma5_above_ma20"].iloc[:20].isna().all(), (
            "ma5_above_ma20 rows 0-19 must be NaN (ma_20 not yet valid)"
        )
        valid = feat["ma5_above_ma20"].iloc[20:]
        assert valid.notna().all()
        # Arithmetic series: ma_5 (trailing 5) > ma_20 (trailing 20) since prices are rising
        assert (valid == 1.0).all(), f"ma5_above_ma20 should be 1.0 in uptrend, got: {valid.tolist()}"


# ── Leakage tests ─────────────────────────────────────────────────────────────

class TestLeakage:
    """
    Five adversarial leakage proofs.
    These tests FAIL if future data contaminates past feature values.
    """

    def test_1_future_blackout(self):
        """
        LEAKAGE TEST 1 — Future Blackout (the critical test).

        Protocol:
          1. Compute features on N rows of flat prices.
          2. Append ONE row with an extreme price (99999.0).
          3. Recompute features on the N+1 row series.
          4. Assert features for rows 0..N-1 are IDENTICAL in both runs.

        If any rolling/EWM operation reads ahead, rows in the first N will differ.
        """
        N = 250   # enough for all features including ma_200 to have valid values

        # Build N rows at flat price 100.0 (no trend, distinctive from extreme)
        prices_base = [100.0] * N
        df_base = _make_ohlcv(prices_base)
        feat_base = compute_features(df_base)

        # Append one extreme row (an impossible future price)
        extreme_price = 99999.0
        extreme_date = pd.bdate_range(
            start=pd.Timestamp(df_base["date"].iloc[-1]) + pd.Timedelta(days=1),
            periods=1
        ).strftime("%Y-%m-%d")[0]
        extreme_row = _make_ohlcv([extreme_price], dates=[extreme_date])
        df_extended = pd.concat([df_base, extreme_row], ignore_index=True)
        feat_extended = compute_features(df_extended)

        # Features for rows 0..N-1 must be IDENTICAL
        feat_past_base = feat_base.drop(columns=["date", "ticker"]).reset_index(drop=True)
        feat_past_ext = feat_extended.iloc[:N].drop(columns=["date", "ticker"]).reset_index(drop=True)

        for col in FEATURE_COLUMNS:
            base_vals = feat_past_base[col].values
            ext_vals = feat_past_ext[col].values
            # Compare NaN positions match
            assert np.array_equal(np.isnan(base_vals), np.isnan(ext_vals)), (
                f"LEAKAGE DETECTED in '{col}': NaN positions changed after appending future row."
            )
            # Compare non-NaN values
            mask = ~np.isnan(base_vals)
            if mask.any():
                max_diff = np.abs(base_vals[mask] - ext_vals[mask]).max()
                assert max_diff < 1e-10, (
                    f"LEAKAGE DETECTED in '{col}': max change = {max_diff:.2e} after appending "
                    f"extreme future row ({extreme_price}). Past features must not change."
                )

    def test_2_append_and_recompute_stability(self):
        """
        LEAKAGE TEST 2 — Append-and-Recompute Stability.

        Compute features on N rows. Then append 20 new rows and recompute.
        The first N rows' features must be bit-for-bit identical in both outputs.

        This catches EMA or stateful initialization bugs where the computation
        is seeded differently depending on the total series length.
        """
        N = 260

        prices_initial = _arithmetic_prices(N, start=100.0)
        df_initial = _make_ohlcv(prices_initial)
        feat_initial = compute_features(df_initial)

        # Append 20 new rows at a different price level
        extra_prices = _arithmetic_prices(20, start=prices_initial[-1] + 1.0)
        extra_dates = pd.bdate_range(
            start=pd.Timestamp(df_initial["date"].iloc[-1]) + pd.Timedelta(days=1),
            periods=20
        ).strftime("%Y-%m-%d").tolist()
        df_extended = pd.concat(
            [df_initial, _make_ohlcv(extra_prices, dates=extra_dates)],
            ignore_index=True,
        )
        feat_extended = compute_features(df_extended)

        # Rows 0..N-1 must be identical
        for col in FEATURE_COLUMNS:
            base_vals = feat_initial[col].values
            ext_vals = feat_extended[col].values[:N]
            nan_match = np.array_equal(np.isnan(base_vals), np.isnan(ext_vals))
            assert nan_match, (
                f"LEAKAGE TEST 2 FAILED in '{col}': NaN positions changed after appending rows."
            )
            mask = ~np.isnan(base_vals)
            if mask.any():
                max_diff = np.abs(base_vals[mask] - ext_vals[mask]).max()
                assert max_diff < 1e-10, (
                    f"LEAKAGE TEST 2 FAILED in '{col}': max change = {max_diff:.2e} "
                    f"after appending 20 rows. Past features must not change."
                )

    def test_3_label_independence_from_future_closes(self):
        """
        LEAKAGE TEST 3 — Label Independence.

        Phase 4 label = does close[D+5] exceed close[D] by >1%?
        This test verifies that modifying future close values (which would be
        used to compute training labels) does NOT affect any feature for rows
        before those modified dates.

        Protocol:
          1. Compute features on 50-row series.
          2. Replace the last 10 rows' close values with random extreme values.
          3. Recompute features.
          4. Features for rows 0..39 must be identical.
        """
        prices = _arithmetic_prices(50, start=100.0)
        df_base = _make_ohlcv(prices)
        feat_base = compute_features(df_base)

        # Replace last 10 closes with extreme random values (simulating label-window data)
        rng = np.random.default_rng(seed=42)
        df_modified = df_base.copy()
        df_modified.loc[40:, "close"] = rng.uniform(50.0, 500.0, size=10)
        df_modified.loc[40:, "high"] = df_modified.loc[40:, "close"] * 1.005
        df_modified.loc[40:, "low"] = df_modified.loc[40:, "close"] * 0.990
        feat_modified = compute_features(df_modified)

        CHECK_ROWS = 39   # rows 0-38: data before modified closes
        for col in FEATURE_COLUMNS:
            base_vals = feat_base[col].values[:CHECK_ROWS]
            mod_vals = feat_modified[col].values[:CHECK_ROWS]
            nan_match = np.array_equal(np.isnan(base_vals), np.isnan(mod_vals))
            assert nan_match, (
                f"LEAKAGE TEST 3 FAILED in '{col}': NaN positions changed when future "
                f"closes were modified."
            )
            mask = ~np.isnan(base_vals)
            if mask.any():
                max_diff = np.abs(base_vals[mask] - mod_vals[mask]).max()
                assert max_diff < 1e-10, (
                    f"LEAKAGE TEST 3 FAILED in '{col}': max change = {max_diff:.2e} "
                    f"when future close values were modified. Feature computation is "
                    f"leaking future label data."
                )

    def test_4_benchmark_join_by_date_not_position(self):
        """
        LEAKAGE TEST 4 — Benchmark Join Date Alignment.

        rel_strength_21 = ticker_roc_21 − spy_roc_21.
        The SPY roc_21 must be joined by date string, not by row position.

        Part A: Identical ticker and SPY prices on same dates → rel_strength_21 = 0.0
                (both roc_21 are equal, so difference is zero).
        Part B: SPY data on completely non-overlapping future dates → all NaN.
                A position-based join would give wrong non-NaN values here.
        """
        prices = _arithmetic_prices(40, start=100.0)
        dates = pd.bdate_range("2020-01-02", periods=40).strftime("%Y-%m-%d").tolist()

        ticker_df = _make_ohlcv(prices, dates=dates, ticker="AAPL")
        spy_df_same = _make_ohlcv(prices, dates=dates, ticker="SPY")

        # Part A: Same prices → rel_strength_21 = 0 wherever both are valid
        feat_a = compute_features(ticker_df, spy_df=spy_df_same)
        valid_rs = feat_a["rel_strength_21"].dropna()
        assert len(valid_rs) > 0, "rel_strength_21 should have valid rows when dates overlap"
        assert (valid_rs.abs() < 1e-9).all(), (
            f"Part A: rel_strength_21 must be ~0 when ticker and SPY have identical prices. "
            f"Got non-zero: {valid_rs[valid_rs.abs() >= 1e-9].tolist()}"
        )

        # Part B: SPY on entirely different (future) dates → all NaN
        future_dates = pd.bdate_range("2030-01-02", periods=40).strftime("%Y-%m-%d").tolist()
        spy_df_future = _make_ohlcv(prices, dates=future_dates, ticker="SPY")

        feat_b = compute_features(ticker_df, spy_df=spy_df_future)
        # A position-based join would return non-NaN values by matching row 0 to row 0.
        # The correct date-based join finds no matching dates → all NaN.
        assert feat_b["rel_strength_21"].isna().all(), (
            f"Part B: rel_strength_21 must be NaN when SPY dates don't overlap ticker dates. "
            f"Got {feat_b['rel_strength_21'].notna().sum()} non-NaN values — "
            f"this indicates a position-based join (leakage bug)."
        )

    def test_5_multi_ticker_isolation(self):
        """
        LEAKAGE TEST 5 — Multi-Ticker Isolation.

        When two tickers are processed together, features for ticker A must be
        identical to what they would be if ticker A were processed alone.
        No cross-ticker contamination via rolling windows or shared state.
        """
        prices_aapl = _arithmetic_prices(50, start=100.0, step=1.0)
        prices_msft = _arithmetic_prices(50, start=200.0, step=2.0)

        df_aapl = _make_ohlcv(prices_aapl, ticker="AAPL")
        df_msft = _make_ohlcv(prices_msft, ticker="MSFT")

        # Solo computation
        feat_aapl_solo = compute_features(df_aapl)

        # Joint computation
        df_combined = pd.concat([df_aapl, df_msft], ignore_index=True)
        feat_combined = compute_features(df_combined)
        feat_aapl_joint = feat_combined[feat_combined["ticker"] == "AAPL"].reset_index(drop=True)

        # AAPL features must be identical solo vs joint
        for col in FEATURE_COLUMNS:
            solo_vals = feat_aapl_solo[col].values
            joint_vals = feat_aapl_joint[col].values
            nan_match = np.array_equal(np.isnan(solo_vals), np.isnan(joint_vals))
            assert nan_match, (
                f"LEAKAGE TEST 5 FAILED in '{col}': AAPL NaN positions differ "
                f"when computed solo vs jointly with MSFT."
            )
            mask = ~np.isnan(solo_vals)
            if mask.any():
                max_diff = np.abs(solo_vals[mask] - joint_vals[mask]).max()
                assert max_diff < 1e-10, (
                    f"LEAKAGE TEST 5 FAILED in '{col}': AAPL feature changed by {max_diff:.2e} "
                    f"when MSFT was added to the batch — cross-ticker contamination."
                )

    def test_6_cross_sectional_scale_invariance(self):
        """
        LEAKAGE TEST 6 — Cross-Sectional Scale Invariance.

        Verify that two tickers with identical relative technical setups
        (same RSI, same relative MA distances, same momentum, same volatility)
        at radically different nominal price levels (e.g. $50 vs $500, a 10x ratio)
        produce IDENTICAL feature values and IDENTICAL model probabilities.

        This ensures the model cannot use nominal share price as an unintended
        proxy feature or suffer from feature distribution shift across tickers.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        n = 250
        dates = pd.bdate_range(start="2020-01-02", periods=n).strftime("%Y-%m-%d").tolist()

        # Generate a synthetic path with realistic price action
        daily_returns = np.random.normal(loc=0.0005, scale=0.015, size=n)
        cum_ret = np.cumprod(1.0 + daily_returns)

        # Stock A: $50 base price
        prices_a = 50.0 * cum_ret
        # Stock B: $500 base price (exactly 10x Stock A)
        prices_b = 500.0 * cum_ret

        # Synthetic SPY benchmark
        spy_returns = np.random.normal(loc=0.0003, scale=0.012, size=n)
        spy_prices = 300.0 * np.cumprod(1.0 + spy_returns)
        spy_df = _make_ohlcv(spy_prices.tolist(), dates=dates, ticker="SPY")

        df_a = _make_ohlcv(prices_a.tolist(), dates=dates, ticker="STOCK_50", volume=1_000_000.0)
        df_b = _make_ohlcv(prices_b.tolist(), dates=dates, ticker="STOCK_500", volume=1_000_000.0)

        feat_a = compute_features(df_a, spy_df=spy_df)
        feat_b = compute_features(df_b, spy_df=spy_df)

        clean_a = drop_warmup_rows(feat_a).reset_index(drop=True)
        clean_b = drop_warmup_rows(feat_b).reset_index(drop=True)

        assert len(clean_a) > 0, "Expected valid clean feature rows for Stock A"
        assert len(clean_a) == len(clean_b), "Stock A and B should produce equal valid rows"

        # 1. Assert all 15 features are IDENTICAL across the $50 and $500 setups
        for col in FEATURE_COLUMNS:
            diff = np.abs(clean_a[col].values - clean_b[col].values)
            max_diff = np.nanmax(diff)
            assert max_diff < 1e-6, (
                f"SCALE CONTAMINATION in feature '{col}': max diff = {max_diff:.2e} "
                f"between $50 and $500 price levels. All features must be scale-invariant."
            )

        # 2. Assert model probabilities are IDENTICAL for both tickers
        rng = np.random.default_rng(42)
        dummy_X = pd.DataFrame(rng.normal(size=(100, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        dummy_y = rng.choice([0, 1], size=100)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ])
        pipeline.fit(dummy_X, dummy_y)

        probs_a = pipeline.predict_proba(clean_a[FEATURE_COLUMNS])[:, 1]
        probs_b = pipeline.predict_proba(clean_b[FEATURE_COLUMNS])[:, 1]

        prob_diff = np.abs(probs_a - probs_b)
        max_prob_diff = np.max(prob_diff)
        assert max_prob_diff < 1e-6, (
            f"Model produced different probabilities for $50 vs $500 setups: "
            f"max diff = {max_prob_diff:.2e}. Scale invariance violated!"
        )

    def test_7_volume_temporal_alignment_no_current_day_leakage(self):
        """
        LEAKAGE TEST 7 — Volume Temporal Alignment (Zero Current-Day Volume Leakage).

        Features at date D must use volume up through D-1, matching the 1-day lag
        applied to price features. Modifying the current day's (last row's) volume
        must NOT change any feature on the current day.
        """
        N = 35
        prices = _arithmetic_prices(N, start=100.0)
        df_base = _make_ohlcv(prices, volume=1_000_000.0)
        feat_base = compute_features(df_base)

        # Mutate ONLY the final row's volume (an extreme spike on today's uncompleted bar)
        df_mutated = df_base.copy()
        df_mutated.loc[N - 1, "volume"] = 999_999_999.0
        feat_mutated = compute_features(df_mutated)

        # The final row's volume features (vol_spike, vol_ratio_5_20) must be identical
        # because they only observe volume up through day N-2 (yesterday's bar).
        assert feat_base["vol_spike"].iloc[N - 1] == feat_mutated["vol_spike"].iloc[N - 1], (
            "LEAKAGE DETECTED: Changing today's volume altered today's vol_spike feature!"
        )
        assert feat_base["vol_ratio_5_20"].iloc[N - 1] == feat_mutated["vol_ratio_5_20"].iloc[N - 1], (
            "LEAKAGE DETECTED: Changing today's volume altered today's vol_ratio_5_20 feature!"
        )


# ── NaN handling tests ────────────────────────────────────────────────────────

class TestNaNHandling:
    """Verify strict NaN policy and the drop_warmup_rows / has_all_features utilities."""

    def test_price_to_ma200_nan_for_first_199_rows(self):
        """price_to_ma200 must be NaN for the first 199 rows (0-indexed), valid from row 199."""
        prices = _arithmetic_prices(210, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        assert feat["price_to_ma200"].iloc[:200].isna().all(), "price_to_ma200 rows 0-199 must be NaN"
        assert feat["price_to_ma200"].iloc[200:].notna().all(), "price_to_ma200 rows 200+ must be valid"

    def test_no_expanding_window_approximation(self):
        """
        price_to_ma200 at row 5 (only 6 data points available) must be NaN — not an expanding MA.
        This confirms strict min_periods=200, not min_periods=1 (expanding).
        """
        prices = _arithmetic_prices(10, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        assert feat["price_to_ma200"].isna().all(), (
            "price_to_ma200 must be NaN for all rows when fewer than 200 data points exist."
        )

    def test_drop_warmup_rows_removes_nan_rows(self):
        """drop_warmup_rows must remove all rows where any feature column is NaN.

        rel_strength_21 is always NaN when spy_df=None, so we provide a matching
        SPY series here. With spy_df present, all 15 features can be valid.
        """
        prices = _arithmetic_prices(220, start=100.0)
        df = _make_ohlcv(prices)
        spy_df = _make_ohlcv(prices, ticker="SPY")  # same prices → rel_strength_21 = 0
        feat = compute_features(df, spy_df=spy_df)

        assert feat[FEATURE_COLUMNS].isna().any(axis=None), (
            "Expected some NaN rows in the full 220-row feature DataFrame."
        )

        clean = drop_warmup_rows(feat)

        # No row in clean should have any NaN in FEATURE_COLUMNS
        assert not clean[FEATURE_COLUMNS].isna().any(axis=None), (
            "drop_warmup_rows must remove ALL rows with NaN features."
        )
        # The clean set must be non-empty (220 > 200 so there should be valid rows)
        assert len(clean) > 0, "clean DataFrame must not be empty for 220-row input."
        # The most restrictive feature is price_to_ma200 — first valid at row 199.
        # With 220 rows, we expect 220 - 199 = 21 valid rows.
        assert len(clean) == 20, (
            f"Expected 20 fully-valid rows (220-200), got {len(clean)}"
        )

    def test_has_all_features_true_for_valid_row(self):
        """has_all_features returns True for a row with no NaN features.

        Provides spy_df so rel_strength_21 is valid (not NaN by default).
        """
        prices = _arithmetic_prices(210, start=100.0)
        df = _make_ohlcv(prices)
        spy_df = _make_ohlcv(prices, ticker="SPY")
        feat = compute_features(df, spy_df=spy_df)
        valid_row = feat.iloc[209]   # last row — all 15 features valid
        assert has_all_features(valid_row), (
            f"Last row of 210-row series (with spy_df) must have all features. "
            f"NaN cols: {[c for c in FEATURE_COLUMNS if pd.isna(valid_row.get(c, float('nan')))]}"
        )

    def test_has_all_features_false_for_warmup_row(self):
        """has_all_features returns False for a row still in warm-up (e.g., row 0)."""
        prices = _arithmetic_prices(210, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        warmup_row = feat.iloc[0]    # price_to_ma200 is NaN, plus many others
        assert not has_all_features(warmup_row), "First row must not have all features."

    def test_rel_strength_21_nan_when_no_spy(self):
        """When spy_df is not provided, rel_strength_21 must be NaN for all rows."""
        prices = _arithmetic_prices(50, start=100.0)
        feat = compute_features(_make_ohlcv(prices))   # no spy_df
        assert feat["rel_strength_21"].isna().all(), (
            "rel_strength_21 must be NaN when spy_df is not provided."
        )

    def test_feature_columns_registry_completeness(self):
        """
        FEATURE_COLUMNS must list exactly the columns that appear in the output,
        and the output must have no extra unlisted feature columns.
        """
        prices = _arithmetic_prices(30, start=100.0)
        feat = compute_features(_make_ohlcv(prices))
        output_feature_cols = [c for c in feat.columns if c not in ("date", "ticker")]
        assert output_feature_cols == FEATURE_COLUMNS, (
            f"Output feature columns don't match FEATURE_COLUMNS registry.\n"
            f"  Output:   {output_feature_cols}\n"
            f"  Registry: {FEATURE_COLUMNS}"
        )
        assert len(FEATURE_COLUMNS) == 15, f"Expected 15 features, got {len(FEATURE_COLUMNS)}"
