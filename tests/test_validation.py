"""
tests/test_validation.py — Adversarial & real-data tests for Phase 2 validation gate.

Each test case documents what adversarial input is used, what the expected
outcome is, and why — so failures are easy to diagnose.
"""

from __future__ import annotations

import logging
import pandas as pd
import pytest

from src.data.validation import (
    REQUIRED_COLUMNS,
    ValidationResult,
    check_price_spikes,
    check_sanity_bounds,
    check_trading_gaps,
    validate_ticker_data,
    validate_universe_data,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a normalized OHLCV DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


def _good_row(date: str, close: float = 150.0, ticker: str = "TEST") -> dict:
    """Return a single valid OHLCV row dict."""
    return {
        "date": date,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 1_000_000,
        "ticker": ticker,
    }


def _clean_test_df(dates: list[str], close: float = 150.0, ticker: str = "TEST") -> pd.DataFrame:
    return _make_df([_good_row(d, close, ticker) for d in dates])


# ── Sanity Bound Tests ─────────────────────────────────────────────────────────

class TestSanityBounds:

    def test_negative_close_rejects_ticker(self):
        """Negative close price → CRITICAL → ticker rejected."""
        df = _make_df([
            _good_row("2024-01-02"),
            {**_good_row("2024-01-03"), "close": -10.0, "low": -12.0},
        ])
        result = validate_ticker_data(df, "AAPL")
        assert not result.is_valid
        assert result.cleaned_df.empty
        assert any("≤ 0" in e or "0" in e for e in result.errors), result.errors

    def test_zero_price_rejects_ticker(self):
        """Zero open price → rejected."""
        df = _make_df([{**_good_row("2024-01-02"), "open": 0.0}])
        result = validate_ticker_data(df, "MSFT")
        assert not result.is_valid

    def test_high_less_than_low_rejects_ticker(self):
        """high < low is physically impossible → rejected."""
        df = _make_df([{
            "date": "2024-01-02",
            "open": 100.0,
            "high": 98.0,    # high < low
            "low": 102.0,
            "close": 100.0,
            "volume": 500_000,
            "ticker": "NVDA",
        }])
        result = validate_ticker_data(df, "NVDA")
        assert not result.is_valid
        assert any("high < low" in e for e in result.errors), result.errors

    def test_close_above_high_rejects_ticker(self):
        """close > high is a bar violation → rejected."""
        df = _make_df([{
            "date": "2024-01-02",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 110.0,  # close > high
            "volume": 500_000,
            "ticker": "GOOGL",
        }])
        result = validate_ticker_data(df, "GOOGL")
        assert not result.is_valid
        assert any("close outside" in e for e in result.errors), result.errors

    def test_close_below_low_rejects_ticker(self):
        """close < low is a bar violation → rejected."""
        df = _make_df([{
            "date": "2024-01-02",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 90.0,  # close < low
            "volume": 500_000,
            "ticker": "META",
        }])
        result = validate_ticker_data(df, "META")
        assert not result.is_valid

    def test_negative_volume_rejects_ticker(self):
        """Negative volume is impossible → rejected."""
        df = _make_df([{**_good_row("2024-01-02"), "volume": -1}])
        result = validate_ticker_data(df, "TSLA")
        assert not result.is_valid
        assert any("volume" in e for e in result.errors), result.errors

    def test_nan_in_close_rejects_ticker(self):
        """NaN in close column → rejected."""
        import math
        df = _make_df([{**_good_row("2024-01-02"), "close": float("nan")}])
        result = validate_ticker_data(df, "JPM")
        assert not result.is_valid

    def test_all_valid_rows_pass(self):
        """Perfectly clean DataFrame → is_valid=True, no errors."""
        df = _clean_test_df(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        result = validate_ticker_data(df, "TEST")
        assert result.is_valid
        assert not result.errors
        assert len(result.cleaned_df) == 5


# ── Spike Detection Tests ──────────────────────────────────────────────────────

class TestSpikeDetection:

    def test_75_pct_drop_rejects_ticker(self):
        """
        75% single-day drop (e.g., unadjusted 4-for-1 split) → rejected.
        Scenario: close goes 200 → 50 overnight (+300% up would also trigger).
        """
        df = _make_df([
            _good_row("2024-01-02", close=200.0),
            {
                "date": "2024-01-03",
                "open": 50.0,
                "high": 52.0,
                "low": 49.0,
                "close": 50.0,   # -75% from 200
                "volume": 1_000_000,
                "ticker": "SPLIT",
            },
        ])
        result = validate_ticker_data(df, "SPLIT")
        assert not result.is_valid
        assert any("spike" in e.lower() for e in result.errors), result.errors

    def test_300_pct_gain_rejects_ticker(self):
        """300% overnight gain is data corruption, not a stock move."""
        df = _make_df([
            _good_row("2024-01-02", close=100.0),
            {
                "date": "2024-01-03",
                "open": 400.0,
                "high": 410.0,
                "low": 395.0,
                "close": 400.0,  # +300% from 100
                "volume": 1_000_000,
                "ticker": "BOGUS",
            },
        ])
        result = validate_ticker_data(df, "BOGUS")
        assert not result.is_valid
        assert any("spike" in e.lower() for e in result.errors), result.errors

    def test_point_in_time_spike_only_excludes_dates_live_would_have(self):
        """
        A spike late in the history must not remove the ticker from earlier backtest dates.
        Live validation (whole window) still rejects; the point-in-time variant excludes only
        [spike, spike + live lookback].
        """
        from src.data.validation import live_validation_lookback_days, validate_ticker_data_point_in_time

        dates = pd.bdate_range("2024-01-02", periods=60).strftime("%Y-%m-%d").tolist()
        spike_date = dates[50]
        rows = [_good_row(d, close=100.0) for d in dates[:50]]
        rows += [{**_good_row(d, close=140.0), "volume": 5_000_000} for d in dates[50:]]  # +40% on day 50
        df = _make_df(rows)

        assert not validate_ticker_data(df, "LATE").is_valid  # live gate behaviour unchanged

        pit = validate_ticker_data_point_in_time(df, "LATE")
        lookback_end = (pd.Timestamp(spike_date) + pd.Timedelta(days=live_validation_lookback_days())).strftime("%Y-%m-%d")
        assert pit.is_valid and len(pit.cleaned_df) == 60
        assert pit.excluded_windows == [(spike_date, lookback_end)]
        assert all(d < pit.excluded_windows[0][0] for d in dates[:50])  # earlier dates usable
        assert any("spike" in e.lower() for e in pit.errors)

    def test_normal_volatile_day_passes(self):
        """A 15% move (large but real — e.g., earnings) must not trigger the spike check."""
        df = _make_df([
            _good_row("2024-01-02", close=100.0),
            {
                "date": "2024-01-03",
                "open": 114.0,
                "high": 116.0,
                "low": 113.0,
                "close": 115.0,  # +15% — large but within threshold
                "volume": 5_000_000,
                "ticker": "EARN",
            },
        ])
        result = validate_ticker_data(df, "EARN")
        assert result.is_valid
        assert not result.errors


# ── Gap Detection Tests ────────────────────────────────────────────────────────

class TestTradingGaps:

    def test_weekend_absence_is_not_a_gap(self):
        """
        Saturday and Sunday are not NYSE trading days — their absence must NOT
        produce any warning. (Tests the core point of using a real calendar.)
        """
        # Mon, Tue, Wed — skipping weekend is correct and expected
        df = _clean_test_df(["2024-01-08", "2024-01-09", "2024-01-10"])
        warnings = check_trading_gaps(df, "TEST")
        assert warnings == [], f"Weekend absence wrongly flagged as gap: {warnings}"

    def test_nyse_holiday_absence_is_not_a_gap(self):
        """
        Martin Luther King Jr. Day (third Monday of January) is a US market
        holiday — its absence must NOT produce a warning.
        In 2024, MLK Day = 2024-01-15. Data spans Jan 12 (Fri) → Jan 16 (Tue).
        """
        df = _clean_test_df(["2024-01-12", "2024-01-16"])
        warnings = check_trading_gaps(df, "TEST")
        assert warnings == [], (
            f"MLK Day holiday (2024-01-15) wrongly flagged as unexplained gap: {warnings}"
        )

    def test_genuine_missing_trading_day_warns(self):
        """
        An unexplained missing trading day (not a holiday, not a weekend)
        should produce a warning — but NOT a rejection.
        Gap: 2024-01-04 (Thursday) is missing.
        """
        # Jan 2, 3 then skip Jan 4, then Jan 5 — Jan 4 is a trading day
        df = _clean_test_df(["2024-01-02", "2024-01-03", "2024-01-05"])
        warnings = check_trading_gaps(df, "TEST")
        assert len(warnings) >= 1, "Expected a warning for missing Jan 4 trading day"
        assert "2024-01-04" in warnings[0], warnings

    def test_gap_does_not_reject_ticker(self):
        """
        A warning-level gap must not flip is_valid to False.
        The ticker should still be valid (rows present are trustworthy).
        """
        df = _clean_test_df(["2024-01-02", "2024-01-03", "2024-01-05"])
        result = validate_ticker_data(df, "GAPPED")
        assert result.is_valid, "Gap warning should not reject ticker"
        assert not result.errors, result.errors
        assert len(result.warnings) >= 1, "Expected gap warning"


# ── Isolation Tests ────────────────────────────────────────────────────────────

class TestIsolation:

    def test_bad_ticker_does_not_contaminate_good_ticker(self):
        """
        AAPL (valid) and BAD (negative price) in same universe.
        AAPL must pass through intact. BAD must be rejected with an error.
        Neither one's outcome affects the other.
        """
        aapl_df = _clean_test_df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            ticker="AAPL",
        )
        bad_df = _make_df([{**_good_row("2024-01-02", ticker="BAD"), "close": -5.0, "low": -6.0}])
        universe = pd.concat([aapl_df, bad_df], ignore_index=True)

        validated_df, results = validate_universe_data(universe)

        assert results["AAPL"].is_valid
        assert not results["BAD"].is_valid
        assert "BAD" not in validated_df["ticker"].values
        assert set(validated_df["ticker"].values) == {"AAPL"}
        assert len(validated_df) == 3

    def test_watch_signal_triggered_on_mass_failure(self, caplog):
        """
        When ≥40% of the universe fails, a CRITICAL watch signal must be logged.
        3 bad tickers out of 5 total = 60% → triggers watch signal.
        """
        good_df = _clean_test_df(["2024-01-02"], ticker="GOOD1")
        good2_df = _clean_test_df(["2024-01-02"], ticker="GOOD2")

        def _bad(ticker):
            return _make_df([{**_good_row("2024-01-02", ticker=ticker), "close": -1.0, "low": -2.0}])

        universe = pd.concat(
            [good_df, good2_df, _bad("BAD1"), _bad("BAD2"), _bad("BAD3")],
            ignore_index=True,
        )
        with caplog.at_level(logging.CRITICAL):
            validate_universe_data(universe)
        assert any("WATCH SIGNAL" in r.message for r in caplog.records), (
            "Expected CRITICAL watch signal when ≥40% of tickers fail"
        )

    def test_held_stock_bad_data_logs_critical(self, caplog):
        """
        If a held position has bad data, a CRITICAL 'HELD POSITION' log
        must be emitted — not just a quiet skip.
        """
        bad_df = _make_df([{**_good_row("2024-01-02", ticker="HELD"), "close": -1.0, "low": -2.0}])
        with caplog.at_level(logging.CRITICAL):
            validate_universe_data(bad_df, held_tickers=["HELD"])
        assert any("HELD POSITION" in r.message for r in caplog.records), (
            "Expected CRITICAL log for held position with bad data"
        )


# ── Real Data Tests ────────────────────────────────────────────────────────────

class TestRealData:
    """
    Live network tests. These hit yfinance and may occasionally return empty
    DataFrames due to rate-limiting or transient session expiry — that is an
    infrastructure issue, not a validation logic failure. If yfinance returns
    empty, the test skips (pytest.skip) rather than failing the suite.
    """

    def test_real_spy_data_passes_validation(self):
        """
        Fetch real SPY data via Phase 1 and run it through the validation gate.
        Clean real-world data must pass without errors and without false-alarm
        rejections (confirms no accidental over-sensitivity in the checks).
        """
        from src.data.market_data import fetch_ticker_data
        df = fetch_ticker_data(
            ticker="SPY",
            start_date="2023-06-01",
            end_date="2023-07-01",
            save_raw=False,
        )
        if df.empty:
            pytest.skip("yfinance returned empty DataFrame for SPY (transient network issue)")

        result = validate_ticker_data(df, "SPY")
        assert result.is_valid, f"Real SPY data failed validation unexpectedly: {result.errors}"
        assert not result.errors
        # Should have ~21 trading days in June 2023
        assert len(result.cleaned_df) >= 20, f"Expected ≥20 rows, got {len(result.cleaned_df)}"

    def test_real_aapl_data_passes_validation(self):
        """
        Real AAPL data (6 months) must pass validation without false positives.
        """
        from src.data.market_data import fetch_ticker_data
        df = fetch_ticker_data(
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2024-07-01",
            save_raw=False,
        )
        if df.empty:
            pytest.skip("yfinance returned empty DataFrame for AAPL (transient network issue)")

        result = validate_ticker_data(df, "AAPL")
        assert result.is_valid, f"Real AAPL data failed validation: {result.errors}"
        assert not result.errors
        assert len(result.cleaned_df) >= 120, f"Expected ≥120 rows, got {len(result.cleaned_df)}"

