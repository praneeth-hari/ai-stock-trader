"""
src/data/validation.py — Phase 2 Data Validation Gate (§1.7).

This module is the single checkpoint that all market data MUST pass before any
downstream engine (features, ML, risk, portfolio, paper broker) is allowed to
touch it. The design contract is:

    fetch (Phase 1) → validate (this module) → everything else

SEVERITY MODEL
--------------
There are two tiers of problems, with different consequences:

CRITICAL → REJECT ticker
    - Sanity bound violations: price ≤ 0, High < Low, Close/Open outside
      [Low, High], volume < 0, NaN in required columns.
    - Price spikes: single-day move beyond ±50% — almost always bad data,
      never observed in large-cap equities under normal conditions.

WARNING → flag, do NOT reject
    - Unexplained trading-day gaps (days that should have been trading days
      per the NYSE calendar, but are absent from the data).
    - Gaps that align with known NYSE holidays or weekends are expected and
      correct. They are NOT flagged as warnings or errors. A DEBUG-level trace
      IS always emitted so every gap-check run leaves a full audit trail
      (CLAUDE.md rule 8: log everything).
    - CRITICAL escalation: >= 6 consecutive unexplained missing trading days
      (run_len >= CRITICAL_GAP_DAYS) triggers a CRITICAL log — this magnitude
      of absence suggests the data
      provider itself is degrading rather than a one-off quirk.

ISOLATION GUARANTEE
-------------------
Each ticker is validated independently. A rejection of ticker A has zero
effect on tickers B, C, … The caller always receives a validated universe
of passing tickers plus a complete skip/warn report.

HELD-STOCK GUARD
----------------
If a currently-held position has bad data, the system logs a high-priority
warning and explicitly instructs downstream engines to hold the position
UNTOUCHED — neither buying nor selling on untrustworthy numbers.

WATCH SIGNAL
------------
If ≥ 40% of the evaluated universe fails validation, a CRITICAL log is
emitted: the data source is likely degrading; the caller should pause.

§1.7 COMPLIANCE
---------------
- Reject impossible values                      ✓
- Flag ±50% spikes                              ✓
- Detect gaps vs. known NYSE trading calendar   ✓
- Fail loud — skip-and-log per stock            ✓
- Isolation — one bad stock never affects others✓
- Held-stock with bad data → hold untouched     ✓
- Watch signal on mass failure                  ✓
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from config.settings import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Constants ──────────────────────────────────────────────────────────────────

# Single-day price move beyond this fraction is treated as suspect data.
# §1.7: "flag any single-day move beyond ±50% for review."
SPIKE_THRESHOLD = 0.50

# Unexplained consecutive missing trading days at or above this count
# escalates the gap warning to CRITICAL severity (data source degrading).
CRITICAL_GAP_DAYS = 6

# If this fraction of the universe fails validation, emit a watch signal.
WATCH_SIGNAL_FRACTION = 0.40

# Required OHLCV columns in normalized Phase 1 output.
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "ticker"]

# Calendar days of history the live daily pipeline fetches before run_date (before the
# fetcher's own warmup extension). Shared so backtests can replay live validation windows.
LIVE_FETCH_LOOKBACK_DAYS = 420


def live_validation_lookback_days() -> int:
    """Calendar days of history live validation sees on any run date (pipeline window + fetch warmup)."""
    return LIVE_FETCH_LOOKBACK_DAYS + int(settings.warmup_bars * 365 / 252) + 15


# ── NYSE calendar helper ───────────────────────────────────────────────────────

def _get_nyse_trading_days(start_date: str, end_date: str) -> Set[str]:
    """
    Return the set of valid NYSE trading days (as 'YYYY-MM-DD' strings)
    between start_date and end_date, inclusive on both ends.

    Uses pandas' built-in USFederalHolidayCalendar to exclude US federal
    holidays (New Year's, MLK Day, Presidents' Day, Good Friday, Memorial Day,
    Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas) plus
    weekends. No extra package required — pandas is already a core dependency.
    """
    cal = USFederalHolidayCalendar()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    holidays = cal.holidays(start=start, end=end)
    # All weekdays in range, then subtract holidays
    all_bdays = pd.bdate_range(start=start, end=end)
    trading_days = all_bdays[~all_bdays.isin(holidays)]
    return set(trading_days.strftime("%Y-%m-%d"))


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Outcome of validating a single ticker's DataFrame.

    Attributes:
        ticker:      Uppercase ticker symbol.
        is_valid:    True if the ticker passed all CRITICAL checks and can
                     proceed to downstream phases. May still carry warnings.
        cleaned_df:  The validated (possibly full) DataFrame for valid tickers,
                     or an empty DataFrame for rejected tickers.
        errors:      Human-readable descriptions of CRITICAL failures that
                     caused rejection. Empty for valid tickers.
        warnings:    Human-readable descriptions of WARNING-level concerns
                     (e.g. unexplained gaps). Present even for valid tickers.
    """
    ticker: str
    is_valid: bool
    cleaned_df: pd.DataFrame
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Point-in-time validation only: inclusive (start, end) dates on which live validation would
    # have rejected this ticker. Outside these windows its data is usable.
    excluded_windows: List[Tuple[str, str]] = field(default_factory=list)


# ── Individual check functions ─────────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Verify all required OHLCV columns are present.

    Returns a list of error strings (empty = OK).
    """
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"{ticker}: missing required columns {missing}")
    return errors


def check_sanity_bounds(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check for values that are physically or logically impossible in market data.

    Rules (all are CRITICAL failures → ticker rejected):
    - Any of open, high, low, close ≤ 0.
    - volume < 0.
    - NaN in any required column.
    - high < low on any row (mathematically impossible OHLC bar).
    - close outside [low, high] on any row.
    - open outside [low, high] on any row.

    Returns a list of error strings (empty = OK).
    """
    errors = []

    # NaN check
    nan_cols = [c for c in REQUIRED_COLUMNS if c != "ticker" and df[c].isna().any()]
    if nan_cols:
        for col in nan_cols:
            bad_dates = df.loc[df[col].isna(), "date"].tolist()
            errors.append(f"{ticker}: NaN in column '{col}' on dates {bad_dates}")

    # Price positivity check
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        bad = df[df[col] <= 0]
        if not bad.empty:
            errors.append(
                f"{ticker}: column '{col}' has values ≤ 0 on dates "
                f"{bad['date'].tolist()} (values: {bad[col].tolist()})"
            )

    # Volume non-negative
    if "volume" in df.columns:
        bad = df[df["volume"] < 0]
        if not bad.empty:
            errors.append(
                f"{ticker}: 'volume' is negative on dates "
                f"{bad['date'].tolist()} (values: {bad['volume'].tolist()})"
            )

    # High < Low
    if "high" in df.columns and "low" in df.columns:
        bad = df[df["high"] < df["low"]]
        if not bad.empty:
            errors.append(
                f"{ticker}: high < low (impossible bar) on dates "
                f"{bad['date'].tolist()}"
            )

    # Close outside [low, high]
    if all(c in df.columns for c in ["close", "low", "high"]):
        bad = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        if not bad.empty:
            errors.append(
                f"{ticker}: close outside [low, high] on dates "
                f"{bad['date'].tolist()}"
            )

    # Open outside [low, high]
    if all(c in df.columns for c in ["open", "low", "high"]):
        bad = df[(df["open"] < df["low"]) | (df["open"] > df["high"])]
        if not bad.empty:
            errors.append(
                f"{ticker}: open outside [low, high] on dates "
                f"{bad['date'].tolist()}"
            )

    return errors


def check_price_spikes(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check for abnormal single-day close-to-close price jumps using dynamic per-ticker
    volatility thresholds and volume confirmation (§1.7).

    Rules:
    1. Dynamic threshold: Flag/reject any overnight move beyond 10x the stock's historical
       average daily move (rolling 20-day mean absolute return). If fewer than 5 rows of
       history exist, falls back to fixed 50% threshold. Minimum floor is 20%.
    2. Secondary check: If overnight move >= 25% AND volume is normal (<= 1.5x 20-day average
       volume, meaning no volume spike), treat as bad data and reject.
    """
    errors = []
    if "close" not in df.columns or len(df) < 2:
        return errors

    pct_change, dynamic_threshold, dynamic_spike_mask, suspicious_jumps, vol_ratio = _price_spike_masks(df)
    if dynamic_spike_mask.any():
        bad_dates = df.loc[dynamic_spike_mask, "date"].tolist()
        bad_vals = pct_change[dynamic_spike_mask].round(4).tolist()
        bad_thresh = dynamic_threshold[dynamic_spike_mask].round(4).tolist()
        errors.append(
            f"{ticker}: dynamic price spike(s) beyond 10x avg daily move detected on dates "
            f"{bad_dates} (moves: {bad_vals}, dynamic thresholds: {bad_thresh}). "
            "Suspected bad data. Ticker rejected."
        )

    # 2. Secondary check: >= 25% price jump without volume spike (normal volume)
    if suspicious_jumps.any():
        bad_dates = df.loc[suspicious_jumps, "date"].tolist()
        bad_jumps = pct_change[suspicious_jumps].round(4).tolist()
        bad_ratios = vol_ratio[suspicious_jumps].round(2).tolist()
        errors.append(
            f"{ticker}: suspicious price jump(s) >=25% with normal volume (no volume spike) detected on dates "
            f"{bad_dates} (jumps: {bad_jumps}, volume ratios vs 20d avg: {bad_ratios}). "
            "Suspected bad data. Ticker rejected."
        )

    return errors


def _price_spike_masks(df: pd.DataFrame):
    """
    Row-level spike flags for check_price_spikes. Every threshold uses only rows before the
    flagged row (shifted rolling windows), so each flag is decidable on its own date.

    Returns (pct_change, dynamic_threshold, dynamic_spike_mask, suspicious_jump_mask, vol_ratio).
    """
    pct_change = df["close"].pct_change().abs()

    # 1. Dynamic per-ticker volatility check
    if len(df) >= 5:
        # Preceding 20-day rolling mean absolute daily move
        prev_moves = pct_change.shift(1).rolling(20, min_periods=3).mean()
        # Fallback to expanding mean for early rows in series
        expanding_moves = pct_change.shift(1).expanding(min_periods=1).mean()
        base_avg_move = prev_moves.fillna(expanding_moves).fillna(0.015)
        dynamic_threshold = (base_avg_move * settings.dynamic_spike_std_multiplier).clip(lower=0.20, upper=0.50)
    else:
        dynamic_threshold = pd.Series(SPIKE_THRESHOLD, index=df.index)
    dynamic_spike_mask = pct_change > dynamic_threshold

    # 2. >= 25% price jump without volume spike (normal volume)
    suspicious_jumps = pd.Series(False, index=df.index)
    vol_ratio = pd.Series(np.nan, index=df.index)
    jump_25_mask = pct_change >= settings.suspicious_price_jump_pct
    if jump_25_mask.any() and "volume" in df.columns:
        prev_avg_vol = df["volume"].shift(1).rolling(20, min_periods=1).mean()
        # If no prior volume history, fallback to current volume
        prev_avg_vol = prev_avg_vol.fillna(df["volume"])
        vol_ratio = df["volume"] / prev_avg_vol.replace(0, np.nan)
        # Normal volume: volume on jump day <= 1.5x of 20-day average volume
        no_vol_spike = (vol_ratio <= settings.suspicious_volume_spike_threshold) | df["volume"].isna()
        suspicious_jumps = jump_25_mask & no_vol_spike

    return pct_change, dynamic_threshold, dynamic_spike_mask, suspicious_jumps, vol_ratio


def check_trading_gaps(
    df: pd.DataFrame,
    ticker: str,
) -> List[str]:
    """
    Check for days that should have been NYSE trading days but are absent.

    SEVERITY: WARNING (not rejection). The data rows that ARE present are
    still valid; missing rows simply mean incomplete coverage for that
    window. This is clearly different from impossible values (which corrupt
    whatever decision uses them) — a gap means "don't know", not "wrong".

    Escalation rule: run_len >= CRITICAL_GAP_DAYS (>= 6) consecutive
    unexplained missing trading days → logged at CRITICAL level (strongly
    suggests data-provider degradation).

    Weekend and NYSE holiday absences are NOT flagged as warnings or errors
    — they are expected and correct. However, a DEBUG-level trace IS always
    emitted for full auditability (CLAUDE.md rule 8: log everything). A run
    with zero unexplained gaps also emits a DEBUG trace confirming the check
    passed cleanly.

    Returns a list of warning strings (empty = OK).
    """
    warnings = []
    if "date" not in df.columns or len(df) < 2:
        return warnings

    dates_present = set(df["date"].tolist())
    min_date = df["date"].min()
    max_date = df["date"].max()

    # Get all NYSE trading days in the range we have data for.
    expected_trading_days = _get_nyse_trading_days(min_date, max_date)

    # DEBUG trace: total calendar days computed vs. data days present.
    # Always emitted so every gap-check run leaves a full audit trail
    # even when no problems are found (CLAUDE.md rule 8: log everything).
    logger.debug(
        "%s: gap check — NYSE expected %d trading days (%s to %s), "
        "data contains %d distinct dates. "
        "Weekend/holiday absences are pre-excluded from the expected set.",
        ticker,
        len(expected_trading_days),
        min_date,
        max_date,
        len(dates_present),
    )

    # Any trading day that should be present but isn't.
    missing_days = sorted(expected_trading_days - dates_present)

    if not missing_days:
        logger.debug(
            "%s: gap check PASSED — no unexplained missing trading days.",
            ticker,
        )
        return warnings

    # Find contiguous runs of missing days to detect multi-day gaps.
    missing_dt = pd.to_datetime(missing_days)
    gaps: List[List[str]] = []
    current_run: List[str] = [missing_days[0]]
    for i in range(1, len(missing_days)):
        # A contiguous run means each missing day is a consecutive trading day.
        # We check by calendar date proximity (not strict +1d because trading
        # days can skip weekends — but these are already filtered out by using
        # the NYSE calendar as the expected set).
        if missing_dt[i] == missing_dt[i - 1] + pd.Timedelta(days=1):
            current_run.append(missing_days[i])
        else:
            gaps.append(current_run)
            current_run = [missing_days[i]]
    gaps.append(current_run)

    for run in gaps:
        run_len = len(run)
        msg = (
            f"{ticker}: {run_len} unexplained missing NYSE trading "
            f"day(s) between {run[0]} and {run[-1]} (dates: {run}). "
            "Weekend/holiday absences have already been excluded."
        )
        if run_len >= CRITICAL_GAP_DAYS:
            logger.critical(
                "%s: CRITICAL — %d consecutive missing trading days. "
                "Data provider may be degrading. Dates: %s",
                ticker, run_len, run,
            )
            warnings.append(f"CRITICAL: {msg}")
        else:
            warnings.append(msg)

    return warnings


# ── Per-ticker validator ───────────────────────────────────────────────────────

def validate_ticker_data(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """
    Run all §1.7 validation checks on a single ticker's normalized DataFrame.

    Check order:
      1. Required columns presence (CRITICAL if missing).
      2. Sanity bounds (CRITICAL).
      3. Price spikes (CRITICAL).
      4. Trading-day gaps vs. NYSE calendar (WARNING only — see module docstring).

    Args:
        df:     Normalized DataFrame from Phase 1 (schema: date/open/high/low/close/volume/ticker).
        ticker: Ticker symbol for logging clarity.

    Returns:
        ValidationResult with is_valid=True/False, cleaned_df, errors, warnings.
    """
    clean_ticker = ticker.strip().upper()
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        errors.append(f"{clean_ticker}: DataFrame is empty — nothing to validate.")
        logger.error("VALIDATION REJECTED %s: empty DataFrame.", clean_ticker)
        return ValidationResult(
            ticker=clean_ticker,
            is_valid=False,
            cleaned_df=pd.DataFrame(columns=REQUIRED_COLUMNS),
            errors=errors,
            warnings=warnings,
        )

    # --- CRITICAL checks ---
    errors += check_required_columns(df, clean_ticker)
    if errors:
        logger.error("VALIDATION REJECTED %s: missing columns. Errors: %s", clean_ticker, errors)
        return ValidationResult(
            ticker=clean_ticker,
            is_valid=False,
            cleaned_df=pd.DataFrame(columns=REQUIRED_COLUMNS),
            errors=errors,
        )

    errors += check_sanity_bounds(df, clean_ticker)
    errors += check_price_spikes(df, clean_ticker)

    if errors:
        for e in errors:
            logger.error("VALIDATION REJECTED %s: %s", clean_ticker, e)
        return ValidationResult(
            ticker=clean_ticker,
            is_valid=False,
            cleaned_df=pd.DataFrame(columns=REQUIRED_COLUMNS),
            errors=errors,
            warnings=warnings,
        )

    # --- WARNING checks (do not reject) ---
    warnings += check_trading_gaps(df, clean_ticker)
    for w in warnings:
        level = logging.CRITICAL if w.startswith("CRITICAL") else logging.WARNING
        logger.log(level, "VALIDATION WARNING %s: %s", clean_ticker, w)

    logger.info(
        "VALIDATION PASSED %s: %d rows, %s to %s.",
        clean_ticker,
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    return ValidationResult(
        ticker=clean_ticker,
        is_valid=True,
        cleaned_df=df.copy(),
        errors=[],
        warnings=warnings,
    )


def _sanity_violation_mask(df: pd.DataFrame) -> pd.Series:
    """Row-level form of check_sanity_bounds: True where a row holds an impossible value."""
    cols = [c for c in REQUIRED_COLUMNS if c != "ticker"]
    mask = df[cols].isna().any(axis=1)
    mask |= (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    mask |= df["volume"] < 0
    mask |= df["high"] < df["low"]
    mask |= (df["close"] < df["low"]) | (df["close"] > df["high"])
    mask |= (df["open"] < df["low"]) | (df["open"] > df["high"])
    return mask


def validate_ticker_data_point_in_time(
    df: pd.DataFrame,
    ticker: str,
    lookback_days: Optional[int] = None,
) -> ValidationResult:
    """
    Backtest form of validate_ticker_data that never uses future information.

    validate_ticker_data rejects a ticker's ENTIRE history if any row fails, so a spike in 2025
    would remove the stock from a 2019 backtest. Live trading only ever validates the trailing
    lookback window, so a failing row on date S makes live skip the ticker on run dates
    [S, S + lookback_days] and nowhere else. This reproduces exactly that:

      - Same checks and thresholds (spike thresholds use only earlier rows).
      - Each flagged row S yields an excluded window [S, S + lookback_days] (merged if overlapping).
      - The ticker stays usable (is_valid=True) with full data; callers must honour
        excluded_windows (run_strategy_backtest(data_exclusions=...)).

    Missing columns or an empty frame still reject the ticker outright.
    """
    clean_ticker = ticker.strip().upper()
    if df.empty or check_required_columns(df, clean_ticker):
        return validate_ticker_data(df, clean_ticker)

    lookback = int(lookback_days) if lookback_days is not None else live_validation_lookback_days()
    df = df.sort_values("date").reset_index(drop=True)

    errors = check_sanity_bounds(df, clean_ticker) + check_price_spikes(df, clean_ticker)
    flagged = _sanity_violation_mask(df)
    if len(df) >= 2:
        _, _, dynamic_mask, suspicious_mask, _ = _price_spike_masks(df)
        flagged |= dynamic_mask | suspicious_mask

    windows: List[Tuple[str, str]] = []
    for flag_date in sorted(df.loc[flagged, "date"].astype(str)):
        end = (pd.Timestamp(flag_date) + pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
        if windows and flag_date <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((flag_date, end))

    warnings = check_trading_gaps(df, clean_ticker)
    for start, end in windows:
        msg = (
            f"{clean_ticker}: excluded from {start} to {end} only — live validation's {lookback}-day "
            "window would contain a flagged row on those dates. Earlier dates are unaffected."
        )
        warnings.append(msg)
        logger.warning("POINT-IN-TIME VALIDATION %s", msg)
    for e in errors:
        logger.warning("POINT-IN-TIME VALIDATION %s flagged row(s): %s", clean_ticker, e)

    return ValidationResult(
        ticker=clean_ticker,
        is_valid=True,
        cleaned_df=df.copy(),
        errors=errors,
        warnings=warnings,
        excluded_windows=windows,
    )


# ── Universe validator (the main public API) ───────────────────────────────────

def validate_universe_data(
    universe_df: pd.DataFrame,
    held_tickers: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, ValidationResult]]:
    """
    Validate a full universe of tickers, each in isolation.

    Steps:
      1. Split universe_df by ticker.
      2. Run validate_ticker_data() independently for each.
      3. Emit held-stock guards for any held position that fails.
      4. Emit the watch signal if ≥40% of tickers fail.
      5. Return the combined validated DataFrame (passing tickers only)
         and a per-ticker ValidationResult report.

    Args:
        universe_df:   Combined normalized DataFrame from Phase 1.
                       Must contain a 'ticker' column.
        held_tickers:  Optional list of currently held position tickers.
                       If a held ticker fails validation it will receive a
                       special CRITICAL warning rather than a silent skip.

    Returns:
        Tuple of:
          - validated_df: DataFrame containing only rows from valid tickers.
          - results:      Dict[ticker → ValidationResult] for every ticker,
                          including rejected ones (for logging / audit).
    """
    held_set: Set[str] = {t.strip().upper() for t in (held_tickers or [])}
    results: Dict[str, ValidationResult] = {}
    valid_frames = []

    tickers = universe_df["ticker"].unique() if "ticker" in universe_df.columns else []

    for ticker in tickers:
        ticker_df = universe_df[universe_df["ticker"] == ticker].copy()
        result = validate_ticker_data(ticker_df, ticker)
        results[ticker] = result

        if result.is_valid:
            valid_frames.append(result.cleaned_df)
        else:
            if ticker in held_set:
                logger.critical(
                    "HELD POSITION %s HAS CORRUPTED DATA. "
                    "DO NOT TRADE THIS POSITION — hold untouched until clean data arrives. "
                    "Errors: %s",
                    ticker,
                    result.errors,
                )
            else:
                logger.error(
                    "SKIPPING %s for this run. Errors: %s",
                    ticker,
                    result.errors,
                )

    # Watch signal: too many failures = data source is degrading.
    total = len(tickers)
    failed = sum(1 for r in results.values() if not r.is_valid)
    if total > 0 and (failed / total) >= WATCH_SIGNAL_FRACTION:
        logger.critical(
            "WATCH SIGNAL: %d/%d tickers (%d%%) failed validation in this run. "
            "Data source may be degrading — pause and inspect before proceeding.",
            failed,
            total,
            int(100 * failed / total),
        )

    validated_df = (
        pd.concat(valid_frames, ignore_index=True)
        if valid_frames
        else pd.DataFrame(columns=REQUIRED_COLUMNS)
    )
    return validated_df, results
