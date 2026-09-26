"""
src/ranking/ranking.py — Phase 7 Stock Ranking & Opportunity Ordering.

PURPOSE
-------
Turn model probabilities P(rise > +1% over 5d) into an ordered opportunity list:
  1. Score all universe candidates using the promoted active model.
  2. Apply deterministic tie-breaking (Relative Strength 21-day, Distance to MA-50).
  3. Classify each opportunity into conviction tiers based on config/settings.py:
       - BUY_CANDIDATE: P >= settings.buy_bar (0.60)
       - NEUTRAL_HOLD:  settings.signal_exit (0.45) <= P < settings.buy_bar (0.60)
       - EXIT_CANDIDATE: P < settings.signal_exit (0.45)
  4. Output the complete ranked opportunity list and un-capped top_buy_candidates.

ARCHITECTURE & CLAUDE.md RULE 2 (NO PRE-CAPPING)
-------------------------------------------------
CRITICAL: top_buy_candidates in RankingResult is NOT pre-capped at 3.
The Ranking Engine's responsibility is solely to rank and qualify ALL candidates
meeting the conviction bar (P >= 0.60).

Per CLAUDE.md Rule 2 and Section 1.4:
  "The Risk Engine always sits between ML and any trade decision, and can veto."
Phase 8's Risk Engine is the ONLY layer authorized to enforce the 3-position cap,
cash reserve constraints, portfolio rebalancing, and regime-filter vetoes.
Pre-capping here would blind the Risk Engine to viable alternatives if a top candidate
is already held or rejected by risk rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config.settings import get_ticker_sector, settings
from src.features.engineer import FEATURE_COLUMNS
from src.ml.evaluate import load_active_model
from src.ml.train import TrainedModel

logger = logging.getLogger(__name__)

TIER_BUY: str = "BUY_CANDIDATE"
TIER_NEUTRAL: str = "NEUTRAL_HOLD"
TIER_EXIT: str = "EXIT_CANDIDATE"


@dataclass
class RankedOpportunity:
    """Represents a single stock's evaluation and ranking for a decision date."""
    ticker: str
    date: str
    probability: float
    rank: int
    conviction_tier: str
    rel_strength_21: float = 0.0
    price_to_ma50: float = 0.0
    is_buy_eligible: bool = False
    is_exit_signal: bool = False
    # Section 5: Intelligence attributes
    base_probability: float = 0.0
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    sentiment_modifier: float = 0.0
    sector: str = "Other"
    sector_rank: Optional[int] = None
    sector_modifier: float = 0.0
    earnings_date: Optional[str] = None
    days_until_earnings: Optional[int] = None
    earnings_status: str = "OK"
    earnings_badge: str = ""
    macro_regime: str = "FAVORABLE"
    macro_buy_bar_shift: float = 0.0
    is_sentiment_veto: bool = False
    is_earnings_blackout: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "probability": round(self.probability, 4),
            "base_probability": round(self.base_probability, 4),
            "rank": self.rank,
            "conviction_tier": self.conviction_tier,
            "rel_strength_21": round(self.rel_strength_21, 4),
            "price_to_ma50": round(self.price_to_ma50, 4),
            "is_buy_eligible": self.is_buy_eligible,
            "is_exit_signal": self.is_exit_signal,
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_label": self.sentiment_label,
            "sentiment_modifier": round(self.sentiment_modifier, 4),
            "sector": self.sector,
            "sector_rank": self.sector_rank,
            "sector_modifier": round(self.sector_modifier, 4),
            "earnings_date": self.earnings_date or "N/A",
            "days_until_earnings": self.days_until_earnings,
            "earnings_status": self.earnings_status,
            "earnings_badge": self.earnings_badge,
            "macro_regime": self.macro_regime,
            "macro_buy_bar_shift": round(self.macro_buy_bar_shift, 4),
            "is_sentiment_veto": self.is_sentiment_veto,
            "is_earnings_blackout": self.is_earnings_blackout,
        }


@dataclass
class RankingResult:
    """
    Container for universe ranking results on a specific decision date.

    NOTE ON SIZING & SELECTION:
    `top_buy_candidates` contains ALL candidates with P >= buy_bar. It is NOT
    pre-capped at 3. Phase 8 Risk Engine is the sole layer enforcing the 3-position cap.
    """
    date: str
    regime_risk_on: bool
    total_evaluated: int
    ranked_opportunities: List[RankedOpportunity] = field(default_factory=list)
    top_buy_candidates: List[RankedOpportunity] = field(default_factory=list)
    neutral_candidates: List[RankedOpportunity] = field(default_factory=list)
    exit_candidates: List[RankedOpportunity] = field(default_factory=list)
    macro_regime: str = "FAVORABLE"
    effective_buy_bar: float = 0.60

    @property
    def buy_candidates(self) -> List[RankedOpportunity]:
        """Alias for top_buy_candidates (un-capped list of all P >= buy_bar)."""
        return self.top_buy_candidates

    def to_markdown_table(self) -> str:
        """Render a formatted markdown table of all ranked opportunities."""
        regime_str = "RISK-ON (SPY >= 200d MA)" if self.regime_risk_on else "RISK-OFF (SPY < 200d MA)"
        header = (
            f"### Universe Ranking for {self.date} — Market Regime: {regime_str} | Macro: {self.macro_regime}\n"
            f"Total Evaluated: {self.total_evaluated} | "
            f"Buy Candidates (P >= {self.effective_buy_bar:.2f}): {len(self.top_buy_candidates)} (Un-capped)\n\n"
            "| Rank | Ticker | Sector | Final P | Base P | Sentiment | Earnings | Buy Eligible? |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for o in self.ranked_opportunities:
            sent_str = f"{o.sentiment_modifier:+0.2f} ({o.sentiment_label})"
            earn_str = o.earnings_badge if o.earnings_badge else o.earnings_status
            rows.append(
                f"| {o.rank} | **{o.ticker}** | {o.sector} | **{o.probability:.4f}** | "
                f"{o.base_probability:.4f} | {sent_str} | {earn_str} | "
                f"{'YES' if o.is_buy_eligible else 'no'} |"
            )
        return header + "\n".join(rows)


def check_market_regime(
    spy_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    as_of_date: Optional[str] = None,
) -> bool:
    """
    Determine if the market is Risk-ON (SPY >= 200-day moving average).

    Parameters
    ----------
    spy_df : Optional[pd.DataFrame]
        SPY OHLCV or feature history DataFrame.
    features_df : Optional[pd.DataFrame]
        Universe feature DataFrame that may contain SPY rows.
    as_of_date : Optional[str]
        Date string (YYYY-MM-DD) to evaluate regime as of.

    Returns
    -------
    bool
        True if Risk-ON (SPY >= 200-day MA), False if Risk-OFF.
    """
    # 1. Check if features_df has price_to_ma200 for SPY
    if features_df is not None and "ticker" in features_df.columns:
        spy_features = features_df[features_df["ticker"].str.upper() == settings.benchmark.upper()]
        if not spy_features.empty:
            if as_of_date:
                spy_features = spy_features[spy_features["date"] <= as_of_date]
            if not spy_features.empty:
                latest = spy_features.iloc[-1]
                if "price_to_ma200" in latest and not pd.isna(latest["price_to_ma200"]):
                    return float(latest["price_to_ma200"]) >= 0.0

    # 2. Check spy_df directly
    if spy_df is not None and not spy_df.empty:
        df = spy_df.copy()
        if "date" in df.columns and as_of_date:
            df = df[df["date"] <= as_of_date]
        if "close" in df.columns and len(df) >= settings.regime_ma_window:
            ma200 = df["close"].rolling(settings.regime_ma_window).mean().iloc[-1]
            latest_close = df["close"].iloc[-1]
            if not pd.isna(ma200):
                return bool(latest_close >= ma200)

    logger.warning(
        "Could not determine SPY 200-day regime (insufficient SPY history). Defaulting to Risk-ON (True)."
    )
    return True


def rank_candidates(
    features_df: pd.DataFrame,
    model: Optional[TrainedModel] = None,
    spy_df: Optional[pd.DataFrame] = None,
    buy_bar: Optional[float] = None,
    signal_exit: Optional[float] = None,
    run_date: Optional[str] = None,
    sentiment_result: Optional[Any] = None,
    sector_result: Optional[Any] = None,
    earnings_result: Optional[Any] = None,
    macro_result: Optional[Any] = None,
) -> RankingResult:
    """
    Rank all universe stocks for a given decision date, incorporating
    Section 5 intelligence modifiers (News Sentiment, Sector Rotation,
    Earnings Calendar, and Macro Regime).
    """
    if features_df.empty:
        raise ValueError("Cannot rank candidates from an empty DataFrame.")

    macro_shift = macro_result.buy_bar_shift if macro_result else 0.0
    macro_label = macro_result.macro_regime if macro_result else "FAVORABLE"
    effective_buy_bar = (buy_bar if buy_bar is not None else settings.buy_bar) + macro_shift
    effective_signal_exit = signal_exit if signal_exit is not None else settings.signal_exit

    # 1. Resolve Active Model
    active_model = model if model is not None else load_active_model()

    # 2. Filter to Target Decision Date
    df = features_df.copy()
    if run_date is not None:
        df = df[df["date"] == run_date].copy()
        if df.empty:
            raise ValueError(f"No feature rows found for specified run_date: {run_date}")
    else:
        target_date = df["date"].max()
        df = df[df["date"] == target_date].copy()

    decision_date = str(df["date"].iloc[0])

    # 3. Check Market Regime
    regime_on = check_market_regime(spy_df=spy_df, features_df=features_df, as_of_date=decision_date)

    # 4. Generate Base Probabilities
    eval_df = df[df["ticker"].str.upper() != settings.benchmark.upper()].copy()
    if eval_df.empty:
        eval_df = df.copy()

    # A ticker with incomplete features (e.g. newly listed, < 200 bars of history) is skipped for
    # this day only; it must never stop the rest of the universe from being scored.
    feature_cols = [c for c in FEATURE_COLUMNS if c in eval_df.columns]
    incomplete = eval_df[feature_cols].isna().any(axis=1)
    if incomplete.any():
        for skipped in eval_df.loc[incomplete, "ticker"]:
            logger.warning(
                "RANKING SKIP %s on %s: incomplete features (insufficient history) — excluded for this day only.",
                skipped, decision_date,
            )
        eval_df = eval_df[~incomplete].copy()
    if eval_df.empty:
        logger.warning(
            "No tickers with complete features on %s: empty ranking (held positions are still "
            "managed by the risk engine's price-based exits).", decision_date,
        )
        return RankingResult(
            date=decision_date,
            regime_risk_on=regime_on,
            total_evaluated=0,
            macro_regime=macro_label,
            effective_buy_bar=effective_buy_bar,
        )

    raw_probs = active_model.predict_proba(eval_df)
    if hasattr(raw_probs, "ndim") and raw_probs.ndim == 2:
        probs = raw_probs[:, 1]
    else:
        probs = raw_probs

    # 4b. Performance-based ensemble weighting (Section 9 Item 4)
    # If multiple models are available, combine their probabilities using
    # performance-based weights from walk-forward evaluation.
    model_probabilities = {active_model.model_type: probs}
    try:
        from src.ml.train import get_performance_weights, get_weighted_ensemble_probability
        from src.db import repository
        wf_history = repository.get_walk_forward_history()
        if not wf_history.empty:
            model_aucs = {}
            for _, row in wf_history.iterrows():
                m = row["model_type"]
                if m not in model_aucs:
                    model_aucs[m] = []
                model_aucs[m].append(float(row["roc_auc"]))
            avg_aucs = {m: sum(v)/len(v) for m, v in model_aucs.items()}
            weights = get_performance_weights(avg_aucs)
            if weights and len(model_probabilities) > 1:
                final_probability = get_weighted_ensemble_probability(
                    model_probabilities, weights
                )
                logger.info("Performance ensemble weights: %s", weights)
    except Exception as exc:
        logger.warning("Performance ensemble failed, using default: %s", exc)

    # 5. Apply Section 5 Intelligence Modifiers & Log Breakdown
    base_probs_list: List[float] = []
    final_probs_list: List[float] = []
    sent_mods: List[float] = []
    sent_scores: List[float] = []
    sent_labels: List[str] = []
    sent_vetoes: List[bool] = []
    sectors: List[str] = []
    sec_ranks: List[Optional[int]] = []
    sec_mods: List[float] = []
    earn_dates: List[Optional[str]] = []
    earn_days_list: List[Optional[int]] = []
    earn_statuses: List[str] = []
    earn_badges: List[str] = []
    earn_blks: List[bool] = []

    for idx, (_, row) in enumerate(eval_df.iterrows()):
        ticker = str(row["ticker"]).upper()
        base_p = float(probs[idx])

        # A. Sentiment
        if sentiment_result and hasattr(sentiment_result, "scores") and ticker in sentiment_result.scores:
            s_obj = sentiment_result.scores[ticker]
            s_mod = s_obj.modifier
            s_score = s_obj.composite_score
            s_lbl = s_obj.sentiment_label
            s_veto = s_obj.is_veto
        else:
            s_mod = 0.0
            s_score = 0.0
            s_lbl = "NEUTRAL"
            s_veto = False

        # B. Sector Rotation
        sec_name = get_ticker_sector(ticker)
        if sector_result and hasattr(sector_result, "get_multiplier_for_ticker"):
            sec_mod = sector_result.get_multiplier_for_ticker(ticker)
            sec_rk = sector_result.get_rank_for_ticker(ticker)
        else:
            sec_mod = 0.0
            sec_rk = None

        # C. Earnings Calendar
        if earnings_result and hasattr(earnings_result, "get_info"):
            e_info = earnings_result.get_info(ticker)
            e_date = e_info.earnings_date if e_info else None
            e_days = e_info.days_until_earnings if e_info else None
            e_stat = e_info.status if e_info else "OK"
            e_bdg = e_info.badge_text if e_info else ""
            e_blk = e_info.is_blocked if e_info else False
        else:
            e_date = None
            e_days = None
            e_stat = "OK"
            e_bdg = ""
            e_blk = False

        # Compute Final Probability
        final_p = min(1.0, max(0.0, base_p + s_mod + sec_mod))

        # Format and log breakdown string
        # e.g.: "AAPL: Base=0.71 -> Sentiment+0.03 -> Sector+0.04 -> Macro-0.00 -> Earnings OK -> Final=0.78"
        earn_text = f"Earnings {e_stat}" + (f" ({e_days}d)" if e_days is not None else "")
        breakdown_msg = (
            f"{ticker}: Base={base_p:.2f} -> Sentiment{s_mod:+0.2f} -> "
            f"Sector{sec_mod:+0.2f} -> Macro{macro_shift:+0.2f} -> {earn_text} -> Final={final_p:.2f}"
        )
        logger.info(breakdown_msg)

        base_probs_list.append(base_p)
        final_probs_list.append(final_p)
        sent_mods.append(s_mod)
        sent_scores.append(s_score)
        sent_labels.append(s_lbl)
        sent_vetoes.append(s_veto)
        sectors.append(sec_name)
        sec_ranks.append(sec_rk)
        sec_mods.append(sec_mod)
        earn_dates.append(e_date)
        earn_days_list.append(e_days)
        earn_statuses.append(e_stat)
        earn_badges.append(e_bdg)
        earn_blks.append(e_blk)

    eval_df["probability"] = final_probs_list
    eval_df["base_probability"] = base_probs_list
    eval_df["sent_mod"] = sent_mods
    eval_df["sent_score"] = sent_scores
    eval_df["sent_label"] = sent_labels
    eval_df["sent_veto"] = sent_vetoes
    eval_df["sector"] = sectors
    eval_df["sec_rank"] = sec_ranks
    eval_df["sec_mod"] = sec_mods
    eval_df["earn_date"] = earn_dates
    eval_df["earn_days"] = earn_days_list
    eval_df["earn_status"] = earn_statuses
    eval_df["earn_badge"] = earn_badges
    eval_df["earn_blk"] = earn_blks

    # Secondary Factor Values for Tie-Breaking
    if "rel_strength_21" not in eval_df.columns:
        eval_df["rel_strength_21"] = 0.0
    if "price_to_ma50" not in eval_df.columns:
        eval_df["price_to_ma50"] = 0.0

    eval_df["rel_strength_21"] = eval_df["rel_strength_21"].fillna(0.0)
    eval_df["price_to_ma50"] = eval_df["price_to_ma50"].fillna(0.0)

    # 6. Multi-Factor Sorting
    eval_df = eval_df.sort_values(
        by=["probability", "rel_strength_21", "price_to_ma50", "ticker"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    ranked_list: List[RankedOpportunity] = []
    buy_list: List[RankedOpportunity] = []
    neutral_list: List[RankedOpportunity] = []
    exit_list: List[RankedOpportunity] = []

    for idx, row in eval_df.iterrows():
        rank = idx + 1
        p = float(row["probability"])
        base_p = float(row["base_probability"])
        ticker = str(row["ticker"])
        rs = float(row["rel_strength_21"])
        ma50 = float(row["price_to_ma50"])
        s_veto = bool(row["sent_veto"])
        e_blk = bool(row["earn_blk"])

        if p >= effective_buy_bar:
            tier = TIER_BUY
            is_buy = (not s_veto) and (not e_blk)
            is_exit = False
        elif p < effective_signal_exit:
            tier = TIER_EXIT
            is_buy = False
            is_exit = True
        else:
            tier = TIER_NEUTRAL
            is_buy = False
            is_exit = False

        opp = RankedOpportunity(
            ticker=ticker,
            date=decision_date,
            probability=p,
            base_probability=base_p,
            rank=rank,
            conviction_tier=tier,
            rel_strength_21=rs,
            price_to_ma50=ma50,
            is_buy_eligible=is_buy,
            is_exit_signal=is_exit,
            sentiment_score=float(row["sent_score"]),
            sentiment_label=str(row["sent_label"]),
            sentiment_modifier=float(row["sent_mod"]),
            sector=str(row["sector"]),
            sector_rank=row["sec_rank"],
            sector_modifier=float(row["sec_mod"]),
            earnings_date=row["earn_date"],
            days_until_earnings=row["earn_days"],
            earnings_status=str(row["earn_status"]),
            earnings_badge=str(row["earn_badge"]),
            macro_regime=macro_label,
            macro_buy_bar_shift=macro_shift,
            is_sentiment_veto=s_veto,
            is_earnings_blackout=e_blk,
        )

        ranked_list.append(opp)
        if is_buy:
            buy_list.append(opp)
        elif is_exit:
            exit_list.append(opp)
        else:
            neutral_list.append(opp)

    logger.info(
        "Ranked %d candidates for %s. Buy candidates (P >= %.2f): %d (un-capped). "
        "Market regime: %s | Macro: %s.",
        len(ranked_list),
        decision_date,
        effective_buy_bar,
        len(buy_list),
        "RISK-ON" if regime_on else "RISK-OFF",
        macro_label,
    )

    return RankingResult(
        date=decision_date,
        regime_risk_on=regime_on,
        total_evaluated=len(ranked_list),
        ranked_opportunities=ranked_list,
        top_buy_candidates=buy_list,
        neutral_candidates=neutral_list,
        exit_candidates=exit_list,
        macro_regime=macro_label,
        effective_buy_bar=effective_buy_bar,
    )


def rank_india_stocks(india_data: dict) -> list:
    """
    Ranks Indian NSE stocks using same logic as US stocks.
    Returns list of dicts with ticker, score, action.

    Parameters
    ----------
    india_data : dict
        Dict of {ticker: DataFrame} from fetch_india_market_data()

    Returns
    -------
    list of {ticker, probability, action, price}
    """
    from config.settings import settings
    from src.ml.train import load_model
    from src.ml.evaluate import ACTIVE_MODEL_FILENAME
    import numpy as np

    results = []

    try:
        active_model = load_model(
            settings.data_models_dir / ACTIVE_MODEL_FILENAME
        )
    except Exception as exc:
        logger.warning("Could not load model for India ranking: %s", exc)
        return results

    for ticker, df in india_data.items():
        try:
            if df.empty or len(df) < 50:
                continue

            from src.features.engineer import engineer_features
            features_df = engineer_features(df, ticker=ticker)
            if features_df.empty:
                continue

            from src.features.engineer import FEATURE_COLUMNS
            latest = features_df.tail(1)[FEATURE_COLUMNS]
            if latest.isnull().any().any():
                continue

            prob = float(active_model.predict_proba(latest)[0][1])

            if prob >= settings.buy_bar:
                action = "BUY"
            elif prob < settings.signal_exit:
                action = "SELL"
            else:
                action = "HOLD"

            results.append({
                "ticker": ticker,
                "probability": round(prob, 4),
                "action": action,
                "price": float(df["close"].iloc[-1]),
            })

        except Exception as exc:
            logger.warning("India ranking failed for %s: %s", ticker, exc)
            continue

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results
