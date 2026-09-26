"""
Decision observability (V1-safe, measurement only).

build_decision_records() explains, for every ticker the model ranked in a trading cycle, what V1
decided and why. It only reads results the pipeline has already produced (ranking, risk engine,
allocation) after the decisions are final; nothing here feeds back into any decision.

decision:
  BUY    - an order was queued for the next open (order_reason says why it won its slot)
  SELL   - an exit order was queued for a held position (reason = exit rule)
  HOLD   - a held position was kept
  REJECT - not bought; reason is explicit (below threshold, vetoed by a named risk rule, ...)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from config.settings import settings
from src.ranking.ranking import TIER_BUY


def _implied_size_tier(p: float) -> str:
    if p < settings.confidence_half_position_max:
        return "HALF"
    if p < settings.confidence_three_quarter_max:
        return "THREE_QUARTER"
    return "FULL"


def _enum_value(v: Any) -> Optional[str]:
    return getattr(v, "value", v) if v is not None else None


def build_decision_records(
    ranking_result: Any,
    risk_result: Any,
    orders: Iterable[Any],
    held_tickers: Iterable[str],
    macro_result: Any,
    intelligence: Dict[str, Dict[str, Any]],
    model_sha256: str,
) -> List[Dict[str, Any]]:
    held = {t.upper() for t in held_tickers}
    order_map = {(o.ticker.upper(), o.action.upper()): o for o in orders}
    exits = {d.ticker.upper(): d for d in risk_result.exit_orders}
    kept = {d.ticker.upper(): d for d in risk_result.held_unchanged}
    approved = {d.ticker.upper(): d for d in risk_result.buy_orders}
    vetoed: Dict[str, Any] = {}
    for d in risk_result.vetoed_orders:
        vetoed.setdefault(d.ticker.upper(), d)          # first veto the ticker hit
    candidates = [t.upper() for t in getattr(risk_result, "evaluated_candidates", [])]
    corr_checks = getattr(risk_result, "correlation_checks", {}) or {}

    ranking_bar = float(ranking_result.effective_buy_bar)
    risk_bar = float(risk_result.active_buy_bar)
    buy_gate = max(ranking_bar, risk_bar)
    macro_state = {
        "regime": getattr(macro_result, "macro_regime", None),
        "size_multiplier": getattr(macro_result, "position_size_multiplier", 1.0) if macro_result else 1.0,
        "buy_bar_shift": getattr(macro_result, "buy_bar_shift", 0.0) if macro_result else 0.0,
        "input_status": (intelligence.get("macro") or {}).get("status"),
    }
    regime_state = {
        "spy_above_200d": bool(ranking_result.regime_risk_on),
        "volatility_regime": risk_result.volatility_regime,
        "circuit_breaker_active": bool(risk_result.circuit_breaker_active),
    }
    slots_available = settings.max_positions - (risk_result.positions_after - len(risk_result.buy_orders))
    bought_in_order = [t for t in candidates if t in approved and (t, "BUY") in order_map]
    lost_for_slots = [t for t in candidates if t in vetoed and _enum_value(vetoed[t].veto_reason) == "MAX_POSITIONS_REACHED"]

    records: List[Dict[str, Any]] = []
    for opp in ranking_result.ranked_opportunities:
        t = opp.ticker.upper()
        p = float(opp.probability)
        corr = corr_checks.get(t) or {"checked": False}
        size_tier, tier_source, order_reason = None, None, None

        if t in held:
            if t in exits and (t, "SELL") in order_map:
                ex = exits[t]
                decision, reason = "SELL", f"{ex.reason}: {ex.details}"
                order_reason = f"Exit rule {ex.reason} fired on a held position ({ex.details})"
            else:
                decision = "HOLD"
                k = kept.get(t)
                reason = f"{k.reason}: {k.details}" if (k and k.reason) else "HELD: no stop-loss, take-profit or signal exit triggered"
        elif t in approved and (t, "BUY") in order_map:
            d = approved[t]
            decision, reason = "BUY", "QUALIFIED_CONVICTION_BUY"
            size_tier, tier_source = d.confidence_tier, "risk_engine"
            k = bought_in_order.index(t) + 1
            corr_txt = (f"max r={corr.get('max_correlation')} vs {corr.get('max_correlated_ticker')}, {corr.get('result')}"
                        if corr.get("result") else "no positions held to compare")
            order_reason = (
                f"Won slot {k} of {slots_available}: evaluated #{candidates.index(t) + 1} of {len(candidates)} buy "
                f"candidates, ordered by final P {p:.4f} then rel_strength_21 {opp.rel_strength_21:.4f}, "
                f"price_to_ma50 {opp.price_to_ma50:.4f}; passed regime, circuit-breaker, correlation ({corr_txt}), "
                f"cash-reserve and min-trade checks; sized {d.confidence_tier} (${d.allocated_amount:.2f})."
                + (f" Lost out for lack of slots: {lost_for_slots}." if lost_for_slots else "")
            )
        elif t in approved:
            decision, reason = "REJECT", "ALLOCATION_DROPPED: approved by the risk engine but no order was generated"
            size_tier, tier_source = approved[t].confidence_tier, "risk_engine"
        elif t in vetoed:
            d = vetoed[t]
            decision, reason = "REJECT", f"{_enum_value(d.veto_reason)}: {d.details}"
            if d.confidence_tier:
                size_tier, tier_source = d.confidence_tier, "risk_engine"
        elif t in candidates:
            decision = "REJECT"
            reason = (f"BELOW_ADAPTIVE_BUY_BAR: final P {p:.4f} < {risk_bar:.2f} risk-engine bar "
                      f"({risk_result.volatility_regime})" if p < risk_bar else "NOT_APPROVED: evaluated but not approved")
        elif opp.conviction_tier == TIER_BUY and not opp.is_buy_eligible:
            decision = "REJECT"
            reason = ("EARNINGS_BLACKOUT: earnings within the blackout window" if opp.earnings_status == "BLACKOUT"
                      else f"SENTIMENT_VETO: sentiment {opp.sentiment_label}")
        else:
            decision, reason = "REJECT", f"BELOW_BUY_THRESHOLD: final P {p:.4f} < {ranking_bar:.2f} buy bar"

        if size_tier is None and decision == "REJECT" and p >= buy_gate:
            size_tier, tier_source = _implied_size_tier(p), "implied_from_probability"

        records.append({
            "ticker": t,
            "model_sha256": model_sha256,
            "raw_probability": round(float(opp.base_probability), 6),
            "sector": opp.sector,
            "sector_rank": opp.sector_rank,
            "sector_modifier": float(opp.sector_modifier),
            "macro_state": macro_state,
            "regime_state": regime_state,
            "correlation": corr,
            "final_probability": round(p, 6),
            "buy_threshold": round(buy_gate, 4),
            "size_tier": size_tier,
            "decision": decision,
            "reason": reason,
            "order_reason": order_reason,
            "details": {
                "rank": opp.rank,
                "conviction_tier": opp.conviction_tier,
                "ranking_buy_bar": ranking_bar,
                "risk_adaptive_buy_bar": risk_bar,
                "sentiment_modifier": float(opp.sentiment_modifier),
                "sentiment_label": opp.sentiment_label,
                "earnings_status": opp.earnings_status,
                "size_tier_source": tier_source,
                "held_before": t in held,
                "intelligence_status": {k: v.get("status") for k, v in intelligence.items()},
            },
        })

    # Held positions the model could not rank (incomplete features) are still managed by the risk
    # engine's price-based exits; log them too so every position decision is visible.
    ranked = {r["ticker"] for r in records}
    for t in sorted(held - ranked):
        if t in exits and (t, "SELL") in order_map:
            ex = exits[t]
            decision, reason = "SELL", f"UNRANKED (incomplete features); {ex.reason}: {ex.details}"
            order_reason = f"Exit rule {ex.reason} fired on a held position ({ex.details})"
        else:
            k = kept.get(t)
            decision = "HOLD"
            reason = "UNRANKED (incomplete features); " + (f"{k.reason}: {k.details}" if (k and k.reason)
                                                           else "no price-based exit triggered")
            order_reason = None
        records.append({
            "ticker": t, "model_sha256": model_sha256, "raw_probability": None, "sector": None,
            "sector_rank": None, "sector_modifier": 0.0, "macro_state": macro_state, "regime_state": regime_state,
            "correlation": {"checked": False}, "final_probability": None, "buy_threshold": round(buy_gate, 4),
            "size_tier": None, "decision": decision, "reason": reason, "order_reason": order_reason,
            "details": {"held_before": True, "ranked": False,
                        "intelligence_status": {k: v.get("status") for k, v in intelligence.items()}},
        })
    return records
