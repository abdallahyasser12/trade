"""Production scanner extension with evidence, microstructure and sleeve supervision.

Trade hit-rate is not confidence: positive expectancy can coexist with sub-50% hit
rate.  Confidence is derived from trade-level expectancy evidence.

Live bid/ask depth is used only as an adverse-selection penalty.  Favorable imbalance
never manufactures alpha.  Adaptive-v2 sleeve supervision is also negative-only:
after enough current-lifecycle real trades, deteriorating sleeves can reduce or lose
entry privilege, but recent winners never receive a confidence boost above baseline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from quant.marketplace import MIN_SAMPLE
from quant.scanner import QuantScanner
from quant.sleeve_supervision import compute_sleeve_health, sleeve_from_entry_reason


def expectancy_confidence(health) -> float:
    """Approximate P(E[R] > 0) from measured trade-level mean/std evidence."""
    if health is None or health.trades < MIN_SAMPLE:
        return 0.0
    sharpe = getattr(health, "sharpe", None)
    expectancy = getattr(health, "expectancy_r", None)
    if sharpe is None or expectancy is None:
        return 0.0
    try:
        sharpe = float(sharpe)
        expectancy = float(expectancy)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sharpe) or not math.isfinite(expectancy):
        return 0.0
    z = sharpe * math.sqrt(float(health.trades))
    confidence = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if expectancy <= 0:
        confidence = min(confidence, 0.5)
    return max(0.0, min(1.0, confidence))


def depth_imbalance(quote) -> float | None:
    """Return (bid_size-ask_size)/(bid_size+ask_size) when depth is measurable."""
    if quote is None:
        return None
    bid_size = getattr(quote, "bid_size", None)
    ask_size = getattr(quote, "ask_size", None)
    try:
        bid = float(bid_size)
        ask = float(ask_size)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0 or ask < 0:
        return None
    total = bid + ask
    if total <= 0:
        return None
    return max(-1.0, min(1.0, (bid - ask) / total))


def adverse_depth_quality(imbalance: float | None) -> float:
    """Neutral/positive depth = 1; adverse ask-heavy depth can only penalize."""
    if imbalance is None or not math.isfinite(imbalance) or imbalance >= 0:
        return 1.0
    return max(0.50, 1.0 + 0.50 * imbalance)


def _state_since(store, strategy_id: str):
    row = store.get_strategy_state(strategy_id)
    raw = row.get("since") if row else None
    if not raw:
        return None
    try:
        timestamp = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class EvidenceQuantScanner(QuantScanner):
    """Scanner whose entry confidence is evidence- and adverse-selection-aware."""

    def _candidate_from_decision(
        self,
        entry,
        decision,
        health,
        prepared,
        regime,
        drawdown_pct,
        quotes,
        equity,
        now,
    ):
        candidate, context = super()._candidate_from_decision(
            entry,
            decision,
            health,
            prepared,
            regime,
            drawdown_pct,
            quotes,
            equity,
            now,
        )
        if candidate is None:
            return candidate, context

        base_confidence = expectancy_confidence(health)
        candidate.confidence = base_confidence
        candidate.details["hit_rate"] = health.hit_rate
        candidate.details["expectancy_confidence"] = base_confidence

        # Adaptive v2: current-state forward evidence can only remove privilege.
        # No sleeve ever receives >1x confidence from recent PnL.
        sleeve = sleeve_from_entry_reason(candidate.signal_reason)
        if sleeve is not None and entry.strategy_id == "crypto_intraday_adaptive":
            sleeve_health = compute_sleeve_health(
                self.store,
                entry.strategy_id,
                sleeve,
                since=_state_since(self.store, entry.strategy_id),
            )
            candidate.details["sleeve_health"] = sleeve_health.to_dict()
            candidate.confidence *= sleeve_health.confidence_multiplier
            candidate.confidence = max(0.0, min(base_confidence, candidate.confidence))

        imbalance = depth_imbalance(quotes.get(decision.symbol))
        quality = adverse_depth_quality(imbalance)
        candidate.details["depth_imbalance"] = imbalance
        candidate.details["adverse_depth_quality"] = quality
        ranker_input = context.get("ranker_input")
        if ranker_input is not None:
            ranker_input.liquidity_quality = min(
                float(ranker_input.liquidity_quality), quality
            )
        return candidate, context
