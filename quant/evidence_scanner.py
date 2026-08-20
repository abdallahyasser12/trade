"""Production scanner extension with evidence, microstructure and setup intelligence.

Adaptive-v2 execution evidence is lifecycle-scoped.  A newly promoted rung cannot
bootstrap from zero trades, so it may temporarily bridge only the immediately
preceding qualified rung.  After 40 current-rung real trades, current-rung evidence
becomes authoritative.  Arbitrary old history is never pooled into v2 entry
confidence.

Live depth, sleeve adaptation and edge intelligence remain negative-only: none of
them can manufacture or boost alpha above the evidence-backed baseline.  The edge
layer only demotes internally weak/noisy setups in the global opportunity ranking.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from quant.contracts import StrategyState
from quant.edge_intelligence import EdgeIntelligenceConfig, assess_edge
from quant.marketplace import MIN_SAMPLE, compute_strategy_health
from quant.scanner import QuantScanner
from quant.sleeve_supervision import compute_sleeve_health, sleeve_from_entry_reason

ADAPTIVE_STRATEGY_ID = "crypto_intraday_adaptive"
CURRENT_RUNG_AUTHORITY_TRADES = 40
_EDGE_CONFIG = EdgeIntelligenceConfig()


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


def _parse_time(raw):
    if not raw:
        return None
    try:
        timestamp = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _state_since(store, strategy_id: str):
    row = store.get_strategy_state(strategy_id)
    return _parse_time(row.get("since") if row else None)


def _immediately_prior_rung_start(store, strategy_id: str, state: StrategyState):
    """Return (previous_state, previous_state_start) from immutable transition history."""
    history = store.get_promotion_history(strategy_id, limit=500)
    current_index = None
    current_event = None
    for index, event in enumerate(history):
        if event.get("to_state") == state.value:
            current_index = index
            current_event = event
            break
    if current_event is None or current_index is None:
        return None, None
    try:
        previous_state = StrategyState(str(current_event.get("from_state")))
    except (TypeError, ValueError):
        return None, None
    for event in history[current_index + 1 :]:
        if event.get("to_state") == previous_state.value:
            return previous_state, _parse_time(event.get("timestamp"))
    return previous_state, None


def adaptive_deployment_health(store, lifecycle, strategy_id: str):
    """Current-rung health, with a bounded one-rung bootstrap bridge when necessary."""
    state = lifecycle.state(strategy_id)
    current_since = _state_since(store, strategy_id)
    current = compute_strategy_health(
        store,
        strategy_id,
        state,
        include_shadow=False,
        since=current_since,
    )
    if current.trades >= CURRENT_RUNG_AUTHORITY_TRADES:
        return current, "current_rung"

    previous_state, previous_since = _immediately_prior_rung_start(
        store,
        strategy_id,
        state,
    )
    if previous_state is None or previous_since is None:
        return current, "current_rung_insufficient"
    if previous_state not in {
        StrategyState.SHADOW,
        StrategyState.PAPER,
        StrategyState.LIVE_CANARY,
        StrategyState.LIVE,
        StrategyState.REDUCED_RISK,
    }:
        return current, "current_rung_insufficient"

    bridge = compute_strategy_health(
        store,
        strategy_id,
        state,
        include_shadow=previous_state is StrategyState.SHADOW,
        since=previous_since,
    )
    if bridge.trades >= MIN_SAMPLE:
        return bridge, f"bridge_{previous_state.value.lower()}_to_{state.value.lower()}"
    return current, "current_rung_insufficient"


class EvidenceQuantScanner(QuantScanner):
    """Scanner whose entries are evidence-, setup- and adverse-selection-aware."""

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
        health_source = "scanner_default"
        if entry.strategy_id == ADAPTIVE_STRATEGY_ID:
            health, health_source = adaptive_deployment_health(
                self.store,
                self.lifecycle,
                entry.strategy_id,
            )

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
        candidate.details["deployment_health_source"] = health_source

        sleeve = sleeve_from_entry_reason(candidate.signal_reason)
        if sleeve is not None and entry.strategy_id == ADAPTIVE_STRATEGY_ID:
            sleeve_health = compute_sleeve_health(
                self.store,
                entry.strategy_id,
                sleeve,
                since=_state_since(self.store, entry.strategy_id),
            )
            candidate.details["sleeve_health"] = sleeve_health.to_dict()
            candidate.confidence *= sleeve_health.confidence_multiplier
            candidate.confidence = max(0.0, min(base_confidence, candidate.confidence))

        quote = quotes.get(decision.symbol)
        imbalance = depth_imbalance(quote)
        quality = adverse_depth_quality(imbalance)
        candidate.details["depth_imbalance"] = imbalance
        candidate.details["adverse_depth_quality"] = quality
        ranker_input = context.get("ranker_input")
        if ranker_input is not None:
            ranker_input.liquidity_quality = min(
                float(ranker_input.liquidity_quality), quality
            )

        edge = assess_edge(
            prepared.get(decision.symbol),
            decision.timestamp,
            strategy_id=entry.strategy_id,
            signal_reason=candidate.signal_reason,
            quote=quote,
            config=_EDGE_CONFIG,
        )
        candidate.details["edge_intelligence"] = edge.to_dict()
        # A non-measurable assessment journals None so aggregate consumers can
        # never mistake "could not measure" for a genuinely measured mid score.
        candidate.details["edge_score"] = edge.score if edge.measurable else None
        if ranker_input is not None:
            ranker_input.setup_quality = min(
                float(ranker_input.setup_quality),
                float(edge.quality_multiplier),
            )
        return candidate, context
