"""Opportunity scoring and the global cross-market ranker (spec §§21-22).

The portfolio's question is "where is the best next unit of risk?", not
"is this symbol a BUY?".  Every candidate — regardless of market or
horizon — is normalized into expected NET alpha (gross minus the
execution estimate minus an uncertainty margin), then scored:

    score = net_alpha_bps
            x confidence
            x regime_compatibility        (measured, neutral when unmeasured)
            x strategy_reliability        (measured, neutral prior when not)
            x liquidity_quality
            x (1 - drawdown_penalty)
            x (1 - tail_risk_penalty)

Every input yields exactly one journaled decision: EXECUTE candidates get
ranks 1..N, everything else keeps its named rejection or abstention
reasons.  Nothing is silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from quant.abstention import AbstentionPolicy
from quant.contracts import (
    DecisionOutcome,
    ExecutionEstimate,
    OpportunityCandidate,
    OpportunityDecision,
    QuantAction,
    RegimeAssessment,
)
from quant.execution import ExecutionQualityEngine


@dataclass(frozen=True)
class OpportunityScoringConfig:
    min_net_alpha_bps: float = 1.0
    # Unmeasured strategy reliability maps to this neutral prior rather
    # than zeroing the score: a new strategy is unproven, not disproven —
    # and it can only reach EXECUTE through the lifecycle gates anyway.
    reliability_neutral: float = 0.5
    drawdown_penalty_per_pct: float = 0.02
    tail_risk_penalty_atr_percentile: float = 90.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.min_net_alpha_bps) or self.min_net_alpha_bps < 0:
            errors.append("Minimum net alpha must be finite and non-negative.")
        if not (0.0 < self.reliability_neutral <= 1.0):
            errors.append("Neutral reliability must be in (0, 1].")
        if not math.isfinite(self.drawdown_penalty_per_pct) or self.drawdown_penalty_per_pct < 0:
            errors.append("Drawdown penalty must be finite and non-negative.")
        if not (0.0 < self.tail_risk_penalty_atr_percentile <= 100.0):
            errors.append("Tail-risk percentile must be in (0, 100].")
        return errors


def score_components(
    config: OpportunityScoringConfig,
    candidate: OpportunityCandidate,
    *,
    net_alpha_bps: float,
    regime_compatibility: Optional[float] = None,
    strategy_reliability: Optional[float] = None,
    liquidity_quality: float = 1.0,
    drawdown_pct: float = 0.0,
    atr_percentile: Optional[float] = None,
) -> dict[str, float]:
    """Multiplier/penalty components, each bounded and documented.

    regime_compatibility / strategy_reliability are in (0, 1] when
    measured; None maps to neutral (1.0 / the neutral prior).  Drawdown
    and tail penalties are in [0, 0.95] so a deep-drawdown book shrinks
    scores without ever flipping their sign.
    """
    regime_multiplier = 1.0 if regime_compatibility is None else max(0.0, min(1.0, regime_compatibility))
    reliability_multiplier = (
        config.reliability_neutral
        if strategy_reliability is None
        else max(0.0, min(1.0, strategy_reliability))
    )
    liquidity = max(0.0, min(1.0, liquidity_quality))
    drawdown_penalty = min(0.95, max(0.0, drawdown_pct) * config.drawdown_penalty_per_pct)
    tail_penalty = 0.0
    if atr_percentile is not None and math.isfinite(atr_percentile):
        threshold = config.tail_risk_penalty_atr_percentile
        if atr_percentile > threshold:
            tail_penalty = min(
                0.95, (atr_percentile - threshold) / max(1e-9, 100.0 - threshold)
            )
    score = (
        net_alpha_bps
        * candidate.confidence
        * regime_multiplier
        * reliability_multiplier
        * liquidity
        * (1.0 - drawdown_penalty)
        * (1.0 - tail_penalty)
    )
    return {
        "net_alpha_bps": net_alpha_bps,
        "confidence": candidate.confidence,
        "regime_multiplier": regime_multiplier,
        "reliability_multiplier": reliability_multiplier,
        "liquidity_quality": liquidity,
        "drawdown_penalty": drawdown_penalty,
        "tail_penalty": tail_penalty,
        "score": score,
    }


@dataclass
class RankerInput:
    """One candidate plus everything the ranker needs to judge it."""

    candidate: OpportunityCandidate
    execution_estimate: ExecutionEstimate
    regime: Optional[RegimeAssessment] = None
    regime_compatibility: Optional[float] = None
    strategy_reliability: Optional[float] = None
    liquidity_quality: float = 1.0
    drawdown_pct: float = 0.0
    atr_percentile: Optional[float] = None
    abstention_kwargs: dict[str, Any] = field(default_factory=dict)


class GlobalOpportunityRanker:
    """One queue across every market, strategy, and horizon."""

    def __init__(
        self,
        scoring_config: Optional[OpportunityScoringConfig],
        abstention_policy: AbstentionPolicy,
        execution_engine: ExecutionQualityEngine,
    ):
        self.scoring_config = scoring_config or OpportunityScoringConfig()
        errors = self.scoring_config.validate()
        if errors:
            raise ValueError("Invalid scoring configuration: " + "; ".join(errors))
        self.abstention_policy = abstention_policy
        self.execution_engine = execution_engine

    def rank(self, inputs: list[RankerInput]) -> list[OpportunityDecision]:
        """Judge every input; return one decision per input, ranked EXECUTEs first.

        Order within outcomes is deterministic: score descending, then
        symbol, then strategy_id.
        """
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, OpportunityDecision]] = []
        others: list[OpportunityDecision] = []

        for item in inputs:
            candidate = item.candidate
            validation_errors = candidate.validate()
            if validation_errors:
                others.append(
                    OpportunityDecision(
                        candidate=candidate,
                        outcome=DecisionOutcome.REJECT,
                        rank=None,
                        expected_cost_bps=None,
                        expected_net_alpha_bps=None,
                        opportunity_score=None,
                        risk_allocation_pct=None,
                        rejection_reasons=validation_errors,
                        execution_estimate=item.execution_estimate,
                        regime=item.regime,
                        decided_at=now,
                    )
                )
                continue

            if candidate.action is not QuantAction.BUY:
                others.append(
                    OpportunityDecision(
                        candidate=candidate,
                        outcome=DecisionOutcome.REJECT,
                        rank=None,
                        expected_cost_bps=None,
                        expected_net_alpha_bps=None,
                        opportunity_score=None,
                        risk_allocation_pct=None,
                        rejection_reasons=["not_an_entry_candidate"],
                        execution_estimate=item.execution_estimate,
                        regime=item.regime,
                        decided_at=now,
                    )
                )
                continue

            verdict = self.abstention_policy.evaluate(
                candidate, **item.abstention_kwargs
            )
            if verdict.abstain:
                others.append(
                    OpportunityDecision(
                        candidate=candidate,
                        outcome=DecisionOutcome.ABSTAIN,
                        rank=None,
                        expected_cost_bps=item.execution_estimate.round_trip_bps,
                        expected_net_alpha_bps=None,
                        opportunity_score=None,
                        risk_allocation_pct=None,
                        rejection_reasons=verdict.reasons,
                        execution_estimate=item.execution_estimate,
                        regime=item.regime,
                        decided_at=now,
                    )
                )
                continue

            tradeable, net_alpha = self.execution_engine.net_alpha_after_costs(
                candidate.expected_gross_alpha_bps, item.execution_estimate
            )
            reasons: list[str] = []
            if not item.execution_estimate.viable:
                reasons.append("execution_not_viable")
                if item.execution_estimate.reason:
                    reasons.append(item.execution_estimate.reason)
            elif not tradeable or net_alpha < self.scoring_config.min_net_alpha_bps:
                reasons.append("net_alpha_below_minimum")
            if reasons:
                others.append(
                    OpportunityDecision(
                        candidate=candidate,
                        outcome=DecisionOutcome.REJECT,
                        rank=None,
                        expected_cost_bps=item.execution_estimate.round_trip_bps,
                        expected_net_alpha_bps=(
                            net_alpha if math.isfinite(net_alpha) else None
                        ),
                        opportunity_score=None,
                        risk_allocation_pct=None,
                        rejection_reasons=reasons,
                        execution_estimate=item.execution_estimate,
                        regime=item.regime,
                        decided_at=now,
                    )
                )
                continue

            components = score_components(
                self.scoring_config,
                candidate,
                net_alpha_bps=net_alpha,
                regime_compatibility=item.regime_compatibility,
                strategy_reliability=item.strategy_reliability,
                liquidity_quality=item.liquidity_quality,
                drawdown_pct=item.drawdown_pct,
                atr_percentile=item.atr_percentile,
            )
            decision = OpportunityDecision(
                candidate=candidate,
                outcome=DecisionOutcome.EXECUTE,
                rank=None,
                expected_cost_bps=item.execution_estimate.round_trip_bps,
                expected_net_alpha_bps=net_alpha,
                opportunity_score=components["score"],
                risk_allocation_pct=None,
                rejection_reasons=[],
                execution_estimate=item.execution_estimate,
                regime=item.regime,
                decided_at=now,
            )
            decision.candidate.details["score_components"] = components
            scored.append((components["score"], decision))

        scored.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].candidate.symbol,
                pair[1].candidate.strategy_id,
            )
        )
        ranked: list[OpportunityDecision] = []
        for position, (_, decision) in enumerate(scored, start=1):
            decision.rank = position
            ranked.append(decision)
        return ranked + others
