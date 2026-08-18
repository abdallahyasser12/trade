"""Production scanner extension with evidence-based confidence.

A strategy's hit rate is not a probability that it has positive expectancy. Trend
systems can be profitable below 50% wins when payoff asymmetry is favorable. The
production scanner therefore converts the measured per-trade Sharpe and sample size
into a conservative normal-approximation confidence that true expectancy is above
zero. Cost viability, lifecycle state, abstention and portfolio gates remain
unchanged.
"""

from __future__ import annotations

import math

from quant.marketplace import MIN_SAMPLE
from quant.scanner import QuantScanner


def expectancy_confidence(health) -> float:
    """Approximate P(E[R] > 0) from measured trade-level mean/std evidence.

    ``StrategyHealth.sharpe`` is the unannualized per-trade mean/std ratio. Under
    the normal approximation, ``sharpe * sqrt(n)`` is the z-like statistic for a
    zero-expectancy null. This is deliberately only a ranking/abstention input; it
    never replaces the walk-forward/DSR promotion gates.
    """
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
    # A non-positive measured expectancy must never be represented as strong
    # positive-alpha confidence because of a pathological/rounded Sharpe input.
    if expectancy <= 0:
        confidence = min(confidence, 0.5)
    return max(0.0, min(1.0, confidence))


class EvidenceQuantScanner(QuantScanner):
    """QuantScanner whose candidate confidence measures evidence, not win rate."""

    def _candidate_from_decision(self, entry, decision, health, *args, **kwargs):
        candidate, context = super()._candidate_from_decision(
            entry, decision, health, *args, **kwargs
        )
        if candidate is not None:
            candidate.confidence = expectancy_confidence(health)
            candidate.details["hit_rate"] = health.hit_rate
            candidate.details["expectancy_confidence"] = candidate.confidence
        return candidate, context
