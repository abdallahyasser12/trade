"""Typed structures crossing quant-desk module boundaries.

Like ``models.py`` and ``crypto_models.py``, this module defines every shape
that crosses a boundary inside the quant layer (features -> strategies ->
ranking -> portfolio -> execution gate -> journal) so the rest of the package
relies on one set of shapes.  It imports only the two existing model modules
and never performs I/O.

Numeric conventions
-------------------
* Alpha and cost figures are expressed in basis points of notional (bps).
* Probabilities and confidences are in [0, 1].
* Timestamps are timezone-aware datetimes serialized as ISO-8601 UTC.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

from models import BotAction


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


class Market(str, Enum):
    """Venue-independent market classification of an instrument."""

    US_EQUITY = "us_equity"
    CRYPTO = "crypto"


class QuantAction(str, Enum):
    """Strategy intents.  Extends ``BotAction`` with an explicit ABSTAIN.

    ABSTAIN means: a prediction exists but uncertainty is too high to
    justify capital.  Execution layers treat it exactly like HOLD; the
    journal records it separately so abstention behavior stays measurable.
    """

    BUY = "BUY"
    EXIT = "EXIT"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"

    def to_bot_action(self) -> BotAction:
        if self is QuantAction.ABSTAIN:
            return BotAction.HOLD
        return BotAction(self.value)


class StrategyState(str, Enum):
    """Deployment lifecycle of one strategy.  Order matters: transitions
    may move one rung at a time toward LIVE and any number of rungs away
    from it (demotion is always allowed to be abrupt)."""

    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE_CANARY = "LIVE_CANARY"
    LIVE = "LIVE"
    REDUCED_RISK = "REDUCED_RISK"
    QUARANTINE = "QUARANTINE"


# States in which a strategy's orders may reach a real-money account.
LIVE_STATES = frozenset({StrategyState.LIVE_CANARY, StrategyState.LIVE, StrategyState.REDUCED_RISK})

# The promotion ladder, in order.  REDUCED_RISK and QUARANTINE sit outside
# the ladder: they are entered by demotion only.
PROMOTION_LADDER = (
    StrategyState.RESEARCH,
    StrategyState.SHADOW,
    StrategyState.PAPER,
    StrategyState.LIVE_CANARY,
    StrategyState.LIVE,
)


class RegimeLabel(str, Enum):
    """Market regimes the regime engine scores.  The engine outputs a
    probability for every label rather than a single hard classification."""

    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    RANGE = "RANGE"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    VOL_EXPANSION = "VOL_EXPANSION"
    VOL_CONTRACTION = "VOL_CONTRACTION"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"


class DecisionOutcome(str, Enum):
    """What the desk decided to do with one opportunity candidate."""

    EXECUTE = "EXECUTE"      # forwarded to an execution engine
    SHADOW = "SHADOW"        # virtual fill only (strategy not deployed)
    REJECT = "REJECT"        # positive signal, failed a deterministic gate
    ABSTAIN = "ABSTAIN"      # uncertainty too high to justify capital


@dataclass(frozen=True)
class RegimeAssessment:
    """Probabilistic regime snapshot for one market at one timestamp.

    ``probabilities`` maps every ``RegimeLabel`` value to [0, 1]; trend
    labels (bull/bear/range) sum to 1, volatility and liquidity labels are
    independent binary probabilities.  ``confidence`` reflects how much
    evidence the inputs carried (short history or missing inputs lower it).
    ``inputs`` snapshots the raw measurements the assessment was built from
    so every regime call is explainable after the fact.
    """

    market: Market
    timestamp: Optional[datetime]
    probabilities: Mapping[str, float]
    confidence: float
    inputs: Mapping[str, Optional[float]] = field(default_factory=dict)

    def probability(self, label: RegimeLabel) -> float:
        return float(self.probabilities.get(label.value, 0.0))

    @property
    def dominant_trend_label(self) -> RegimeLabel:
        trend_labels = (
            RegimeLabel.STRONG_BULL,
            RegimeLabel.WEAK_BULL,
            RegimeLabel.RANGE,
            RegimeLabel.WEAK_BEAR,
            RegimeLabel.STRONG_BEAR,
        )
        return max(trend_labels, key=self.probability)

    @property
    def risk_on_probability(self) -> float:
        """P(risk-on): bullish trend mass, discounted by liquidity stress."""
        bull = self.probability(RegimeLabel.STRONG_BULL) + self.probability(
            RegimeLabel.WEAK_BULL
        )
        stress = self.probability(RegimeLabel.LIQUIDITY_STRESS)
        return max(0.0, min(1.0, bull * (1.0 - stress)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "timestamp": _iso(self.timestamp),
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "inputs": dict(self.inputs),
        }


@dataclass(frozen=True)
class ExecutionEstimate:
    """Pre-trade execution-quality estimate for one prospective order.

    ``cost_source`` is ``"measured"`` when the slippage figure comes from
    the rolling execution-calibration measurements, ``"assumed"`` when it
    falls back to configured conservative assumptions.  ``viable`` is False
    when the estimate itself could not be built safely (missing quote,
    zero liquidity) — consumers must fail closed on it.
    """

    symbol: str
    spread_bps: float
    slippage_bps: float
    fee_bps: float
    round_trip_bps: float
    participation_pct: Optional[float]
    cost_source: str
    sample_count: int
    viable: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpportunityCandidate:
    """One scored, explainable trade candidate produced by a strategy.

    ``expected_gross_alpha_bps`` is the strategy's own estimate of edge
    before costs; the ranker computes ``expected_net_alpha_bps`` by
    subtracting the execution estimate plus an uncertainty margin.  A
    candidate is comparable across markets and horizons because every
    field is normalized (bps, [0,1] confidences).
    """

    symbol: str
    market: Market
    timeframe: str
    strategy_id: str
    strategy_version: str
    action: QuantAction
    timestamp: Optional[datetime]
    close: Optional[float]
    stop_price: Optional[float]
    atr: Optional[float]
    expected_gross_alpha_bps: float
    confidence: float
    signal_reason: str
    horizon_bars: int = 0
    filters: dict[str, bool] = field(default_factory=dict)
    features: dict[str, Optional[float]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.symbol:
            errors.append("Candidate requires a symbol.")
        if not self.strategy_id:
            errors.append("Candidate requires a strategy id.")
        if not math.isfinite(self.expected_gross_alpha_bps):
            errors.append("Expected gross alpha must be finite.")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("Confidence must be within [0, 1].")
        if self.action is QuantAction.BUY:
            close = _finite_or_none(self.close)
            stop = _finite_or_none(self.stop_price)
            if close is None or close <= 0:
                errors.append("BUY candidate requires a positive close.")
            if stop is None or stop <= 0:
                errors.append("BUY candidate requires a positive stop price.")
            elif close is not None and stop >= close:
                errors.append("BUY candidate stop must sit below the close.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["market"] = self.market.value
        d["action"] = self.action.value
        d["timestamp"] = _iso(self.timestamp)
        return d


@dataclass
class OpportunityDecision:
    """The desk's final, journaled verdict on one candidate."""

    candidate: OpportunityCandidate
    outcome: DecisionOutcome
    rank: Optional[int]
    expected_cost_bps: Optional[float]
    expected_net_alpha_bps: Optional[float]
    opportunity_score: Optional[float]
    risk_allocation_pct: Optional[float]
    rejection_reasons: list[str] = field(default_factory=list)
    execution_estimate: Optional[ExecutionEstimate] = None
    regime: Optional[RegimeAssessment] = None
    decided_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "outcome": self.outcome.value,
            "rank": self.rank,
            "expected_cost_bps": self.expected_cost_bps,
            "expected_net_alpha_bps": self.expected_net_alpha_bps,
            "opportunity_score": self.opportunity_score,
            "risk_allocation_pct": self.risk_allocation_pct,
            "rejection_reasons": list(self.rejection_reasons),
            "execution_estimate": (
                self.execution_estimate.to_dict() if self.execution_estimate else None
            ),
            "regime": self.regime.to_dict() if self.regime else None,
            "decided_at": _iso(self.decided_at),
        }


@dataclass
class AttributionRecord:
    """Complete post-trade attribution for one round trip (section-50 data).

    ``max_favorable_excursion_pct`` / ``max_adverse_excursion_pct`` are
    positive magnitudes relative to the entry price, measured on completed
    bars while the position was open.  ``r_multiple`` is net P&L divided by
    the initial at-risk amount (entry minus initial stop, times quantity).
    """

    strategy_id: str
    strategy_version: str
    symbol: str
    market: Market
    timeframe: str
    session_id: Optional[str]
    signal_timestamp: Optional[datetime]
    entry_timestamp: Optional[datetime]
    exit_timestamp: Optional[datetime]
    qty: float
    entry_price: float
    exit_price: Optional[float]
    initial_stop_price: Optional[float]
    fees: float
    expected_cost_bps: Optional[float]
    realized_slippage_bps: Optional[float]
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    r_multiple: Optional[float]
    max_favorable_excursion_pct: Optional[float]
    max_adverse_excursion_pct: Optional[float]
    entry_reason: str = ""
    exit_reason: str = ""
    regime_at_entry: Optional[str] = None
    opportunity_score: Optional[float] = None
    shadow: bool = False
    features_at_entry: dict[str, Optional[float]] = field(default_factory=dict)
    # Stable identity of the booked exit fill (broker client order id plus
    # side).  Present on engine-sourced rows so a reconciliation that
    # re-books the same fill cannot double-count it; ``None`` for shadow
    # and backfilled rows, which are unconstrained.
    event_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["market"] = self.market.value
        for key in ("signal_timestamp", "entry_timestamp", "exit_timestamp"):
            d[key] = _iso(getattr(self, key))
        return d


@dataclass(frozen=True)
class ProviderAvailability:
    """Whether one optional data source can be used right now, and why not.

    Strategies declare required feature sources; a missing source disables
    the strategy with this record as the logged reason.  Nothing in the
    platform may substitute fabricated values for an unavailable source.
    """

    source: str
    available: bool
    reason: Optional[str] = None
    as_of: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = _iso(self.as_of)
        return d


@dataclass
class StrategyHealth:
    """Rolling marketplace statistics for one strategy.

    Every field except counts may be None when the sample is too small to
    be meaningful — consumers must not present a None as a measurement.
    """

    strategy_id: str
    state: StrategyState
    trades: int
    wins: int
    losses: int
    hit_rate: Optional[float]
    expectancy_r: Optional[float]
    avg_r: Optional[float]
    profit_factor: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown_pct: Optional[float]
    tail_loss_r: Optional[float]
    net_pnl: Optional[float]
    regime_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d
