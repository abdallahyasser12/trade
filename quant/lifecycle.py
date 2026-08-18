"""Evidence-based strategy lifecycle.

RESEARCH -> SHADOW -> PAPER -> LIVE_CANARY -> LIVE advances one rung at a
time. Every forward rung must demonstrate both adequate sample size *and* acceptable
health; accumulating enough losing trades is never sufficient for promotion.
Any live transition also requires an explicit operator acknowledgement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from quant.contracts import (
    LIVE_STATES,
    PROMOTION_LADDER,
    StrategyHealth,
    StrategyState,
)
from quant.store import QuantStore

_STATE_RISK_MULTIPLIER = {
    StrategyState.RESEARCH: 0.0,
    StrategyState.SHADOW: 0.0,
    StrategyState.PAPER: 1.0,
    StrategyState.LIVE_CANARY: 0.25,
    StrategyState.LIVE: 1.0,
    StrategyState.REDUCED_RISK: 0.5,
    StrategyState.QUARANTINE: 0.0,
}

_PRIVILEGE_ORDER = {
    StrategyState.RESEARCH: 0,
    StrategyState.QUARANTINE: 0,
    StrategyState.SHADOW: 1,
    StrategyState.PAPER: 2,
    StrategyState.REDUCED_RISK: 3,
    StrategyState.LIVE_CANARY: 3,
    StrategyState.LIVE: 4,
}


@dataclass(frozen=True)
class PromotionRequirements:
    min_research_trades: int = 15

    min_shadow_trades: int = 20
    min_shadow_expectancy_r: float = 0.0
    min_shadow_profit_factor: float = 1.0
    max_shadow_drawdown_r: float = 6.0

    min_paper_trades: int = 30
    min_paper_expectancy_r: float = 0.0
    min_paper_profit_factor: float = 1.0
    max_paper_drawdown_r: float = 8.0
    min_measured_execution_samples: int = 20

    min_canary_trades: int = 20
    min_canary_expectancy_r: float = 0.0
    min_canary_profit_factor: float = 1.0
    max_canary_drawdown_r: float = 6.0

    max_open_operational_defects: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        counts = {
            "research trades": self.min_research_trades,
            "shadow trades": self.min_shadow_trades,
            "paper trades": self.min_paper_trades,
            "execution samples": self.min_measured_execution_samples,
            "canary trades": self.min_canary_trades,
        }
        for label, value in counts.items():
            if value < 1:
                errors.append(f"Minimum {label} must be at least 1.")

        for label, value in {
            "shadow expectancy floor": self.min_shadow_expectancy_r,
            "paper expectancy floor": self.min_paper_expectancy_r,
            "canary expectancy floor": self.min_canary_expectancy_r,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")

        for label, value in {
            "shadow profit-factor floor": self.min_shadow_profit_factor,
            "paper profit-factor floor": self.min_paper_profit_factor,
            "canary profit-factor floor": self.min_canary_profit_factor,
            "shadow drawdown ceiling": self.max_shadow_drawdown_r,
            "paper drawdown ceiling": self.max_paper_drawdown_r,
            "canary drawdown ceiling": self.max_canary_drawdown_r,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and positive.")

        if self.max_open_operational_defects < 0:
            errors.append("Operational-defect ceiling cannot be negative.")
        return errors


@dataclass
class PromotionEvidence:
    research_run: Optional[dict[str, Any]] = None
    shadow_health: Optional[StrategyHealth] = None
    paper_health: Optional[StrategyHealth] = None
    canary_health: Optional[StrategyHealth] = None
    measured_execution_samples: int = 0
    operational_defects: int = 0
    research_rerun_with_measured_costs: Optional[dict[str, Any]] = None


@dataclass
class LifecyclePromotionDecision:
    approved: bool
    from_state: StrategyState
    to_state: StrategyState
    failures: list[str] = field(default_factory=list)


def _research_run_trades(run: dict[str, Any]) -> Optional[int]:
    result = run.get("result_json") if "result_json" in run else run.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return None
    if not isinstance(result, dict):
        return None
    trades = result.get("total_oos_trades")
    return int(trades) if isinstance(trades, (int, float)) else None


def _run_passed(run: Optional[dict[str, Any]]) -> bool:
    return isinstance(run, dict) and bool(run.get("passed"))


def _health_failures(
    prefix: str,
    health: Optional[StrategyHealth],
    *,
    min_trades: int,
    min_expectancy_r: float,
    min_profit_factor: float,
    max_drawdown_r: float,
) -> list[str]:
    """Fail closed on every forward-health dimension required for promotion."""
    if health is None:
        return [f"{prefix}_health_missing"]

    failures: list[str] = []
    if health.trades < min_trades:
        failures.append(f"{prefix}_trades_{health.trades}_below_{min_trades}")

    expectancy = health.expectancy_r
    if expectancy is None or not math.isfinite(expectancy):
        failures.append(f"{prefix}_expectancy_unmeasured")
    elif expectancy < min_expectancy_r:
        failures.append(
            f"{prefix}_expectancy_{expectancy:.4f}R_below_{min_expectancy_r:.4f}R"
        )

    profit_factor = health.profit_factor
    if profit_factor is None or not math.isfinite(profit_factor):
        failures.append(f"{prefix}_profit_factor_unmeasured")
    elif profit_factor < min_profit_factor:
        failures.append(
            f"{prefix}_profit_factor_{profit_factor:.4f}_below_{min_profit_factor:.4f}"
        )

    drawdown = health.max_drawdown_pct
    if drawdown is None or not math.isfinite(drawdown):
        failures.append(f"{prefix}_drawdown_unmeasured")
    elif drawdown > max_drawdown_r:
        failures.append(
            f"{prefix}_drawdown_{drawdown:.2f}R_above_{max_drawdown_r:.2f}R"
        )
    return failures


class LifecycleManager:
    """Own every legal state transition and persist its evidence."""

    def __init__(
        self,
        store: QuantStore,
        requirements: Optional[PromotionRequirements] = None,
    ):
        self.store = store
        self.requirements = requirements or PromotionRequirements()
        errors = self.requirements.validate()
        if errors:
            raise ValueError("Invalid promotion requirements: " + "; ".join(errors))

    def state(self, strategy_id: str) -> StrategyState:
        row = self.store.get_strategy_state(strategy_id)
        return StrategyState.RESEARCH if row is None else StrategyState(row["state"])

    def risk_multiplier(self, strategy_id: str) -> float:
        return _STATE_RISK_MULTIPLIER[self.state(strategy_id)]

    def request_promotion(
        self,
        strategy_id: str,
        evidence: PromotionEvidence,
        *,
        operator_ack: bool = False,
        reason: str = "",
    ) -> LifecyclePromotionDecision:
        req = self.requirements
        current = self.state(strategy_id)
        if current not in PROMOTION_LADDER:
            return LifecyclePromotionDecision(
                approved=False,
                from_state=current,
                to_state=current,
                failures=[
                    f"promotion_from_{current.value.lower()}_requires_explicit_demote_first"
                ],
            )
        position = PROMOTION_LADDER.index(current)
        if position == len(PROMOTION_LADDER) - 1:
            return LifecyclePromotionDecision(
                approved=False,
                from_state=current,
                to_state=current,
                failures=["already_at_top_of_ladder"],
            )
        target = PROMOTION_LADDER[position + 1]
        failures: list[str] = []

        if target is StrategyState.SHADOW:
            if not _run_passed(evidence.research_run):
                failures.append("research_run_missing_or_failed")
            else:
                trades = _research_run_trades(evidence.research_run)
                if trades is None:
                    failures.append("research_run_trade_count_unreadable")
                elif trades < req.min_research_trades:
                    failures.append(
                        f"research_trades_{trades}_below_{req.min_research_trades}"
                    )
                if not evidence.research_run.get("trial_count"):
                    failures.append("research_run_missing_trial_count")

        elif target is StrategyState.PAPER:
            failures.extend(
                _health_failures(
                    "shadow",
                    evidence.shadow_health,
                    min_trades=req.min_shadow_trades,
                    min_expectancy_r=req.min_shadow_expectancy_r,
                    min_profit_factor=req.min_shadow_profit_factor,
                    max_drawdown_r=req.max_shadow_drawdown_r,
                )
            )

        elif target is StrategyState.LIVE_CANARY:
            failures.extend(
                _health_failures(
                    "paper",
                    evidence.paper_health,
                    min_trades=req.min_paper_trades,
                    min_expectancy_r=req.min_paper_expectancy_r,
                    min_profit_factor=req.min_paper_profit_factor,
                    max_drawdown_r=req.max_paper_drawdown_r,
                )
            )
            if evidence.measured_execution_samples < req.min_measured_execution_samples:
                failures.append(
                    f"measured_execution_samples_{evidence.measured_execution_samples}"
                    f"_below_{req.min_measured_execution_samples}"
                )
            if not _run_passed(evidence.research_rerun_with_measured_costs):
                failures.append("research_rerun_with_measured_costs_missing_or_failed")
            if evidence.operational_defects > req.max_open_operational_defects:
                failures.append("open_operational_defects")
            if not operator_ack:
                failures.append("operator_ack_required_for_live_states")

        elif target is StrategyState.LIVE:
            failures.extend(
                _health_failures(
                    "canary",
                    evidence.canary_health,
                    min_trades=req.min_canary_trades,
                    min_expectancy_r=req.min_canary_expectancy_r,
                    min_profit_factor=req.min_canary_profit_factor,
                    max_drawdown_r=req.max_canary_drawdown_r,
                )
            )
            if not operator_ack:
                failures.append("operator_ack_required_for_live_states")

        if (
            target in LIVE_STATES
            and not operator_ack
            and "operator_ack_required_for_live_states" not in failures
        ):
            failures.append("operator_ack_required_for_live_states")

        if failures:
            return LifecyclePromotionDecision(
                approved=False,
                from_state=current,
                to_state=target,
                failures=failures,
            )

        self.store.set_strategy_state(
            strategy_id,
            current,
            target,
            reason=reason or f"promotion_to_{target.value.lower()}",
            evidence={
                "research_run_id": (evidence.research_run or {}).get("id"),
                "measured_execution_samples": evidence.measured_execution_samples,
                "operator_ack": operator_ack,
            },
            operator_ack=operator_ack,
        )
        return LifecyclePromotionDecision(
            approved=True,
            from_state=current,
            to_state=target,
            failures=[],
        )

    def demote(
        self,
        strategy_id: str,
        to_state: StrategyState,
        reason: str,
        evidence: Optional[dict[str, Any]] = None,
    ) -> None:
        current = self.state(strategy_id)
        if _PRIVILEGE_ORDER[to_state] > _PRIVILEGE_ORDER[current]:
            raise ValueError(
                f"demote() cannot raise privilege: {current.value} -> {to_state.value}."
            )
        self.store.set_strategy_state(
            strategy_id,
            current,
            to_state,
            reason=reason,
            evidence=evidence,
        )


@dataclass(frozen=True)
class DegradationThresholds:
    min_profit_factor: float = 0.9
    min_expectancy_r: float = -0.05
    max_drawdown_r: float = 8.0
    max_calibration_ece: float = 0.15
    max_cost_inflation_ratio: float = 2.0
    min_sample_trades: int = 20

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in {
            "profit-factor floor": self.min_profit_factor,
            "drawdown ceiling": self.max_drawdown_r,
            "calibration ceiling": self.max_calibration_ece,
            "cost-inflation ceiling": self.max_cost_inflation_ratio,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and positive.")
        if not math.isfinite(self.min_expectancy_r):
            errors.append("Expectancy floor must be finite.")
        if self.min_sample_trades < 5:
            errors.append("Degradation checks need at least 5 trades of sample.")
        return errors


@dataclass
class DegradationVerdict:
    action: str
    reasons: list[str] = field(default_factory=list)


class DegradationMonitor:
    """Detect decay and propose reduce-risk/quarantine responses."""

    def __init__(self, thresholds: Optional[DegradationThresholds] = None):
        self.thresholds = thresholds or DegradationThresholds()
        errors = self.thresholds.validate()
        if errors:
            raise ValueError("Invalid degradation thresholds: " + "; ".join(errors))

    def evaluate(
        self,
        health: StrategyHealth,
        *,
        calibration_ece: Optional[float] = None,
        cost_inflation_ratio: Optional[float] = None,
    ) -> DegradationVerdict:
        thresholds = self.thresholds
        if health.trades < thresholds.min_sample_trades:
            return DegradationVerdict(action="none", reasons=["insufficient_sample"])

        soft: list[str] = []
        hard: list[str] = []
        if (
            health.profit_factor is not None
            and health.profit_factor < thresholds.min_profit_factor
        ):
            soft.append("profit_factor_collapse")
        if (
            health.expectancy_r is not None
            and health.expectancy_r < thresholds.min_expectancy_r
        ):
            soft.append("expectancy_collapse")
        if (
            calibration_ece is not None
            and calibration_ece > thresholds.max_calibration_ece
        ):
            soft.append("calibration_failure")
        if (
            health.max_drawdown_pct is not None
            and health.max_drawdown_pct > thresholds.max_drawdown_r
        ):
            hard.append("unexpected_drawdown")
        if (
            cost_inflation_ratio is not None
            and cost_inflation_ratio > thresholds.max_cost_inflation_ratio
        ):
            hard.append("execution_cost_inflation")

        if hard or len(soft) >= 2:
            return DegradationVerdict(
                action="quarantine",
                reasons=sorted(hard + soft),
            )
        if soft:
            return DegradationVerdict(action="reduce_risk", reasons=sorted(soft))
        return DegradationVerdict(action="none", reasons=[])
