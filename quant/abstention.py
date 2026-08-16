"""Dedicated uncertainty/abstention layer (spec §§19-20).

ABSTAIN is a first-class verdict: a prediction exists but uncertainty is
too high to justify capital.  Every trigger is a named reason, journaled
alongside executions and rejections, so abstention behavior is itself
measurable.  Model/strategy DISAGREEMENT is treated as explicit evidence
(spec §20), not averaged away: a direction model that contradicts a BUY
abstains the trade rather than diluting its confidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from quant.contracts import OpportunityCandidate, QuantAction


@dataclass(frozen=True)
class AbstentionConfig:
    min_confidence: float = 0.55
    max_feature_z: float = 4.0
    max_data_age_bars: int = 2
    # Disagreement in [0, 1]; a candidate abstains when disagreement
    # exceeds 1 - min_model_agreement.  0 disables the check.
    min_model_agreement: float = 0.0
    require_model_when_available: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not (0.0 <= self.min_confidence <= 1.0):
            errors.append("Minimum confidence must be in [0, 1].")
        if not math.isfinite(self.max_feature_z) or self.max_feature_z <= 0:
            errors.append("Feature z-score bound must be finite and positive.")
        if self.max_data_age_bars < 0:
            errors.append("Data-age tolerance cannot be negative.")
        if not (0.0 <= self.min_model_agreement <= 1.0):
            errors.append("Model-agreement floor must be in [0, 1].")
        return errors


@dataclass
class AbstentionVerdict:
    abstain: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AbstentionPolicy:
    """Deterministic abstention checks; any triggered reason abstains."""

    def __init__(self, config: Optional[AbstentionConfig] = None):
        self.config = config or AbstentionConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid abstention configuration: " + "; ".join(errors))

    def evaluate(
        self,
        candidate: OpportunityCandidate,
        *,
        model_prob_up: Optional[float] = None,
        model_ood: bool = False,
        model_disagreement: Optional[float] = None,
        data_age_bars: Optional[int] = None,
        feature_zscores: Optional[Mapping[str, float]] = None,
    ) -> AbstentionVerdict:
        cfg = self.config
        reasons: list[str] = []
        details: dict[str, Any] = {}

        numeric_inputs = {"model_prob_up": model_prob_up, "model_disagreement": model_disagreement}
        for name, value in numeric_inputs.items():
            if value is not None and not math.isfinite(value):
                reasons.append("non_finite_input")
                details[name] = "non_finite"

        if candidate.confidence < cfg.min_confidence:
            reasons.append("low_confidence")
            details["confidence"] = candidate.confidence

        if model_ood:
            reasons.append("out_of_distribution")
        if feature_zscores:
            extreme = {
                name: z
                for name, z in feature_zscores.items()
                if not math.isfinite(z) or abs(z) > cfg.max_feature_z
            }
            if extreme:
                if "out_of_distribution" not in reasons:
                    reasons.append("out_of_distribution")
                details["extreme_features"] = {
                    k: (v if math.isfinite(v) else None) for k, v in extreme.items()
                }

        if data_age_bars is not None and data_age_bars > cfg.max_data_age_bars:
            reasons.append("stale_data")
            details["data_age_bars"] = data_age_bars

        if (
            cfg.min_model_agreement > 0
            and model_disagreement is not None
            and math.isfinite(model_disagreement)
            and model_disagreement > 1.0 - cfg.min_model_agreement
        ):
            reasons.append("model_disagreement")
            details["disagreement"] = model_disagreement

        if (
            cfg.require_model_when_available
            and model_prob_up is not None
            and math.isfinite(model_prob_up)
            and candidate.action is QuantAction.BUY
            and model_prob_up < 0.5
        ):
            reasons.append("model_contradicts")
            details["model_prob_up"] = model_prob_up

        return AbstentionVerdict(
            abstain=bool(reasons), reasons=sorted(set(reasons)), details=details
        )
