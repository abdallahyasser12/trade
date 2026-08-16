"""Drift detection: features, expectancy, calibration (spec §41).

The goal is distinguishing "temporary drawdown" from "the world changed"
early enough to act before catastrophic loss.  Three cheap, dependency-
free detectors:

* population stability index over quantile bins of the BASELINE — bins
  are anchored to the reference distribution so a shifted live
  distribution piles into tail bins instead of redefining them;
* Kolmogorov–Smirnov statistic (max ECDF distance);
* rolling expectancy shift — recent-window mean R against prior history,
  as a z-score.

Insufficient data yields an informative non-breach finding, never a
silent skip and never a fabricated statistic.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from quant.store import QuantStore

# PSI over fewer observations than this is noise wearing a metric costume.
MIN_PSI_OBSERVATIONS = 30


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], bins: int = 10
) -> float:
    """Standard PSI on quantile bins of the EXPECTED (baseline) sample."""
    expected_values = np.asarray(list(expected), dtype=float)
    actual_values = np.asarray(list(actual), dtype=float)
    if len(expected_values) < MIN_PSI_OBSERVATIONS or len(actual_values) < MIN_PSI_OBSERVATIONS:
        raise ValueError(
            f"PSI needs at least {MIN_PSI_OBSERVATIONS} observations on each side."
        )
    if not np.isfinite(expected_values).all() or not np.isfinite(actual_values).all():
        raise ValueError("PSI inputs must be finite.")
    edges = np.quantile(expected_values, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_pct = np.maximum(expected_counts / len(expected_values), 1e-6)
    actual_pct = np.maximum(actual_counts / len(actual_values), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: max |ECDF_a - ECDF_b|."""
    a_values = np.sort(np.asarray(list(a), dtype=float))
    b_values = np.sort(np.asarray(list(b), dtype=float))
    if len(a_values) == 0 or len(b_values) == 0:
        raise ValueError("KS statistic needs non-empty samples.")
    grid = np.concatenate([a_values, b_values])
    cdf_a = np.searchsorted(a_values, grid, side="right") / len(a_values)
    cdf_b = np.searchsorted(b_values, grid, side="right") / len(b_values)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def rolling_expectancy_shift(
    r_multiples: Sequence[float], recent_window: int = 20
) -> Optional[dict[str, float]]:
    """Recent-window expectancy vs prior history, as a z-score.

    None below 2x the window: with less history the 'baseline' is itself
    noise.  Pooled ddof=1 std — both windows estimate the same population
    under the null, so pooling is the correct variance under H0.
    """
    values = [float(v) for v in r_multiples if math.isfinite(float(v))]
    if len(values) < recent_window * 2:
        return None
    recent = np.asarray(values[-recent_window:])
    baseline = np.asarray(values[:-recent_window])
    pooled = np.asarray(values)
    pooled_std = float(pooled.std(ddof=1))
    if pooled_std <= 0:
        return None
    z = float(
        (recent.mean() - baseline.mean())
        / (pooled_std * math.sqrt(1.0 / len(recent) + 1.0 / len(baseline)))
    )
    return {
        "recent_mean": float(recent.mean()),
        "baseline_mean": float(baseline.mean()),
        "z": z,
    }


@dataclass(frozen=True)
class DriftThresholds:
    psi_warn: float = 0.10
    psi_breach: float = 0.25
    ks_breach: float = 0.20
    expectancy_z_breach: float = -2.0
    calibration_ece_breach: float = 0.15

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in {
            "PSI warn": self.psi_warn,
            "PSI breach": self.psi_breach,
            "KS breach": self.ks_breach,
            "calibration breach": self.calibration_ece_breach,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} threshold must be finite and positive.")
        if self.psi_warn >= self.psi_breach:
            errors.append("PSI warn threshold must sit below the breach threshold.")
        if not math.isfinite(self.expectancy_z_breach) or self.expectancy_z_breach >= 0:
            errors.append("Expectancy z-score breach must be negative.")
        return errors


@dataclass
class DriftFinding:
    subject: str
    metric: str
    value: Optional[float]
    threshold: Optional[float]
    breached: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DriftMonitor:
    """Runs detectors, persists findings when a store is attached."""

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        store: Optional[QuantStore] = None,
    ):
        self.thresholds = thresholds or DriftThresholds()
        errors = self.thresholds.validate()
        if errors:
            raise ValueError("Invalid drift thresholds: " + "; ".join(errors))
        self.store = store

    def _persist(self, finding: DriftFinding) -> DriftFinding:
        if self.store is not None:
            self.store.record_drift(
                subject=finding.subject,
                metric=finding.metric,
                value=finding.value,
                threshold=finding.threshold,
                breached=finding.breached,
                details=finding.details,
            )
        return finding

    def check_feature_drift(
        self, name: str, baseline: Sequence[float], recent: Sequence[float]
    ) -> DriftFinding:
        try:
            psi = population_stability_index(baseline, recent)
        except ValueError as error:
            return self._persist(
                DriftFinding(
                    subject=f"feature:{name}",
                    metric="psi",
                    value=None,
                    threshold=self.thresholds.psi_breach,
                    breached=False,
                    details={"reason": str(error)},
                )
            )
        return self._persist(
            DriftFinding(
                subject=f"feature:{name}",
                metric="psi",
                value=psi,
                threshold=self.thresholds.psi_breach,
                breached=psi > self.thresholds.psi_breach,
                details={"warn": psi > self.thresholds.psi_warn},
            )
        )

    def check_expectancy_drift(
        self, strategy_id: str, r_multiples: Sequence[float], recent_window: int = 20
    ) -> DriftFinding:
        shift = rolling_expectancy_shift(r_multiples, recent_window)
        if shift is None:
            return self._persist(
                DriftFinding(
                    subject=f"strategy:{strategy_id}",
                    metric="expectancy_z",
                    value=None,
                    threshold=self.thresholds.expectancy_z_breach,
                    breached=False,
                    details={"reason": "insufficient_sample"},
                )
            )
        return self._persist(
            DriftFinding(
                subject=f"strategy:{strategy_id}",
                metric="expectancy_z",
                value=shift["z"],
                threshold=self.thresholds.expectancy_z_breach,
                breached=shift["z"] < self.thresholds.expectancy_z_breach,
                details=shift,
            )
        )

    def check_calibration_drift(self, model_id: str, ece: float) -> DriftFinding:
        value = ece if math.isfinite(ece) else None
        return self._persist(
            DriftFinding(
                subject=f"model:{model_id}",
                metric="calibration_ece",
                value=value,
                threshold=self.thresholds.calibration_ece_breach,
                breached=value is not None and value > self.thresholds.calibration_ece_breach,
                details={} if value is not None else {"reason": "non_finite_ece"},
            )
        )
