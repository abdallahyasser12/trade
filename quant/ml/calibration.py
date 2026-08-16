"""Probability calibration measurement and correction (spec §18).

A model that says 0.7 must be right about 70% of the time or its 0.7 is
a lie the sizing layer would act on.  This module measures the lie
(Brier score, reliability table, expected calibration error) and offers a
dependency-free isotonic correction (pool-adjacent-violators) fitted on
held-out data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate(probs: Sequence[float], outcomes: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("Probabilities and outcomes must be equal-length and non-empty.")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Calibration inputs must be finite.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Outcomes must be 0 or 1.")
    return p, y


def brier_score(probs: Sequence[float], outcomes: Sequence[float]) -> float:
    p, y = _validate(probs, outcomes)
    return float(np.mean((p - y) ** 2))


def reliability_table(
    probs: Sequence[float], outcomes: Sequence[float], bins: int = 10
) -> list[dict[str, float]]:
    p, y = _validate(probs, outcomes)
    if bins < 2:
        raise ValueError("At least two bins are required.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[dict[str, float]] = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (p >= low) & (p < high) if i < bins - 1 else (p >= low) & (p <= high)
        count = int(mask.sum())
        table.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": float(count),
                "mean_prob": float(p[mask].mean()) if count else float("nan"),
                "event_rate": float(y[mask].mean()) if count else float("nan"),
            }
        )
    return table


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[float], bins: int = 10
) -> float:
    p, _ = _validate(probs, outcomes)
    table = reliability_table(probs, outcomes, bins)
    total = float(len(p))
    ece = 0.0
    for row in table:
        if row["count"] == 0:
            continue
        ece += row["count"] / total * abs(row["mean_prob"] - row["event_rate"])
    return ece


class PoolAdjacentViolators:
    """Isotonic regression via PAV: a monotone map from raw probability to
    empirical event rate.  Fitted on VALIDATION predictions only."""

    def __init__(self) -> None:
        self._thresholds: np.ndarray | None = None
        self._values: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._thresholds is not None

    def fit(self, probs: Sequence[float], outcomes: Sequence[float]) -> "PoolAdjacentViolators":
        p, y = _validate(probs, outcomes)
        order = np.argsort(p, kind="stable")
        p_sorted = p[order]
        y_sorted = y[order]
        # Pools of (value_sum, weight); merge backward while decreasing.
        values: list[float] = []
        weights: list[float] = []
        for target in y_sorted:
            values.append(float(target))
            weights.append(1.0)
            while len(values) > 1 and values[-2] > values[-1]:
                merged_weight = weights[-2] + weights[-1]
                merged_value = (
                    values[-2] * weights[-2] + values[-1] * weights[-1]
                ) / merged_weight
                values[-2:] = [merged_value]
                weights[-2:] = [merged_weight]
        # Expand pools back over the sorted probabilities.
        fitted = np.empty(len(p_sorted))
        position = 0
        for value, weight in zip(values, weights):
            span = int(round(weight))
            fitted[position : position + span] = value
            position += span
        self._thresholds = p_sorted
        self._values = fitted
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        if self._thresholds is None or self._values is None:
            raise RuntimeError("Calibrator is not fitted.")
        p = np.asarray(list(probs), dtype=float)
        if not np.isfinite(p).all():
            raise ValueError("Probabilities must be finite.")
        positions = np.searchsorted(self._thresholds, p, side="right") - 1
        positions = np.clip(positions, 0, len(self._values) - 1)
        return self._values[positions]
