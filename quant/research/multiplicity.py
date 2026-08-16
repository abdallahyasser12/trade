"""Multiple-testing control: deflated Sharpe ratio (spec §37).

The research factory evaluates many candidates; the best of N trials has
an inflated Sharpe purely from selection.  The deflated Sharpe ratio
(Bailey & López de Prado) asks: what is the probability that the observed
Sharpe exceeds the Sharpe the BEST of N zero-skill trials would show?
Only that probability may argue for promotion.

Everything is closed-form and dependency-free: the standard normal
quantile uses the Acklam rational approximation (documented, unit-tested
against known quantiles), the CDF uses ``math.erf``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# Euler–Mascheroni constant, used by the expected-maximum approximation.
_EULER_GAMMA = 0.5772156649015329


def normal_ppf(p: float) -> float:
    """Standard normal quantile via Acklam's rational approximation.

    Absolute error below 1.15e-9 over (0, 1) — far tighter than any gate
    threshold here needs.  Raises on p outside (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"Quantile probability must be in (0, 1); got {p}.")
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_max_sharpe(n_trials: int, variance_sharpe: float = 1.0) -> float:
    """Expected maximum Sharpe among ``n_trials`` zero-skill strategies.

    E[max] ≈ sqrt(V) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))).  A single
    trial carries no selection bias, so N=1 returns exactly 0.
    """
    if n_trials < 1:
        raise ValueError("Trial count must be at least 1.")
    if not math.isfinite(variance_sharpe) or variance_sharpe <= 0:
        raise ValueError("Sharpe variance must be finite and positive.")
    if n_trials == 1:
        return 0.0
    scale = math.sqrt(variance_sharpe)
    return scale * (
        (1.0 - _EULER_GAMMA) * normal_ppf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


@dataclass
class DeflatedSharpeReport:
    dsr_probability: float
    expected_max_sharpe: float
    observed_sharpe: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deflated_sharpe_probability(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    variance_sharpe: float = 1.0,
) -> DeflatedSharpeReport:
    """P(true Sharpe > best-of-N-noise Sharpe), per the PSR/DSR formula.

    ``observed_sharpe`` must be in PER-PERIOD units matching
    ``n_observations`` (do not pass an annualized figure with bar counts).
    An uncomputable deflation returns probability 0.0 with a reason — it
    must never pass a gate by accident.
    """
    for label, value in {
        "observed Sharpe": observed_sharpe,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
    if n_observations < 2:
        raise ValueError("Deflated Sharpe needs at least two observations.")
    benchmark = expected_max_sharpe(n_trials, variance_sharpe)
    variance_term = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe * observed_sharpe
    )
    if variance_term <= 0:
        return DeflatedSharpeReport(
            dsr_probability=0.0,
            expected_max_sharpe=benchmark,
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
            reason="variance_term_not_positive",
        )
    statistic = (
        (observed_sharpe - benchmark)
        * math.sqrt(n_observations - 1.0)
        / math.sqrt(variance_term)
    )
    return DeflatedSharpeReport(
        dsr_probability=normal_cdf(statistic),
        expected_max_sharpe=benchmark,
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
