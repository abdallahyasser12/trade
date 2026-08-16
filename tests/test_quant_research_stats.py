"""Monte Carlo stress and deflated-Sharpe correctness tests."""

import math

import pytest

from quant.research.multiplicity import (
    deflated_sharpe_probability,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
)
from quant.research.stress import StressConfig, run_trade_stress

PROFITABLE = [1.5, -1.0, 2.0, -1.0, 0.8, 1.2, -1.0, 2.5, -0.5, 1.0, -1.0, 1.8]
HEAVY_LOSS = [-1.0, -1.0, -1.0, 0.5, -1.0, -1.0, -1.0, -1.0, 0.3, -1.0, -1.0, -1.0]


def test_stress_is_deterministic():
    a = run_trade_stress(PROFITABLE, StressConfig(num_paths=500, seed=11))
    b = run_trade_stress(PROFITABLE, StressConfig(num_paths=500, seed=11))
    assert a.to_dict() == b.to_dict()


def test_stressed_block_is_worse_than_baseline():
    report = run_trade_stress(PROFITABLE, StressConfig(num_paths=500))
    assert report.stressed.expectancy_r < report.baseline.expectancy_r
    assert report.stressed.final_r_p50 <= report.baseline.final_r_p50


def test_ruin_zero_for_all_winners_positive_for_heavy_losses():
    winners = [1.0] * 12
    report = run_trade_stress(winners, StressConfig(num_paths=300, ruin_drawdown_r=5.0))
    assert report.baseline.risk_of_ruin == 0.0
    losses = run_trade_stress(
        HEAVY_LOSS, StressConfig(num_paths=300, ruin_drawdown_r=5.0)
    )
    assert losses.baseline.risk_of_ruin > 0.0


def test_stress_refuses_tiny_samples():
    with pytest.raises(ValueError, match="at least 5"):
        run_trade_stress([1.0, -1.0], StressConfig())


def test_stress_refuses_non_finite():
    with pytest.raises(ValueError, match="finite"):
        run_trade_stress([1.0, float("nan")] * 4, StressConfig())


def test_normal_ppf_matches_known_quantiles():
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert normal_ppf(0.05) == pytest.approx(-1.644854, abs=1e-6)
    assert normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


def test_expected_max_sharpe_grows_with_trials_and_single_trial_is_zero():
    assert expected_max_sharpe(1) == 0.0
    small = expected_max_sharpe(10)
    large = expected_max_sharpe(1000)
    assert 0 < small < large


def test_dsr_decreases_with_trial_count():
    few = deflated_sharpe_probability(0.15, n_trials=1, n_observations=500)
    many = deflated_sharpe_probability(0.15, n_trials=500, n_observations=500)
    assert few.dsr_probability > many.dsr_probability
    assert few.dsr_probability > 0.9


def test_dsr_fails_closed_on_bad_variance_term():
    # Extreme negative skew with a large Sharpe can push the variance term
    # non-positive; the probability must be 0 with a reason, never an error
    # that a gate might swallow as "passed".
    report = deflated_sharpe_probability(
        3.0, n_trials=10, n_observations=100, skewness=5.0
    )
    assert report.dsr_probability == 0.0
    assert report.reason == "variance_term_not_positive"


def test_dsr_input_validation():
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_probability(float("nan"), n_trials=5, n_observations=100)
    with pytest.raises(ValueError, match="observations"):
        deflated_sharpe_probability(1.0, n_trials=5, n_observations=1)
    with pytest.raises(ValueError, match="at least 1"):
        expected_max_sharpe(0)
