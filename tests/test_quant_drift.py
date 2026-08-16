"""Drift detector correctness and persistence tests."""

import numpy as np
import pytest

from quant.drift import (
    DriftMonitor,
    DriftThresholds,
    ks_statistic,
    population_stability_index,
    rolling_expectancy_shift,
)
from quant.store import QuantStore


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 500)
    recent = rng.normal(0, 1, 500)
    # Two independent draws of the SAME distribution carry sampling noise;
    # the conventional "no shift" line is 0.10.
    assert population_stability_index(baseline, recent) < 0.10


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0, 1, 500)
    shifted = rng.normal(2.0, 1, 500)
    assert population_stability_index(baseline, shifted) > 0.5


def test_psi_refuses_tiny_samples():
    with pytest.raises(ValueError, match="at least 30"):
        population_stability_index([1.0] * 10, [1.0] * 100)


def test_ks_sanity():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 400)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 3.0) > 0.8


def test_expectancy_shift_detects_collapse():
    healthy = [0.5] * 40
    collapsed = healthy + [-1.0] * 20
    shift = rolling_expectancy_shift(collapsed, recent_window=20)
    assert shift is not None
    assert shift["z"] < -2.0
    assert rolling_expectancy_shift([0.5] * 10, recent_window=20) is None


def test_monitor_persists_findings():
    store = QuantStore(":memory:")
    monitor = DriftMonitor(DriftThresholds(), store=store)
    rng = np.random.default_rng(4)
    finding = monitor.check_feature_drift(
        "atr_pct", rng.normal(0, 1, 200), rng.normal(3, 1, 200)
    )
    assert finding.breached
    rows = store.get_drift("feature:atr_pct")
    assert len(rows) == 1 and rows[0]["breached"] == 1

    insufficient = monitor.check_expectancy_drift("donchian_trend", [0.5] * 5)
    assert not insufficient.breached
    assert insufficient.value is None
    assert insufficient.details["reason"] == "insufficient_sample"

    calibration = monitor.check_calibration_drift("direction_5bar", 0.3)
    assert calibration.breached
    store.close()


def test_thresholds_validation():
    assert DriftThresholds(psi_warn=0.5, psi_breach=0.2).validate()
    assert DriftThresholds(expectancy_z_breach=1.0).validate()
    assert not DriftThresholds().validate()
