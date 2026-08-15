"""Round-trip friction expressed in units of the trade's own risk."""

import math

import pytest

from cost_model import (
    break_even_win_rate,
    cost_to_risk,
    round_trip_cost_bps,
    stop_distance_bps,
)


def test_round_trip_charges_both_legs_of_every_component():
    # Two crossings of half the spread, two fees, two slippage hits.
    assert round_trip_cost_bps(
        slippage_bps=10.0, spread_bps=5.0, taker_fee_bps=25.0
    ) == pytest.approx(75.0)
    assert round_trip_cost_bps(
        slippage_bps=0.0, spread_bps=0.0, taker_fee_bps=0.0
    ) == 0.0


def test_negative_or_nonfinite_cost_assumptions_are_rejected():
    with pytest.raises(ValueError):
        round_trip_cost_bps(slippage_bps=-1.0, spread_bps=5.0, taker_fee_bps=25.0)
    with pytest.raises(ValueError):
        round_trip_cost_bps(
            slippage_bps=float("nan"), spread_bps=5.0, taker_fee_bps=25.0
        )


def test_stop_distance_requires_a_usable_long_stop():
    assert stop_distance_bps(100.0, 99.0) == pytest.approx(100.0)
    for entry, stop in ((100.0, 100.0), (100.0, 101.0), (0.0, -1.0), (100.0, 0.0)):
        assert stop_distance_bps(entry, stop) == math.inf


def test_cost_to_risk_is_friction_measured_in_R():
    # 1R = 100bps here, so a 75bps round trip is 0.75R.
    assert cost_to_risk(
        entry_price=100.0, stop_price=99.0, round_trip_bps=75.0
    ) == pytest.approx(0.75)
    # A wider stop makes the same friction cheaper in risk terms.
    assert cost_to_risk(
        entry_price=100.0, stop_price=97.0, round_trip_bps=75.0
    ) == pytest.approx(0.25)


def test_unusable_inputs_fail_closed_at_infinity():
    """A gate on a maximum must reject, never admit, on bad inputs."""

    assert cost_to_risk(entry_price=100.0, stop_price=100.0, round_trip_bps=75.0) == math.inf
    assert cost_to_risk(entry_price=100.0, stop_price=99.0, round_trip_bps=math.nan) == math.inf
    assert cost_to_risk(entry_price=math.nan, stop_price=99.0, round_trip_bps=75.0) == math.inf


def test_break_even_win_rate_rises_with_cost():
    free = break_even_win_rate(0.0, reward_to_risk=2.0)
    assert free == pytest.approx(1.0 / 3.0)
    costly = break_even_win_rate(0.86, reward_to_risk=2.0)
    assert costly > free
    assert costly == pytest.approx(0.619, abs=0.005)
    # Friction at or above the reward makes break-even unreachable.
    assert math.isnan(break_even_win_rate(2.0, reward_to_risk=2.0))
