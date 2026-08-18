from datetime import datetime, timedelta, timezone

from quant.sleeve_supervision import (
    SleeveSupervisionConfig,
    _evaluate_values,
    compute_sleeve_health,
    sleeve_from_entry_reason,
)


class Store:
    def __init__(self, rows):
        self.rows = rows

    def get_attributions(self, strategy_id=None, include_shadow=True, limit=500):
        assert strategy_id == "crypto_intraday_adaptive"
        assert include_shadow is False
        return self.rows[:limit]


def test_entry_reason_maps_only_known_adaptive_sleeves():
    assert sleeve_from_entry_reason("adaptive_momentum_entry") == "momentum"
    assert sleeve_from_entry_reason("adaptive_reversion_entry") == "reversion"
    assert sleeve_from_entry_reason("other_signal") is None


def test_insufficient_sample_never_penalizes_or_boosts():
    health = _evaluate_values(
        "momentum",
        [0.10] * 39,
        SleeveSupervisionConfig(min_trades=40),
    )
    assert health.verdict == "insufficient_sample"
    assert health.confidence_multiplier == 1.0


def test_strong_positive_sleeve_stays_at_baseline_not_above_one():
    values = [0.30] * 28 + [-0.20] * 12
    health = _evaluate_values("breakout", values, SleeveSupervisionConfig())
    assert health.expectancy_r > 0
    assert health.confidence_multiplier == 1.0
    assert health.verdict == "baseline"


def test_mild_negative_sleeve_only_reduces_confidence():
    values = [0.20] * 19 + [-0.20] * 21
    health = _evaluate_values("pullback", values, SleeveSupervisionConfig())
    assert health.expectancy_r < 0
    assert health.verdict == "reduced_negative_edge"
    assert 0.0 < health.confidence_multiplier < 1.0


def test_statistically_negative_sleeve_is_suspended():
    values = [-0.25] * 40
    health = _evaluate_values("reversion", values, SleeveSupervisionConfig())
    assert health.upper_expectancy_bound_r <= 0
    assert health.verdict == "suspended_negative_edge"
    assert health.confidence_multiplier == 0.0


def test_only_current_lifecycle_rows_are_counted():
    now = datetime(2026, 8, 18, 2, 0, tzinfo=timezone.utc)
    since = now - timedelta(hours=1)
    old = {
        "entry_reason": "adaptive_momentum_entry",
        "r_multiple": -2.0,
        "recorded_at": (since - timedelta(minutes=1)).isoformat(),
    }
    current = [
        {
            "entry_reason": "adaptive_momentum_entry",
            "r_multiple": 0.20,
            "recorded_at": (since + timedelta(minutes=index + 1)).isoformat(),
        }
        for index in range(40)
    ]
    health = compute_sleeve_health(
        Store([old] + current),
        "crypto_intraday_adaptive",
        "momentum",
        since=since,
    )
    assert health.trades == 40
    assert health.expectancy_r == 0.20
    assert health.confidence_multiplier == 1.0
