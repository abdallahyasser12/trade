from types import SimpleNamespace

from quant.evidence_scanner import (
    adverse_depth_quality,
    depth_imbalance,
    expectancy_confidence,
)


def test_positive_expectancy_low_hit_rate_can_have_high_evidence_confidence():
    health = SimpleNamespace(
        trades=200,
        expectancy_r=0.12,
        sharpe=0.18,
        hit_rate=0.42,
    )
    confidence = expectancy_confidence(health)
    assert confidence > 0.95
    assert health.hit_rate < 0.55


def test_non_positive_expectancy_never_gets_positive_alpha_confidence():
    health = SimpleNamespace(
        trades=200,
        expectancy_r=-0.01,
        sharpe=0.20,
        hit_rate=0.80,
    )
    assert expectancy_confidence(health) <= 0.5


def test_small_sample_has_zero_confidence():
    health = SimpleNamespace(
        trades=5,
        expectancy_r=1.0,
        sharpe=2.0,
        hit_rate=1.0,
    )
    assert expectancy_confidence(health) == 0.0


def test_depth_imbalance_is_bounded_and_ask_heavy_depth_only_penalizes():
    quote = SimpleNamespace(bid_size=25.0, ask_size=75.0)
    imbalance = depth_imbalance(quote)
    assert imbalance == -0.5
    assert adverse_depth_quality(imbalance) == 0.75


def test_positive_depth_never_boosts_above_neutral_quality():
    quote = SimpleNamespace(bid_size=90.0, ask_size=10.0)
    imbalance = depth_imbalance(quote)
    assert imbalance == 0.8
    assert adverse_depth_quality(imbalance) == 1.0


def test_missing_or_invalid_depth_is_neutral_not_fabricated():
    assert depth_imbalance(SimpleNamespace(bid_size=None, ask_size=None)) is None
    assert adverse_depth_quality(None) == 1.0
