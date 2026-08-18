from types import SimpleNamespace

from quant.evidence_scanner import expectancy_confidence


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
