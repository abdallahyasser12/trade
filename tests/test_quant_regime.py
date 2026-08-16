"""Behavioral tests for the probabilistic regime engine."""

import numpy as np
import pandas as pd
import pytest

from quant.contracts import Market, RegimeLabel
from quant.regime import RegimeConfig, RegimeEngine


def make_benchmark(closes, start="2022-01-01", freq="1D"):
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.full(len(closes), 1_000.0),
        },
        index=index,
    )


def compact_config():
    return RegimeConfig(
        trend_ma_period=30,
        trend_slope_period=5,
        fast_ma_period=10,
        momentum_period=20,
        vol_short_period=10,
        vol_long_period=40,
        vol_percentile_lookback=60,
        drawdown_lookback=60,
        correlation_window=20,
    )


def trending_up(periods=200, drift=0.004, seed=1):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, periods))


def trending_down(periods=200, drift=-0.004, seed=2):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, periods))


def flat(periods=200, seed=3):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 0.05, periods))


def test_uptrend_scores_bullish_and_risk_on():
    engine = RegimeEngine(compact_config())
    assessment = engine.assess(make_benchmark(trending_up()), market=Market.US_EQUITY)
    bull = assessment.probability(RegimeLabel.STRONG_BULL) + assessment.probability(
        RegimeLabel.WEAK_BULL
    )
    bear = assessment.probability(RegimeLabel.STRONG_BEAR) + assessment.probability(
        RegimeLabel.WEAK_BEAR
    )
    assert bull > 0.6
    assert bear < 0.1
    assert assessment.risk_on_probability > 0.5
    assert assessment.dominant_trend_label in (RegimeLabel.STRONG_BULL, RegimeLabel.WEAK_BULL)


def test_downtrend_scores_bearish():
    engine = RegimeEngine(compact_config())
    assessment = engine.assess(make_benchmark(trending_down()), market=Market.US_EQUITY)
    bear = assessment.probability(RegimeLabel.STRONG_BEAR) + assessment.probability(
        RegimeLabel.WEAK_BEAR
    )
    assert bear > 0.6
    assert assessment.risk_on_probability < 0.2


def test_flat_market_scores_range():
    engine = RegimeEngine(compact_config())
    assessment = engine.assess(make_benchmark(flat()), market=Market.US_EQUITY)
    assert assessment.probability(RegimeLabel.RANGE) > 0.4


def test_trend_probabilities_sum_to_one():
    engine = RegimeEngine(compact_config())
    for closes in (trending_up(), trending_down(), flat()):
        assessment = engine.assess(make_benchmark(closes), market=Market.CRYPTO)
        total = sum(
            assessment.probability(label)
            for label in (
                RegimeLabel.STRONG_BULL,
                RegimeLabel.WEAK_BULL,
                RegimeLabel.RANGE,
                RegimeLabel.WEAK_BEAR,
                RegimeLabel.STRONG_BEAR,
            )
        )
        assert total == pytest.approx(1.0)


def test_vol_spike_raises_expansion_and_stress():
    rng = np.random.default_rng(9)
    # The crash must be shorter than the long-vol window (40 bars) so the
    # short window sits inside the crash while the long window still spans
    # mostly calm history — otherwise there is no expansion to detect.
    calm = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.002, 185))
    crash = calm[-1] * np.cumprod(1.0 + rng.normal(-0.02, 0.05, 15))
    closes = np.concatenate([calm, crash])
    engine = RegimeEngine(compact_config())
    assessment = engine.assess(make_benchmark(closes), market=Market.CRYPTO)
    assert assessment.probability(RegimeLabel.VOL_EXPANSION) > 0.5
    assert assessment.probability(RegimeLabel.LIQUIDITY_STRESS) > 0.3
    assert assessment.probability(RegimeLabel.VOL_CONTRACTION) == pytest.approx(0.0)


def test_insufficient_history_zero_confidence():
    engine = RegimeEngine(compact_config())
    assessment = engine.assess(make_benchmark(trending_up(30)), market=Market.US_EQUITY)
    assert assessment.confidence == 0.0
    assert all(value == 0.0 for value in assessment.probabilities.values())


def test_member_inputs_raise_confidence_and_feed_breadth():
    engine = RegimeEngine(compact_config())
    bars = make_benchmark(trending_up())
    bare = engine.assess(bars, market=Market.CRYPTO)
    members = {
        f"SYM{i}": pd.Series(
            trending_up(seed=10 + i), index=bars.index, dtype=float
        )
        for i in range(4)
    }
    enriched = engine.assess(bars, market=Market.CRYPTO, member_closes=members)
    assert enriched.confidence > bare.confidence
    assert enriched.inputs["breadth"] == pytest.approx(1.0)


def test_assessment_is_deterministic():
    engine = RegimeEngine(compact_config())
    bars = make_benchmark(trending_up())
    a = engine.assess(bars, market=Market.US_EQUITY)
    b = engine.assess(bars, market=Market.US_EQUITY)
    assert a.probabilities == b.probabilities
    assert a.confidence == b.confidence


def test_config_validation():
    assert RegimeConfig(fast_ma_period=300).validate()
    assert RegimeConfig(vol_long_period=5, vol_short_period=20).validate()
    assert not RegimeConfig().validate()
