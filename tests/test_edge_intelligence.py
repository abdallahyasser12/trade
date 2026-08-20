from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant.edge_intelligence import (
    EdgeIntelligenceConfig,
    assess_edge,
    edge_profile,
)
from quant.opportunity import OpportunityScoringConfig, score_components


def _bars(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.12 * x + 0.28 * np.sin(x / 3.0)
    open_ = close - 0.05 + 0.03 * np.sin(x / 5.0)
    high = np.maximum(open_, close) + 0.22 + 0.02 * np.cos(x / 4.0)
    low = np.minimum(open_, close) - 0.20 - 0.02 * np.sin(x / 6.0)
    volume = 1_000.0 + 120.0 * (1.0 + np.sin(x / 7.0))
    volume[-1] = 1_900.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_edge_assessment_is_bounded_negative_only_and_measurable():
    bars = _bars()
    quote = SimpleNamespace(
        bid_price=113.9,
        ask_price=114.0,
        bid_size=600.0,
        ask_size=400.0,
    )
    assessment = assess_edge(
        bars,
        bars.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="cross_sectional_momentum",
        quote=quote,
    )
    assert assessment.measurable
    assert 0.0 <= assessment.score <= 100.0
    assert EdgeIntelligenceConfig().min_quality_multiplier <= assessment.quality_multiplier <= 1.0
    assert set(assessment.components) == {
        "trend_alignment",
        "impulse_to_noise",
        "path_efficiency",
        "range_location",
        "participation",
        "vwap_location",
        "volatility_release",
        "candle_pressure",
    }
    assert assessment.diagnostics["spread_bps"] is not None


def test_edge_assessment_is_strictly_causal_when_future_bars_are_appended():
    bars = _bars(120)
    timestamp = bars.index[99]
    before = assess_edge(
        bars.iloc[:100],
        timestamp,
        strategy_id="opening_range_breakout",
        signal_reason="opening_range_breakout",
    )

    future = _bars(20).copy()
    future.index = pd.date_range(
        timestamp + pd.Timedelta(minutes=5), periods=20, freq="5min", tz="UTC"
    )
    future["close"] = np.linspace(250.0, 40.0, len(future))
    future["open"] = future["close"] + 3.0
    future["high"] = future[["open", "close"]].max(axis=1) + 8.0
    future["low"] = future[["open", "close"]].min(axis=1) - 8.0
    future["volume"] = 50_000.0
    combined = pd.concat([bars.iloc[:100], future])

    after = assess_edge(
        combined,
        timestamp,
        strategy_id="opening_range_breakout",
        signal_reason="opening_range_breakout",
    )
    assert before.to_dict() == after.to_dict()


def test_uninitialized_edge_history_is_neutral_not_fabricated():
    bars = _bars(40)
    assessment = assess_edge(
        bars,
        bars.index[-1],
        strategy_id="volatility_squeeze",
        signal_reason="volatility_squeeze",
    )
    assert not assessment.measurable
    assert assessment.quality_multiplier == 1.0
    assert "insufficient_edge_history" in assessment.reasons


def test_strategy_aware_profiles_cover_current_intraday_families():
    assert edge_profile("vwap_reversion", "entry") == "mean_reversion"
    assert edge_profile("opening_range_breakout", "entry") == "breakout"
    assert edge_profile("volatility_squeeze", "expansion") == "breakout"
    assert edge_profile("crypto_intraday_adaptive", "trend_pullback") == "trend_pullback"
    assert edge_profile("crypto_intraday_adaptive", "cross_sectional_momentum") == "trend"


def test_setup_quality_can_only_reduce_opportunity_score():
    candidate = SimpleNamespace(confidence=0.80)
    config = OpportunityScoringConfig()
    baseline = score_components(
        config,
        candidate,
        net_alpha_bps=10.0,
        regime_compatibility=1.0,
        strategy_reliability=1.0,
        liquidity_quality=1.0,
        setup_quality=1.0,
    )
    penalized = score_components(
        config,
        candidate,
        net_alpha_bps=10.0,
        regime_compatibility=1.0,
        strategy_reliability=1.0,
        liquidity_quality=1.0,
        setup_quality=0.40,
    )
    boosted_attempt = score_components(
        config,
        candidate,
        net_alpha_bps=10.0,
        regime_compatibility=1.0,
        strategy_reliability=1.0,
        liquidity_quality=1.0,
        setup_quality=3.0,
    )
    assert penalized["score"] == baseline["score"] * 0.40
    assert boosted_attempt["score"] == baseline["score"]
    assert boosted_attempt["setup_quality"] == 1.0
