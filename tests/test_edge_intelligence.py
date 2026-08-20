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
        "flow_pressure",
        "wick_rejection",
        "structure_persistence",
        "horizon_agreement",
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


def test_profile_weights_are_normalized_with_identical_component_sets():
    from quant.edge_intelligence import _weights

    profiles = ("breakout", "mean_reversion", "trend_pullback", "trend", "scalp", "generic")
    reference_keys = set(_weights("generic"))
    for profile in profiles:
        weights = _weights(profile)
        assert set(weights) == reference_keys, profile
        assert abs(sum(weights.values()) - 1.0) < 1e-9, profile


def test_scalp_profile_is_recognized():
    assert edge_profile("crypto_intraday_scalp", "scalp_momentum_entry") == "scalp"
    assert edge_profile("anything", "scalp_breakout") == "scalp"


def test_unsorted_or_duplicated_index_refuses_measurement():
    bars = _bars()
    shuffled = bars.iloc[::-1]
    assessment = assess_edge(
        shuffled,
        bars.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
    )
    assert not assessment.measurable
    assert assessment.quality_multiplier == 1.0
    assert "unsorted_time_index" in assessment.reasons

    import pandas as pd

    duplicated = pd.concat([bars, bars.iloc[[-1]]])
    assessment2 = assess_edge(
        duplicated,
        bars.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
    )
    assert not assessment2.measurable
    assert "unsorted_time_index" in assessment2.reasons


def test_nan_on_evaluated_bar_is_refused_not_substituted():
    bars = _bars()
    bars.loc[bars.index[-1], "volume"] = np.nan
    assessment = assess_edge(
        bars,
        bars.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
    )
    assert not assessment.measurable
    assert assessment.quality_multiplier == 1.0
    assert "evaluated_bar_invalid" in assessment.reasons


def test_min_bars_boundary_is_exact():
    config = EdgeIntelligenceConfig()
    just_enough = _bars(config.min_bars)
    ok = assess_edge(
        just_enough,
        just_enough.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
        config=config,
    )
    assert ok.measurable

    one_short = _bars(config.min_bars - 1)
    starved = assess_edge(
        one_short,
        one_short.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
        config=config,
    )
    assert not starved.measurable
    assert "insufficient_edge_history" in starved.reasons


def test_zero_range_candle_is_neutral_not_bearish():
    bars = _bars()
    last = bars.index[-1]
    price = float(bars.loc[last, "close"])
    bars.loc[last, ["open", "high", "low", "close"]] = price
    assessment = assess_edge(
        bars,
        last,
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
    )
    assert assessment.measurable
    assert assessment.components["candle_pressure"] == 0.5


def test_flat_price_history_is_refused_not_scored():
    rows = 120
    index = pd.date_range("2026-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    flat = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000.0,
        },
        index=index,
    )
    assessment = assess_edge(
        flat,
        flat.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="momentum",
    )
    assert not assessment.measurable
    assert assessment.quality_multiplier == 1.0


def test_reversion_dip_is_not_punished_by_the_agreement_factor():
    """A textbook dip must not pin the agreement factor to its floor.

    Raw momentum on a controlled dip is bearish; the profile transformation
    re-expresses the evidence as reversion fit, so the coherence factor for a
    clean dip should sit meaningfully above its 0.78 floor.
    """
    rows = 120
    index = pd.date_range("2026-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.10 * x
    close[-8:] = close[-9] - 0.22 * np.arange(1, 9)  # controlled pullback
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.30  # long lower wicks: dip buying
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    assessment = assess_edge(
        frame,
        frame.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="adaptive_reversion_entry",
    )
    assert assessment.measurable
    assert assessment.profile == "mean_reversion"
    assert assessment.diagnostics["agreement_factor"] > 0.82
