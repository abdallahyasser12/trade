"""Hold-intelligence contract tests: causal, bounded, neutral when blind."""

import numpy as np
import pandas as pd

from quant.exit_intelligence import (
    HoldIntelligenceConfig,
    HoldStance,
    assess_hold,
    hold_profile,
)

COMPONENTS = {
    "trend_health",
    "momentum_health",
    "pressure_health",
    "flow_health",
    "advance_health",
    "efficiency_health",
}


def _frame(closes: np.ndarray, volumes: np.ndarray | None = None) -> pd.DataFrame:
    rows = len(closes)
    index = pd.date_range("2026-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) + 0.15
    low = np.minimum(open_, closes) - 0.15
    if volumes is None:
        volumes = np.full(rows, 1_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": volumes},
        index=index,
    )


def _advancing(rows: int = 90) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return _frame(100.0 + 0.30 * x + 0.05 * np.sin(x / 3.0))


def _collapsing(rows: int = 90) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    closes = 100.0 + 0.30 * x
    # The last 12 bars roll over hard: lower closes, rising volume.
    closes[-12:] = closes[-13] - 0.55 * np.arange(1, 13)
    volumes = np.full(rows, 1_000.0)
    volumes[-12:] = 2_400.0
    return _frame(closes, volumes)


def test_hold_assessment_is_bounded_with_named_components():
    frame = _advancing()
    assessment = assess_hold(
        frame,
        frame.index[-1],
        strategy_id="crypto_intraday_adaptive",
        signal_reason="adaptive_momentum_entry",
    )
    assert assessment.measurable
    assert 0.0 <= assessment.hold_score <= 100.0
    assert set(assessment.components) == COMPONENTS
    assert all(0.0 <= value <= 1.0 for value in assessment.components.values())


def test_advancing_market_holds_and_collapse_demands_exit():
    advancing = _advancing()
    collapsing = _collapsing()
    healthy = assess_hold(
        advancing,
        advancing.index[-1],
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    broken = assess_hold(
        collapsing,
        collapsing.index[-1],
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    assert healthy.hold_score > broken.hold_score
    assert healthy.stance is HoldStance.HOLD
    assert broken.stance is HoldStance.EXIT


def test_hold_assessment_is_strictly_causal():
    frame = _advancing(100)
    timestamp = frame.index[89]
    before = assess_hold(
        frame.iloc[:90],
        timestamp,
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    future = _collapsing(20).copy()
    future.index = pd.date_range(
        timestamp + pd.Timedelta(minutes=5), periods=20, freq="5min", tz="UTC"
    )
    combined = pd.concat([frame.iloc[:90], future])
    after = assess_hold(
        combined,
        timestamp,
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    assert before.to_dict() == after.to_dict()


def test_insufficient_or_malformed_history_is_neutral_hold():
    short = _advancing(30)
    starved = assess_hold(
        short,
        short.index[-1],
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    assert not starved.measurable
    assert starved.stance is HoldStance.HOLD
    assert starved.hold_score == 50.0
    assert "insufficient_hold_history" in starved.reasons

    frame = _advancing()
    shuffled = frame.iloc[::-1]
    unsorted = assess_hold(
        shuffled,
        frame.index[-1],
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    assert not unsorted.measurable
    assert "unsorted_time_index" in unsorted.reasons

    poisoned = frame.copy()
    poisoned.loc[poisoned.index[-1], "volume"] = np.nan
    dropped = assess_hold(
        poisoned,
        poisoned.index[-1],
        strategy_id="",
        signal_reason="adaptive_momentum_entry",
    )
    assert not dropped.measurable
    assert "evaluated_bar_invalid" in dropped.reasons


def test_hold_profiles_extend_entry_mapping_with_scalp():
    assert hold_profile("crypto_intraday_scalp", "scalp_momentum_entry") == "scalp"
    assert hold_profile("crypto_intraday_adaptive", "adaptive_reversion_entry") == "mean_reversion"
    assert hold_profile("crypto_intraday_adaptive", "adaptive_pullback_entry") == "trend_pullback"
    assert hold_profile("crypto_intraday_adaptive", "adaptive_momentum_entry") == "trend"


def test_invalid_configuration_is_rejected():
    bad = HoldIntelligenceConfig(exit_threshold=80.0, tighten_threshold=55.0)
    try:
        assess_hold(
            _advancing(),
            _advancing().index[-1],
            strategy_id="",
            signal_reason="x",
            config=bad,
        )
    except ValueError as error:
        assert "Thresholds" in str(error)
    else:  # pragma: no cover - the assertion above must fire
        raise AssertionError("invalid config was accepted")
