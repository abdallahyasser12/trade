"""Opening-range breakout behavior tests over engineered sessions."""

import numpy as np
import pandas as pd
import pytest

from models import BotAction
from quant.contracts import Market
from quant.strategies.opening_range import OpeningRangeBreakoutStrategy, OpeningRangeConfig


def compact_config(**overrides):
    base = dict(
        symbols=("TEST",),
        market=Market.US_EQUITY,
        timeframe="5Min",
        atr_period=5,
        atr_percentile_lookback=20,
        volume_period=5,
        max_atr_percentile=100.0,
        opening_range_bars=4,
        max_entry_session_bar=48,
        min_or_volume_ratio=1.5,
    )
    base.update(overrides)
    return OpeningRangeConfig(**base)


def two_session_bars():
    """Day 1: flat warmup.  Day 2: 4-bar range then a volume breakout."""
    rows = []
    # Day 1: 40 flat bars, mild noise so ATR initializes.
    for i in range(40):
        rows.append((100.0, 100.6, 99.4, 100.0 + (0.1 if i % 2 else -0.1), 1000.0))
    # Day 2 bars 0-3: the opening range (high 101, low 99).
    day2 = [
        (100.0, 101.0, 99.0, 100.5, 1000.0),
        (100.5, 100.9, 99.2, 100.2, 1000.0),
        (100.2, 100.8, 99.1, 100.4, 1000.0),
        (100.4, 100.7, 99.3, 100.3, 1000.0),
        # bar 4: inside the range (previous-close reference for first-ness)
        (100.3, 100.9, 100.0, 100.5, 1000.0),
        # bar 5: FIRST breakout on 3x volume
        (100.6, 102.4, 100.5, 102.0, 3000.0),
        # bar 6: continues above the range (must NOT re-enter)
        (102.0, 103.2, 101.8, 103.0, 3000.0),
        (103.0, 103.5, 102.5, 103.2, 1500.0),
    ]
    rows.extend(day2)
    index = list(
        pd.date_range("2024-01-02 14:30", periods=40, freq="5min", tz="UTC")
    ) + list(pd.date_range("2024-01-03 14:30", periods=len(day2), freq="5min", tz="UTC"))
    frame = pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"], dtype=float
    )
    frame.index = pd.DatetimeIndex(index)
    return frame


def test_first_breakout_generates_buy_with_structure_stop():
    bars = two_session_bars()
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared = strategy.prepare_universe({"TEST": bars})
    breakout_ts = bars.index[45]  # day 2, bar 5
    decision = strategy.evaluate_prepared(prepared, breakout_ts, set())[0]
    assert decision.action is BotAction.BUY, decision.details
    assert all(decision.filters.values())
    # Stop is the tighter of or_low and the ATR stop, i.e. at least or_low.
    assert decision.stop_price >= 99.0
    assert decision.stop_price < decision.close


def test_no_entry_before_range_completes():
    bars = two_session_bars()
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared = strategy.prepare_universe({"TEST": bars})
    in_range_ts = bars.index[42]  # day 2, bar 2 — range still forming
    decision = strategy.evaluate_prepared(prepared, in_range_ts, set())[0]
    assert decision.action is BotAction.HOLD


def test_second_close_above_range_is_not_reentered():
    bars = two_session_bars()
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared = strategy.prepare_universe({"TEST": bars})
    continuation_ts = bars.index[46]  # day 2, bar 6
    decision = strategy.evaluate_prepared(prepared, continuation_ts, set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["first_breakout"] is False


def test_low_volume_breakout_blocked():
    bars = two_session_bars()
    bars.iloc[45, bars.columns.get_loc("volume")] = 900.0
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared = strategy.prepare_universe({"TEST": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[45], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["volume"] is False


def test_held_position_exits_below_vwap():
    bars = two_session_bars()
    # Collapse the continuation bar well below the session VWAP.
    bars.iloc[46] = (100.0, 100.1, 96.0, 96.5, 2000.0)
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared = strategy.prepare_universe({"TEST": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[46], {"TEST"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason == "vwap_loss"


def test_causality_under_future_appends():
    bars = two_session_bars()
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    prepared_full = strategy.prepare_universe({"TEST": bars})
    prepared_short = strategy.prepare_universe({"TEST": bars.iloc[:46]})
    ts = bars.index[45]
    full = strategy.evaluate_prepared(prepared_full, ts, set())[0]
    short = strategy.evaluate_prepared(prepared_short, ts, set())[0]
    assert full.action == short.action
    assert full.stop_price == pytest.approx(short.stop_price)


def test_ratchet_stop_never_loosens():
    strategy = OpeningRangeBreakoutStrategy(compact_config())
    stop, highest = strategy.ratchet_stop(99.0, 103.0, 104.0, 1.0)
    assert stop >= 99.0
    lower, _ = strategy.ratchet_stop(stop, 104.0, 90.0, 50.0)
    assert lower >= stop


def test_config_validation():
    assert compact_config(opening_range_bars=0).validate()
    assert compact_config(max_entry_session_bar=2).validate()
    assert not compact_config().validate()
    assert np is not None
