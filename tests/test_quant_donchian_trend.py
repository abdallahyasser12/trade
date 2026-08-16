"""Adversarial causal and deterministic Donchian trend decision scenarios."""

import numpy as np
import pandas as pd
import pytest

from models import BotAction
from quant.contracts import Market
from quant.strategies.donchian_trend import DonchianTrendConfig, DonchianTrendStrategy


def compact_config(**overrides) -> DonchianTrendConfig:
    values = dict(
        symbols=("BTC/USD",),
        market=Market.CRYPTO,
        benchmark_symbol="BTC/USD",
        timeframe="1Hour",
        breakout_period=3,
        trend_ma_period=5,
        momentum_period=2,
        min_efficiency_ratio=0.0,
        exit_channel_fraction=0.5,
        atr_period=2,
        atr_stop_multiplier=2.0,
        volume_period=2,
        min_volume_ratio=0.1,
        atr_percentile_lookback=20,
        max_atr_percentile=100.0,
    )
    values.update(overrides)
    return DonchianTrendConfig(**values)


def breakout_bars(
    *,
    periods: int = 35,
    breakout_at: int = 30,
    base_close=None,
    gap_at: int | None = None,
    zero_volume_at: int | None = None,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Hourly UTC uptrend with a plateau and a fresh channel breakout.

    ``base_close`` lets tests swap the pre-plateau path (e.g. a downtrend)
    while keeping the identical plateau/breakout closes at the event bar.
    """
    if periods <= breakout_at or breakout_at < 27:
        raise ValueError("Fixture needs the breakout bar and 27 warmup bars.")
    if base_close is None:
        close = np.asarray([100.0 + index * 0.4 for index in range(periods)], dtype=float)
    else:
        close = np.asarray([base_close(index) for index in range(periods)], dtype=float)
    close[breakout_at - 3 : breakout_at + 1] = (111.0, 110.8, 110.9, 112.5)
    for index in range(breakout_at + 1, periods):
        close[index] = 112.5 + (index - breakout_at) * 0.5
    if gap_at is not None:
        close[gap_at] = 90.0
        for index in range(gap_at + 1, periods):
            close[index] = 90.0 + (index - gap_at)
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    open_price = close.copy()
    open_price[1:] = (close[:-1] + close[1:]) / 2.0
    volume = np.full(periods, 1_000.0)
    if zero_volume_at is not None:
        volume[zero_volume_at] = 0.0
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def downtrend_pop_bars(**kwargs) -> pd.DataFrame:
    """Same breakout event, but preceded by a decline from much higher prices."""
    return breakout_bars(base_close=lambda index: 150.0 - index * 1.3, **kwargs)


def test_clean_breakout_generates_buy():
    bars = breakout_bars()
    strategy = DonchianTrendStrategy(compact_config())
    # Fixture's latest bars move away from the breakout; inspect the exact
    # breakout close to prove the causal event rule.
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.action is BotAction.BUY
    assert all(decision.filters.values())
    assert decision.stop_price is not None
    assert 0 < decision.stop_price < decision.close


def test_breakout_with_zero_volume_is_hold():
    bars = breakout_bars(zero_volume_at=30)
    strategy = DonchianTrendStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["volume"] is False


def test_breakout_with_low_efficiency_is_hold():
    # Over a 3-bar window the plateau detour makes the path inefficient.
    bars = breakout_bars()
    strategy = DonchianTrendStrategy(
        compact_config(momentum_period=3, min_efficiency_ratio=0.95)
    )
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.efficiency_ratio < 0.95
    assert decision.action is BotAction.HOLD
    assert decision.filters["efficiency"] is False


def test_breakout_with_negative_momentum_is_hold():
    # A 6-bar lookback reaches back into the decline, so momentum is negative
    # even though the breakout itself is genuine.
    bars = downtrend_pop_bars()
    strategy = DonchianTrendStrategy(compact_config(momentum_period=6))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.momentum < 0
    assert decision.action is BotAction.HOLD
    assert decision.filters["momentum"] is False
    assert decision.filters["new_breakout"] is True


def test_breakout_below_trend_ma_is_hold():
    # An 8-bar trend MA still carries the earlier decline above the close.
    bars = downtrend_pop_bars()
    strategy = DonchianTrendStrategy(compact_config(trend_ma_period=8))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["trend"] is False
    assert decision.filters["new_breakout"] is True


def test_breakout_in_volatility_stress_is_hold():
    bars = breakout_bars()
    strategy = DonchianTrendStrategy(compact_config(max_atr_percentile=50.0))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.atr_percentile > 50.0
    assert decision.action is BotAction.HOLD
    assert decision.filters["volatility"] is False


def test_persistent_breakout_does_not_repeat_buy():
    bars = breakout_bars(periods=42)
    strategy = DonchianTrendStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decisions = [
        strategy.evaluate_prepared(prepared, bars.index[position], set())[0]
        for position in range(30, 41)
    ]
    assert decisions[0].action is BotAction.BUY
    assert [decision.action for decision in decisions[1:]] == [BotAction.HOLD] * 10
    assert all(decision.filters["new_breakout"] is False for decision in decisions[1:])


def test_held_position_channel_break_exits():
    bars = breakout_bars(gap_at=32)
    strategy = DonchianTrendStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[32], {"BTC/USD"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason == "channel_exit"
    assert decision.close < decision.details["channel_midpoint"]


def test_held_position_volatility_stress_exits():
    bars = breakout_bars()
    strategy = DonchianTrendStrategy(compact_config(max_atr_percentile=50.0))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], {"BTC/USD"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason == "volatility_stress_exit"


def test_held_position_above_midpoint_holds():
    bars = breakout_bars()
    strategy = DonchianTrendStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], {"BTC/USD"})[0]
    assert decision.action is BotAction.HOLD
    assert decision.reason == "position_healthy"


def test_future_prices_cannot_change_historical_decision():
    bars = breakout_bars(periods=40)
    strategy = DonchianTrendStrategy(compact_config())
    timestamp = bars.index[30]
    prefix = bars.loc[:timestamp]
    prefix_decision = strategy.evaluate_prepared(
        strategy.prepare_universe({"BTC/USD": prefix}), timestamp, set()
    )[0]
    altered = bars.copy()
    altered.loc[altered.index > timestamp, ["open", "high", "low", "close"]] *= 10
    full_decision = strategy.evaluate_prepared(
        strategy.prepare_universe({"BTC/USD": altered}), timestamp, set()
    )[0]
    assert full_decision.to_dict() == prefix_decision.to_dict()


def test_trailing_stop_can_only_tighten_from_completed_close():
    strategy = DonchianTrendStrategy(compact_config())
    tighter, high = strategy.ratchet_stop(95.0, 100.0, 110.0, 2.0)
    assert high == 110.0
    assert tighter >= 95.0
    unchanged, _ = strategy.ratchet_stop(tighter, high, 105.0, 10.0)
    assert unchanged == tighter


def test_disabled_trailing_stop_never_moves():
    strategy = DonchianTrendStrategy(compact_config(enable_trailing_stop=False))
    stop, high = strategy.ratchet_stop(95.0, 100.0, 130.0, 1.0)
    assert stop == 95.0
    assert high == 130.0


def test_config_validate_rejects_bad_values():
    assert compact_config().validate() == []
    errors = compact_config(
        breakout_period=1,
        trend_ma_period=1,
        momentum_period=1,
        min_efficiency_ratio=1.5,
        exit_channel_fraction=-0.1,
    ).validate()
    joined = " ".join(errors)
    assert "Breakout period" in joined
    assert "Trend MA period" in joined
    assert "Momentum period" in joined
    assert "efficiency ratio" in joined
    assert "channel fraction" in joined
    # Inherited structural validation still applies.
    assert compact_config(symbols=()).validate()


def test_strategy_rejects_invalid_config():
    with pytest.raises(ValueError):
        DonchianTrendStrategy(compact_config(breakout_period=1))


def test_warmup_covers_every_indicator_window():
    config = compact_config(trend_ma_period=40, momentum_period=35, breakout_period=45)
    assert config.warmup_bars() >= 48  # breakout_period + shift + previous row + pad
    assert config.warmup_bars() >= 42  # trend MA + pad
    assert config.warmup_bars() >= 37  # momentum/efficiency + pad
