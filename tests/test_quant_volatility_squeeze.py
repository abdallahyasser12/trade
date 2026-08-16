"""Adversarial causal and deterministic squeeze-expansion decision scenarios."""

import numpy as np
import pandas as pd
import pytest

from models import BotAction
from quant.contracts import Market
from quant.strategies.volatility_squeeze import (
    VolatilitySqueezeConfig,
    VolatilitySqueezeStrategy,
)


def compact_config(**overrides) -> VolatilitySqueezeConfig:
    values = dict(
        symbols=("BTC/USD",),
        market=Market.CRYPTO,
        benchmark_symbol="BTC/USD",
        timeframe="1Hour",
        squeeze_lookback=20,
        squeeze_percentile_max=30.0,
        breakout_period=3,
        trend_ma_period=5,
        min_expansion_volume_ratio=1.2,
        atr_period=2,
        atr_stop_multiplier=2.0,
        volume_period=2,
        min_volume_ratio=0.1,
        atr_percentile_lookback=20,
        max_atr_percentile=100.0,
    )
    values.update(overrides)
    return VolatilitySqueezeConfig(**values)


def squeeze_bars(
    *,
    periods: int = 40,
    breakout_at: int = 34,
    tight_bars: int = 5,
    compressed: bool = True,
    wide_a: float = 100.0,
    wide_b: float = 104.0,
    plateau: float = 110.0,
    breakout_close: float = 113.0,
    volume_spike: float = 3.0,
    gap_at: int | None = None,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Hourly UTC bars: a wide oscillation, an optional tight coil, a breakout.

    With ``compressed=False`` the oscillation (and its wide high/low padding)
    simply continues until the identical two-bar breakout event, so the ONLY
    difference between the two fixtures is the prior compression.
    """
    if breakout_at >= periods or breakout_at - tight_bars < 25:
        raise ValueError("Fixture needs the breakout bar and 25 warmup bars.")
    tight_start = breakout_at - tight_bars
    close = np.empty(periods, dtype=float)
    pad = np.empty(periods, dtype=float)
    for index in range(periods):
        if index < tight_start:
            close[index] = wide_a if index % 2 == 0 else wide_b
            pad[index] = 5.0
        elif index < breakout_at:
            if compressed:
                close[index] = plateau
                pad[index] = 0.5
            else:
                close[index] = wide_a if index % 2 == 0 else wide_b
                pad[index] = 5.0
        elif index == breakout_at:
            close[index] = breakout_close
            pad[index] = 0.5
        else:
            close[index] = breakout_close + (index - breakout_at) * 0.5
            pad[index] = 0.5
    if gap_at is not None:
        close[gap_at] = 90.0
        for index in range(gap_at + 1, periods):
            close[index] = 90.0 + (index - gap_at) * 0.5
    index_range = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    open_price = close.copy()
    open_price[1:] = (close[:-1] + close[1:]) / 2.0
    volume = np.full(periods, 1_000.0)
    volume[breakout_at] = 1_000.0 * volume_spike
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + pad,
            "low": np.minimum(open_price, close) - pad,
            "close": close,
            "volume": volume,
        },
        index=index_range,
    )


def test_compressed_breakout_generates_buy():
    bars = squeeze_bars()
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[34], set())[0]
    assert decision.action is BotAction.BUY
    assert all(decision.filters.values())
    assert decision.details["squeeze_percentile"] <= 30.0
    assert decision.stop_price is not None
    assert 0 < decision.stop_price < decision.close


def test_identical_breakout_without_compression_is_hold():
    # Same two-bar breakout event, same volume spike, same trend position —
    # only the prior compression is missing, and that alone must block entry.
    bars = squeeze_bars(compressed=False)
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[34], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["squeeze"] is False
    assert decision.filters["new_breakout"] is True
    assert decision.filters["volume"] is True
    assert decision.filters["trend"] is True


def test_breakout_without_expansion_volume_is_hold():
    bars = squeeze_bars(volume_spike=1.0)
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[34], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["volume"] is False


def test_breakout_below_trend_ma_is_hold():
    # The coil sits far below the earlier wide range, so a longer trend MA
    # still carries the old prices above the breakout close.
    bars = squeeze_bars(wide_a=120.0, wide_b=124.0, breakout_close=111.5)
    strategy = VolatilitySqueezeStrategy(compact_config(trend_ma_period=8))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[34], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["trend"] is False
    assert decision.filters["new_breakout"] is True


def test_persistent_extension_does_not_repeat_buy():
    bars = squeeze_bars(periods=40)
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decisions = [
        strategy.evaluate_prepared(prepared, bars.index[position], set())[0]
        for position in range(34, 40)
    ]
    assert decisions[0].action is BotAction.BUY
    assert [decision.action for decision in decisions[1:]] == [BotAction.HOLD] * 5
    assert all(decision.filters["new_breakout"] is False for decision in decisions[1:])


def test_held_position_trend_break_exits():
    bars = squeeze_bars(gap_at=37)
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[37], {"BTC/USD"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason == "trend_break"


def test_held_position_above_trend_ma_holds():
    bars = squeeze_bars()
    strategy = VolatilitySqueezeStrategy(compact_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[34], {"BTC/USD"})[0]
    assert decision.action is BotAction.HOLD
    assert decision.reason == "position_healthy"


def test_future_prices_cannot_change_historical_decision():
    bars = squeeze_bars(periods=40)
    strategy = VolatilitySqueezeStrategy(compact_config())
    timestamp = bars.index[34]
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
    strategy = VolatilitySqueezeStrategy(compact_config())
    tighter, high = strategy.ratchet_stop(95.0, 100.0, 110.0, 2.0)
    assert high == 110.0
    assert tighter >= 95.0
    unchanged, _ = strategy.ratchet_stop(tighter, high, 105.0, 10.0)
    assert unchanged == tighter


def test_disabled_trailing_stop_never_moves():
    strategy = VolatilitySqueezeStrategy(compact_config(enable_trailing_stop=False))
    stop, high = strategy.ratchet_stop(95.0, 100.0, 130.0, 1.0)
    assert stop == 95.0
    assert high == 130.0


def test_config_validate_rejects_bad_values():
    assert compact_config().validate() == []
    errors = compact_config(
        squeeze_lookback=5,
        squeeze_percentile_max=0.0,
        breakout_period=1,
        trend_ma_period=1,
        min_expansion_volume_ratio=0.0,
    ).validate()
    joined = " ".join(errors)
    assert "Squeeze lookback" in joined
    assert "percentile ceiling" in joined
    assert "Breakout period" in joined
    assert "Trend MA period" in joined
    assert "expansion volume ratio" in joined
    # Inherited structural validation still applies.
    assert compact_config(symbols=()).validate()


def test_strategy_rejects_invalid_config():
    with pytest.raises(ValueError):
        VolatilitySqueezeStrategy(compact_config(squeeze_lookback=5))


def test_warmup_covers_squeeze_percentile_chain():
    config = compact_config()
    # ATR seed + range window + percentile lookback + shift + pad.
    assert config.warmup_bars() >= 2 + 3 + 20 + 3
