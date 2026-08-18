"""Causality, ranking and stop-safety tests for the high-velocity hypothesis."""

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig
from models import BotAction
from quant.strategies.intraday_momentum import (
    IntradayMomentumConfig,
    IntradayMomentumStrategy,
)

SYMBOLS = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD")


def make_bars(seed, periods=180, drift=0.0002):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-08-01", periods=periods, freq="5min", tz="UTC")
    returns = drift + rng.normal(0, 0.0015, periods)
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(10_000, 50_000, periods).astype(float),
        },
        index=index,
    )


def config():
    return CryptoStrategyConfig(
        symbols=SYMBOLS,
        benchmark_symbol="BTC/USD",
        timeframe="5Min",
        fast_ma_period=5,
        slow_ma_period=20,
        regime_ma_period=30,
        regime_slope_period=5,
        atr_period=10,
        atr_percentile_lookback=20,
        adx_period=5,
        efficiency_period=5,
    )


def strategy():
    return IntradayMomentumStrategy(
        config(),
        IntradayMomentumConfig(
            fast_lookback=2,
            slow_lookback=6,
            vol_lookback=12,
            volume_lookback=8,
            top_k=2,
            min_score=0.0,
            min_volume_ratio=0.1,
            benchmark_fast_ma=6,
            benchmark_slow_ma=12,
        ),
    )


def universe():
    return {
        "BTC/USD": make_bars(1, drift=0.0001),
        "ETH/USD": make_bars(2, drift=0.0005),
        "SOL/USD": make_bars(3, drift=0.0004),
        "XRP/USD": make_bars(4, drift=-0.0001),
    }


def test_future_append_does_not_change_past_features():
    s = strategy()
    bars = make_bars(42, periods=160)
    prepared = s.prepare_symbol(bars)
    cutoff = bars.index[-20]
    past = prepared.loc[cutoff, ["atr", "realized_vol", "volume_ratio", "score"]]

    future = make_bars(99, periods=20)
    future.index = pd.date_range(
        bars.index[-1] + pd.Timedelta(minutes=5), periods=20, freq="5min", tz="UTC"
    )
    extended = pd.concat([bars, future])
    after = s.prepare_symbol(extended).loc[
        cutoff, ["atr", "realized_vol", "volume_ratio", "score"]
    ]
    pd.testing.assert_series_equal(past, after)


def test_buy_decisions_are_ranked_best_first():
    s = strategy()
    prepared = s.prepare_universe(universe())
    timestamp = next(iter(prepared.values())).index[-1]
    decisions = s.evaluate_prepared(prepared, timestamp)
    buys = [d for d in decisions if d.action is BotAction.BUY]
    assert len(buys) <= 2
    assert buys == sorted(buys, key=lambda d: (-(d.score or float("-inf")), d.symbol))


def test_held_symbol_exits_after_rank_decay_or_negative_momentum():
    s = strategy()
    prepared = s.prepare_universe(universe())
    timestamp = next(iter(prepared.values())).index[-1]
    worst = max(
        SYMBOLS,
        key=lambda symbol: float(prepared[symbol].loc[timestamp, "rank"]),
    )
    decisions = s.evaluate_prepared(prepared, timestamp, held_symbols={worst})
    decision = next(d for d in decisions if d.symbol == worst)
    assert decision.action in {BotAction.EXIT, BotAction.HOLD}
    if float(prepared[worst].loc[timestamp, "rank"]) > 3:
        assert decision.action is BotAction.EXIT


def test_trailing_stop_never_moves_down():
    s = strategy()
    stop, highest = s.ratchet_stop(95.0, 100.0, 104.0, 2.0)
    assert stop >= 95.0
    assert highest == 104.0
    next_stop, _ = s.ratchet_stop(stop, highest, 101.0, 3.0)
    assert next_stop >= stop
