"""Causality, regime-routing and stop-safety tests for adaptive intraday v2."""

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig
from models import BotAction
from quant.strategies.intraday_adaptive import (
    AdaptiveIntradayConfig,
    AdaptiveIntradayStrategy,
)

SYMBOLS = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "LTC/USD")


def make_bars(seed, periods=260, drift=0.00025):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-07-01", periods=periods, freq="5min", tz="UTC")
    returns = drift + rng.normal(0, 0.0018, periods)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0025,
            "low": np.minimum(open_, close) * 0.9975,
            "close": close,
            "volume": rng.integers(20_000, 80_000, periods).astype(float),
        },
        index=index,
    )


def strategy():
    config = CryptoStrategyConfig(
        symbols=SYMBOLS,
        benchmark_symbol="BTC/USD",
        timeframe="5Min",
        atr_period=10,
        atr_percentile_lookback=20,
        adx_period=5,
        efficiency_period=5,
    )
    adaptive = AdaptiveIntradayConfig(
        fast_lookback=2,
        medium_lookback=6,
        slow_lookback=12,
        vol_lookback=16,
        volume_lookback=8,
        breakout_lookback=8,
        reversion_lookback=10,
        trend_fast_span=6,
        trend_slow_span=12,
        atr_baseline_lookback=20,
        top_k=3,
        min_entry_score=0.10,
        exit_score=0.03,
        min_volume_ratio=0.10,
    )
    return AdaptiveIntradayStrategy(config, adaptive)


def universe():
    return {
        symbol: make_bars(i + 1, drift=0.00005 + i * 0.00008)
        for i, symbol in enumerate(SYMBOLS)
    }


def test_future_append_does_not_change_past_adaptive_features():
    s = strategy()
    bars = make_bars(42, periods=220)
    prepared = s.prepare_symbol(bars)
    cutoff = bars.index[-30]
    columns = [
        "atr",
        "realized_vol",
        "alpha_core",
        "trend_atr",
        "breakout_atr",
        "reversion_z",
        "volume_ratio",
        "atr_ratio",
    ]
    before = prepared.loc[cutoff, columns]

    future = make_bars(99, periods=25)
    future.index = pd.date_range(
        bars.index[-1] + pd.Timedelta(minutes=5), periods=25, freq="5min", tz="UTC"
    )
    after = s.prepare_symbol(pd.concat([bars, future])).loc[cutoff, columns]
    pd.testing.assert_series_equal(before, after)


def test_adaptive_decisions_are_complete_and_buy_priority_is_deterministic():
    s = strategy()
    prepared = s.prepare_universe(universe())
    timestamp = next(iter(prepared.values())).index[-1]
    decisions = s.evaluate_prepared(prepared, timestamp)
    assert {decision.symbol for decision in decisions} == set(SYMBOLS)
    buys = [decision for decision in decisions if decision.action is BotAction.BUY]
    assert buys == sorted(
        buys,
        key=lambda decision: (-(decision.score or float("-inf")), decision.symbol),
    )


def test_stress_probability_blocks_new_entries_and_forces_held_exit():
    s = strategy()
    prepared = s.prepare_universe(universe())
    timestamp = next(iter(prepared.values())).index[-1]
    for frame in prepared.values():
        frame.loc[timestamp, "stress_probability"] = 1.0
        frame.loc[timestamp, "risk_off_probability"] = 1.0
    decisions = s.evaluate_prepared(prepared, timestamp)
    assert all(decision.action is not BotAction.BUY for decision in decisions)

    held = SYMBOLS[0]
    held_decisions = s.evaluate_prepared(prepared, timestamp, held_symbols={held})
    held_decision = next(decision for decision in held_decisions if decision.symbol == held)
    assert held_decision.action is BotAction.EXIT
    assert held_decision.reason == "adaptive_stress_exit"


def test_strategy_exposes_regime_and_sleeve_diagnostics_after_warmup():
    s = strategy()
    prepared = s.prepare_universe(universe())
    timestamp = next(iter(prepared.values())).index[-1]
    decisions = s.evaluate_prepared(prepared, timestamp)
    enriched = [decision for decision in decisions if decision.details]
    assert enriched
    for decision in enriched:
        assert "momentum_sleeve" in decision.details
        assert "reversion_sleeve" in decision.details
        assert 0.0 <= decision.details["stress_probability"] <= 1.0


def test_adaptive_trailing_stop_never_moves_down():
    s = strategy()
    stop, highest = s.ratchet_stop(95.0, 100.0, 104.0, 2.0)
    assert stop >= 95.0
    assert highest == 104.0
    next_stop, _ = s.ratchet_stop(stop, highest, 101.0, 3.0)
    assert next_stop >= stop
