"""Deterministic OHLCV fixtures for crypto production-code tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig


def compact_crypto_config(
    symbols: tuple[str, ...] = ("BTC/USD",),
    **overrides,
) -> CryptoStrategyConfig:
    values = dict(
        symbols=symbols,
        benchmark_symbol="BTC/USD",
        timeframe="1Hour",
        fast_ma_period=3,
        slow_ma_period=5,
        regime_ma_period=8,
        regime_slope_period=2,
        breakout_period=3,
        momentum_period=2,
        relative_strength_period=3,
        rsi_period=2,
        min_rsi=0.0,
        max_rsi=100.0,
        atr_period=2,
        atr_percentile_lookback=20,
        max_atr_percentile=100.0,
        adx_period=2,
        min_adx=0.0,
        efficiency_period=2,
        min_efficiency_ratio=0.0,
        volume_period=2,
        min_volume_ratio=0.1,
        min_relative_strength=-1.0,
    )
    values.update(overrides)
    return CryptoStrategyConfig(**values)


def breakout_bars(
    *,
    periods: int = 35,
    breakout_at: int = 30,
    gap_at: int | None = None,
    zero_volume_at: int | None = None,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    if periods <= breakout_at or breakout_at < 27:
        raise ValueError("Fixture needs the breakout bar and 27 warmup bars.")
    close = np.asarray([100.0 + index * 0.4 for index in range(periods)], dtype=float)
    close[breakout_at - 3 : breakout_at + 1] = (111.0, 110.8, 110.9, 112.5)
    if breakout_at + 1 < periods:
        close[breakout_at + 1] = 113.0
    if gap_at is not None:
        close[gap_at] = 90.0
        for index in range(gap_at + 1, periods):
            close[index] = 90.0 + (index - gap_at)
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    open_price = close.copy()
    open_price[1:] = (close[:-1] + close[1:]) / 2.0
    if breakout_at + 1 < periods:
        open_price[breakout_at + 1] = 112.8
    if gap_at is not None:
        open_price[gap_at] = 90.0
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
