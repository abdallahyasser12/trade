"""Deterministic technical indicators built on pandas/numpy.

Every function is causal: the value at bar N uses information from bars
<= N only.  No future data ever leaks into an indicator value, which is
what makes the backtest trustworthy.

Conventions
-----------
* Inputs are pandas Series aligned on the bar index.
* Warmup values (where the indicator is not yet fully initialized) are NaN.
* RSI and ATR use Wilder smoothing, seeded with the simple mean of the first
  full period and then updated recursively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average.  NaN until ``period`` observations exist."""
    if period < 1:
        raise ValueError("SMA period must be >= 1")
    return series.rolling(window=period, min_periods=period).mean()


def _wilder_average(series: pd.Series, period: int) -> pd.Series:
    """Canonical Wilder average with an SMA seed and recursive updates."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.astype(float).to_numpy()
    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) < period:
        return result
    seed_positions = valid_positions[:period]
    seed_pos = int(seed_positions[-1])
    previous = float(np.mean(values[seed_positions]))
    result.iloc[seed_pos] = previous
    for pos in range(seed_pos + 1, len(values)):
        value = values[pos]
        if np.isnan(value):
            previous = np.nan
        elif np.isnan(previous):
            preceding = values[max(0, pos - period + 1) : pos + 1]
            previous = float(np.mean(preceding)) if np.isfinite(preceding).all() else np.nan
        else:
            previous = (previous * (period - 1) + value) / period
        result.iloc[pos] = previous
    return result


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing.

    RSI = 100 - 100 / (1 + RS) where RS = avg_gain / avg_loss, and the
    averages are Wilder-smoothed over ``period`` bars.  The first
    ``period`` bars are NaN (insufficient history).  A window with zero
    average loss yields RSI 100; zero average gain yields RSI 0.
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_average(gain, period)
    avg_loss = _wilder_average(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # Where avg_loss == 0 the ratio is inf -> RSI 100; both zero -> flat -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max(high-low, |high-prev_close|, |low-prev_close|).

    The first bar has no previous close, so its TR is NaN (not high-low)
    to keep the ATR warmup honest.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr[prev_close.isna()] = np.nan
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing over ``period`` bars."""
    if period < 1:
        raise ValueError("ATR period must be >= 1")
    tr = true_range(high, low, close)
    return _wilder_average(tr, period)


def rolling_volume_average(volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling mean of volume.  NaN until ``period`` observations exist."""
    if period < 1:
        raise ValueError("Volume average period must be >= 1")
    return volume.rolling(window=period, min_periods=period).mean()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current bar volume divided by its rolling average (incl. current bar)."""
    avg = rolling_volume_average(volume, period)
    ratio = volume / avg
    return ratio.where(avg > 0)


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    """Percentile rank (0-100) of the current ATR within its own recent past.

    For bar N the window is the last ``lookback`` ATR values ending at and
    including bar N — strictly causal.  The value is the percentage of
    window values that are <= the current value, so an unusually volatile
    bar scores near 100.  NaN until the window is full.
    """
    if lookback < 2:
        raise ValueError("ATR percentile lookback must be >= 2")

    def _rank(window: np.ndarray) -> float:
        current = window[-1]
        if np.isnan(current):
            return np.nan
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= current).sum()) / float(len(valid)) * 100.0

    return atr_series.rolling(window=lookback, min_periods=lookback).apply(_rank, raw=True)


def compute_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Return a copy of ``df`` with all strategy indicator columns added.

    Input must contain columns: open, high, low, close, volume.
    Added columns: short_ma, long_ma, rsi, atr, avg_volume, volume_ratio,
    atr_percentile (NaN when the volatility filter is disabled or warming up).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price data is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["short_ma"] = sma(out["close"], config.short_ma_period)
    out["long_ma"] = sma(out["close"], config.long_ma_period)
    out["rsi"] = rsi(out["close"], config.rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], config.atr_period)
    out["avg_volume"] = rolling_volume_average(out["volume"], config.volume_period)
    out["volume_ratio"] = volume_ratio(out["volume"], config.volume_period)
    if config.enable_volatility_filter:
        out["atr_percentile"] = atr_percentile(out["atr"], config.atr_percentile_lookback)
    else:
        out["atr_percentile"] = np.nan
    return out
