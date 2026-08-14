"""Unit tests for indicators.py.

All data is deterministic and synthetic; no network access.  Assertions use
positional (iloc) indexing so warmup boundaries are stated as row indexes,
while the underlying indexes are timezone-aware UTC timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from indicators import (
    adx,
    atr,
    atr_percentile,
    compute_indicators,
    efficiency_ratio,
    rolling_volume_average,
    rsi,
    sma,
    true_range,
    volume_ratio,
)


def _utc_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")


def _series(values) -> pd.Series:
    values = list(values)
    return pd.Series(values, index=_utc_index(len(values)), dtype=float)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def test_sma_hand_computed_values():
    out = sma(_series([1, 2, 3, 4, 5, 6]), period=3)
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(out.to_numpy(), expected, equal_nan=True)


def test_sma_warmup_first_period_minus_one_values_nan():
    period = 4
    out = sma(_series(range(10)), period=period)
    assert out.iloc[: period - 1].isna().all()
    assert out.iloc[period - 1 :].notna().all()


def test_sma_period_one_is_identity():
    s = _series([3.0, 1.5, 7.25])
    out = sma(s, period=1)
    assert np.allclose(out.to_numpy(), s.to_numpy())


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def test_rsi_all_gains_is_100():
    """A strictly rising close has zero average loss, so RSI pins at 100."""
    close = _series(np.arange(1.0, 31.0))
    out = rsi(close, period=14)
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid == 100.0).all()


def test_rsi_all_losses_is_0():
    """A strictly falling close has zero average gain, so RSI pins at 0."""
    close = _series(np.arange(100.0, 70.0, -1.0))
    out = rsi(close, period=14)
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid == 0.0).all()


def test_rsi_flat_series_is_50():
    """No gains and no losses is defined as neutral RSI 50."""
    close = _series([10.0] * 20)
    out = rsi(close, period=5)
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid == 50.0).all()


def test_rsi_first_valid_value_at_row_index_period():
    """RSI needs period+1 bars: the diff burns one bar, then ``period``
    changes must accumulate, so the first non-NaN sits at row index == period."""
    period = 14
    close = _series(np.linspace(100.0, 130.0, 25))
    out = rsi(close, period=period)
    assert out.iloc[:period].isna().all()
    assert out.iloc[period:].notna().all()


def test_rsi_hand_computed_mixed_case():
    """Hand-derived Wilder RSI, period=2, close = 10, 11, 10.5, 11.5, 11.

    gains  = [-, 1.0, 0.0, 1.0, 0.0], losses = [-, 0.0, 0.5, 0.0, 0.5]
    Wilder recursion (alpha=1/2, seeded from the first change):
      avg_gain = [-, 1.0, 0.5, 0.75, 0.375]
      avg_loss = [-, 0.0, 0.25, 0.125, 0.3125]
    => RSI[2] = 100 - 100/(1+2) = 66.6667, RSI[3] = 100 - 100/7 = 85.7143,
       RSI[4] = 100 - 100/2.2 = 54.5455 — all strictly inside (0, 100).
    """
    close = _series([10.0, 11.0, 10.5, 11.5, 11.0])
    out = rsi(close, period=2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(200.0 / 3.0)
    assert out.iloc[3] == pytest.approx(600.0 / 7.0)
    assert out.iloc[4] == pytest.approx(600.0 / 11.0)
    assert ((out.dropna() > 0.0) & (out.dropna() < 100.0)).all()


def test_rsi_matches_manual_wilder_recursion():
    """Wilder smoothing uses an SMA seed, then the recursive update."""
    period = 5
    rng = np.random.default_rng(7)
    close = _series(100.0 + np.cumsum(rng.normal(0.0, 1.0, 60)))
    out = rsi(close, period=period)

    deltas = np.diff(close.to_numpy())
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()
    expected = np.full(len(close), np.nan)
    seed_row = period
    if avg_loss == 0.0:
        expected[seed_row] = 50.0 if avg_gain == 0.0 else 100.0
    else:
        expected[seed_row] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain += (gains[i] - avg_gain) / period
        avg_loss += (losses[i] - avg_loss) / period
        row = i + 1  # deltas[i] belongs to close row i+1
        if row >= period:
            if avg_loss == 0.0:
                expected[row] = 50.0 if avg_gain == 0.0 else 100.0
            else:
                expected[row] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    assert np.allclose(out.to_numpy(), expected, equal_nan=True)


def test_rsi_invalid_period_raises():
    close = _series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rsi(close, period=0)
    with pytest.raises(ValueError):
        rsi(close, period=-3)


def test_sma_invalid_period_raises():
    with pytest.raises(ValueError):
        sma(_series([1.0, 2.0]), period=0)


# ---------------------------------------------------------------------------
# True range
# ---------------------------------------------------------------------------


def test_true_range_first_bar_nan_and_crafted_rows():
    """Each crafted bar exercises a different arm of the TR max():

    bar1: gap up   -> |high - prev_close| = 1.0 dominates
    bar2: gap down -> |low - prev_close| = 2.2 dominates
    bar3: plain    -> high - low = 0.6 dominates
    """
    high = _series([10.0, 10.5, 10.0, 9.0])
    low = _series([9.0, 9.8, 8.0, 8.4])
    close = _series([9.5, 10.2, 8.5, 8.8])
    tr = true_range(high, low, close)
    assert np.isnan(tr.iloc[0])  # no previous close on the first bar
    assert tr.iloc[1] == pytest.approx(1.0)
    assert tr.iloc[2] == pytest.approx(2.2)
    assert tr.iloc[3] == pytest.approx(0.6)


def test_true_range_first_bar_nan_even_with_wide_range():
    """The first bar's TR must be NaN, not high-low, to keep warmup honest."""
    high = _series([120.0, 101.0])
    low = _series([80.0, 100.0])
    close = _series([100.0, 100.5])
    tr = true_range(high, low, close)
    assert np.isnan(tr.iloc[0])
    assert tr.iloc[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def _constant_tr_frame(n: int, spread: float = 0.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    """OHLC with constant close, so every TR (after the first NaN) is 2*spread."""
    close = _series([100.0] * n)
    high = close + spread
    low = close - spread
    return high, low, close


def test_atr_constant_true_range_equals_tr0():
    """With TR constant from the very first measurable bar, the Wilder
    recursion is seeded at TR0 and stays exactly there."""
    period = 3
    high, low, close = _constant_tr_frame(20)
    out = atr(high, low, close, period=period)
    assert out.iloc[:period].isna().all()
    assert np.allclose(out.iloc[period:].to_numpy(), 1.0)


def test_atr_converges_to_constant_tr_after_shock():
    """After an initial volatility shock, constant TR0=1.0 bars pull the
    Wilder average asymptotically down to TR0."""
    n = 60
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    highs[1], lows[1] = 110.0, 90.0  # TR at bar1 = 20.0, then 1.0 forever
    high, low, close = _series(highs), _series(lows), _series(closes)
    out = atr(high, low, close, period=3)
    valid = out.dropna()
    # Strictly decreasing toward TR0, always from above, and very close by the end.
    assert (valid.diff().dropna() < 0).all()
    assert (valid > 1.0).all()
    assert valid.iloc[-1] == pytest.approx(1.0, abs=1e-4)


def test_atr_warmup_nan_behavior():
    """TR is NaN at bar 0, so ``period`` TR observations exist only at row
    index ``period`` — everything before that is NaN."""
    period = 5
    high, low, close = _constant_tr_frame(12)
    out = atr(high, low, close, period=period)
    assert out.iloc[:period].isna().all()
    assert out.iloc[period:].notna().all()


def test_atr_invalid_period_raises():
    high, low, close = _constant_tr_frame(5)
    with pytest.raises(ValueError):
        atr(high, low, close, period=0)


# ---------------------------------------------------------------------------
# Volume average / ratio
# ---------------------------------------------------------------------------


def test_rolling_volume_average_hand_computed():
    out = rolling_volume_average(_series([100, 200, 300, 400]), period=2)
    expected = [np.nan, 150.0, 250.0, 350.0]
    assert np.allclose(out.to_numpy(), expected, equal_nan=True)


def test_rolling_volume_average_invalid_period_raises():
    with pytest.raises(ValueError):
        rolling_volume_average(_series([1, 2, 3]), period=0)


def test_volume_ratio_hand_computed():
    """ratio = volume / rolling_mean(volume) including the current bar."""
    out = volume_ratio(_series([100, 300, 200, 600]), period=2)
    assert np.isnan(out.iloc[0])  # warming up
    assert out.iloc[1] == pytest.approx(300.0 / 200.0)
    assert out.iloc[2] == pytest.approx(200.0 / 250.0)
    assert out.iloc[3] == pytest.approx(600.0 / 400.0)


def test_volume_ratio_nan_when_average_is_zero_or_warming_up():
    """Zero-volume windows must yield NaN, never inf or a divide error."""
    out = volume_ratio(_series([0.0, 0.0, 0.0, 100.0]), period=2)
    assert np.isnan(out.iloc[0])  # warmup
    assert np.isnan(out.iloc[1])  # avg == 0
    assert np.isnan(out.iloc[2])  # avg == 0
    assert out.iloc[3] == pytest.approx(100.0 / 50.0)


def test_volume_ratio_invalid_period_raises():
    with pytest.raises(ValueError):
        volume_ratio(_series([1, 2, 3]), period=0)


# ---------------------------------------------------------------------------
# ATR percentile
# ---------------------------------------------------------------------------


def test_atr_percentile_nan_until_window_full():
    lookback = 5
    s = _series(np.linspace(1.0, 2.0, 12))
    out = atr_percentile(s, lookback=lookback)
    assert out.iloc[: lookback - 1].isna().all()
    assert out.iloc[lookback - 1 :].notna().all()


def test_atr_percentile_near_100_when_current_is_window_max():
    """When the current ATR is >= every value in its window, its percentile
    rank is exactly 100."""
    s = _series([1.0, 2.0, 3.0, 4.0, 10.0])
    out = atr_percentile(s, lookback=5)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_atr_percentile_minimum_scores_lowest_rank():
    """A current value strictly below the rest of the window ranks 1/lookback."""
    s = _series([5.0, 4.0, 3.0, 2.0, 1.0])
    out = atr_percentile(s, lookback=5)
    assert out.iloc[-1] == pytest.approx(20.0)


def test_atr_percentile_strictly_causal_under_appended_rows():
    """Appending future rows must not change any previously computed value —
    the whole point of a rolling, backward-looking percentile."""
    rng = np.random.default_rng(42)
    full_values = rng.uniform(0.5, 3.0, 45)
    lookback = 7
    n_short = 30

    idx_full = _utc_index(len(full_values))
    s_full = pd.Series(full_values, index=idx_full)
    s_short = s_full.iloc[:n_short]

    p_short = atr_percentile(s_short, lookback=lookback)
    p_full = atr_percentile(s_full, lookback=lookback)
    pd.testing.assert_series_equal(p_short, p_full.iloc[:n_short])


def test_atr_percentile_invalid_lookback_raises():
    s = _series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        atr_percentile(s, lookback=1)
    with pytest.raises(ValueError):
        atr_percentile(s, lookback=0)


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

INDICATOR_COLUMNS = {
    "adx",
    "short_ma",
    "long_ma",
    "rsi",
    "atr",
    "avg_volume",
    "volume_ratio",
    "atr_percentile",
    "efficiency_ratio",
}


def _small_config(**overrides) -> StrategyConfig:
    params = dict(
        symbol="SPY",
        timeframe="1Day",
        short_ma_period=3,
        long_ma_period=5,
        rsi_period=3,
        atr_period=3,
        volume_period=3,
        enable_volatility_filter=True,
        atr_percentile_lookback=10,
    )
    params.update(overrides)
    return StrategyConfig(**params)


def _ohlcv_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = (high + low) / 2.0
    volume = rng.integers(1_000, 50_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=_utc_index(n),
    )


def test_compute_indicators_adds_exactly_expected_columns():
    df = _ohlcv_frame()
    out = compute_indicators(df, _small_config())
    assert set(out.columns) == set(df.columns) | INDICATOR_COLUMNS
    # Original frame is returned as a copy, not mutated in place.
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    # Each indicator column has at least one real value on this long frame.
    for col in INDICATOR_COLUMNS:
        assert out[col].notna().any(), f"{col} is unexpectedly all-NaN"


def test_compute_indicators_columns_match_direct_calls():
    """The composed frame must agree with calling each indicator directly."""
    df = _ohlcv_frame()
    cfg = _small_config()
    out = compute_indicators(df, cfg)
    atr_direct = atr(df["high"], df["low"], df["close"], cfg.atr_period)
    checks = {
        "short_ma": sma(df["close"], cfg.short_ma_period),
        "long_ma": sma(df["close"], cfg.long_ma_period),
        "rsi": rsi(df["close"], cfg.rsi_period),
        "atr": atr_direct,
        "adx": adx(df["high"], df["low"], df["close"], cfg.adx_period),
        "efficiency_ratio": efficiency_ratio(df["close"], cfg.efficiency_period),
        "avg_volume": rolling_volume_average(df["volume"], cfg.volume_period),
        "volume_ratio": volume_ratio(df["volume"], cfg.volume_period),
        "atr_percentile": atr_percentile(atr_direct, cfg.atr_percentile_lookback),
    }
    for col, expected in checks.items():
        assert np.allclose(
            out[col].to_numpy(), expected.to_numpy(), equal_nan=True
        ), f"{col} differs from direct computation"


@pytest.mark.parametrize("missing", ["open", "high", "low", "close", "volume"])
def test_compute_indicators_missing_column_raises(missing):
    df = _ohlcv_frame(20).drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        compute_indicators(df, _small_config())


def test_compute_indicators_atr_percentile_all_nan_when_filter_disabled():
    df = _ohlcv_frame()
    out = compute_indicators(df, _small_config(enable_volatility_filter=False))
    assert "atr_percentile" in out.columns
    assert out["atr_percentile"].isna().all()


def test_compute_indicators_does_not_mutate_input():
    df = _ohlcv_frame(30)
    original = df.copy(deep=True)
    compute_indicators(df, _small_config())
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# Trend-quality indicators
# ---------------------------------------------------------------------------


def test_efficiency_ratio_hand_computed_clean_and_noisy_paths():
    clean = efficiency_ratio(_series([1, 2, 3, 4, 5]), period=4)
    noisy = efficiency_ratio(_series([1, 2, 1, 2, 1]), period=4)
    assert clean.iloc[-1] == pytest.approx(1.0)
    assert noisy.iloc[-1] == pytest.approx(0.0)
    assert clean.iloc[:4].isna().all()


def test_efficiency_ratio_flat_initialized_window_is_zero():
    out = efficiency_ratio(_series([10.0] * 8), period=4)
    assert out.iloc[:4].isna().all()
    assert (out.iloc[4:] == 0.0).all()


def test_adx_strict_uptrend_converges_to_100_after_two_stage_warmup():
    close = _series(np.arange(100.0, 140.0))
    high = close + 1.0
    low = close - 1.0
    out = adx(high, low, close, period=5)
    assert out.first_valid_index() == close.index[9]
    assert np.allclose(out.dropna().to_numpy(), 100.0)


def test_adx_empty_input_is_empty_and_invalid_period_rejected():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    assert adx(empty, empty, empty, period=5).empty
    with pytest.raises(ValueError):
        adx(_series([1, 2, 3]), _series([1, 2, 3]), _series([1, 2, 3]), period=1)


def test_adx_flat_initialized_market_is_zero_not_nan():
    flat = _series([100.0] * 10)

    out = adx(flat, flat, flat, period=3)

    assert out.iloc[:5].isna().all()
    assert out.iloc[5:].eq(0.0).all()
