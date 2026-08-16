"""Causality and session-mechanics tests for the quant feature engine."""

import numpy as np
import pandas as pd
import pytest

from quant.features import (
    SESSION_ANCHOR_US_EQUITY,
    SESSION_ANCHOR_UTC,
    add_context_features,
    compute_price_features,
    feature_payload,
    opening_range,
    rolling_zscore,
    session_vwap,
)


def make_bars(periods=200, start="2024-01-02 09:30", freq="5min", tz="UTC", seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq, tz=tz)
    close = 100.0 + np.cumsum(rng.normal(0.02, 0.5, periods))
    close = np.maximum(close, 5.0)
    high = close + rng.uniform(0.1, 0.6, periods)
    low = close - rng.uniform(0.1, 0.6, periods)
    open_ = close + rng.normal(0.0, 0.2, periods)
    volume = rng.uniform(1_000, 5_000, periods)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def make_daily_bars(periods=300, seed=3):
    return make_bars(periods=periods, start="2023-01-02", freq="1D", seed=seed)


def test_all_features_are_causal_under_future_appends():
    bars = make_bars(periods=260)
    full = compute_price_features(bars, session_anchor=SESSION_ANCHOR_UTC, opening_range_bars=4)
    truncated = compute_price_features(
        bars.iloc[:200], session_anchor=SESSION_ANCHOR_UTC, opening_range_bars=4
    )
    check_ts = truncated.index[-1]
    for column in truncated.columns:
        a = truncated.loc[check_ts, column]
        b = full.loc[check_ts, column]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), f"feature {column} changed when future bars were appended"


def test_vwap_resets_at_session_boundary():
    # Two US-equity sessions: VWAP on the first bar of day two must depend
    # only on that day's single bar (typical price of that bar).
    day1 = make_bars(periods=78, start="2024-01-02 09:30", freq="5min")
    day2 = make_bars(periods=78, start="2024-01-03 09:30", freq="5min", seed=11)
    bars = pd.concat([day1, day2])
    bars.index = bars.index.tz_convert("UTC")
    vwap = session_vwap(bars, SESSION_ANCHOR_US_EQUITY)
    first_day2 = day2.index[0]
    row = bars.loc[first_day2]
    expected = (row["high"] + row["low"] + row["close"]) / 3.0
    assert vwap.loc[first_day2] == pytest.approx(expected)


def test_opening_range_hidden_while_forming_and_stable_after():
    day = make_bars(periods=30, start="2024-01-02 09:30", freq="5min")
    frame = opening_range(day, SESSION_ANCHOR_US_EQUITY, range_bars=4)
    # During the forming window the range must be invisible.
    assert frame["or_high"].iloc[:4].isna().all()
    assert not bool(frame["or_complete"].iloc[3])
    # After completion it equals the extremes of the first four bars, everywhere.
    expected_high = day["high"].iloc[:4].max()
    expected_low = day["low"].iloc[:4].min()
    assert np.allclose(frame["or_high"].iloc[4:], expected_high)
    assert np.allclose(frame["or_low"].iloc[4:], expected_low)
    assert frame["or_complete"].iloc[4:].all()


def test_gap_pct_only_on_session_change():
    day1 = make_bars(periods=10, start="2024-01-02 09:30", freq="5min")
    day2 = make_bars(periods=10, start="2024-01-03 09:30", freq="5min", seed=5)
    bars = pd.concat([day1, day2])
    features = compute_price_features(bars, session_anchor=SESSION_ANCHOR_US_EQUITY)
    gaps = features["gap_pct"]
    # Within a session: no gap value; at the boundary: exactly one.
    assert gaps.iloc[1:10].isna().all()
    boundary = day2.index[0]
    prev_close = day1["close"].iloc[-1]
    expected = (day2["open"].iloc[0] - prev_close) / prev_close
    assert gaps.loc[boundary] == pytest.approx(expected)


def test_context_join_respects_availability_time():
    # Daily context joined onto hourly bars: a daily bar stamped Tuesday
    # 00:00 UTC completes Wednesday 00:00 and must be invisible to every
    # hourly decision made during Tuesday.
    hourly = make_bars(periods=72, start="2024-01-02 00:00", freq="1h")
    daily = make_daily_bars(periods=10, seed=9)
    daily = daily.loc[daily.index <= hourly.index[-1]]
    daily = daily.assign(marker=np.arange(float(len(daily))))
    joined = add_context_features(
        hourly,
        daily,
        base_timeframe="1Hour",
        context_timeframe="1Day",
        prefix="d1",
        columns=["marker"],
    )
    for ts in hourly.index:
        decision_time = ts + pd.Timedelta(hours=1)
        visible = daily[(daily.index + pd.Timedelta(days=1)) <= decision_time]
        expected = visible["marker"].iloc[-1] if len(visible) else np.nan
        got = joined.loc[ts, "d1_marker"]
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == expected


def test_context_join_rejects_faster_context():
    hourly = make_bars(periods=40, start="2024-01-02 00:00", freq="1h")
    minute = make_bars(periods=40, start="2024-01-02 00:00", freq="1min")
    with pytest.raises(ValueError, match="slower"):
        add_context_features(
            hourly, minute, base_timeframe="1Hour", context_timeframe="1Min", prefix="m1"
        )


def test_rolling_zscore_warmup_and_value():
    series = pd.Series([1.0] * 19 + [2.0], dtype=float)
    z = rolling_zscore(series, 20)
    assert z.iloc[:19].isna().all()
    window = series.iloc[-20:]
    expected = (2.0 - window.mean()) / window.std(ddof=0)
    assert z.iloc[-1] == pytest.approx(expected)


def test_rolling_zscore_zero_std_is_nan():
    series = pd.Series([3.0] * 25, dtype=float)
    z = rolling_zscore(series, 20)
    assert z.iloc[-1] != z.iloc[-1]  # NaN


def test_feature_payload_sanitizes_non_finite():
    row = pd.Series({"a": 1.5, "b": np.nan, "c": np.inf, "d": "text"})
    payload = feature_payload(row, ["a", "b", "c", "d", "missing"])
    assert payload == {"a": 1.5, "b": None, "c": None, "d": None, "missing": None}


def test_naive_index_rejected():
    bars = make_bars()
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_price_features(bars)
