"""Multi-timeframe causal feature engine.

Builds the standard quant-desk feature frame on top of ``indicators.py``.
Every column obeys the same invariant as the indicator module: the value
at bar N uses information from bars <= N only, so appending future bars
can never change an already-computed value.  The causality test suite
enforces this property mechanically for every feature column.

Two additional causal mechanics live here:

* Session anchoring — VWAP, opening range, and gap features restart at
  each session boundary.  US equities anchor to the America/New_York
  trading date, 24/7 crypto anchors to the UTC date.
* Higher-timeframe context — ``add_context_features`` joins a slower
  frame onto a faster one using AVAILABILITY time: a context bar stamped
  T with duration D becomes visible only to base bars whose own decision
  moment (bar close) is at or after T + D.  This mirrors the equivalent
  logic in ``strategy.py`` and prevents a daily close from leaking into
  an intraday decision made before that close existed.

The engine is deliberately price-only.  Optional sources (fundamentals,
news, macro, sentiment, derivatives) enter through the provider layer and
the feature store, never by fabricating columns here.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from config import timeframe_to_timedelta
from indicators import atr, atr_percentile, efficiency_ratio, rsi, sma, volume_ratio

# Bump on ANY change to feature definitions: persisted snapshots and trained
# models are keyed by this version, and silently changing a definition under
# an existing key corrupts every downstream measurement.
FEATURE_VERSION = "1.0.0"

# Anchors a session calendar for session-scoped features.
SESSION_ANCHOR_UTC = "UTC"
SESSION_ANCHOR_US_EQUITY = "America/New_York"


def _validate_bars(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
        raise ValueError("Feature engine requires open/high/low/close/volume columns.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Feature engine requires a timezone-aware DatetimeIndex.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Bars must be chronological and de-duplicated.")


def session_ids(index: pd.DatetimeIndex, session_anchor: str) -> pd.Series:
    """The session (trading date) each bar belongs to, in the anchor zone."""
    local = index.tz_convert(session_anchor)
    return pd.Series(local.date, index=index)


def rolling_zscore(series: pd.Series, period: int) -> pd.Series:
    """Causal z-score of the latest value against its own trailing window."""
    if period < 2:
        raise ValueError("Z-score period must be >= 2")
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    z = (series - mean) / std
    return z.where(std > 0)


def session_vwap(bars: pd.DataFrame, session_anchor: str) -> pd.Series:
    """Volume-weighted average price anchored to each session's open.

    Uses the typical price (H+L+C)/3 per completed bar.  Bars with zero
    cumulative session volume yield NaN rather than a fabricated price.
    """
    _validate_bars(bars)
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    weighted = typical * bars["volume"]
    session = session_ids(bars.index, session_anchor)
    cum_weighted = weighted.groupby(session.values).cumsum()
    cum_volume = bars["volume"].groupby(session.values).cumsum()
    vwap = cum_weighted / cum_volume
    return vwap.where(cum_volume > 0)


def opening_range(
    bars: pd.DataFrame, session_anchor: str, range_bars: int
) -> pd.DataFrame:
    """Per-bar opening-range levels, causal within each session.

    ``or_high``/``or_low`` are the extremes of the first ``range_bars``
    bars of the session and are exposed only AFTER the range completes
    (NaN while the range is still forming, so no bar can trade against a
    range that includes itself or future bars).  ``or_complete`` marks
    bars strictly after the range window.
    """
    if range_bars < 1:
        raise ValueError("Opening range must span at least one bar.")
    _validate_bars(bars)
    session = session_ids(bars.index, session_anchor)
    position = session.groupby(session.values).cumcount()

    in_range = position < range_bars
    # groupby cummax/cummin leave NaN at masked rows instead of carrying the
    # running extreme forward, so the completed range must be forward-filled
    # within each session (never across a session boundary).
    range_high = bars["high"].where(in_range).groupby(session.values).cummax()
    range_high = range_high.groupby(session.values).ffill()
    range_low = bars["low"].where(in_range).groupby(session.values).cummin()
    range_low = range_low.groupby(session.values).ffill()
    complete = position >= range_bars

    out = pd.DataFrame(index=bars.index)
    # A bar during the range window must not see the (partial) range.
    out["or_high"] = range_high.where(complete)
    out["or_low"] = range_low.where(complete)
    out["or_complete"] = complete
    out["session_bar_index"] = position
    return out


def compute_price_features(
    bars: pd.DataFrame,
    *,
    session_anchor: str = SESSION_ANCHOR_UTC,
    opening_range_bars: int = 0,
) -> pd.DataFrame:
    """The standard causal feature frame for one symbol at one timeframe.

    Columns added (all NaN during warmup, never fabricated):

    ret_1 / ret_5 / ret_20      simple returns over 1/5/20 completed bars
    mom_accel                   5-bar return minus the previous 5-bar return
    realized_vol_20             std-dev of 1-bar returns over 20 bars
    atr_14 / atr_pct            ATR and ATR as a fraction of close
    atr_percentile_100          ATR percentile in its trailing 100 values
    rsi_14                      Wilder RSI
    efficiency_20               Kaufman efficiency ratio
    zscore_20                   close z-scored against its 20-bar window
    volume_ratio_20             volume vs its trailing 20-bar average
    dollar_volume_20            trailing average close*volume (liquidity)
    donchian_high_20 / _dist    prior 20-bar high (excl. current) & distance
    donchian_low_20 / _dist     prior 20-bar low (excl. current) & distance
    range_compression_100       percentile of 20-bar range/ATR (low = squeeze)
    gap_pct                     session open vs previous session close
    vwap / vwap_z               session VWAP and ATR-scaled distance from it
    or_high / or_low / ...      opening-range columns when requested
    """
    _validate_bars(bars)
    out = bars.copy()
    close = out["close"].astype(float)

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["mom_accel"] = out["ret_5"] - out["ret_5"].shift(5)
    out["realized_vol_20"] = out["ret_1"].rolling(20, min_periods=20).std(ddof=0)

    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["atr_pct"] = (out["atr_14"] / close).where(close > 0)
    out["atr_percentile_100"] = atr_percentile(out["atr_14"], 100)
    out["rsi_14"] = rsi(close, 14)
    out["efficiency_20"] = efficiency_ratio(close, 20)
    out["zscore_20"] = rolling_zscore(close, 20)
    out["volume_ratio_20"] = volume_ratio(out["volume"], 20)
    out["dollar_volume_20"] = (
        (close * out["volume"]).rolling(20, min_periods=20).mean()
    )

    # shift(1) keeps the current bar OUT of its own breakout reference —
    # the same no-look-ahead invariant the production crypto strategy uses.
    prior_high = out["high"].shift(1).rolling(20, min_periods=20).max()
    prior_low = out["low"].shift(1).rolling(20, min_periods=20).min()
    out["donchian_high_20"] = prior_high
    out["donchian_low_20"] = prior_low
    out["donchian_high_dist"] = ((close - prior_high) / prior_high).where(prior_high > 0)
    out["donchian_low_dist"] = ((close - prior_low) / prior_low).where(prior_low > 0)

    bar_range = (
        out["high"].rolling(20, min_periods=20).max()
        - out["low"].rolling(20, min_periods=20).min()
    )
    range_vs_atr = (bar_range / out["atr_14"]).where(out["atr_14"] > 0)
    out["range_compression_100"] = atr_percentile(range_vs_atr, 100)

    session = session_ids(out.index, session_anchor)
    session_changed = session != session.shift(1)
    prev_close = close.shift(1)
    gap = (out["open"] - prev_close) / prev_close
    out["gap_pct"] = gap.where(session_changed & (prev_close > 0))

    out["vwap"] = session_vwap(bars, session_anchor)
    out["vwap_z"] = ((close - out["vwap"]) / out["atr_14"]).where(out["atr_14"] > 0)

    if opening_range_bars > 0:
        or_frame = opening_range(bars, session_anchor, opening_range_bars)
        for column in or_frame.columns:
            out[column] = or_frame[column]
    return out


def add_context_features(
    base: pd.DataFrame,
    context: pd.DataFrame,
    *,
    base_timeframe: str,
    context_timeframe: str,
    prefix: str,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Join slower-timeframe context onto a faster frame, causally.

    A context bar stamped ``T`` with duration ``D`` describes the interval
    [T, T+D) and is knowable only at ``T+D``.  A base bar stamped ``t``
    is acted on at its own close ``t + base_duration``.  The join therefore
    attaches the newest context row satisfying ``T + D <= t + base_dur``.

    Returns a copy of ``base`` with ``{prefix}_{column}`` columns added.
    Bars with no completed context yet get NaN — never a peeked value.
    """
    for frame, name in ((base, "base"), (context, "context")):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{name} frame requires a timezone-aware DatetimeIndex.")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"{name} frame must be chronological and de-duplicated.")

    base_duration = pd.Timedelta(timeframe_to_timedelta(base_timeframe))
    context_duration = pd.Timedelta(timeframe_to_timedelta(context_timeframe))
    if context_duration < base_duration:
        raise ValueError(
            "Context timeframe must be slower than or equal to the base timeframe; "
            f"got base={base_timeframe!r} context={context_timeframe!r}."
        )

    selected = list(columns) if columns is not None else list(context.columns)
    missing = set(selected).difference(context.columns)
    if missing:
        raise ValueError(f"Context frame lacks columns: {sorted(missing)}")

    context_view = context[selected].copy()
    context_view.index = context_view.index + context_duration  # availability time
    base_decision_time = base.index + base_duration

    merged = pd.merge_asof(
        pd.DataFrame(index=base_decision_time),
        context_view,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )
    out = base.copy()
    for column in selected:
        out[f"{prefix}_{column}"] = merged[column].to_numpy()
    return out


def feature_payload(row: pd.Series, columns: list[str]) -> dict[str, Optional[float]]:
    """Extract a JSON-safe snapshot of selected feature values from one row."""
    payload: dict[str, Optional[float]] = {}
    for column in columns:
        value = row.get(column)
        try:
            converted = float(value)
        except (TypeError, ValueError):
            payload[column] = None
            continue
        payload[column] = converted if math.isfinite(converted) else None
    return payload
