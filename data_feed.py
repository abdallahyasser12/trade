"""Market data access: historical bars, completed-bar detection,
normalization, staleness rules and bounded retry behavior.

Bar timestamp convention (matches Alpaca): a bar's timestamp is the
START of its interval.  A bar starting at T with timeframe duration D is
complete once ``now >= T + D + tolerance`` — never trust "the last bar
in the response is complete", because it frequently is not.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd

from config import timeframe_to_alpaca, timeframe_to_timedelta

logger = logging.getLogger(__name__)


class DataFeedError(Exception):
    """A market-data operation failed after retries."""


class DataSubscriptionError(DataFeedError):
    """The request exceeded the account's market-data entitlement.

    On Alpaca's free tier this usually means SIP data or very recent
    history was requested; switch to the IEX feed or widen the window.
    """


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _status_code_of(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def is_rate_limit_error(exc: Exception) -> bool:
    """True for HTTP 429 (rate limit) errors — handled distinctly."""
    return _status_code_of(exc) == 429


def is_subscription_error(exc: Exception) -> bool:
    """True when the failure looks like a data-entitlement problem."""
    code = _status_code_of(exc)
    text = str(exc).lower()
    return code == 403 and ("subscription" in text or "sip" in text) or (
        "subscription does not permit" in text
    )


def is_permanent_error(exc: Exception) -> bool:
    """Deterministic failures that must not be retried."""
    code = _status_code_of(exc)
    return code in (400, 401, 403, 404, 422)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Honor the Retry-After header on 429 responses when present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def with_retries(
    fn: Callable[[], Any],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    on_rate_limit: Optional[Callable[[float], None]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Run ``fn`` with bounded exponential backoff on transient failures.

    * 429: backs off using the API's Retry-After window when present,
      reports through ``on_rate_limit`` so the caller can surface it.
    * Permanent errors (400/401/403/404/422) are raised immediately.
    * Subscription errors are converted to DataSubscriptionError.
    * Anything else transient is retried up to ``max_attempts`` total
      attempts, then the last error is raised.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            last_exc = exc
            if is_subscription_error(exc):
                raise DataSubscriptionError(
                    "Alpaca rejected the market-data request due to subscription limits. "
                    "The free tier serves IEX data only and restricts very recent history; "
                    "use the 'iex' feed and avoid requesting the most recent minutes with SIP."
                ) from exc
            if is_rate_limit_error(exc):
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = base_delay * (2 ** (attempt - 1))
                if on_rate_limit is not None:
                    on_rate_limit(delay)
                logger.warning("Alpaca rate limit hit (429); backing off %.1fs", delay)
                if attempt < max_attempts:
                    sleep_fn(delay)
                    continue
                raise DataFeedError(
                    "Alpaca rate limit (429) persisted after retries; consider a longer "
                    "polling interval."
                ) from exc
            if is_permanent_error(exc):
                raise
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient data error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                sleep_fn(delay)
    raise DataFeedError(f"Data request failed after {max_attempts} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Bar completion / staleness / normalization (pure functions — unit tested)
# ---------------------------------------------------------------------------


def bar_is_complete(
    bar_start: datetime,
    timeframe: str,
    now: datetime,
    tolerance_seconds: float = 5.0,
) -> bool:
    """Whether the bar starting at ``bar_start`` has fully closed.

    ``tolerance_seconds`` absorbs publication delay: the bar only counts
    as complete once its end time plus the tolerance has passed.
    """
    if bar_start.tzinfo is None:
        raise ValueError("bar_start must be timezone-aware")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    end = bar_start + timeframe_to_timedelta(timeframe)
    return now >= end + timedelta(seconds=tolerance_seconds)


def drop_incomplete_bars(
    df: pd.DataFrame,
    timeframe: str,
    now: datetime,
    tolerance_seconds: float = 5.0,
) -> pd.DataFrame:
    """Remove any trailing bars that are still forming."""
    if df.empty:
        return df
    duration = timeframe_to_timedelta(timeframe)
    cutoff = pd.Timestamp(now) - duration - pd.Timedelta(seconds=tolerance_seconds)
    return df[df.index <= cutoff]


def max_staleness(timeframe: str, staleness_bars: int = 3) -> timedelta:
    """How old the newest completed bar may be before data is 'stale'.

    Intraday: ``staleness_bars`` bar-durations (plus a two-minute grace
    for publication).  Daily bars complete at the close, so during the
    next session the newest daily bar is legitimately up to several days
    old (weekends/holidays) — allow 5 calendar days.
    """
    if timeframe == "1Day":
        return timedelta(days=5)
    return timeframe_to_timedelta(timeframe) * staleness_bars + timedelta(minutes=2)


def is_data_stale(
    last_bar_start: datetime,
    timeframe: str,
    now: datetime,
    staleness_bars: int = 3,
) -> bool:
    """True when the newest completed bar is too old to trade from."""
    bar_end = last_bar_start + timeframe_to_timedelta(timeframe)
    return (now - bar_end) > max_staleness(timeframe, staleness_bars)


def normalize_bars_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize an alpaca-py bars frame to the internal shape.

    Output: DatetimeIndex named 'timestamp' (UTC, tz-aware, ascending,
    de-duplicated) with float columns open/high/low/close/volume.  Rows
    with non-positive prices or missing values are dropped — the bot
    never trades from malformed data.
    """
    if df is None or len(df) == 0:
        return _empty_bars_frame()

    out = df
    # alpaca-py returns a (symbol, timestamp) MultiIndex for BarSet.df.
    if isinstance(out.index, pd.MultiIndex):
        if "symbol" in (out.index.names or []):
            try:
                out = out.xs(symbol, level="symbol")
            except KeyError:
                return _empty_bars_frame()
        else:
            out = out.droplevel(0)

    out = out.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    if len(keep) < 5:
        raise DataFeedError(
            f"Bar data for {symbol} is missing OHLCV columns; got {list(out.columns)}"
        )
    out = out[keep].astype(float)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    valid = (
        out[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & out["volume"].ge(0)
        & out.notna().all(axis=1)
    )
    dropped = int((~valid).sum())
    if dropped:
        logger.warning("Dropped %d malformed bar(s) for %s", dropped, symbol)
    return out[valid]


def _empty_bars_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Alpaca data feed
# ---------------------------------------------------------------------------


class AlpacaDataFeed:
    """Historical/recent OHLCV bars from Alpaca market data (alpaca-py)."""

    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        from alpaca.data.historical import StockHistoricalDataClient

        if feed not in ("iex", "sip"):
            raise ValueError(f"Unsupported data feed: {feed!r} (use 'iex' or 'sip')")
        self.feed = feed
        self._client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self._rate_limit_events = 0

    @property
    def rate_limit_events(self) -> int:
        """How many 429 responses this feed has absorbed (for the UI)."""
        return self._rate_limit_events

    def _note_rate_limit(self, _delay: float) -> None:
        self._rate_limit_events += 1

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: str = "split",
    ) -> pd.DataFrame:
        """Fetch historical bars, normalized and split-adjusted by default.

        ``adjustment`` is explicit because corporate actions (splits)
        otherwise create fake crossovers in backtests.
        """
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_to_alpaca(timeframe),
            start=start,
            end=end,
            limit=limit,
            adjustment=Adjustment(adjustment),
            feed=DataFeed(self.feed),
        )

        def _fetch() -> pd.DataFrame:
            bars = self._client.get_stock_bars(request)
            return bars.df

        raw = with_retries(_fetch, on_rate_limit=self._note_rate_limit)
        return normalize_bars_df(raw, symbol)

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
        adjustment: str = "split",
    ) -> pd.DataFrame:
        """Fetch the most recent COMPLETED bars, newest last.

        Fetches a window generously sized for weekends/holidays and
        off-market hours, drops any still-forming trailing bar, and
        returns at most the trailing ``lookback_bars`` rows.
        """
        now = now or datetime.now(timezone.utc)
        duration = timeframe_to_timedelta(timeframe)
        if timeframe == "1Day":
            window = timedelta(days=int(lookback_bars * 1.7) + 10)
        else:
            # Intraday bars only print during ~6.5 market hours per day;
            # scale the calendar window accordingly and pad for weekends.
            bars_per_day = max(1, int(timedelta(hours=6.5) / duration))
            days_needed = lookback_bars / bars_per_day
            window = timedelta(days=days_needed * 1.6 + 5)
        start = now - window

        df = self.get_bars(symbol, timeframe, start=start, adjustment=adjustment)
        df = drop_incomplete_bars(df, timeframe, now, tolerance_seconds)
        if len(df) > lookback_bars:
            df = df.iloc[-lookback_bars:]
        return df
