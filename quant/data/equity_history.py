"""Historical US equity bars for research: chunked, cached, accounted for.

The live path (``data_feed.AlpacaDataFeed.get_recent_bars``) fetches a
short trailing window every poll. Research needs years, across many
symbols, and needs to know afterwards exactly what it got. This module
provides that without modifying the live path.

What it adds over a raw ``get_bars`` call:

* **Deterministic chunking.** Ranges are split on calendar boundaries
  (month for intraday, year for daily) so that two different request
  windows covering the same month reuse the same cache entry instead of
  each fetching their own overlapping slab.
* **Caching.** Every chunk is stored under the fingerprint of its
  request. A parameter sweep pays the provider once.
* **Session awareness.** Extended-hours prints are dropped by default,
  and completeness is measured against the real NYSE calendar rather
  than against elapsed wall-clock time.
* **Provenance and quality.** Every fetch returns what was requested,
  what arrived, how complete it is, and which sessions are missing.
  A symbol with no data is MISSING, not an empty frame that silently
  reads as "no signals".

It deliberately does NOT interpolate, forward-fill, or synthesize bars.
A gap stays a gap and is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Iterable, Optional, Protocol, Sequence

import pandas as pd

from config import timeframe_to_timedelta
from quant.data.cache import BarCache
from quant.data.calendar import (
    MARKET_TZ,
    filter_regular_hours,
    is_trading_day,
    session_dates,
    trading_days,
)
from quant.data.provenance import (
    BarProvenance,
    Completeness,
    DataQuality,
    assess_equity_completeness,
    classify_quality,
)

logger = logging.getLogger(__name__)

# Timeframes the equity research path supports. 4Hour is excluded: the
# 6.5-hour session does not divide into 4-hour buckets, so a "4Hour"
# equity bar is two ragged buckets a day whose meaning changes with
# daylight saving. Research asks for 1Hour or 1Day instead.
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1Min", "5Min", "15Min", "1Hour", "1Day")

# Chunk granularity per timeframe. Smaller chunks for denser data keeps
# any single provider response bounded and makes cache reuse finer.
_CHUNK_MONTHS: dict[str, int] = {
    "1Min": 1,
    "5Min": 3,
    "15Min": 6,
    "1Hour": 12,
}


class BarSource(Protocol):
    """The subset of ``AlpacaDataFeed`` this module needs.

    Declared structurally so tests can supply a recorded-fixture source
    with no network and no credentials.
    """

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: str = "split",
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class EquityHistory:
    """A fetched series plus everything needed to trust and reproduce it."""

    symbol: str
    timeframe: str
    bars: pd.DataFrame
    provenance: BarProvenance

    @property
    def quality(self) -> DataQuality:
        return self.provenance.quality

    @property
    def is_usable(self) -> bool:
        return not self.bars.empty and self.quality is not DataQuality.MISSING

    def __len__(self) -> int:
        return len(self.bars)


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _add_months(day: date, months: int) -> date:
    total = day.year * 12 + (day.month - 1) + months
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


def chunk_ranges(
    start: date, end: date, timeframe: str
) -> list[tuple[date, date]]:
    """Split ``[start, end]`` on stable calendar boundaries.

    Boundaries do not depend on the requested window, so a request for
    Feb 10 - Apr 20 and one for Mar 1 - Mar 31 share the March chunk.
    That shared key is what makes the cache actually hit during a sweep.
    """
    if end < start:
        return []
    if timeframe == "1Day":
        out: list[tuple[date, date]] = []
        year = start.year
        while year <= end.year:
            lo = max(start, date(year, 1, 1))
            hi = min(end, date(year, 12, 31))
            if lo <= hi:
                out.append((lo, hi))
            year += 1
        return out

    months = _CHUNK_MONTHS.get(timeframe)
    if months is None:
        raise ValueError(f"Unsupported equity research timeframe: {timeframe!r}")
    out = []
    cursor = _month_start(start)
    while cursor <= end:
        nxt = _add_months(cursor, months)
        lo = max(start, cursor)
        hi = min(end, nxt - timedelta(days=1))
        if lo <= hi:
            out.append((lo, hi))
        cursor = nxt
    return out


def _utc_window(start: date, end: date) -> tuple[datetime, datetime]:
    """Widen a date range to UTC instants covering both sessions fully.

    The window is widened by a day on each side because a US session
    spans two UTC dates in winter; trimming happens after the
    regular-hours filter, on real session boundaries.
    """
    lo = datetime.combine(start - timedelta(days=1), time(0, 0), tzinfo=timezone.utc)
    hi = datetime.combine(end + timedelta(days=1), time(0, 0), tzinfo=timezone.utc)
    return lo, hi


def _trim_to_sessions(
    bars: pd.DataFrame, start: date, end: date, timeframe: str
) -> pd.DataFrame:
    """Keep only rows whose session date falls inside the range."""
    if bars.empty:
        return bars
    keep = [start <= day <= end for day in session_dates(bars.index, timeframe)]
    return bars[pd.Series(keep, index=bars.index).to_numpy()]


class EquityHistoryFetcher:
    """Fetches, caches and accounts for historical equity bars."""

    def __init__(
        self,
        source: BarSource,
        *,
        cache: Optional[BarCache] = None,
        feed_name: str = "iex",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.source = source
        self.cache = cache
        self.feed_name = feed_name
        self._clock = clock
        self._api_calls = 0
        self._cache_hits = 0

    @property
    def api_calls(self) -> int:
        """Provider requests issued — the number a sweep should minimize."""
        return self._api_calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    # ------------------------------------------------------------------
    # Single symbol
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        *,
        adjustment: str = "split",
        regular_hours_only: bool = True,
        refresh: bool = False,
    ) -> EquityHistory:
        """Fetch ``symbol`` over ``[start, end]`` (market-local dates).

        Returns an ``EquityHistory`` even when the provider has nothing:
        the caller inspects ``quality``/``is_usable`` rather than having
        to distinguish an exception from an empty frame.
        """
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported equity research timeframe: {timeframe!r}. "
                f"Supported: {list(SUPPORTED_TIMEFRAMES)}"
            )
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        if adjustment not in ("raw", "split", "dividend", "all"):
            raise ValueError(f"Unsupported adjustment: {adjustment!r}")

        sessions = trading_days(start, end)
        requested_lo, requested_hi = _utc_window(start, end)
        notes: list[str] = []

        if not sessions:
            provenance = BarProvenance(
                provider="alpaca",
                feed=self.feed_name,
                market="us_equity",
                symbol=symbol,
                timeframe=timeframe,
                requested_start=requested_lo,
                requested_end=requested_hi,
                observed_start=None,
                observed_end=None,
                adjustment=adjustment,
                fetched_at=self._clock(),
                quality=DataQuality.MISSING,
                completeness=Completeness(0, 0, (), 1.0),
                regular_hours_only=regular_hours_only,
                notes=("requested range contains no US equity sessions",),
            )
            return EquityHistory(symbol, timeframe, _empty(), provenance)

        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in chunk_ranges(start, end, timeframe):
            frames.append(
                self._fetch_chunk(
                    symbol,
                    timeframe,
                    chunk_start,
                    chunk_end,
                    adjustment=adjustment,
                    regular_hours_only=regular_hours_only,
                    refresh=refresh,
                    notes=notes,
                )
            )

        bars = _concat(frames)
        bars = _trim_to_sessions(bars, start, end, timeframe)

        completeness = assess_equity_completeness(bars, timeframe, start, end)
        observed_start = bars.index[0].to_pydatetime() if not bars.empty else None
        observed_end = bars.index[-1].to_pydatetime() if not bars.empty else None
        # Staleness only means something when the caller asked for data
        # up to now.  A window that deliberately ends in the past is not
        # degraded for having done so.
        now = self._clock()
        asks_for_current = end >= now.astimezone(MARKET_TZ).date()
        quality, quality_notes = classify_quality(
            completeness,
            observed_end=observed_end,
            timeframe=timeframe,
            now=now,
            is_crypto=False,
            check_staleness=asks_for_current,
        )

        provenance = BarProvenance(
            provider="alpaca",
            feed=self.feed_name,
            market="us_equity",
            symbol=symbol,
            timeframe=timeframe,
            requested_start=requested_lo,
            requested_end=requested_hi,
            observed_start=observed_start,
            observed_end=observed_end,
            adjustment=adjustment,
            fetched_at=self._clock(),
            quality=quality,
            completeness=completeness,
            regular_hours_only=regular_hours_only,
            notes=tuple(notes) + tuple(quality_notes),
        )
        return EquityHistory(symbol, timeframe, bars, provenance)

    def _fetch_chunk(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: date,
        chunk_end: date,
        *,
        adjustment: str,
        regular_hours_only: bool,
        refresh: bool,
        notes: list[str],
    ) -> pd.DataFrame:
        lo, hi = _utc_window(chunk_start, chunk_end)
        key_provenance = BarProvenance(
            provider="alpaca",
            feed=self.feed_name,
            market="us_equity",
            symbol=symbol,
            timeframe=timeframe,
            requested_start=lo,
            requested_end=hi,
            observed_start=None,
            observed_end=None,
            adjustment=adjustment,
            fetched_at=self._clock(),
            quality=DataQuality.OK,
            completeness=Completeness(0, 0, (), 0.0),
            regular_hours_only=regular_hours_only,
        )
        fingerprint = key_provenance.fingerprint()

        if self.cache is not None and not refresh:
            hit = self.cache.get(fingerprint)
            if hit is not None:
                self._cache_hits += 1
                return hit[0]

        try:
            raw = self.source.get_bars(
                symbol, timeframe, start=lo, end=hi, adjustment=adjustment
            )
            self._api_calls += 1
        except Exception as error:
            # One bad chunk must not lose the whole history: record the
            # failure, return nothing for this window, and let the
            # completeness check downgrade the series honestly.
            logger.warning(
                "Bar fetch failed for %s %s %s..%s: %s",
                symbol, timeframe, chunk_start, chunk_end, error,
            )
            notes.append(
                f"provider error for {chunk_start.isoformat()}..{chunk_end.isoformat()}: {error}"
            )
            self._api_calls += 1
            return _empty()

        bars = raw if raw is not None else _empty()
        if not bars.empty and regular_hours_only:
            bars = filter_regular_hours(bars, timeframe)
        bars = _trim_to_sessions(bars, chunk_start, chunk_end, timeframe)

        if self.cache is not None:
            chunk_completeness = assess_equity_completeness(
                bars, timeframe, chunk_start, chunk_end
            )
            self.cache.put(
                replace(
                    key_provenance,
                    observed_start=bars.index[0].to_pydatetime() if not bars.empty else None,
                    observed_end=bars.index[-1].to_pydatetime() if not bars.empty else None,
                    completeness=chunk_completeness,
                ),
                bars,
            )
        return bars

    # ------------------------------------------------------------------
    # Many symbols
    # ------------------------------------------------------------------

    def fetch_many(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: date,
        end: date,
        *,
        adjustment: str = "split",
        regular_hours_only: bool = True,
        refresh: bool = False,
        skip_unusable: bool = True,
    ) -> dict[str, EquityHistory]:
        """Fetch a universe. One symbol's failure never aborts the rest.

        With ``skip_unusable`` the result contains only series a strategy
        could actually trade; the dropped symbols are logged with their
        quality so a research report can state how many were excluded and
        why, rather than silently researching a smaller universe.
        """
        out: dict[str, EquityHistory] = {}
        excluded: list[tuple[str, str]] = []
        for symbol in symbols:
            try:
                history = self.fetch(
                    symbol,
                    timeframe,
                    start,
                    end,
                    adjustment=adjustment,
                    regular_hours_only=regular_hours_only,
                    refresh=refresh,
                )
            except Exception as error:
                logger.warning("Skipping %s: %s", symbol, error)
                excluded.append((symbol, f"fetch raised {type(error).__name__}: {error}"))
                continue
            if skip_unusable and not history.is_usable:
                excluded.append((symbol, history.quality.value))
                continue
            out[symbol] = history
        if excluded:
            logger.info(
                "Universe fetch excluded %d/%d symbols: %s",
                len(excluded), len(symbols),
                ", ".join(f"{s} ({why})" for s, why in excluded[:10]),
            )
        return out


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")},
        index=index,
    )


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return _empty()
    out = pd.concat(usable).sort_index()
    return out[~out.index.duplicated(keep="last")]
