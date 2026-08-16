"""US equity market calendar computed from NYSE/Nasdaq holiday rules.

Crypto trades 24/7; equities do not.  Every equity research question —
"how many bars should this day have?", "is this gap a holiday or missing
data?", "where does the opening range start?" — needs a session
calendar.  Getting it wrong silently corrupts research: a half-day
treated as a full day looks like missing data, and a holiday treated as
a trading day makes a complete history look incomplete.

This module derives the calendar from the published NYSE rules rather
than shipping a static table, so it stays correct for future years and
adds no dependency.  Rules implemented (NYSE Rule 7.2):

* New Year's Day, Martin Luther King Jr. Day, Washington's Birthday,
  Good Friday, Memorial Day, Juneteenth (from 2022), Independence Day,
  Labor Day, Thanksgiving Day, Christmas Day.
* Saturday holidays observe the preceding Friday; Sunday holidays
  observe the following Monday — except that a Saturday holiday is NOT
  observed at all when the preceding Friday falls in the prior year
  (New Year's Day), which matches NYSE practice.
* Early closes (13:00 ET): the day after Thanksgiving, and Christmas Eve
  and July 3rd when they fall on a weekday that is itself a session.

Ad-hoc closures (national days of mourning, hurricanes, 9/11) are NOT
rule-derivable.  The known ones in Alpaca's data era are listed
explicitly in ``_AD_HOC_CLOSURES``; anything outside that list is
treated as a normal session, and callers that care report the day as
INCOMPLETE rather than silently assuming data loss.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")

REGULAR_OPEN_ET = time(9, 30)
REGULAR_CLOSE_ET = time(16, 0)
EARLY_CLOSE_ET = time(13, 0)

# Unscheduled full-day closures inside the era Alpaca serves data for.
# Rule-derivable holidays are NOT listed here.
_AD_HOC_CLOSURES: frozenset[date] = frozenset(
    {
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),  # Hurricane Sandy
        date(2018, 12, 5),  # National day of mourning, George H. W. Bush
        date(2025, 1, 9),  # National day of mourning, Jimmy Carter
    }
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` means last."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last_day = (date(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _observed(holiday: date) -> Optional[date]:
    """Weekend holidays shift to the adjacent weekday, per NYSE practice.

    A Saturday holiday observed on the preceding Friday is skipped when
    that Friday belongs to the previous calendar year — the NYSE does not
    close on December 31 for a January 1 that falls on a Saturday.
    """
    if holiday.weekday() == 5:  # Saturday -> preceding Friday
        observed = holiday - timedelta(days=1)
        if observed.year != holiday.year:
            return None
        return observed
    if holiday.weekday() == 6:  # Sunday -> following Monday
        return holiday + timedelta(days=1)
    return holiday


@lru_cache(maxsize=64)
def market_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures in ``year`` (observed dates)."""
    raw: list[date] = [
        date(year, 1, 1),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Jr. Day (3rd Monday, from 1998)
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday (3rd Monday)
        _easter(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day (last Monday)
        date(year, 7, 4),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day (1st Monday)
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving (4th Thursday)
        date(year, 12, 25),  # Christmas Day
    ]
    if year >= 2022:  # Juneteenth became a market holiday in 2022
        raw.append(date(year, 6, 19))
    if year >= 1998:
        pass  # MLK already included; kept explicit for the reader
    observed = {day for day in (_observed(holiday) for holiday in raw) if day is not None}
    observed |= {day for day in _AD_HOC_CLOSURES if day.year == year}
    return frozenset(observed)


@lru_cache(maxsize=64)
def early_closes(year: int) -> frozenset[date]:
    """Sessions that close at 13:00 ET instead of 16:00 ET."""
    holidays = market_holidays(year)
    candidates: list[date] = [
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),  # day after Thanksgiving
        date(year, 7, 3),  # July 3rd
        date(year, 12, 24),  # Christmas Eve
    ]
    return frozenset(
        day
        for day in candidates
        if day.weekday() < 5 and day not in holidays
    )


def is_trading_day(day: date) -> bool:
    """True when the US equity market holds a session on ``day``."""
    if day.weekday() >= 5:
        return False
    return day not in market_holidays(day.year)


def is_early_close(day: date) -> bool:
    """True when ``day`` is a session that closes early (13:00 ET)."""
    return is_trading_day(day) and day in early_closes(day.year)


def session_close_et(day: date) -> time:
    return EARLY_CLOSE_ET if is_early_close(day) else REGULAR_CLOSE_ET


def regular_session_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC open/close instants of ``day``'s regular session.

    Raises ``ValueError`` for non-sessions so a caller can never silently
    build a window over a holiday.
    """
    if not is_trading_day(day):
        raise ValueError(f"{day.isoformat()} is not a US equity trading day")
    open_local = datetime.combine(day, REGULAR_OPEN_ET, tzinfo=MARKET_TZ)
    close_local = datetime.combine(day, session_close_et(day), tzinfo=MARKET_TZ)
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


def trading_days(start: date, end: date) -> list[date]:
    """Every session in ``[start, end]`` inclusive, ascending."""
    if end < start:
        return []
    out: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += timedelta(days=1)
    return out


def session_count(start: date, end: date) -> int:
    return len(trading_days(start, end))


def iter_sessions(start: date, end: date) -> Iterator[tuple[date, datetime, datetime]]:
    for day in trading_days(start, end):
        open_utc, close_utc = regular_session_bounds(day)
        yield day, open_utc, close_utc


def _bucket_start(moment: datetime, duration: timedelta) -> datetime:
    """Floor ``moment`` to the start of its bar bucket.

    Alpaca aligns intraday buckets to the epoch in UTC, so a 1Hour bucket
    starts on the clock hour and a 4Hour bucket on 00/04/08/12/16/20 UTC.
    The 09:30 ET open does not divide either, which is why session
    membership is decided by OVERLAP rather than containment.
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed = moment.astimezone(timezone.utc) - epoch
    return epoch + (elapsed // duration) * duration


def expected_session_bars(day: date, timeframe: str) -> int:
    """How many regular-hours bars ``day`` should contain.

    Daily bars: one per session.  Intraday: the number of bar buckets
    that OVERLAP the regular session.  A 6.5-hour session spans seven
    1Hour buckets (09:00 through 15:00 ET), not six — the 09:00 bucket
    carries the first thirty minutes of trading.  Counting six would make
    every complete day look like it was missing a bar.
    """
    from config import timeframe_to_timedelta

    if not is_trading_day(day):
        return 0
    if timeframe == "1Day":
        return 1
    open_utc, close_utc = regular_session_bounds(day)
    duration = timeframe_to_timedelta(timeframe)
    if duration <= timedelta(0):
        raise ValueError(f"non-positive bar duration for {timeframe!r}")
    first = _bucket_start(open_utc, duration)
    # Buckets are [b, b+duration); one overlaps the session when it starts
    # strictly before the close.  Count them without materializing.
    span = close_utc - first
    return int(-(-span // duration))  # ceil division on timedeltas


def expected_bar_count(start: date, end: date, timeframe: str) -> int:
    """Expected regular-hours bar count across ``[start, end]``."""
    return sum(expected_session_bars(day, timeframe) for day in trading_days(start, end))


def session_dates(index: pd.DatetimeIndex, timeframe: str) -> list[date]:
    """The session each bar belongs to.

    Daily bars use the **UTC** date; intraday bars use the market-local
    date.  This is not an inconsistency, it is the only rule that is
    correct for both.

    Providers stamp a daily bar with one of several conventions — 00:00
    UTC, midnight ET (04:00/05:00 UTC), the session open, or the session
    close.  Every one of those falls on the session's own UTC date, but
    the 00:00-UTC convention converts to the *previous* evening in ET, so
    reading the ET date would shift a whole daily history back by one day
    for part of the year.  Regular-hours intraday bars (13:30-21:00 UTC)
    never cross UTC midnight either, but extended-hours bars do, so
    intraday attribution has to use market-local time.
    """
    if timeframe == "1Day":
        return [ts.date() for ts in index.tz_convert("UTC")]
    return [ts.date() for ts in index.tz_convert(MARKET_TZ)]


def filter_regular_hours(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop pre/post-market rows and any row outside a real session.

    Alpaca's equity bars include extended-hours prints for intraday
    timeframes.  Extended-hours bars have different liquidity and spread
    behavior; mixing them into a regular-hours strategy is a silent
    research error, so research callers filter explicitly.

    Daily bars are returned unchanged apart from dropping non-sessions.
    """
    if bars.empty:
        return bars
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by timestamp")
    if bars.index.tz is None:
        raise ValueError("bars index must be timezone-aware")

    days = session_dates(bars.index, timeframe)

    if timeframe == "1Day":
        keep = [is_trading_day(day) for day in days]
        return bars[pd.Series(keep, index=bars.index).to_numpy()]

    from config import timeframe_to_timedelta

    duration = timeframe_to_timedelta(timeframe)
    starts_utc = bars.index.tz_convert("UTC").to_pydatetime()

    # A bar belongs to the session when its interval [start, start+d)
    # OVERLAPS [open, close).  Containment would drop the 09:00 hourly
    # bucket that carries the first half hour of trading, and the 15:00
    # bucket that carries the last hour, because neither is contained in
    # a 09:30-16:00 window.
    keep: list[bool] = []
    for day, start in zip(days, starts_utc):
        if not is_trading_day(day):
            keep.append(False)
            continue
        open_utc, close_utc = regular_session_bounds(day)
        keep.append(start < close_utc and start + duration > open_utc)
    return bars[pd.Series(keep, index=bars.index).to_numpy()]
