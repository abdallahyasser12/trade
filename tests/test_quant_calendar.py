"""The US equity session calendar, pinned to independently known dates.

Every date asserted here is a published NYSE session or closure, not a
value read back from the implementation.  A calendar bug is invisible at
runtime — it shows up as "missing" bars on holidays and as complete
histories that fail their completeness check — so these are the tests
that keep equity research honest.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from quant.data.calendar import (
    EARLY_CLOSE_ET,
    MARKET_TZ,
    REGULAR_CLOSE_ET,
    expected_bar_count,
    expected_session_bars,
    filter_regular_hours,
    is_early_close,
    is_trading_day,
    market_holidays,
    regular_session_bounds,
    session_close_et,
    session_count,
    trading_days,
)


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day",
    [
        date(2024, 1, 1),  # New Year's Day (Mon)
        date(2024, 1, 15),  # MLK Jr. Day
        date(2024, 2, 19),  # Washington's Birthday
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ],
)
def test_2024_market_holidays_are_closed(day):
    assert not is_trading_day(day)


@pytest.mark.parametrize(
    "day",
    [
        date(2024, 1, 2),  # first session of 2024
        date(2024, 3, 28),  # Thursday before Good Friday
        date(2024, 7, 5),  # Friday after Independence Day
        date(2024, 11, 29),  # day after Thanksgiving (early, but open)
        date(2024, 12, 24),  # Christmas Eve (early, but open)
        date(2024, 12, 26),
    ],
)
def test_2024_ordinary_sessions_are_open(day):
    assert is_trading_day(day)


def test_weekends_are_never_sessions():
    assert not is_trading_day(date(2024, 6, 15))  # Saturday
    assert not is_trading_day(date(2024, 6, 16))  # Sunday


def test_sunday_holiday_observed_on_monday():
    # New Year's Day 2023 fell on a Sunday; the market closed Monday Jan 2.
    assert date(2023, 1, 2) in market_holidays(2023)
    assert not is_trading_day(date(2023, 1, 2))
    assert is_trading_day(date(2023, 1, 3))


def test_saturday_holiday_observed_on_friday():
    # Christmas 2021 fell on a Saturday; the market closed Friday Dec 24.
    assert not is_trading_day(date(2021, 12, 24))
    assert is_trading_day(date(2021, 12, 23))


def test_new_years_on_saturday_is_not_observed_in_the_prior_year():
    # Jan 1 2022 was a Saturday.  The NYSE did NOT close Friday Dec 31
    # 2021 for it, and Monday Jan 3 2022 was a normal session.
    assert is_trading_day(date(2021, 12, 31))
    assert is_trading_day(date(2022, 1, 3))


def test_juneteenth_only_from_2022():
    assert is_trading_day(date(2021, 6, 18))  # Friday, before it was a holiday
    assert not is_trading_day(date(2022, 6, 20))  # observed Monday for Sunday Jun 19


def test_good_friday_tracks_easter():
    # Easter 2023 was April 9, so Good Friday was April 7.
    assert not is_trading_day(date(2023, 4, 7))
    # Easter 2025 was April 20, so Good Friday was April 18.
    assert not is_trading_day(date(2025, 4, 18))


def test_ad_hoc_closure_is_honored():
    # National day of mourning for President Carter.
    assert not is_trading_day(date(2025, 1, 9))


# ---------------------------------------------------------------------------
# Early closes
# ---------------------------------------------------------------------------


def test_day_after_thanksgiving_closes_early():
    assert is_early_close(date(2024, 11, 29))
    assert session_close_et(date(2024, 11, 29)) == EARLY_CLOSE_ET


def test_christmas_eve_weekday_closes_early():
    assert is_early_close(date(2024, 12, 24))


def test_christmas_eve_is_not_an_early_close_when_it_is_a_full_holiday():
    # Dec 24 2021 was the observed Christmas holiday, not a short session.
    assert not is_early_close(date(2021, 12, 24))
    assert not is_trading_day(date(2021, 12, 24))


def test_christmas_eve_on_a_weekend_is_not_an_early_close():
    assert not is_early_close(date(2023, 12, 24))  # Sunday


def test_ordinary_session_closes_at_four():
    assert session_close_et(date(2024, 6, 12)) == REGULAR_CLOSE_ET
    assert not is_early_close(date(2024, 6, 12))


# ---------------------------------------------------------------------------
# Session bounds
# ---------------------------------------------------------------------------


def test_session_bounds_are_utc_and_track_daylight_saving():
    # June is EDT (UTC-4): 09:30 ET == 13:30 UTC.
    open_utc, close_utc = regular_session_bounds(date(2024, 6, 12))
    assert open_utc == datetime(2024, 6, 12, 13, 30, tzinfo=timezone.utc)
    assert close_utc == datetime(2024, 6, 12, 20, 0, tzinfo=timezone.utc)

    # January is EST (UTC-5): 09:30 ET == 14:30 UTC.
    open_utc, close_utc = regular_session_bounds(date(2024, 1, 12))
    assert open_utc == datetime(2024, 1, 12, 14, 30, tzinfo=timezone.utc)
    assert close_utc == datetime(2024, 1, 12, 21, 0, tzinfo=timezone.utc)


def test_early_close_session_is_three_and_a_half_hours():
    open_utc, close_utc = regular_session_bounds(date(2024, 11, 29))
    assert close_utc - open_utc == timedelta(hours=3, minutes=30)


def test_session_bounds_refuse_a_non_session():
    with pytest.raises(ValueError, match="not a US equity trading day"):
        regular_session_bounds(date(2024, 7, 4))


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_trading_days_skips_weekend_and_holiday():
    days = trading_days(date(2024, 7, 1), date(2024, 7, 8))
    assert days == [
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
    ]


def test_2024_had_252_sessions():
    # Independently known: the NYSE held 252 sessions in 2024.
    assert session_count(date(2024, 1, 1), date(2024, 12, 31)) == 252


def test_2023_had_250_sessions():
    assert session_count(date(2023, 1, 1), date(2023, 12, 31)) == 250


def test_reversed_range_is_empty_not_an_error():
    assert trading_days(date(2024, 6, 10), date(2024, 6, 3)) == []


@pytest.mark.parametrize(
    "timeframe,expected",
    [
        ("1Min", 390),  # 6.5h * 60
        ("5Min", 78),
        ("15Min", 26),
        ("1Day", 1),
    ],
)
def test_expected_bars_for_an_aligned_timeframe(timeframe, expected):
    assert expected_session_bars(date(2024, 6, 12), timeframe) == expected


def test_hourly_session_spans_seven_buckets_not_six():
    # 09:30-16:00 ET overlaps the 09:00,10:00,...,15:00 buckets.  Counting
    # 6 would make every complete day look short by one bar.
    assert expected_session_bars(date(2024, 6, 12), "1Hour") == 7


def test_early_close_day_has_fewer_bars():
    assert expected_session_bars(date(2024, 11, 29), "5Min") == 42  # 3.5h * 12
    assert expected_session_bars(date(2024, 11, 29), "1Hour") == 4


def test_non_session_expects_zero_bars():
    assert expected_session_bars(date(2024, 7, 4), "5Min") == 0
    assert expected_session_bars(date(2024, 6, 15), "1Day") == 0


def test_expected_bar_count_sums_sessions_and_respects_early_closes():
    # Nov 25-29 2024: Mon-Wed full, Thu Thanksgiving closed, Fri early.
    total = expected_bar_count(date(2024, 11, 25), date(2024, 11, 29), "5Min")
    assert total == 78 * 3 + 0 + 42


# ---------------------------------------------------------------------------
# Regular-hours filtering
# ---------------------------------------------------------------------------


def _frame(timestamps) -> pd.DataFrame:
    index = pd.DatetimeIndex(timestamps, name="timestamp")
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=index,
    )


def test_filter_drops_premarket_and_afterhours_minutes():
    et = MARKET_TZ
    stamps = [
        datetime(2024, 6, 12, 8, 0, tzinfo=et),  # pre-market
        datetime(2024, 6, 12, 9, 29, tzinfo=et),  # pre-market
        datetime(2024, 6, 12, 9, 30, tzinfo=et),  # first session bar
        datetime(2024, 6, 12, 15, 59, tzinfo=et),  # last session bar
        datetime(2024, 6, 12, 16, 0, tzinfo=et),  # after hours
        datetime(2024, 6, 12, 18, 0, tzinfo=et),  # after hours
    ]
    kept = filter_regular_hours(_frame(stamps), "1Min")
    local = kept.index.tz_convert(et)
    assert [t.strftime("%H:%M") for t in local] == ["09:30", "15:59"]


def test_filter_drops_the_extra_minute_of_an_early_close_day():
    et = MARKET_TZ
    stamps = [
        datetime(2024, 11, 29, 12, 59, tzinfo=et),  # in session
        datetime(2024, 11, 29, 13, 0, tzinfo=et),  # at/after the early close
        datetime(2024, 11, 29, 15, 30, tzinfo=et),  # would be valid on a full day
    ]
    kept = filter_regular_hours(_frame(stamps), "1Min")
    local = kept.index.tz_convert(et)
    assert [t.strftime("%H:%M") for t in local] == ["12:59"]


def test_filter_keeps_the_overlapping_hourly_buckets():
    et = MARKET_TZ
    stamps = [
        datetime(2024, 6, 12, 8, 0, tzinfo=et),  # 08:00-09:00, no overlap
        datetime(2024, 6, 12, 9, 0, tzinfo=et),  # 09:00-10:00, overlaps open
        datetime(2024, 6, 12, 15, 0, tzinfo=et),  # 15:00-16:00, ends at close
        datetime(2024, 6, 12, 16, 0, tzinfo=et),  # 16:00-17:00, no overlap
    ]
    kept = filter_regular_hours(_frame(stamps), "1Hour")
    local = kept.index.tz_convert(et)
    assert [t.strftime("%H:%M") for t in local] == ["09:00", "15:00"]


def test_filter_drops_holiday_rows_entirely():
    et = MARKET_TZ
    stamps = [
        datetime(2024, 7, 3, 10, 0, tzinfo=et),  # session
        datetime(2024, 7, 4, 10, 0, tzinfo=et),  # Independence Day
        datetime(2024, 7, 5, 10, 0, tzinfo=et),  # session
    ]
    kept = filter_regular_hours(_frame(stamps), "1Min")
    assert [t.date() for t in kept.index.tz_convert(et)] == [
        date(2024, 7, 3),
        date(2024, 7, 5),
    ]


def test_filter_keeps_daily_bars_on_sessions_only():
    stamps = [
        datetime(2024, 7, 3, 4, 0, tzinfo=timezone.utc),
        datetime(2024, 7, 4, 4, 0, tzinfo=timezone.utc),  # holiday
        datetime(2024, 7, 6, 4, 0, tzinfo=timezone.utc),  # Saturday
        datetime(2024, 7, 8, 4, 0, tzinfo=timezone.utc),
    ]
    kept = filter_regular_hours(_frame(stamps), "1Day")
    assert len(kept) == 2


def test_filter_is_idempotent():
    et = MARKET_TZ
    stamps = [
        datetime(2024, 6, 12, 9, 0, tzinfo=et),
        datetime(2024, 6, 12, 12, 0, tzinfo=et),
        datetime(2024, 6, 12, 20, 0, tzinfo=et),
    ]
    once = filter_regular_hours(_frame(stamps), "1Hour")
    twice = filter_regular_hours(once, "1Hour")
    assert once.index.equals(twice.index)


def test_filter_rejects_naive_timestamps():
    naive = pd.DataFrame(
        {c: [1.0] for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([datetime(2024, 6, 12, 14, 0)], name="timestamp"),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_regular_hours(naive, "1Min")


def test_filter_on_empty_frame_is_a_noop():
    empty = _frame([]).iloc[0:0]
    assert filter_regular_hours(empty, "1Min").empty
