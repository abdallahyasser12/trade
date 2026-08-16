"""Historical equity retrieval: chunking, caching, completeness, failure.

Every test runs against a recorded-fixture bar source, so the suite needs
no credentials and touches no network — which is also the only way this
path can be exercised in CI. The fixture source records the exact windows
it was asked for, so the tests can assert on API-call behavior (the thing
that actually costs money and rate limit) and not just on the frame.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest

from quant.data.cache import BarCache
from quant.data.calendar import MARKET_TZ, regular_session_bounds, trading_days
from quant.data.equity_history import (
    EquityHistoryFetcher,
    chunk_ranges,
    _trim_to_sessions,
)
from quant.data.provenance import DataQuality

NOW = datetime(2024, 7, 15, 20, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture bar source
# ---------------------------------------------------------------------------


def _session_bars(day: date, timeframe: str, *, price: float = 100.0) -> pd.DataFrame:
    """Synthetic but calendar-correct bars for one session."""
    from config import timeframe_to_timedelta

    open_utc, close_utc = regular_session_bounds(day)
    duration = timeframe_to_timedelta(timeframe)
    if timeframe == "1Day":
        stamps = [datetime.combine(day, time(4, 0), tzinfo=timezone.utc)]
    else:
        stamps = []
        cursor = open_utc
        while cursor < close_utc:
            stamps.append(cursor)
            cursor += duration
    index = pd.DatetimeIndex(stamps, name="timestamp", tz="UTC")
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000.0,
        },
        index=index,
    )


class FixtureSource:
    """A ``BarSource`` that serves calendar-correct bars from memory."""

    def __init__(
        self,
        *,
        available: set[str] | None = None,
        first_session: date | None = None,
        drop_days: set[date] | None = None,
        fail_on: set[str] | None = None,
        include_premarket: bool = False,
    ) -> None:
        self.available = available
        self.first_session = first_session
        self.drop_days = drop_days or set()
        self.fail_on = fail_on or set()
        self.include_premarket = include_premarket
        self.requests: list[tuple[str, str, datetime, datetime]] = []

    def get_bars(self, symbol, timeframe, start, end=None, limit=None, adjustment="split"):
        self.requests.append((symbol, timeframe, start, end))
        if symbol in self.fail_on:
            raise RuntimeError(f"provider blew up for {symbol}")
        if self.available is not None and symbol not in self.available:
            return pd.DataFrame()

        lo = start.astimezone(MARKET_TZ).date()
        hi = (end or start).astimezone(MARKET_TZ).date()
        frames = []
        for day in trading_days(lo, hi):
            if day in self.drop_days:
                continue
            if self.first_session is not None and day < self.first_session:
                continue
            frames.append(_session_bars(day, timeframe))
            if self.include_premarket and timeframe != "1Day":
                pre = datetime.combine(day, time(8, 0), tzinfo=MARKET_TZ).astimezone(
                    timezone.utc
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "open": 99.0, "high": 99.0, "low": 99.0,
                            "close": 99.0, "volume": 5.0,
                        },
                        index=pd.DatetimeIndex([pre], name="timestamp", tz="UTC"),
                    )
                )
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()


def _fetcher(source, tmp_path=None, **kwargs):
    cache = BarCache(tmp_path / "cache") if tmp_path is not None else None
    return EquityHistoryFetcher(
        source, cache=cache, clock=lambda: NOW, **kwargs
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_daily_chunks_split_on_year_boundaries():
    chunks = chunk_ranges(date(2022, 6, 1), date(2024, 3, 15), "1Day")
    assert chunks == [
        (date(2022, 6, 1), date(2022, 12, 31)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 3, 15)),
    ]


def test_minute_chunks_split_on_month_boundaries():
    chunks = chunk_ranges(date(2024, 2, 10), date(2024, 4, 20), "1Min")
    assert chunks == [
        (date(2024, 2, 10), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 20)),
    ]


def test_chunk_boundaries_do_not_depend_on_the_request_window():
    # This is what makes the cache hit across differently-framed requests.
    wide = chunk_ranges(date(2024, 2, 10), date(2024, 4, 20), "1Min")
    narrow = chunk_ranges(date(2024, 3, 1), date(2024, 3, 31), "1Min")
    assert narrow[0] == (date(2024, 3, 1), date(2024, 3, 31))
    assert (date(2024, 3, 1), date(2024, 3, 31)) in wide


def test_reversed_range_chunks_to_nothing():
    assert chunk_ranges(date(2024, 4, 1), date(2024, 3, 1), "1Day") == []


def test_unsupported_timeframe_is_rejected_at_chunking():
    with pytest.raises(ValueError, match="Unsupported"):
        chunk_ranges(date(2024, 1, 1), date(2024, 2, 1), "4Hour")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fetches_a_full_month_of_daily_bars_and_reports_ok(tmp_path):
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 28))

    expected_sessions = len(trading_days(date(2024, 6, 3), date(2024, 6, 28)))
    assert len(history) == expected_sessions
    assert history.quality is DataQuality.OK
    assert history.provenance.completeness.coverage == pytest.approx(1.0)
    assert history.is_usable


def test_intraday_fetch_drops_premarket_prints(tmp_path):
    source = FixtureSource(include_premarket=True)
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "5Min", date(2024, 6, 3), date(2024, 6, 4)
    )
    local_times = {ts.strftime("%H:%M") for ts in history.bars.index.tz_convert(MARKET_TZ)}
    assert "08:00" not in local_times
    assert "09:30" in local_times


def test_extended_hours_can_be_kept_explicitly(tmp_path):
    source = FixtureSource(include_premarket=True)
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "5Min", date(2024, 6, 3), date(2024, 6, 4), regular_hours_only=False
    )
    local_times = {ts.strftime("%H:%M") for ts in history.bars.index.tz_convert(MARKET_TZ)}
    assert "08:00" in local_times
    assert history.provenance.regular_hours_only is False


def test_holidays_are_absent_without_being_reported_as_missing(tmp_path):
    # July 4 2024 is a holiday: no bars, and completeness must still be 100%.
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Day", date(2024, 7, 1), date(2024, 7, 8)
    )
    days = {ts.date() for ts in history.bars.index.tz_convert(MARKET_TZ)}
    assert date(2024, 7, 4) not in days
    assert history.quality is DataQuality.OK
    assert history.provenance.completeness.missing_sessions == ()


def test_early_close_day_is_complete_with_fewer_bars(tmp_path):
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "5Min", date(2024, 11, 29), date(2024, 11, 29)
    )
    assert len(history) == 42
    assert history.quality is DataQuality.OK


def test_bars_are_trimmed_to_the_requested_market_local_range(tmp_path):
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Day", date(2024, 6, 10), date(2024, 6, 12)
    )
    days = sorted({ts.date() for ts in history.bars.index.tz_convert(MARKET_TZ)})
    assert days == [date(2024, 6, 10), date(2024, 6, 11), date(2024, 6, 12)]


# ---------------------------------------------------------------------------
# Incompleteness is reported, never hidden
# ---------------------------------------------------------------------------


def test_missing_sessions_downgrade_quality_to_incomplete(tmp_path):
    dropped = {date(2024, 6, 11), date(2024, 6, 12)}
    source = FixtureSource(drop_days=dropped)
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Day", date(2024, 6, 3), date(2024, 6, 28)
    )
    assert history.quality is DataQuality.INCOMPLETE
    assert "2024-06-11" in history.provenance.completeness.missing_sessions
    assert history.provenance.completeness.coverage < 1.0
    assert any("coverage" in note for note in history.provenance.notes)


def test_a_symbol_with_no_data_is_missing_not_empty_ok(tmp_path):
    source = FixtureSource(available={"SPY"})
    history = _fetcher(source, tmp_path).fetch(
        "DELISTED", "1Day", date(2024, 6, 3), date(2024, 6, 28)
    )
    assert history.quality is DataQuality.MISSING
    assert not history.is_usable
    assert len(history) == 0


def test_short_history_symbol_reports_the_gap(tmp_path):
    # Listed mid-range: the earlier sessions are genuinely absent.
    source = FixtureSource(first_session=date(2024, 6, 17))
    history = _fetcher(source, tmp_path).fetch(
        "IPO", "1Day", date(2024, 6, 3), date(2024, 6, 28)
    )
    assert history.quality is DataQuality.INCOMPLETE
    assert "2024-06-03" in history.provenance.completeness.missing_sessions


def test_range_with_no_sessions_is_missing(tmp_path):
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Day", date(2024, 7, 6), date(2024, 7, 7)  # Sat + Sun
    )
    assert history.quality is DataQuality.MISSING
    assert source.requests == []  # never bothered the provider


def test_a_past_window_is_not_downgraded_for_staleness(tmp_path):
    # Research over 2023 must not be flagged STALE just for being historical.
    source = FixtureSource()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Day", date(2023, 3, 1), date(2023, 3, 31)
    )
    assert history.quality is DataQuality.OK
    assert not any("stale" in note for note in history.provenance.notes)


# ---------------------------------------------------------------------------
# Provider failure containment
# ---------------------------------------------------------------------------


def test_a_failing_chunk_does_not_lose_the_other_chunks(tmp_path):
    class FlakyOnMarch(FixtureSource):
        def get_bars(self, symbol, timeframe, start, end=None, limit=None, adjustment="split"):
            if start.astimezone(MARKET_TZ).date().month == 2:
                self.requests.append((symbol, timeframe, start, end))
                raise RuntimeError("503 from provider")
            return super().get_bars(symbol, timeframe, start, end, limit, adjustment)

    source = FlakyOnMarch()
    history = _fetcher(source, tmp_path).fetch(
        "SPY", "1Min", date(2024, 2, 1), date(2024, 3, 8)
    )
    # March survived; February is reported as a provider error, not silence.
    assert len(history) > 0
    assert history.quality is DataQuality.INCOMPLETE
    assert any("provider error" in note for note in history.provenance.notes)


def test_fetch_many_isolates_one_bad_symbol(tmp_path):
    source = FixtureSource(fail_on={"BROKEN"})
    out = _fetcher(source, tmp_path).fetch_many(
        ["SPY", "BROKEN", "QQQ"], "1Day", date(2024, 6, 3), date(2024, 6, 7)
    )
    assert set(out) == {"SPY", "QQQ"}


def test_fetch_many_drops_unusable_symbols_by_default(tmp_path):
    source = FixtureSource(available={"SPY"})
    out = _fetcher(source, tmp_path).fetch_many(
        ["SPY", "GONE"], "1Day", date(2024, 6, 3), date(2024, 6, 7)
    )
    assert set(out) == {"SPY"}


def test_fetch_many_can_surface_unusable_symbols_when_asked(tmp_path):
    source = FixtureSource(available={"SPY"})
    out = _fetcher(source, tmp_path).fetch_many(
        ["SPY", "GONE"], "1Day", date(2024, 6, 3), date(2024, 6, 7),
        skip_unusable=False,
    )
    assert set(out) == {"SPY", "GONE"}
    assert out["GONE"].quality is DataQuality.MISSING


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_second_identical_fetch_hits_the_cache_and_skips_the_provider(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    first = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 28))
    calls_after_first = fetcher.api_calls

    second = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 28))
    assert fetcher.api_calls == calls_after_first
    assert fetcher.cache_hits >= 1
    assert second.bars.equals(first.bars)


def test_cache_is_shared_across_overlapping_windows(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    fetcher.fetch("SPY", "1Min", date(2024, 2, 1), date(2024, 4, 30))
    calls = fetcher.api_calls
    # March is a whole cached chunk; asking only for March costs nothing.
    fetcher.fetch("SPY", "1Min", date(2024, 3, 1), date(2024, 3, 31))
    assert fetcher.api_calls == calls


def test_refresh_bypasses_the_cache(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))
    calls = fetcher.api_calls
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7), refresh=True)
    assert fetcher.api_calls > calls


def test_cache_distinguishes_adjustment(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7), adjustment="split")
    calls = fetcher.api_calls
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7), adjustment="all")
    assert fetcher.api_calls > calls, "a different adjustment must not reuse the entry"


def test_cache_survives_a_new_fetcher_instance(tmp_path):
    source_a = FixtureSource()
    _fetcher(source_a, tmp_path).fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))

    source_b = FixtureSource()
    fetcher_b = _fetcher(source_b, tmp_path)
    fetcher_b.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))
    assert source_b.requests == []


def test_a_damaged_cache_entry_is_a_miss_not_a_crash(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))

    for path in (tmp_path / "cache").rglob("bars.*"):
        path.write_text("this is not a dataframe")

    history = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))
    assert history.quality is DataQuality.OK
    assert len(history) > 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_records_the_request_and_the_result(tmp_path):
    source = FixtureSource()
    history = _fetcher(source, tmp_path, feed_name="sip").fetch(
        "SPY", "1Hour", date(2024, 6, 3), date(2024, 6, 7), adjustment="all"
    )
    p = history.provenance
    assert p.provider == "alpaca"
    assert p.feed == "sip"
    assert p.market == "us_equity"
    assert p.symbol == "SPY"
    assert p.timeframe == "1Hour"
    assert p.adjustment == "all"
    assert p.observed_start is not None and p.observed_end is not None
    assert p.fetched_at == NOW
    assert p.to_dict()["completeness"]["observed_bars"] == len(history)


def test_fingerprint_is_stable_for_the_same_request_and_differs_otherwise(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    a = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7)).provenance
    b = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7)).provenance
    c = fetcher.fetch("QQQ", "1Day", date(2024, 6, 3), date(2024, 6, 7)).provenance
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_provenance_json_round_trips_through_the_cache(tmp_path):
    source = FixtureSource()
    fetcher = _fetcher(source, tmp_path)
    fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))
    cache = BarCache(tmp_path / "cache")
    found = [p for p in (tmp_path / "cache").rglob("provenance.json")]
    assert found
    fingerprint = found[0].parent.name
    bars, provenance = cache.get(fingerprint)
    assert provenance.symbol == "SPY"
    assert provenance.timeframe == "1Day"
    assert not bars.empty


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unsupported_timeframe_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported equity research timeframe"):
        _fetcher(FixtureSource(), tmp_path).fetch(
            "SPY", "4Hour", date(2024, 6, 3), date(2024, 6, 7)
        )


def test_reversed_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="precedes start"):
        _fetcher(FixtureSource(), tmp_path).fetch(
            "SPY", "1Day", date(2024, 6, 7), date(2024, 6, 3)
        )


def test_bad_adjustment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported adjustment"):
        _fetcher(FixtureSource(), tmp_path).fetch(
            "SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7), adjustment="magic"
        )


def test_works_without_a_cache_at_all():
    source = FixtureSource()
    fetcher = EquityHistoryFetcher(source, cache=None, clock=lambda: NOW)
    history = fetcher.fetch("SPY", "1Day", date(2024, 6, 3), date(2024, 6, 7))
    assert len(history) == 5
    assert fetcher.cache_hits == 0
