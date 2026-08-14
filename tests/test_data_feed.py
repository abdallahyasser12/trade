"""Unit tests for data_feed.py pure functions and retry logic.

Everything runs on deterministic synthetic data and in-file fake exception
classes — AlpacaDataFeed is never constructed and no network is touched.
All timestamps are timezone-aware UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data_feed import (
    DataFeedError,
    DataSubscriptionError,
    bar_is_complete,
    drop_incomplete_bars,
    is_data_stale,
    is_permanent_error,
    is_rate_limit_error,
    is_subscription_error,
    max_staleness,
    normalize_bars_df,
    with_retries,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for an HTTP response carrying headers."""

    def __init__(self, headers: dict):
        self.headers = headers


class FakeAPIError(Exception):
    """Exception shaped like an alpaca-py APIError: status_code + response."""

    def __init__(self, message: str = "", status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class SleepRecorder:
    """Injectable sleep_fn that records requested delays without sleeping."""

    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class InterruptingSleep(SleepRecorder):
    def __call__(self, seconds: float) -> bool:
        self.delays.append(seconds)
        return True


def test_retry_backoff_can_be_interrupted_by_engine_stop_event():
    sleeper = InterruptingSleep()
    with pytest.raises(DataFeedError, match="interrupted"):
        with_retries(lambda: (_ for _ in ()).throw(RuntimeError("down")), sleep_fn=sleeper)
    assert sleeper.delays == [1.0]


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


OHLCV = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# bar_is_complete
# ---------------------------------------------------------------------------


class TestBarIsComplete:
    START = _utc(2026, 8, 10, 14, 30)  # 5Min bar covering 14:30-14:35 UTC

    def test_5min_bar_not_complete_before_interval_ends(self):
        assert not bar_is_complete(self.START, "5Min", _utc(2026, 8, 10, 14, 34, 59))

    def test_5min_bar_not_complete_within_tolerance_window(self):
        """The bar ends at 14:35 but a 5s publication tolerance means it only
        counts as complete from 14:35:05 — at 14:35:02 it is still forming."""
        now = _utc(2026, 8, 10, 14, 35, 2)
        assert not bar_is_complete(self.START, "5Min", now, tolerance_seconds=5.0)

    def test_5min_bar_complete_after_end_plus_tolerance(self):
        now = _utc(2026, 8, 10, 14, 35, 6)
        assert bar_is_complete(self.START, "5Min", now, tolerance_seconds=5.0)

    def test_1day_bar_not_complete_before_interval_ends(self):
        start = _utc(2026, 8, 10)  # covers 2026-08-10 00:00 -> 2026-08-11 00:00
        assert not bar_is_complete(start, "1Day", _utc(2026, 8, 10, 23, 59, 59))

    def test_1day_bar_not_complete_within_tolerance_window(self):
        start = _utc(2026, 8, 10)
        now = _utc(2026, 8, 11, 0, 0, 2)
        assert not bar_is_complete(start, "1Day", now, tolerance_seconds=5.0)

    def test_1day_bar_complete_after_end_plus_tolerance(self):
        start = _utc(2026, 8, 10)
        now = _utc(2026, 8, 11, 0, 0, 6)
        assert bar_is_complete(start, "1Day", now, tolerance_seconds=5.0)

    def test_naive_bar_start_raises(self):
        with pytest.raises(ValueError):
            bar_is_complete(datetime(2026, 8, 10, 14, 30), "5Min", _utc(2026, 8, 10, 15, 0))

    def test_naive_now_raises(self):
        with pytest.raises(ValueError):
            bar_is_complete(self.START, "5Min", datetime(2026, 8, 10, 15, 0))

    def test_zero_tolerance_boundary_now_equals_end_is_complete(self):
        """With tolerance 0 the comparison is now >= end, so the exact end
        instant already counts as complete."""
        end = self.START + timedelta(minutes=5)
        assert bar_is_complete(self.START, "5Min", end, tolerance_seconds=0.0)
        assert not bar_is_complete(
            self.START, "5Min", end - timedelta(microseconds=1), tolerance_seconds=0.0
        )


# ---------------------------------------------------------------------------
# drop_incomplete_bars
# ---------------------------------------------------------------------------


def _bars_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=index,
    )


class TestDropIncompleteBars:
    def test_trailing_forming_bar_removed_completed_kept(self):
        """Four 5Min bars starting 14:00; at 14:16:00 the 14:15 bar is still
        forming and must be dropped while all earlier bars survive."""
        idx = pd.date_range("2026-08-10 14:00", periods=4, freq="5min", tz="UTC")
        df = _bars_frame(idx)
        out = drop_incomplete_bars(df, "5Min", _utc(2026, 8, 10, 14, 16), tolerance_seconds=5.0)
        assert list(out.index) == list(idx[:3])
        pd.testing.assert_frame_equal(out, df.iloc[:3])

    def test_multiple_trailing_bars_removed(self):
        """If 'now' has fallen behind (e.g. a data gap), every not-yet-closed
        trailing bar goes, not just the very last one."""
        idx = pd.date_range("2026-08-10 14:00", periods=4, freq="5min", tz="UTC")
        df = _bars_frame(idx)
        out = drop_incomplete_bars(df, "5Min", _utc(2026, 8, 10, 14, 12), tolerance_seconds=5.0)
        assert list(out.index) == list(idx[:2])

    def test_all_bars_complete_frame_unchanged(self):
        idx = pd.date_range("2026-08-10 14:00", periods=4, freq="5min", tz="UTC")
        df = _bars_frame(idx)
        out = drop_incomplete_bars(df, "5Min", _utc(2026, 8, 10, 15, 0), tolerance_seconds=5.0)
        pd.testing.assert_frame_equal(out, df)

    def test_tolerance_holds_back_bar_that_just_ended(self):
        """A bar whose interval ended 2s ago is still inside the 5s tolerance
        window and must be dropped."""
        idx = pd.date_range("2026-08-10 14:00", periods=2, freq="5min", tz="UTC")
        df = _bars_frame(idx)
        out = drop_incomplete_bars(df, "5Min", _utc(2026, 8, 10, 14, 10, 2), tolerance_seconds=5.0)
        assert list(out.index) == [idx[0]]

    def test_empty_frame_ok(self):
        empty = _bars_frame(pd.DatetimeIndex([], tz="UTC"))
        out = drop_incomplete_bars(empty, "5Min", _utc(2026, 8, 10, 14, 0))
        assert out.empty


# ---------------------------------------------------------------------------
# max_staleness / is_data_stale
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_intraday_max_staleness_is_bars_plus_grace(self):
        """Intraday staleness = staleness_bars * bar duration + 2min grace."""
        assert max_staleness("5Min", staleness_bars=3) == timedelta(minutes=17)
        assert max_staleness("1Min", staleness_bars=3) == timedelta(minutes=5)
        assert max_staleness("1Hour", staleness_bars=2) == timedelta(hours=2, minutes=2)

    def test_daily_max_staleness_allows_several_days(self):
        assert max_staleness("1Day") == timedelta(days=5)

    def test_fresh_intraday_bar_not_stale(self):
        """A 5Min bar that closed one minute ago is well within 3 bars + 2min."""
        now = _utc(2026, 8, 10, 15, 0)
        last_start = now - timedelta(minutes=6)  # bar ended 1 minute ago
        assert not is_data_stale(last_start, "5Min", now, staleness_bars=3)

    def test_old_intraday_bar_is_stale(self):
        """A 5Min bar that closed 20 minutes ago exceeds 15min + 2min grace."""
        now = _utc(2026, 8, 10, 15, 0)
        last_start = now - timedelta(minutes=25)  # bar ended 20 minutes ago
        assert is_data_stale(last_start, "5Min", now, staleness_bars=3)

    def test_intraday_staleness_boundary_is_exclusive(self):
        """Staleness triggers strictly beyond the allowance: a bar whose age
        equals exactly 3 bars + 2min grace is still acceptable."""
        now = _utc(2026, 8, 10, 15, 0)
        last_start = now - timedelta(minutes=5) - timedelta(minutes=17)
        assert not is_data_stale(last_start, "5Min", now, staleness_bars=3)
        assert is_data_stale(last_start - timedelta(seconds=1), "5Min", now, staleness_bars=3)

    def test_friday_daily_bar_not_stale_on_monday(self):
        """2026-08-07 is a Friday; its daily bar seen on Monday afternoon is
        ~2.6 days old, inside the 5-day daily allowance."""
        friday_bar_start = _utc(2026, 8, 7)
        monday = _utc(2026, 8, 10, 15, 30)
        assert not is_data_stale(friday_bar_start, "1Day", monday)

    def test_week_old_daily_bar_is_stale(self):
        old_bar_start = _utc(2026, 8, 1)
        now = _utc(2026, 8, 10, 15, 30)
        assert is_data_stale(old_bar_start, "1Day", now)


# ---------------------------------------------------------------------------
# normalize_bars_df
# ---------------------------------------------------------------------------


def _alpaca_style_frame(
    symbol: str = "AAPL",
    timestamps: pd.DatetimeIndex | None = None,
    extra_columns: bool = True,
) -> pd.DataFrame:
    """Build a frame shaped like alpaca-py BarSet.df: (symbol, timestamp)
    MultiIndex plus OHLCV (and by default the extra alpaca columns)."""
    if timestamps is None:
        timestamps = pd.date_range("2026-08-10 13:30", periods=4, freq="5min", tz="UTC")
    index = pd.MultiIndex.from_product([[symbol], timestamps], names=["symbol", "timestamp"])
    n = len(timestamps)
    data = {
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000 + i for i in range(n)],  # ints on purpose: must cast
        "trade_count": [50] * n,
        "vwap": [100.2] * n,
    }
    if not extra_columns:
        data.pop("trade_count")
        data.pop("vwap")
    return pd.DataFrame(data, index=index)


class TestNormalizeBarsDf:
    def test_multiindex_frame_normalized(self):
        """A (symbol, timestamp) MultiIndex frame collapses to a UTC tz-aware
        DatetimeIndex named 'timestamp' with only float OHLCV columns."""
        out = normalize_bars_df(_alpaca_style_frame("AAPL"), "AAPL")
        assert len(out) == 4
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.name == "timestamp"
        assert out.index.tz is not None
        assert str(out.index.tz) == "UTC"
        assert out.index.is_monotonic_increasing
        assert list(out.columns) == OHLCV
        assert all(out[c].dtype == float for c in OHLCV)
        assert out["close"].iloc[0] == 100.5
        assert out["volume"].iloc[-1] == 1003.0

    def test_unsorted_input_returned_ascending(self):
        ts = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-08-10 13:40", tz="UTC"),
                pd.Timestamp("2026-08-10 13:30", tz="UTC"),
                pd.Timestamp("2026-08-10 13:35", tz="UTC"),
            ]
        )
        out = normalize_bars_df(_alpaca_style_frame("AAPL", timestamps=ts), "AAPL")
        assert out.index.is_monotonic_increasing
        assert len(out) == 3

    def test_duplicate_timestamps_keep_last(self):
        """When the same bar timestamp appears twice (e.g. an amended bar in a
        re-fetch), the later row wins."""
        t1 = pd.Timestamp("2026-08-10 13:30", tz="UTC")
        t2 = pd.Timestamp("2026-08-10 13:35", tz="UTC")
        index = pd.MultiIndex.from_tuples(
            [("AAPL", t1), ("AAPL", t2), ("AAPL", t2)], names=["symbol", "timestamp"]
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.2, 101.2, 102.2],
                "volume": [1000.0, 2000.0, 3000.0],
            },
            index=index,
        )
        out = normalize_bars_df(df, "AAPL")
        assert len(out) == 2
        assert out.loc[t2, "close"] == 102.2
        assert out.loc[t2, "volume"] == 3000.0

    def test_rows_with_non_positive_prices_dropped(self):
        df = _alpaca_style_frame("AAPL", extra_columns=False)
        plain = df.copy()
        plain.iloc[1, plain.columns.get_loc("low")] = 0.0
        plain.iloc[2, plain.columns.get_loc("close")] = -5.0
        out = normalize_bars_df(plain, "AAPL")
        assert len(out) == 2
        assert (out[["open", "high", "low", "close"]] > 0).all().all()

    def test_non_finite_and_internally_inconsistent_bars_are_dropped(self):
        idx = pd.date_range("2026-08-10", periods=3, freq="min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, float("inf"), 99.0],
                "low": [99.0, 99.0, 98.0],
                "close": [100.5, 100.0, 100.0],
                "volume": [1_000.0, 1_000.0, 1_000.0],
            },
            index=idx,
        )
        out = normalize_bars_df(df, "AAPL")
        assert list(out.index) == [idx[0]]

    def test_rows_with_nans_dropped(self):
        df = _alpaca_style_frame("AAPL", extra_columns=False)
        plain = df.copy()
        plain.iloc[0, plain.columns.get_loc("high")] = float("nan")
        out = normalize_bars_df(plain, "AAPL")
        assert len(out) == 3
        assert not out.isna().any().any()

    def test_zero_volume_row_kept(self):
        """Volume 0 is a legitimate quiet bar — only negative volume or bad
        prices disqualify a row."""
        df = _alpaca_style_frame("AAPL", extra_columns=False)
        plain = df.copy()
        plain.iloc[0, plain.columns.get_loc("volume")] = 0
        out = normalize_bars_df(plain, "AAPL")
        assert len(out) == 4
        assert out["volume"].iloc[0] == 0.0

    def test_unknown_symbol_returns_empty_frame(self):
        out = normalize_bars_df(_alpaca_style_frame("MSFT"), "AAPL")
        assert out.empty
        assert list(out.columns) == OHLCV
        assert out.index.name == "timestamp"
        assert out.index.tz is not None

    def test_missing_ohlcv_column_raises(self):
        df = _alpaca_style_frame("AAPL", extra_columns=False).drop(columns=["volume"])
        with pytest.raises(DataFeedError):
            normalize_bars_df(df, "AAPL")

    def test_empty_input_returns_empty_frame(self):
        out = normalize_bars_df(pd.DataFrame(), "AAPL")
        assert out.empty
        assert list(out.columns) == OHLCV

    def test_none_input_returns_empty_frame(self):
        out = normalize_bars_df(None, "AAPL")
        assert out.empty
        assert list(out.columns) == OHLCV

    def test_unnamed_multiindex_drops_first_level(self):
        """An unnamed two-level index (no 'symbol' level name) falls back to
        dropping the outer level instead of selecting by symbol."""
        ts = pd.date_range("2026-08-10 13:30", periods=3, freq="5min", tz="UTC")
        index = pd.MultiIndex.from_product([["whatever"], ts])
        df = pd.DataFrame(
            {
                "open": [100.0] * 3,
                "high": [101.0] * 3,
                "low": [99.0] * 3,
                "close": [100.5] * 3,
                "volume": [1000.0] * 3,
            },
            index=index,
        )
        out = normalize_bars_df(df, "AAPL")
        assert len(out) == 3
        assert list(out.index) == list(ts)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_429_is_rate_limit_not_permanent(self):
        exc = FakeAPIError("too many requests", status_code=429)
        assert is_rate_limit_error(exc)
        assert not is_permanent_error(exc)

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_deterministic_http_codes_are_permanent(self, code):
        exc = FakeAPIError("nope", status_code=code)
        assert is_permanent_error(exc)
        assert not is_rate_limit_error(exc)

    @pytest.mark.parametrize("code", [500, 502, 503])
    def test_server_errors_are_neither_permanent_nor_rate_limit(self, code):
        exc = FakeAPIError("server hiccup", status_code=code)
        assert not is_permanent_error(exc)
        assert not is_rate_limit_error(exc)

    def test_plain_exception_without_status_code_is_transient(self):
        exc = ConnectionError("connection reset")
        assert not is_permanent_error(exc)
        assert not is_rate_limit_error(exc)
        assert not is_subscription_error(exc)

    def test_non_int_status_code_ignored(self):
        exc = FakeAPIError("weird", status_code="401")
        assert not is_permanent_error(exc)

    def test_subscription_error_by_403_and_text(self):
        exc = FakeAPIError("your subscription tier lacks SIP", status_code=403)
        assert is_subscription_error(exc)

    def test_subscription_error_by_message_alone(self):
        exc = FakeAPIError("subscription does not permit querying recent SIP data")
        assert is_subscription_error(exc)

    def test_plain_403_without_subscription_text_not_subscription(self):
        exc = FakeAPIError("forbidden", status_code=403)
        assert not is_subscription_error(exc)
        assert is_permanent_error(exc)


# ---------------------------------------------------------------------------
# with_retries
# ---------------------------------------------------------------------------


class TestWithRetries:
    def test_succeeds_after_transient_failures(self):
        """Two transient failures then success: three attempts total, with
        exponential backoff sleeps of base_delay and 2*base_delay."""
        sleeps = SleepRecorder()
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("transient network blip")
            return "payload"

        result = with_retries(fn, max_attempts=3, base_delay=1.0, sleep_fn=sleeps)
        assert result == "payload"
        assert len(attempts) == 3
        assert sleeps.delays == [1.0, 2.0]

    def test_first_try_success_no_sleep(self):
        sleeps = SleepRecorder()
        assert with_retries(lambda: 42, sleep_fn=sleeps) == 42
        assert sleeps.delays == []

    def test_gives_up_after_max_attempts_with_data_feed_error(self):
        sleeps = SleepRecorder()
        attempts = []
        boom = ConnectionError("still down")

        def fn():
            attempts.append(1)
            raise boom

        with pytest.raises(DataFeedError) as excinfo:
            with_retries(fn, max_attempts=3, base_delay=1.0, sleep_fn=sleeps)
        assert len(attempts) == 3
        assert "3 attempts" in str(excinfo.value)
        assert excinfo.value.__cause__ is boom
        assert sleeps.delays == [1.0, 2.0]  # no sleep after the final attempt

    def test_permanent_401_raises_immediately_without_retry(self):
        sleeps = SleepRecorder()
        attempts = []
        exc = FakeAPIError("unauthorized", status_code=401)

        def fn():
            attempts.append(1)
            raise exc

        with pytest.raises(FakeAPIError) as excinfo:
            with_retries(fn, max_attempts=5, sleep_fn=sleeps)
        assert excinfo.value is exc
        assert len(attempts) == 1
        assert sleeps.delays == []

    def test_rate_limit_honors_retry_after_and_notifies(self):
        """A 429 carrying Retry-After: 7 must back off exactly 7.0s and report
        that delay through on_rate_limit before retrying."""
        sleeps = SleepRecorder()
        rate_limit_delays = []
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) == 1:
                raise FakeAPIError(
                    "rate limited",
                    status_code=429,
                    response=FakeResponse({"Retry-After": "7"}),
                )
            return "ok"

        result = with_retries(
            fn,
            max_attempts=3,
            base_delay=1.0,
            on_rate_limit=rate_limit_delays.append,
            sleep_fn=sleeps,
        )
        assert result == "ok"
        assert len(attempts) == 2
        assert sleeps.delays == [7.0]
        assert rate_limit_delays == [7.0]

    def test_rate_limit_without_retry_after_uses_backoff(self):
        sleeps = SleepRecorder()
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) == 1:
                raise FakeAPIError("rate limited", status_code=429)
            return "ok"

        assert with_retries(fn, max_attempts=3, base_delay=2.0, sleep_fn=sleeps) == "ok"
        assert sleeps.delays == [2.0]

    def test_persistent_rate_limit_exhausts_into_data_feed_error(self):
        sleeps = SleepRecorder()
        attempts = []

        def fn():
            attempts.append(1)
            raise FakeAPIError("rate limited", status_code=429)

        with pytest.raises(DataFeedError) as excinfo:
            with_retries(fn, max_attempts=2, base_delay=1.0, sleep_fn=sleeps)
        assert len(attempts) == 2
        assert "429" in str(excinfo.value)
        assert len(sleeps.delays) == 1  # only between attempts, not after the last

    def test_subscription_error_raises_immediately(self):
        """The Alpaca free-tier entitlement failure must convert to
        DataSubscriptionError on the first attempt, never retried."""
        sleeps = SleepRecorder()
        attempts = []
        original = FakeAPIError("subscription does not permit querying recent SIP data")

        def fn():
            attempts.append(1)
            raise original

        with pytest.raises(DataSubscriptionError) as excinfo:
            with_retries(fn, max_attempts=5, sleep_fn=sleeps)
        assert len(attempts) == 1
        assert sleeps.delays == []
        assert excinfo.value.__cause__ is original

    def test_subscription_error_is_a_data_feed_error(self):
        """Callers catching DataFeedError must also catch the subscription
        variant."""
        assert issubclass(DataSubscriptionError, DataFeedError)
