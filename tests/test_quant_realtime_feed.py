from datetime import datetime, timedelta, timezone

import pandas as pd

from quant.realtime_feed import CompletedBarCachingCryptoFeed


class Backend:
    def __init__(self):
        self.bar_calls = 0
        self.quote_calls = 0

    def get_recent_bars(self, symbol, timeframe, lookback_bars, **kwargs):
        self.bar_calls += 1
        index = pd.date_range("2026-08-18", periods=lookback_bars, freq="5min", tz="UTC")
        return pd.DataFrame(
            {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            index=index,
        )

    def get_latest_quotes(self, symbols):
        self.quote_calls += 1
        return {symbol: object() for symbol in symbols}


def test_repeated_fast_cycles_fetch_history_once_per_completed_bar():
    backend = Backend()
    feed = CompletedBarCachingCryptoFeed(backend)
    now = datetime(2026, 8, 18, 12, 3, tzinfo=timezone.utc)
    for seconds in (0, 5, 10, 30, 60):
        frame = feed.get_recent_bars("BTC/USD", "5Min", 100, now=now + timedelta(seconds=seconds), tolerance_seconds=2)
        assert len(frame) == 100
    assert backend.bar_calls == 1


def test_cache_rolls_exactly_when_next_bar_is_safely_complete():
    backend = Backend()
    feed = CompletedBarCachingCryptoFeed(backend)
    before = datetime(2026, 8, 18, 12, 4, 59, tzinfo=timezone.utc)
    after = datetime(2026, 8, 18, 12, 5, 3, tzinfo=timezone.utc)
    feed.get_recent_bars("BTC/USD", "5Min", 100, now=before, tolerance_seconds=2)
    feed.get_recent_bars("BTC/USD", "5Min", 100, now=after, tolerance_seconds=2)
    assert backend.bar_calls == 2


def test_quotes_are_never_cached():
    backend = Backend()
    feed = CompletedBarCachingCryptoFeed(backend)
    for _ in range(10):
        feed.get_latest_quotes(["BTC/USD", "ETH/USD"])
    assert backend.quote_calls == 10
