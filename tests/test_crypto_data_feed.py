"""24/7 completed-bar, stale-data, and execution-quote tests."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from crypto_data_feed import (
    AlpacaCryptoDataFeed,
    is_crypto_data_stale,
    quote_is_safe,
)
from crypto_models import CryptoQuote


UTC = timezone.utc


def test_crypto_weekend_data_does_not_receive_stock_market_grace():
    bar = datetime(2026, 8, 14, 0, tzinfo=UTC)
    now = bar + timedelta(days=4)
    assert is_crypto_data_stale(bar, "1Day", now, staleness_bars=2) is True


def test_quote_guard_rejects_stale_wide_or_crossed_quotes():
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    safe = CryptoQuote("BTC/USD", now, 99.9, 100.1)
    assert quote_is_safe(safe, now, max_age_seconds=30, max_spread_bps=30)[0]
    stale = CryptoQuote("BTC/USD", now - timedelta(seconds=31), 99.9, 100.1)
    assert not quote_is_safe(stale, now, max_age_seconds=30, max_spread_bps=30)[0]
    wide = CryptoQuote("BTC/USD", now, 90, 110)
    assert not quote_is_safe(wide, now, max_age_seconds=30, max_spread_bps=30)[0]
    crossed = CryptoQuote("BTC/USD", now, 101, 100)
    assert not quote_is_safe(crossed, now, max_age_seconds=30, max_spread_bps=30)[0]


def test_recent_crypto_bars_drop_forming_bar_and_keep_zero_volume():
    now = datetime(2026, 8, 14, 12, 30, 10, tzinfo=UTC)
    index = pd.date_range("2026-08-14 10:00", periods=3, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [10, 0, 12],
        },
        index=index,
    )
    feed = AlpacaCryptoDataFeed.__new__(AlpacaCryptoDataFeed)
    feed._interrupt_event = None
    feed._rate_limit_events = 0
    feed.get_bars = lambda *args, **kwargs: raw
    completed = feed.get_recent_bars("BTC/USD", "1Hour", 10, now=now)
    assert list(completed.index) == list(index[:2])
    assert completed.iloc[-1]["volume"] == 0


def test_latest_quote_adapter_uses_real_sdk_shape_without_network():
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    client = SimpleNamespace(
        get_crypto_latest_quote=lambda request, feed: {
            "BTC/USD": SimpleNamespace(
                timestamp=now,
                bid_price="99.5",
                ask_price="100.5",
                bid_size="2",
                ask_size="3",
            )
        }
    )
    feed = AlpacaCryptoDataFeed.__new__(AlpacaCryptoDataFeed)
    feed._client = client
    feed._interrupt_event = None
    feed._rate_limit_events = 0
    quote = feed.get_latest_quotes(["BTC/USD"])["BTC/USD"]
    assert quote.midpoint == 100
    assert quote.bid_size == 2
