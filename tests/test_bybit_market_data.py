from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bybit_burnin import BybitBurnInConfig
from bybit_management import PolicyManagedStrategy
from bybit_market_data import (
    BybitPrivateWebSocketManager,
    BybitPublicWebSocketManager,
    _OrderBook,
)
from crypto_venue import CryptoExecutionEnvironment
from data_feed import DataFeedError


class DummyStrategy:
    marker = "delegate"

    def ratchet_stop(self, current_stop, highest, close, atr):
        return current_stop + 10, highest + 10


class BookHTTP:
    def __init__(self):
        self.resyncs = 0

    def get_orderbook(self, **_kwargs):
        self.resyncs += 1
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        return {
            "retCode": 0,
            "result": {
                "b": [["100", "4"]],
                "a": [["101", "5"]],
                "u": 50,
                "seq": 60,
                "ts": now,
            },
        }


def message(kind, u, seq, *, bid="100", ask="101", ts=None):
    return {
        "type": kind,
        "ts": ts or int(datetime.now(timezone.utc).timestamp() * 1000),
        "data": {"s": "BTCUSDT", "u": u, "seq": seq, "b": [[bid, "2"]], "a": [[ask, "3"]]},
    }


def test_order_book_tolerates_duplicates_and_detects_gap():
    book = _OrderBook()
    assert book.ingest(message("snapshot", 10, 20)) is True
    assert book.ingest(message("delta", 10, 20)) is False
    assert book.gap_detected is False
    assert book.ingest(message("delta", 12, 22)) is True
    assert book.gap_detected is True


def test_gap_forces_rest_snapshot_before_market_context():
    http = BookHTTP()
    manager = BybitPublicWebSocketManager(http, symbols=("BTC/USD",), timeframe="5Min")
    manager._books["BTC/USD"].ingest(message("snapshot", 10, 20))
    manager._books["BTC/USD"].ingest(message("delta", 12, 22))
    snap = manager.snapshot("BTC/USD")
    assert http.resyncs == 1
    assert snap.bid == 100
    assert snap.ask == 101
    assert snap.gap_detected is False


def test_stale_order_book_fails_closed():
    http = BookHTTP()
    manager = BybitPublicWebSocketManager(http, symbols=("BTC/USD",), timeframe="5Min")
    old = int((datetime.now(timezone.utc) - timedelta(seconds=30)).timestamp() * 1000)
    manager._books["BTC/USD"].ingest(message("snapshot", 10, 20, ts=old))
    with pytest.raises(DataFeedError, match="stale"):
        manager.snapshot("BTC/USD", max_age_seconds=1.0)


def test_crossed_book_fails_closed():
    book = _OrderBook()
    book.ingest(message("snapshot", 10, 20, bid="102", ask="101"))
    with pytest.raises(DataFeedError, match="crossed"):
        book.best()


def test_private_execution_duplicate_forwarded_once():
    seen = []
    manager = BybitPrivateWebSocketManager(
        "key",
        "secret",
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        on_execution=lambda payload: seen.extend(payload["data"]),
    )
    payload = {"data": [{"execId": "x1", "execQty": "1"}]}
    manager._handle_execution(payload)
    manager._handle_execution(payload)
    assert len(seen) == 1
    assert manager.health()["private_ws_duplicates"] == 1


def test_policy_managed_strategy_disables_only_legacy_ratchet():
    wrapped = PolicyManagedStrategy(DummyStrategy())
    assert wrapped.marker == "delegate"
    stop, high = wrapped.ratchet_stop(95, 101, 100, 2)
    assert stop == 95
    assert high == 101


def test_burnin_hash_changes_when_parameters_change():
    a = BybitBurnInConfig(hours=72)
    b = BybitBurnInConfig(hours=120)
    assert a.config_hash == BybitBurnInConfig(hours=72).config_hash
    assert a.config_hash != b.config_hash
    assert a.max_entries_rolling_24h == 5
