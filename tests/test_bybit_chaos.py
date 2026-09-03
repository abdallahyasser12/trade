from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from broker import BrokerError
from bybit_market_data import BybitPrivateWebSocketManager, BybitPublicWebSocketManager
from bybit_spot import BYBIT_DEMO_REST_HOST, BybitSpotBroker
from bybit_state import BybitStateStore
from crypto_venue import CryptoExecutionEnvironment


class ChaosHTTP:
    endpoint = BYBIT_DEMO_REST_HOST

    def __init__(self) -> None:
        self.place_calls = []
        self.server_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.wallet_ret_code = 0
        self.wallet_ret_msg = "OK"
        self.place_error: Exception | None = None

    def get_server_time(self):
        return {"retCode": 0, "time": self.server_ms, "result": {}}

    def get_wallet_balance(self, **_kwargs):
        if self.wallet_ret_code:
            return {
                "retCode": self.wallet_ret_code,
                "retMsg": self.wallet_ret_msg,
                "result": {},
            }
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "coin": [
                            {"coin": "USDT", "walletBalance": "1000", "locked": "0", "spotBorrow": "0"},
                            {"coin": "BTC", "walletBalance": "0", "locked": "0", "spotBorrow": "0"},
                        ]
                    }
                ]
            },
        }

    def get_instruments_info(self, **_kwargs):
        return {
            "retCode": 0,
            "time": self.server_ms,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "Trading",
                        "priceFilter": {"tickSize": "0.01"},
                        "lotSizeFilter": {
                            "basePrecision": "0.000001",
                            "minOrderQty": "0.000001",
                            "minOrderAmt": "5",
                            "maxOrderQty": "100",
                            "maxMarketOrderQty": "10",
                        },
                    }
                ]
            },
        }

    def get_tickers(self, **_kwargs):
        return {"retCode": 0, "result": {"list": [{"lastPrice": "100"}]}}

    def get_executions(self, **_kwargs):
        return {"retCode": 0, "result": {"list": []}}

    def get_open_orders(self, **_kwargs):
        return {"retCode": 0, "result": {"list": []}}

    def get_order_history(self, **_kwargs):
        return {"retCode": 0, "result": {"list": []}}

    def place_order(self, **payload):
        self.place_calls.append(dict(payload))
        if self.place_error is not None:
            raise self.place_error
        return {
            "retCode": 0,
            "result": {"orderId": "chaos-order", "orderLinkId": payload["orderLinkId"]},
        }

    def get_orderbook(self, **_kwargs):
        return {
            "retCode": 0,
            "result": {
                "b": [["99.9", "1"]],
                "a": [["100.1", "1"]],
                "u": 1,
                "seq": 1,
                "ts": self.server_ms,
            },
        }


def _broker(tmp_path, http: ChaosHTTP):
    state = BybitStateStore(str(tmp_path / "venue.db"))
    broker = BybitSpotBroker(
        "demo-key",
        "demo-secret",
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        state=state,
        strategy_capital_budget_usdt=100,
        permitted_symbols=("BTC/USD",),
        http_client=http,
        allow_live=False,
    )
    return broker, state


def _submit_buy(broker):
    return broker.submit_crypto_market_order(
        "BTC/USD",
        0.1,
        "buy",
        "trendbot-12345678-BTCUSD-entry-1",
        trade_increment=0.000001,
        min_order_size=0.000001,
        reference_price=100,
    )


def test_bad_demo_credentials_fail_environment_verification(tmp_path):
    http = ChaosHTTP()
    http.wallet_ret_code = 10003
    http.wallet_ret_msg = "API key is invalid"
    broker, state = _broker(tmp_path, http)
    try:
        with pytest.raises(BrokerError, match="wallet balance"):
            broker.verify_environment()
        assert broker.broker_health()["rest_failures"] >= 1
    finally:
        broker.close()
        state.close()


def test_rate_limit_response_is_counted_and_fails_closed(tmp_path):
    http = ChaosHTTP()
    http.wallet_ret_code = 10006
    http.wallet_ret_msg = "Too many visits. Exceeded the API Rate Limit."
    broker, state = _broker(tmp_path, http)
    try:
        with pytest.raises(BrokerError):
            broker.verify_environment()
        health = broker.broker_health()
        assert health["rate_limit_events"] >= 1
        assert health["rest_failures"] >= 1
    finally:
        broker.close()
        state.close()


def test_network_failure_before_order_acceptance_never_blind_retries_economic_submit(tmp_path):
    http = ChaosHTTP()
    http.place_error = ConnectionError("provider unavailable before acceptance")
    broker, state = _broker(tmp_path, http)
    try:
        with pytest.raises(BrokerError):
            _submit_buy(broker)
        # Economic submission is one-shot. The deterministic intent remains for
        # reconciliation, but there is never a second blind place_order call.
        assert len(http.place_calls) == 1
        assert state.order_by_intent("trendbot-12345678-BTCUSD-entry-1") is not None
    finally:
        broker.close()
        state.close()


def test_database_failure_before_persisted_intent_prevents_network_order(tmp_path, monkeypatch):
    http = ChaosHTTP()
    broker, state = _broker(tmp_path, http)

    def fail_persistence(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(state, "record_order_intent", fail_persistence)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            _submit_buy(broker)
        assert http.place_calls == []
    finally:
        broker.close()
        state.close()


class FakeSocket:
    def __init__(self, **_kwargs) -> None:
        self.exited = False
        self.callbacks = []

    def orderbook_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def trade_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def kline_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def order_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def execution_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def wallet_stream(self, **kwargs):
        self.callbacks.append(kwargs["callback"])

    def exit(self):
        self.exited = True


def test_public_websocket_shutdown_does_not_leave_socket_owned():
    http = ChaosHTTP()
    created = []

    def factory(**kwargs):
        socket = FakeSocket(**kwargs)
        created.append(socket)
        return socket

    manager = BybitPublicWebSocketManager(
        http,
        symbols=("BTC/USD",),
        timeframe="5Min",
        websocket_factory=factory,
    )
    manager._connect_locked()
    assert manager.health()["public_ws_up"] is True
    manager.stop()
    assert created[0].exited is True
    assert manager.health()["public_ws_up"] is False


def test_private_demo_websocket_shutdown_does_not_leave_socket_owned():
    created = []

    def factory(**kwargs):
        assert kwargs["demo"] is True
        assert kwargs["testnet"] is False
        socket = FakeSocket(**kwargs)
        created.append(socket)
        return socket

    manager = BybitPrivateWebSocketManager(
        "key",
        "secret",
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        websocket_factory=factory,
    )
    manager._connect_locked()
    assert manager.health()["private_ws_up"] is True
    manager.stop()
    assert created[0].exited is True
    assert manager.health()["private_ws_up"] is False
