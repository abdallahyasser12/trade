from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from broker import BrokerError
from bybit_spot import (
    BYBIT_DEMO_REST_HOST,
    BybitInstrumentProvider,
    BybitSpotBroker,
    assert_spot_only_payload,
    decimal_ceil,
    decimal_floor,
    make_bybit_order_link_id,
)
from bybit_state import BybitStateStore
from crypto_venue import CryptoExecutionEnvironment, bybit_instrument


class FakeHTTP:
    endpoint = BYBIT_DEMO_REST_HOST

    def __init__(self):
        self.place_calls = []
        self.cancel_calls = []
        self.amend_calls = []
        self.order = None
        self.executions = []
        self.wallet = {
            "USDT": {"walletBalance": "1000", "locked": "0", "spotBorrow": "0"},
            "BTC": {"walletBalance": "0", "locked": "0", "spotBorrow": "0"},
        }
        self.server_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.raise_after_accept = False
        self.reject_code = 0
        self.min_order_amt = "5"
        self.base_precision = "0.000001"

    def get_server_time(self):
        return {"retCode": 0, "time": self.server_ms, "result": {}}

    def get_wallet_balance(self, **_kwargs):
        if self.reject_code:
            return {"retCode": self.reject_code, "retMsg": "bad credentials", "result": {}}
        coins = []
        for coin, values in self.wallet.items():
            coins.append({"coin": coin, **values})
        return {"retCode": 0, "result": {"list": [{"coin": coins}]}}

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
                            "basePrecision": self.base_precision,
                            "minOrderQty": "0.000001",
                            "minOrderAmt": self.min_order_amt,
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
        return {"retCode": 0, "result": {"list": list(self.executions)}}

    def place_order(self, **payload):
        self.place_calls.append(dict(payload))
        self.order = {
            "orderId": "order-1",
            "orderLinkId": payload["orderLinkId"],
            "symbol": payload["symbol"],
            "side": payload["side"],
            "qty": payload["qty"],
            "cumExecQty": "0",
            "avgPrice": "",
            "orderStatus": "New",
            "createdTime": str(self.server_ms),
            "updatedTime": str(self.server_ms),
            "triggerPrice": payload.get("triggerPrice", ""),
            "price": payload.get("price", ""),
        }
        if self.raise_after_accept:
            raise TimeoutError("socket timed out after venue accepted order")
        return {"retCode": 0, "result": {"orderId": "order-1", "orderLinkId": payload["orderLinkId"]}}

    def get_open_orders(self, **kwargs):
        if self.order and (
            not kwargs.get("orderLinkId") or kwargs["orderLinkId"] == self.order["orderLinkId"]
        ):
            return {"retCode": 0, "result": {"list": [self.order]}}
        return {"retCode": 0, "result": {"list": []}}

    def get_order_history(self, **kwargs):
        return self.get_open_orders(**kwargs)

    def cancel_order(self, **payload):
        self.cancel_calls.append(dict(payload))
        if self.order:
            self.order["orderStatus"] = "Cancelled"
        return {"retCode": 0, "result": {"orderId": payload.get("orderId")}}

    def amend_order(self, **payload):
        self.amend_calls.append(dict(payload))
        if self.order:
            self.order["qty"] = payload.get("qty", self.order["qty"])
            self.order["triggerPrice"] = payload.get("triggerPrice", self.order.get("triggerPrice", ""))
            self.order["price"] = payload.get("price", self.order.get("price", ""))
        return {"retCode": 0, "result": {"orderId": "order-1"}}


def broker(tmp_path, http=None, **kwargs):
    state = BybitStateStore(str(tmp_path / "venue.db"))
    instance = BybitSpotBroker(
        "demo-key",
        "demo-secret",
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        state=state,
        strategy_capital_budget_usdt=100,
        permitted_symbols=("BTC/USD",),
        http_client=http or FakeHTTP(),
        **kwargs,
    )
    return instance, state


def test_symbol_mapping_is_explicit():
    item = bybit_instrument("BTC/USD")
    assert item.base_asset == "BTC"
    assert item.quote_asset == "USDT"
    assert item.canonical_symbol == "BTC/USDT"
    assert item.venue_symbol == "BTCUSDT"
    assert item.strategy_symbol == "BTC/USD"
    with pytest.raises(ValueError):
        bybit_instrument("DOGE/USD")


def test_rounding_is_increment_aware():
    assert decimal_floor(1.239, 0.01) == pytest.approx(1.23)
    assert decimal_ceil(1.231, 0.01) == pytest.approx(1.24)


def test_order_link_id_is_deterministic_bounded_and_purpose_specific():
    entry = make_bybit_order_link_id("trendbot-session-BTCUSD-entry-1", "entry")
    same = make_bybit_order_link_id("trendbot-session-BTCUSD-entry-1", "entry")
    stop = make_bybit_order_link_id("trendbot-session-BTCUSD-entry-1", "stop")
    retry = make_bybit_order_link_id("trendbot-session-BTCUSD-entry-1", "entry", attempt=2)
    assert entry == same
    assert entry.startswith("tb-e-")
    assert len(entry) <= 36
    assert len(stop) <= 36
    assert len(retry) <= 36
    assert len({entry, stop, retry}) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "linear", "isLeverage": 0, "side": "Buy"},
        {"category": "spot", "isLeverage": 1, "side": "Buy"},
        {"category": "spot", "isLeverage": 0, "side": "Sell", "reduceOnly": True},
    ],
)
def test_spot_only_invariant_fails_before_network(payload):
    with pytest.raises(BrokerError):
        assert_spot_only_payload(payload)


def test_live_requires_two_explicit_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_BYBIT_LIVE_TRADING", raising=False)
    with pytest.raises(BrokerError):
        BybitSpotBroker(
            "live-key",
            "live-secret",
            environment=CryptoExecutionEnvironment.BYBIT_LIVE,
            state=BybitStateStore(str(tmp_path / "live.db")),
            strategy_capital_budget_usdt=100,
            permitted_symbols=("BTC/USD",),
            http_client=FakeHTTP(),
            allow_live=False,
        )
    monkeypatch.setenv("ENABLE_BYBIT_LIVE_TRADING", "true")
    with pytest.raises(BrokerError):
        BybitSpotBroker(
            "live-key",
            "live-secret",
            environment=CryptoExecutionEnvironment.BYBIT_LIVE,
            state=BybitStateStore(str(tmp_path / "live2.db")),
            strategy_capital_budget_usdt=100,
            permitted_symbols=("BTC/USD",),
            http_client=FakeHTTP(),
            allow_live=False,
        )


def test_manual_wallet_holdings_are_not_adopted(tmp_path):
    http = FakeHTTP()
    http.wallet["BTC"]["walletBalance"] = "2"
    instance, state = broker(tmp_path, http)
    try:
        assert instance.get_position("BTC/USD") is None
        assert instance.bot_owned_quantity("BTC/USD") == 0
    finally:
        instance.close()
        state.close()


def test_unowned_sell_is_refused_even_when_wallet_has_btc(tmp_path):
    http = FakeHTTP()
    http.wallet["BTC"]["walletBalance"] = "2"
    instance, state = broker(tmp_path, http)
    try:
        with pytest.raises(BrokerError, match="safe bot-owned/available bound"):
            instance.submit_crypto_market_order(
                "BTC/USD",
                0.1,
                "sell",
                "trendbot-12345678-BTCUSD-exit-1",
                trade_increment=0.000001,
                min_order_size=0.000001,
                reference_price=100,
            )
        assert http.place_calls == []
    finally:
        instance.close()
        state.close()


def test_buy_payload_is_spot_and_unleveraged(tmp_path):
    http = FakeHTTP()
    instance, state = broker(tmp_path, http)
    try:
        instance.submit_crypto_market_order(
            "BTC/USD",
            0.1,
            "buy",
            "trendbot-12345678-BTCUSD-entry-1",
            trade_increment=0.000001,
            min_order_size=0.000001,
            reference_price=100,
        )
        assert len(http.place_calls) == 1
        payload = http.place_calls[0]
        assert payload["category"] == "spot"
        assert payload["isLeverage"] == 0
        assert payload["symbol"] == "BTCUSDT"
        assert payload["marketUnit"] == "baseCoin"
        assert "reduceOnly" not in payload
    finally:
        instance.close()
        state.close()


def test_timeout_after_acceptance_is_reconciled_without_second_submit(tmp_path):
    http = FakeHTTP()
    http.raise_after_accept = True
    instance, state = broker(tmp_path, http)
    try:
        order = instance.submit_crypto_market_order(
            "BTC/USD",
            0.1,
            "buy",
            "trendbot-12345678-BTCUSD-entry-1",
            trade_increment=0.000001,
            min_order_size=0.000001,
            reference_price=100,
        )
        assert order.broker_order_id == "order-1"
        assert len(http.place_calls) == 1
        assert state.order_by_intent("trendbot-12345678-BTCUSD-entry-1") is not None
    finally:
        instance.close()
        state.close()


def test_bot_owned_inventory_is_execution_idempotent_and_wallet_deficit_fails(tmp_path):
    http = FakeHTTP()
    instance, state = broker(tmp_path, http)
    try:
        internal = "trendbot-12345678-BTCUSD-entry-1"
        link = make_bybit_order_link_id(internal, "entry")
        state.record_order_intent(
            internal_intent_id=internal,
            order_link_id=link,
            strategy_symbol="BTC/USD",
            venue_symbol="BTCUSDT",
            purpose="entry",
            attempt=1,
            requested_qty=0.1,
        )
        http.executions = [
            {
                "execId": "exec-1",
                "orderId": "order-1",
                "orderLinkId": link,
                "side": "Buy",
                "execQty": "0.1",
                "execPrice": "100",
                "execFee": "0",
                "feeCurrency": "USDT",
                "execTime": str(http.server_ms),
            }
        ]
        http.wallet["BTC"]["walletBalance"] = "0.1"
        assert instance.sync_bot_executions("BTC/USD") == 1
        assert instance.sync_bot_executions("BTC/USD") == 0
        assert instance.bot_owned_quantity("BTC/USD") == pytest.approx(0.1)
        http.wallet["BTC"]["walletBalance"] = "0.05"
        with pytest.raises(BrokerError, match="wallet deficit"):
            instance.get_position("BTC/USD")
    finally:
        instance.close()
        state.close()


def test_dynamic_metadata_refresh_observes_precision_and_minimum_changes():
    http = FakeHTTP()
    provider = BybitInstrumentProvider(http, cache_ttl_seconds=999)
    first = provider.get_instrument_metadata("BTC/USD")
    http.base_precision = "0.001"
    http.min_order_amt = "25"
    cached = provider.get_instrument_metadata("BTC/USD")
    changed = provider.get_instrument_metadata("BTC/USD", force_refresh=True)
    assert cached.quantity_step == first.quantity_step
    assert changed.quantity_step == pytest.approx(0.001)
    assert changed.min_order_amount == pytest.approx(25)


def test_clock_skew_fails_closed(tmp_path):
    http = FakeHTTP()
    http.server_ms += 30_000
    instance, state = broker(tmp_path, http, max_clock_skew_ms=1000)
    try:
        with pytest.raises(BrokerError, match="clock"):
            instance.verify_environment()
    finally:
        instance.close()
        state.close()
