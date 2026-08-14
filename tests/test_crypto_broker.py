"""Exact environment, fractional request, and live notional boundary tests."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from broker import (
    AlpacaBroker,
    AlpacaLiveBroker,
    BrokerError,
    make_client_order_id,
)
from config import AccountMode


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


class CaptureClient:
    def __init__(self):
        self.request = None

    def submit_order(self, request):
        self.request = request
        return SimpleNamespace(
            id="crypto-order",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            qty=request.qty,
            filled_qty="0",
            filled_avg_price=None,
            stop_price=getattr(request, "stop_price", None),
            limit_price=getattr(request, "limit_price", None),
            status="new",
            side=request.side,
            submitted_at=NOW,
            updated_at=NOW,
        )


def bare_broker(mode=AccountMode.PAPER, cap=1_000.0):
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._account_mode = mode
    broker._max_live_order_notional = cap
    broker._client = CaptureClient()
    return broker


def test_live_broker_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    with pytest.raises(BrokerError, match="Live trading is disabled"):
        AlpacaLiveBroker("key", "secret")


def test_live_broker_uses_exact_official_endpoint_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = AlpacaLiveBroker("key", "secret")
    assert broker._client._base_url.value == "https://api.alpaca.markets"
    broker.get_account = lambda: SimpleNamespace(
        status="ACTIVE", trading_blocked=False, can_trade=True
    )
    assert broker.verify_account().status == "ACTIVE"


def test_crypto_client_id_sanitizes_pair_and_remains_deterministic():
    value = make_client_order_id("1234567890", "entry", NOW, "BTC/USD")
    assert "BTCUSD" in value
    assert "/" not in value
    assert len(value) <= 48
    assert value == make_client_order_id("1234567890", "entry", NOW, "BTC/USD")


def test_crypto_market_order_uses_fractional_gtc_request():
    broker = bare_broker()
    record = broker.submit_crypto_market_order(
        "BTC/USD",
        0.1234567899,
        "buy",
        "trendbot-12345678-BTCUSD-entry-1",
        trade_increment=1e-9,
        min_order_size=1e-6,
        reference_price=100.0,
    )
    assert record.requested_qty == pytest.approx(0.123456789)
    assert broker._client.request.time_in_force.value == "gtc"
    assert broker._client.request.type.value == "market"


def test_live_crypto_buy_rechecks_hard_notional_cap_but_exit_is_allowed():
    broker = bare_broker(AccountMode.LIVE, cap=100.0)
    with pytest.raises(BrokerError, match="exceeds hard cap"):
        broker.submit_crypto_market_order(
            "BTC/USD",
            2.0,
            "buy",
            "trendbot-12345678-BTCUSD-entry-1",
            trade_increment=0.001,
            min_order_size=0.001,
            reference_price=100.0,
        )
    broker.submit_crypto_market_order(
        "BTC/USD",
        2.0,
        "sell",
        "trendbot-12345678-BTCUSD-exit-1",
        trade_increment=0.001,
        min_order_size=0.001,
        reference_price=100.0,
    )


def test_crypto_protective_order_is_sell_stop_limit_with_limit_below_stop():
    broker = bare_broker()
    record = broker.submit_crypto_stop_limit_order(
        "ETH/USD",
        0.5,
        95.0,
        94.5,
        "trendbot-12345678-ETHUSD-stop-1",
        trade_increment=1e-6,
        min_order_size=1e-5,
        price_increment=0.01,
    )
    request = broker._client.request
    assert request.side.value == "sell"
    assert request.type.value == "stop_limit"
    assert float(request.limit_price) < float(request.stop_price)
    assert record.limit_price == pytest.approx(94.5)
