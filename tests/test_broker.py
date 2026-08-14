"""Broker boundary tests using alpaca-py's real request models, without I/O."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from broker import (
    AlpacaPaperBroker,
    BrokerError,
    make_client_order_id,
    normalize_sell_stop_price,
)
from models import OrderRecord


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


class CapturingClient:
    def __init__(self):
        self.request = None

    def submit_order(self, request):
        self.request = request
        return SimpleNamespace(
            id="order-1",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            qty=request.qty,
            filled_qty="0",
            filled_avg_price=None,
            stop_price=getattr(request, "stop_price", None),
            status="new",
            side=request.side,
            submitted_at=NOW,
            updated_at=NOW,
        )


def test_trading_client_is_hardwired_to_paper_endpoint():
    broker = AlpacaPaperBroker("dummy-key", "dummy-secret")
    assert "paper-api.alpaca.markets" in broker._client._base_url.value


def test_paper_verification_understands_alpaca_base_url_enum(monkeypatch):
    broker = AlpacaPaperBroker("dummy-key", "dummy-secret")
    account = SimpleNamespace(status="ACTIVE", trading_blocked=False)
    monkeypatch.setattr(broker, "get_account", lambda: account)
    assert broker.verify_paper_account() is account


def test_client_order_ids_are_deterministic_retry_scoped_and_bounded():
    first = make_client_order_id("1234567890", "exit", NOW, symbol="ABCDEFGHIJ")
    retry = make_client_order_id(
        "1234567890", "exit", NOW, symbol="ABCDEFGHIJ", attempt=2
    )
    assert first == make_client_order_id("1234567890", "exit", NOW, symbol="ABCDEFGHIJ")
    assert first != retry
    assert len(retry) <= 48
    assert retry.split("-")[-2] == "exit"
    assert len(
        make_client_order_id(
            "1234567890", "exit", NOW, symbol="ABCDEFGHIJ", attempt=1_000
        )
    ) <= 48


@pytest.mark.parametrize(
    "raw, expected",
    [(95.123, 95.13), (0.987654, 0.9877), (0.00001, 0.0001)],
)
def test_sell_stop_rounding_is_valid_and_never_moves_down(raw, expected):
    normalized = normalize_sell_stop_price(raw)
    assert normalized == expected
    assert normalized >= raw


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_sell_stop_rejects_non_finite_or_non_positive_prices(price):
    with pytest.raises(BrokerError):
        normalize_sell_stop_price(price)


def test_submit_market_order_rejects_unknown_side():
    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker._client = CapturingClient()
    with pytest.raises(BrokerError, match="Unsupported order side"):
        broker.submit_market_order("SPY", 1, "short", "trendbot-12345678-entry-1")
    assert broker._client.request is None


def test_submit_stop_uses_real_alpaca_request_model_and_four_decimals_below_one():
    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker._client = CapturingClient()
    record = broker.submit_stop_order(
        "PENNY", 7, 0.987654, "trendbot-12345678-PENNY-stop-1"
    )
    assert isinstance(record, OrderRecord)
    assert float(broker._client.request.stop_price) == pytest.approx(0.9877)


@pytest.mark.parametrize(
    "status, terminal, open_",
    [
        ("done_for_day", False, True),
        ("replaced", True, False),
        ("calculated", False, True),
        ("partially_filled", False, True),
    ],
)
def test_order_status_classification_matches_alpaca_lifecycle(status, terminal, open_):
    order = OrderRecord("b", "c", "SPY", "entry", "buy", 1, 0, status)
    assert order.is_terminal is terminal
    assert order.is_open is open_
