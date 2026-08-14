"""Crypto stop-limit persistence and fractional position serialization."""

from datetime import datetime, timezone

from models import OrderRecord
from state_store import StateStore


def test_stop_limit_and_fractional_position_round_trip(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    order = OrderRecord(
        "broker-1",
        "trendbot-session-BTCUSD-stop-1",
        "BTC/USD",
        "stop",
        "sell",
        0.123456789,
        0.02,
        "partially_filled",
        stop_price=60_000.0,
        limit_price=59_700.0,
        submitted_at=now,
        updated_at=now,
    )
    store.record_order(order, "session")
    restored = store.get_order_by_client_id(order.client_order_id)
    assert restored.requested_qty == order.requested_qty
    assert restored.filled_qty == order.filled_qty
    assert restored.limit_price == order.limit_price
    payload = {"qty": 0.103456789, "entry_price": 62_000.0}
    store.set_bot_position("BTC/USD", payload)
    assert store.get_bot_position("BTC/USD") == payload
    assert store.bot_net_filled_qty("BTC/USD") == order.filled_qty * -1
