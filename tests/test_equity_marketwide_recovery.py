import json
from datetime import datetime, timezone
from types import SimpleNamespace

from equity_marketwide_deployment import ManifestLockedMarketWideEquityTradeEngine
from equity_marketwide_engine import EQUITY_MARKETWIDE_SESSION_SYMBOL
from models import OrderRecord

NOW = datetime(2026, 8, 18, 14, 30, tzinfo=timezone.utc)


class FakeLocalOrder:
    def __init__(self, client_order_id, *, terminal=False):
        self.client_order_id = client_order_id
        self.is_terminal = terminal


class ReconcileStore:
    def __init__(self):
        self.orders = {
            "current": FakeLocalOrder("current"),
            "prior": FakeLocalOrder("prior"),
            "other": FakeLocalOrder("other"),
            "done": FakeLocalOrder("done", terminal=True),
        }

    def get_orders(self, limit=5000):
        return [
            {"client_order_id": "current", "session_id": "session-now"},
            {"client_order_id": "prior", "session_id": "session-old"},
            {"client_order_id": "other", "session_id": "other-engine"},
            {"client_order_id": "done", "session_id": "session-old"},
        ]

    def get_order_by_client_id(self, client_id):
        return self.orders.get(client_id)

    def get_session(self, session_id):
        if session_id == "session-old":
            return {"symbol": EQUITY_MARKETWIDE_SESSION_SYMBOL}
        if session_id == "other-engine":
            return {"symbol": "SPY"}
        return None


class ReconcileBroker:
    def get_order_by_client_id(self, client_id):
        return SimpleNamespace(client_order_id=client_id)


def test_restart_reconciles_current_and_prior_marketwide_orders_only():
    engine = object.__new__(ManifestLockedMarketWideEquityTradeEngine)
    engine.session_id = "session-now"
    engine.store = ReconcileStore()
    engine.broker = ReconcileBroker()
    synced = []
    engine._sync_order = lambda broker_order, local, now: synced.append(
        (broker_order.client_order_id, local.client_order_id)
    )

    engine._reconcile_orders(NOW)

    assert synced == [("current", "current"), ("prior", "prior")]


class GapStore:
    def __init__(self):
        self.positions = {}
        self.deleted = []
        self.values = {
            "equity_marketwide_pending_entry:AAPL": json.dumps(
                {
                    "strategy_id": "opening_range_breakout",
                    "strategy_version": "1.0.0",
                    "stop_price": 100.0,
                    "entry_reason": "opening_range_breakout",
                }
            )
        }

    def get_kv(self, key):
        return self.values.get(key)

    def set_bot_position(self, symbol, record):
        self.positions[symbol] = dict(record)

    def get_bot_position(self, symbol):
        return self.positions.get(symbol)

    def delete_kv(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)


class GapBroker:
    def get_position(self, symbol):
        assert symbol == "AAPL"
        return SimpleNamespace(qty=5.0, avg_entry_price=95.0)


def test_gap_through_signal_stop_is_owned_then_immediately_flattened():
    engine = object.__new__(ManifestLockedMarketWideEquityTradeEngine)
    engine.store = GapStore()
    engine.broker = GapBroker()
    exits = []
    engine._initiate_exit = lambda *args: exits.append(args)

    order = OrderRecord(
        broker_order_id="broker-1",
        client_order_id="entry-1",
        symbol="AAPL",
        purpose="entry",
        side="buy",
        requested_qty=5.0,
        filled_qty=5.0,
        status="filled",
        fill_price=95.0,
        submitted_at=NOW,
        updated_at=NOW,
    )

    engine._adopt_entry_fill("AAPL", order, NOW)

    owned = engine.store.positions["AAPL"]
    assert owned["strategy_id"] == "opening_range_breakout"
    assert owned["market"] == "us_equity"
    assert owned["stop_price"] == 94.05
    assert owned["recovery_reason"] == "entry_filled_through_signal_stop"
    assert engine.store.deleted == ["equity_marketwide_pending_entry:AAPL"]
    assert len(exits) == 1
    assert exits[0][0] == "AAPL"
    assert exits[0][2] == "opening_range_breakout"
    assert exits[0][3] == "entry_filled_through_signal_stop"
