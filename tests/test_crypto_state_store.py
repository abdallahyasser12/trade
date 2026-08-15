"""Crypto stop-limit persistence and fractional position serialization."""

from datetime import datetime, timezone

import pytest

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


def test_schema_version_is_stamped_and_a_newer_database_is_refused(tmp_path):
    """The database defines what the bot believes it owns; never guess at it."""

    import sqlite3

    import pytest

    from state_store import SCHEMA_VERSION, StateStore

    path = tmp_path / "versioned.db"
    store = StateStore(str(path))
    conn = sqlite3.connect(str(path))
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    # Simulate a database written by a future build.
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    store.close()

    with pytest.raises(RuntimeError, match="newer build"):
        StateStore(str(path))


def test_execution_slippage_is_signed_against_the_trade_direction(tmp_path):
    """Positive must mean 'worse than reference' for a buy and a sell alike."""

    from state_store import StateStore

    store = StateStore(str(tmp_path / "exec.db"))
    # Buy filled above reference: adverse.
    store.record_execution(
        "s1", "cid-buy", "BTC/USD", "entry", "buy", 100.0, 101.0, 1.0, event_key="a"
    )
    # Sell filled below reference: also adverse.
    store.record_execution(
        "s1", "cid-sell", "BTC/USD", "exit", "sell", 100.0, 99.0, 1.0, event_key="b"
    )
    rows = {row["client_order_id"]: row for row in store.get_executions("s1")}
    assert rows["cid-buy"]["slippage_bps"] == pytest.approx(100.0)
    assert rows["cid-sell"]["slippage_bps"] == pytest.approx(100.0)

    summary = store.execution_quality("s1")
    assert summary["samples"] == 2
    assert summary["avg_slippage_bps"] == pytest.approx(100.0)
    store.close()


def test_execution_quality_reports_zero_samples_rather_than_a_fake_average(tmp_path):
    from state_store import StateStore

    store = StateStore(str(tmp_path / "empty.db"))
    summary = store.execution_quality()
    assert summary["samples"] == 0
    assert summary["avg_slippage_bps"] is None
    store.close()


def test_executions_are_idempotent_on_event_key(tmp_path):
    from state_store import StateStore

    store = StateStore(str(tmp_path / "idem.db"))
    first = store.record_execution(
        "s1", "cid", "BTC/USD", "entry", "buy", 100.0, 100.5, 1.0, event_key="same"
    )
    second = store.record_execution(
        "s1", "cid", "BTC/USD", "entry", "buy", 100.0, 100.5, 1.0, event_key="same"
    )
    assert first is True and second is False
    assert len(store.get_executions()) == 1
    store.close()
