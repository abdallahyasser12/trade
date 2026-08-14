"""Tests for state_store.StateStore.

All stores are backed by real SQLite files under pytest's tmp_path so the
WAL journal mode and restart-persistence behaviour are exercised for real.
No network, no Alpaca — everything here is synthetic and deterministic.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from models import OrderRecord
from state_store import StateStore

UTC = timezone.utc

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = StateStore(str(tmp_path / "state.db"))
    yield s
    s.close()


def _make_order(
    client_order_id: str = "tbot-entry-abc123",
    *,
    broker_order_id: str = "broker-1",
    status: str = "accepted",
    filled_qty: float = 0.0,
    fill_price=None,
    stop_price=None,
    submitted_at: datetime = T0,
    updated_at: datetime = T0,
) -> OrderRecord:
    return OrderRecord(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        symbol="AAPL",
        purpose="entry",
        side="buy",
        requested_qty=10.0,
        filled_qty=filled_qty,
        status=status,
        fill_price=fill_price,
        stop_price=stop_price,
        submitted_at=submitted_at,
        updated_at=updated_at,
    )


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


class TestSessions:
    def test_create_and_get_session_round_trip(self, store):
        """A created session comes back with every field intact and all
        timestamps timezone-aware (the store persists ISO-8601 UTC strings)."""
        stop_at = T0 + timedelta(hours=6)
        store.create_session(
            "sess-1", "AAPL", "5Min", T0, stop_at, 100_000.0, config_json='{"risk": 1}'
        )

        sess = store.get_session("sess-1")
        assert sess is not None
        assert sess["session_id"] == "sess-1"
        assert sess["symbol"] == "AAPL"
        assert sess["timeframe"] == "5Min"
        assert sess["status"] == "running"
        assert sess["starting_equity"] == 100_000.0
        assert sess["config_json"] == '{"risk": 1}'
        assert sess["halted"] is False
        assert sess["stop_reason"] is None
        assert sess["stopped_at"] is None

        assert sess["started_at"] == T0
        assert sess["started_at"].tzinfo is not None
        assert sess["started_at"].utcoffset() == timedelta(0)
        assert sess["intended_stop_at"] == stop_at
        assert sess["intended_stop_at"].tzinfo is not None

    def test_get_session_missing_returns_none(self, store):
        assert store.get_session("nope") is None
        assert store.get_latest_session() is None
        assert store.get_active_session() is None

    def test_end_session_sets_stopped_status_and_reason(self, store):
        store.create_session("sess-1", "AAPL", "5Min", T0, None, 100_000.0)
        stopped_at = T0 + timedelta(hours=2)
        store.end_session("sess-1", stopped_at, "USER_STOP")

        sess = store.get_session("sess-1")
        assert sess["status"] == "stopped"
        assert sess["stop_reason"] == "USER_STOP"
        assert sess["stopped_at"] == stopped_at
        assert sess["stopped_at"].tzinfo is not None

    def test_set_session_halted(self, store):
        store.create_session("sess-1", "AAPL", "5Min", T0, None, 100_000.0)
        store.set_session_halted("sess-1", "daily loss limit reached")

        sess = store.get_session("sess-1")
        assert sess["halted"] is True
        assert sess["halt_reason"] == "daily loss limit reached"

    def test_latest_and_active_session_selection(self, store):
        """get_latest_session picks the newest started_at regardless of
        status; get_active_session only considers status='running'."""
        store.create_session("old", "AAPL", "5Min", T0, None, 100_000.0)
        store.create_session("new", "MSFT", "1Min", T0 + timedelta(hours=1), None, 50_000.0)

        assert store.get_latest_session()["session_id"] == "new"
        assert store.get_active_session()["session_id"] == "new"

        store.end_session("new", T0 + timedelta(hours=2), "USER_STOP")
        # Newest overall is still "new", but the only running one is "old".
        assert store.get_latest_session()["session_id"] == "new"
        assert store.get_active_session()["session_id"] == "old"

        store.end_session("old", T0 + timedelta(hours=3), "RUNTIME_EXPIRED")
        assert store.get_active_session() is None


# ----------------------------------------------------------------------
# Processed bars
# ----------------------------------------------------------------------


class TestProcessedBars:
    def test_first_mark_true_then_false_even_across_sessions(self, store):
        """Dedup key is (symbol, timeframe, bar_timestamp) globally: a second
        engine session must not be able to act on the same bar again."""
        bar_ts = T0
        assert store.mark_bar_processed("AAPL", "5Min", bar_ts, "sess-1") is True
        assert store.mark_bar_processed("AAPL", "5Min", bar_ts, "sess-1") is False
        # Different session id, same bar -> still deduplicated.
        assert store.mark_bar_processed("AAPL", "5Min", bar_ts, "sess-2") is False

    def test_distinct_keys_are_independent(self, store):
        assert store.mark_bar_processed("AAPL", "5Min", T0, "s") is True
        assert store.mark_bar_processed("MSFT", "5Min", T0, "s") is True
        assert store.mark_bar_processed("AAPL", "1Min", T0, "s") is True
        assert store.mark_bar_processed("AAPL", "5Min", T0 + timedelta(minutes=5), "s") is True

    def test_is_bar_processed(self, store):
        assert store.is_bar_processed("AAPL", "5Min", T0) is False
        store.mark_bar_processed("AAPL", "5Min", T0, "sess-1")
        assert store.is_bar_processed("AAPL", "5Min", T0) is True
        assert store.is_bar_processed("AAPL", "1Min", T0) is False

    def test_get_processed_bars_ordering_and_limit(self, store):
        """get_processed_bars returns newest bar_timestamp first and honours
        the limit; the session_id filter narrows to one session's bars."""
        stamps = pd.date_range("2026-01-05 14:30", periods=6, freq="5min", tz="UTC")
        for i, ts in enumerate(stamps):
            sess = "sess-a" if i < 3 else "sess-b"
            assert store.mark_bar_processed("AAPL", "5Min", ts.to_pydatetime(), sess) is True

        rows = store.get_processed_bars(limit=4)
        assert len(rows) == 4
        got = [r["bar_timestamp"] for r in rows]
        assert got == sorted(got, reverse=True)
        assert got[0] == stamps[-1].to_pydatetime().isoformat()

        rows_a = store.get_processed_bars(limit=50, session_id="sess-a")
        assert len(rows_a) == 3
        assert all(r["session_id"] == "sess-a" for r in rows_a)


# ----------------------------------------------------------------------
# Orders
# ----------------------------------------------------------------------


class TestOrders:
    def test_record_and_get_order_by_client_id(self, store):
        order = _make_order(fill_price=None, stop_price=189.5)
        store.record_order(order, "sess-1")

        got = store.get_order_by_client_id("tbot-entry-abc123")
        assert got is not None
        assert isinstance(got, OrderRecord)
        assert got.client_order_id == "tbot-entry-abc123"
        assert got.broker_order_id == "broker-1"
        assert got.symbol == "AAPL"
        assert got.purpose == "entry"
        assert got.side == "buy"
        assert got.requested_qty == 10.0
        assert got.filled_qty == 0.0
        assert got.status == "accepted"
        assert got.fill_price is None
        assert got.stop_price == 189.5
        assert got.submitted_at == T0
        assert got.submitted_at.tzinfo is not None
        assert got.updated_at == T0

    def test_record_order_upserts_status_and_fill_fields(self, store):
        """A second record_order with the same client_order_id must update
        the fill/status fields in place, not insert a duplicate row."""
        store.record_order(_make_order(), "sess-1")

        later = T0 + timedelta(minutes=1)
        store.record_order(
            _make_order(
                broker_order_id="broker-1",
                status="filled",
                filled_qty=10.0,
                fill_price=190.25,
                updated_at=later,
            ),
            "sess-1",
        )

        got = store.get_order_by_client_id("tbot-entry-abc123")
        assert got.status == "filled"
        assert got.filled_qty == 10.0
        assert got.fill_price == 190.25
        assert got.updated_at == later
        # Still exactly one row for this client order id.
        assert len(store.get_orders(limit=100)) == 1

    def test_get_orders_limit_and_ordering(self, store):
        for i in range(5):
            store.record_order(
                _make_order(
                    client_order_id=f"tbot-entry-{i}",
                    updated_at=T0 + timedelta(minutes=i),
                ),
                "sess-1",
            )

        rows = store.get_orders(limit=3)
        assert len(rows) == 3
        # Newest updated_at first.
        assert [r["client_order_id"] for r in rows] == [
            "tbot-entry-4",
            "tbot-entry-3",
            "tbot-entry-2",
        ]

    def test_get_orders_session_filter(self, store):
        store.record_order(_make_order(client_order_id="a"), "sess-1")
        store.record_order(
            _make_order(client_order_id="b", updated_at=T0 + timedelta(minutes=1)), "sess-2"
        )
        rows = store.get_orders(session_id="sess-2")
        assert [r["client_order_id"] for r in rows] == ["b"]

    def test_is_bot_client_order_id(self, store):
        assert store.is_bot_client_order_id("tbot-entry-abc123") is False
        store.record_order(_make_order(), "sess-1")
        assert store.is_bot_client_order_id("tbot-entry-abc123") is True
        assert store.is_bot_client_order_id("someone-elses-id") is False


# ----------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------


class TestTrades:
    def test_signal_snapshot_round_trips_through_signal_json(self, store):
        snapshot = {
            "action": "BUY",
            "reason": "bullish crossover",
            "short_ma": 101.5,
            "filters": {"trend": True, "rsi": True},
        }
        store.record_trade(
            "sess-1", "AAPL", "BUY", 10, 190.0, None, "entry fill", snapshot, timestamp=T0
        )

        rows = store.get_trades(session_id="sess-1")
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "AAPL"
        assert row["action"] == "BUY"
        assert row["qty"] == 10
        assert row["fill_price"] == 190.0
        assert row["realized_pnl"] is None
        assert row["reason"] == "entry fill"
        assert json.loads(row["signal_json"]) == snapshot

    def test_record_trade_without_snapshot_stores_null(self, store):
        store.record_trade("sess-1", "AAPL", "EXIT", 10, 191.0, 10.0, "stop", None, timestamp=T0)
        assert store.get_trades()[0]["signal_json"] is None

    def test_get_trades_newest_first(self, store):
        for i in range(3):
            store.record_trade(
                "sess-1",
                "AAPL",
                "BUY",
                1,
                100.0 + i,
                None,
                f"trade-{i}",
                None,
                timestamp=T0 + timedelta(minutes=i),
            )
        reasons = [r["reason"] for r in store.get_trades(session_id="sess-1")]
        assert reasons == ["trade-2", "trade-1", "trade-0"]

    def test_get_trades_limit(self, store):
        for i in range(5):
            store.record_trade(
                "sess-1", "AAPL", "BUY", 1, 100.0, None, f"t{i}", None,
                timestamp=T0 + timedelta(minutes=i),
            )
        assert len(store.get_trades(limit=2)) == 2

    def test_session_realized_pnl_sums_only_non_null_rows(self, store):
        """Entry fills carry realized_pnl=None and must not affect the sum;
        SQL SUM skips NULLs so only exit rows count."""
        store.record_trade("sess-1", "AAPL", "BUY", 10, 190.0, None, "entry", None, timestamp=T0)
        store.record_trade(
            "sess-1", "AAPL", "EXIT", 10, 195.0, 50.0, "exit", None,
            timestamp=T0 + timedelta(minutes=1),
        )
        store.record_trade(
            "sess-1", "AAPL", "EXIT", 5, 186.0, -20.0, "stop", None,
            timestamp=T0 + timedelta(minutes=2),
        )
        # A different session's pnl must not leak in.
        store.record_trade(
            "sess-2", "AAPL", "EXIT", 5, 200.0, 999.0, "exit", None,
            timestamp=T0 + timedelta(minutes=3),
        )

        assert store.session_realized_pnl("sess-1") == pytest.approx(30.0)

    def test_session_realized_pnl_zero_when_only_entries(self, store):
        store.record_trade("sess-1", "AAPL", "BUY", 10, 190.0, None, "entry", None, timestamp=T0)
        assert store.session_realized_pnl("sess-1") == 0.0
        assert store.session_realized_pnl("no-such-session") == 0.0


# ----------------------------------------------------------------------
# Equity snapshots
# ----------------------------------------------------------------------


class TestEquity:
    def test_equity_curve_chronological_even_if_inserted_out_of_order(self, store):
        """get_equity_curve must sort by timestamp ascending regardless of
        insertion order, and return timezone-aware timestamps."""
        stamps = [T0 + timedelta(minutes=m) for m in (10, 0, 5)]
        for ts, eq in zip(stamps, (100_100.0, 100_000.0, 100_050.0)):
            store.record_equity("sess-1", ts, eq, buying_power=eq * 2)

        curve = store.get_equity_curve("sess-1")
        assert [p["timestamp"] for p in curve] == sorted(stamps)
        assert [p["equity"] for p in curve] == [100_000.0, 100_050.0, 100_100.0]
        assert all(p["timestamp"].tzinfo is not None for p in curve)
        assert curve[0]["buying_power"] == 200_000.0

    def test_equity_curve_scoped_to_session(self, store):
        store.record_equity("sess-1", T0, 100_000.0)
        store.record_equity("sess-2", T0, 55_000.0)
        curve = store.get_equity_curve("sess-1")
        assert len(curve) == 1
        assert curve[0]["equity"] == 100_000.0


# ----------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------


class TestLogs:
    def test_severity_stored_uppercase_and_context_json(self, store):
        store.log("warning", "stop order rejected", context={"code": 42}, session_id="sess-1")
        rows = store.get_logs()
        assert len(rows) == 1
        assert rows[0]["severity"] == "WARNING"
        assert rows[0]["message"] == "stop order rejected"
        assert json.loads(rows[0]["context"]) == {"code": 42}
        assert rows[0]["session_id"] == "sess-1"

    def test_get_logs_session_filter_and_limit(self, store):
        for i in range(4):
            store.log("info", f"a-{i}", session_id="sess-a")
        store.log("error", "b-0", session_id="sess-b")

        rows_a = store.get_logs(limit=100, session_id="sess-a")
        assert len(rows_a) == 4
        assert all(r["session_id"] == "sess-a" for r in rows_a)
        # Newest (highest id) first.
        assert [r["message"] for r in rows_a] == ["a-3", "a-2", "a-1", "a-0"]

        assert len(store.get_logs(limit=2)) == 2
        assert [r["message"] for r in store.get_logs(limit=100, session_id="sess-b")] == ["b-0"]


# ----------------------------------------------------------------------
# KV + bot position helpers
# ----------------------------------------------------------------------


class TestKvAndBotPosition:
    def test_kv_set_get_overwrite_delete(self, store):
        assert store.get_kv("kill_switch") is None
        store.set_kv("kill_switch", "armed")
        assert store.get_kv("kill_switch") == "armed"
        store.set_kv("kill_switch", "disarmed")  # upsert overwrites
        assert store.get_kv("kill_switch") == "disarmed"
        store.delete_kv("kill_switch")
        assert store.get_kv("kill_switch") is None
        # Deleting a missing key is a no-op, not an error.
        store.delete_kv("never-existed")

    def test_bot_position_round_trip_and_clear(self, store):
        pos = {
            "qty": 10,
            "entry_price": 190.25,
            "atr_at_entry": 1.8,
            "stop_price": 186.65,
            "stop_client_order_id": "tbot-stop-xyz",
            "session_id": "sess-1",
            "entry_time": T0.isoformat(),
        }
        assert store.get_bot_position("AAPL") is None
        store.set_bot_position("AAPL", pos)
        assert store.get_bot_position("AAPL") == pos
        # Positions are keyed per symbol.
        assert store.get_bot_position("MSFT") is None

        store.clear_bot_position("AAPL")
        assert store.get_bot_position("AAPL") is None


# ----------------------------------------------------------------------
# Restart persistence
# ----------------------------------------------------------------------


class TestRestartPersistence:
    def test_everything_survives_close_and_reopen(self, tmp_path):
        """The store is the bot's crash-recovery source of truth: every table
        written by one StateStore instance must be readable by a fresh
        instance opened on the same file after close()."""
        db_path = str(tmp_path / "restart.db")
        snapshot = {"action": "BUY", "rsi": 55.5}
        pos = {"qty": 5, "entry_price": 100.0, "session_id": "sess-1"}

        s1 = StateStore(db_path)
        s1.create_session("sess-1", "AAPL", "5Min", T0, None, 100_000.0)
        s1.mark_bar_processed("AAPL", "5Min", T0, "sess-1")
        s1.record_order(_make_order(), "sess-1")
        s1.record_trade("sess-1", "AAPL", "BUY", 5, 100.0, None, "entry", snapshot, timestamp=T0)
        s1.record_equity("sess-1", T0, 100_000.0)
        s1.log("info", "hello", session_id="sess-1")
        s1.set_kv("kill_switch", "armed")
        s1.set_bot_position("AAPL", pos)
        s1.close()

        s2 = StateStore(db_path)
        try:
            sess = s2.get_session("sess-1")
            assert sess is not None
            assert sess["symbol"] == "AAPL"
            assert sess["started_at"] == T0

            assert s2.is_bar_processed("AAPL", "5Min", T0) is True
            # Global dedup still holds across restart.
            assert s2.mark_bar_processed("AAPL", "5Min", T0, "sess-2") is False

            order = s2.get_order_by_client_id("tbot-entry-abc123")
            assert order is not None and order.status == "accepted"
            assert s2.is_bot_client_order_id("tbot-entry-abc123") is True

            trades = s2.get_trades(session_id="sess-1")
            assert len(trades) == 1
            assert json.loads(trades[0]["signal_json"]) == snapshot

            curve = s2.get_equity_curve("sess-1")
            assert len(curve) == 1 and curve[0]["equity"] == 100_000.0

            logs = s2.get_logs(session_id="sess-1")
            assert len(logs) == 1 and logs[0]["severity"] == "INFO"

            assert s2.get_kv("kill_switch") == "armed"
            assert s2.get_bot_position("AAPL") == pos
        finally:
            s2.close()


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


class TestConcurrency:
    def test_parallel_logs_and_bar_marks_on_shared_store(self, tmp_path):
        """8 threads hammer ONE shared StateStore (its documented usage:
        engine thread + UI thread share a connection behind an RLock).
        Every write must land exactly once with no exceptions."""
        store = StateStore(str(tmp_path / "concurrent.db"))
        n_threads, n_rows = 8, 25
        errors: list[BaseException] = []
        true_marks = [0] * n_threads
        barrier = threading.Barrier(n_threads)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                for j in range(n_rows):
                    store.log("info", f"t{idx}-msg{j}", session_id=f"sess-{idx}")
                    bar_ts = T0 + timedelta(minutes=idx * 1000 + j)
                    if store.mark_bar_processed("AAPL", "1Min", bar_ts, f"sess-{idx}"):
                        true_marks[idx] += 1
            except BaseException as exc:  # noqa: BLE001 - captured for the assert below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        try:
            assert not any(t.is_alive() for t in threads)
            assert errors == []
            total = n_threads * n_rows
            assert len(store.get_logs(limit=total * 2)) == total
            assert len(store.get_processed_bars(limit=total * 2)) == total
            # Every (symbol, timeframe, bar_ts) was distinct, so every mark
            # must have reported "first time".
            assert true_marks == [n_rows] * n_threads
            for i in range(n_threads):
                assert len(store.get_logs(limit=total, session_id=f"sess-{i}")) == n_rows
        finally:
            store.close()
