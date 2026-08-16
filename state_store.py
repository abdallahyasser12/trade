"""Durable SQLite state store shared by the engine thread and the
Streamlit UI thread.

Design:
* One connection opened with ``check_same_thread=False``, guarded by a
  re-entrant lock; every public method is a short transaction.
* WAL journal mode so a reader (dashboard) never blocks the writer
  (engine) and vice versa.
* Timestamps are stored as ISO-8601 UTC strings; helpers convert on the
  way in/out so callers always deal with timezone-aware datetimes.
* Streamlit ``session_state`` is never authoritative — everything the
  bot needs to survive a rerun or restart lives here.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import CLIENT_ORDER_PREFIX, DEFAULT_DB_PATH
from models import OrderRecord

# Bumped whenever a migration is added below. Opening a database written by
# a NEWER build refuses rather than silently operating on a schema this code
# does not understand — the database defines what the bot believes it owns.
SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    intended_stop_at TEXT,
    stopped_at       TEXT,
    status           TEXT NOT NULL DEFAULT 'running',
    stop_reason      TEXT,
    starting_equity  REAL,
    halted           INTEGER NOT NULL DEFAULT 0,
    halt_reason      TEXT,
    config_json      TEXT
);

CREATE TABLE IF NOT EXISTS processed_bars (
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    bar_timestamp TEXT NOT NULL,
    session_id    TEXT,
    processed_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, bar_timestamp)
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    broker_order_id TEXT,
    session_id      TEXT,
    symbol          TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    side            TEXT NOT NULL,
    requested_qty   REAL NOT NULL,
    filled_qty      REAL NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,
    fill_price      REAL,
    stop_price      REAL,
    limit_price     REAL,
    submitted_at    TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key       TEXT,
    session_id      TEXT,
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,
    qty             REAL NOT NULL,
    fill_price      REAL,
    realized_pnl    REAL,
    reason          TEXT,
    signal_json     TEXT,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    timestamp    TEXT NOT NULL,
    equity       REAL NOT NULL,
    buying_power REAL
);

CREATE TABLE IF NOT EXISTS logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp  TEXT NOT NULL,
    severity   TEXT NOT NULL,
    message    TEXT NOT NULL,
    context    TEXT
);

-- Execution quality: what the decision assumed versus what the market gave.
-- Without this the simulator's slippage assumption is never checked against
-- reality, and a live shortfall is indistinguishable from a bad signal.
CREATE TABLE IF NOT EXISTS executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key       TEXT UNIQUE,
    session_id      TEXT,
    client_order_id TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    side            TEXT NOT NULL,
    reference_price REAL,
    fill_price      REAL NOT NULL,
    filled_qty      REAL NOT NULL,
    slippage_bps    REAL,
    fees            REAL,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_session ON orders (session_id);
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades (session_id);
CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_snapshots (session_id);
CREATE INDEX IF NOT EXISTS idx_logs_session ON logs (session_id);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


class StateStore:
    """Thread-safe persistent store for all bot-owned state."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            # FULL, not NORMAL: the engine's crash-safety design requires
            # the pending-order intent written BEFORE each broker submit to
            # actually be on disk if power is lost right after the submit.
            self._conn.execute("PRAGMA synchronous=FULL;")
            self._conn.executescript(_SCHEMA)
            found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if found > SCHEMA_VERSION:
                self._conn.close()
                raise RuntimeError(
                    f"State database {self.db_path!r} was written by a newer build "
                    f"(schema v{found}; this build understands v{SCHEMA_VERSION}). "
                    "Upgrade the application rather than downgrading the database."
                )
            trade_columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
            }
            if "event_key" not in trade_columns:
                self._conn.execute("ALTER TABLE trades ADD COLUMN event_key TEXT")
            if "fees" not in trade_columns:
                # Realized P&L was booked gross of fees, so every stored row and
                # every rendered total overstated performance by the round trip.
                self._conn.execute("ALTER TABLE trades ADD COLUMN fees REAL")
            order_columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(orders)").fetchall()
            }
            if "limit_price" not in order_columns:
                self._conn.execute("ALTER TABLE orders ADD COLUMN limit_price REAL")
            if "reference_price" not in order_columns:
                # The price the decision was made against, kept so realized
                # slippage can be measured against what the model assumed.
                self._conn.execute("ALTER TABLE orders ADD COLUMN reference_price REAL")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_event_key "
                "ON trades (event_key) WHERE event_key IS NOT NULL"
            )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self.release_engine_lock()
            self._conn.close()

    # ------------------------------------------------------------------
    # Cross-process engine lock
    # ------------------------------------------------------------------
    #
    # Exactly one live trading engine may run against one state database,
    # across ALL processes.  An OS advisory lock on a sibling lockfile is
    # held from engine startup to engine finalize; a second engine (second
    # `streamlit run`, or an orphan left by a dev auto-reload) fails to
    # acquire it and refuses to trade.  The lock dies with the process, so
    # a crash can never leave it stuck.

    def acquire_engine_lock(self) -> bool:
        """Take the exclusive cross-process engine lock.  Non-blocking;
        returns True on success, False when another engine holds it."""
        with self._lock:
            if self.db_path == ":memory:":
                return True  # private database; nothing to protect
            if getattr(self, "_engine_lock_file", None) is not None:
                return False  # this store's engine already holds it
            lock_path = self.db_path + ".engine.lock"
            handle = open(lock_path, "a+")
            try:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except ImportError:  # pragma: no cover — Windows
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                return False
            self._engine_lock_file = handle
            return True

    def release_engine_lock(self) -> None:
        """Release the engine lock if this store holds it.  Idempotent."""
        with self._lock:
            handle = getattr(self, "_engine_lock_file", None)
            if handle is None:
                return
            self._engine_lock_file = None
            try:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover — Windows
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                handle.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run one statement inside the lock and commit."""
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        symbol: str,
        timeframe: str,
        started_at: datetime,
        intended_stop_at: Optional[datetime],
        starting_equity: Optional[float],
        config_json: str = "{}",
    ) -> None:
        self._execute(
            "INSERT INTO sessions (session_id, symbol, timeframe, started_at,"
            " intended_stop_at, status, starting_equity, config_json)"
            " VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (
                session_id,
                symbol,
                timeframe,
                _to_iso(started_at),
                _to_iso(intended_stop_at),
                starting_equity,
                config_json,
            ),
        )

    def end_session(self, session_id: str, stopped_at: datetime, stop_reason: str) -> None:
        self._execute(
            "UPDATE sessions SET stopped_at = ?, status = 'stopped', stop_reason = ?"
            " WHERE session_id = ?",
            (_to_iso(stopped_at), stop_reason, session_id),
        )

    def set_session_halted(self, session_id: str, reason: str) -> None:
        self._execute(
            "UPDATE sessions SET halted = 1, halt_reason = ? WHERE session_id = ?",
            (reason, session_id),
        )

    def _session_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("started_at", "intended_stop_at", "stopped_at"):
            d[key] = _from_iso(d.get(key))
        d["halted"] = bool(d.get("halted"))
        return d

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        rows = self._query("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return self._session_row_to_dict(rows[0]) if rows else None

    def get_latest_session(self) -> Optional[dict[str, Any]]:
        rows = self._query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1")
        return self._session_row_to_dict(rows[0]) if rows else None

    def get_active_session(self) -> Optional[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM sessions WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
        )
        return self._session_row_to_dict(rows[0]) if rows else None

    # ------------------------------------------------------------------
    # Processed bars (duplicate-execution protection)
    # ------------------------------------------------------------------

    def mark_bar_processed(
        self, symbol: str, timeframe: str, bar_timestamp: datetime, session_id: str
    ) -> bool:
        """Record a bar as processed.  Returns False if it already was.

        The uniqueness key is (symbol, timeframe, bar_timestamp) across ALL
        sessions, so an engine restart can never act on the same bar twice.
        """
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO processed_bars"
                    " (symbol, timeframe, bar_timestamp, session_id, processed_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (symbol, timeframe, _to_iso(bar_timestamp), session_id, _to_iso(_utc_now())),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except Exception:
                self._conn.rollback()
                raise

    def is_bar_processed(self, symbol: str, timeframe: str, bar_timestamp: datetime) -> bool:
        rows = self._query(
            "SELECT 1 FROM processed_bars WHERE symbol = ? AND timeframe = ? AND bar_timestamp = ?",
            (symbol, timeframe, _to_iso(bar_timestamp)),
        )
        return bool(rows)

    def get_processed_bars(
        self, limit: int = 50, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM processed_bars WHERE session_id = ?"
                " ORDER BY bar_timestamp DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM processed_bars ORDER BY bar_timestamp DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def record_order(self, order: OrderRecord, session_id: Optional[str]) -> None:
        """Insert or update the persisted view of a bot-owned order."""
        self._execute(
            "INSERT INTO orders (client_order_id, broker_order_id, session_id, symbol,"
            " purpose, side, requested_qty, filled_qty, status, fill_price, stop_price,"
            " limit_price, submitted_at, updated_at, reference_price)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(client_order_id) DO UPDATE SET"
            " broker_order_id = excluded.broker_order_id,"
            " requested_qty = excluded.requested_qty,"
            " filled_qty = excluded.filled_qty,"
            " status = excluded.status,"
            " fill_price = excluded.fill_price,"
            " stop_price = excluded.stop_price,"
            " limit_price = excluded.limit_price,"
            " updated_at = excluded.updated_at,"
            # The broker's own record carries no reference price, so a later
            # overwrite must never erase what the intent recorded.
            " reference_price = COALESCE(excluded.reference_price, orders.reference_price)",
            (
                order.client_order_id,
                order.broker_order_id,
                session_id,
                order.symbol,
                order.purpose,
                order.side,
                order.requested_qty,
                order.filled_qty,
                order.status,
                order.fill_price,
                order.stop_price,
                order.limit_price,
                _to_iso(order.submitted_at),
                _to_iso(order.updated_at or _utc_now()),
                order.reference_price,
            ),
        )

    def update_order(self, order: OrderRecord) -> None:
        self._execute(
            "UPDATE orders SET broker_order_id = ?, filled_qty = ?, status = ?,"
            " fill_price = ?, stop_price = ?, limit_price = ?, updated_at = ?"
            " WHERE client_order_id = ?",
            (
                order.broker_order_id,
                order.filled_qty,
                order.status,
                order.fill_price,
                order.stop_price,
                order.limit_price,
                _to_iso(order.updated_at or _utc_now()),
                order.client_order_id,
            ),
        )

    def _order_row_to_record(self, row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            broker_order_id=row["broker_order_id"] or "",
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            purpose=row["purpose"],
            side=row["side"],
            requested_qty=row["requested_qty"],
            filled_qty=row["filled_qty"],
            status=row["status"],
            fill_price=row["fill_price"],
            stop_price=row["stop_price"],
            limit_price=row["limit_price"],
            reference_price=(
                row["reference_price"] if "reference_price" in row.keys() else None
            ),
            submitted_at=_from_iso(row["submitted_at"]),
            updated_at=_from_iso(row["updated_at"]),
        )

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]:
        rows = self._query(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        )
        return self._order_row_to_record(rows[0]) if rows else None

    def get_orders(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM orders WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = self._query("SELECT * FROM orders ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def is_bot_client_order_id(self, client_order_id: str) -> bool:
        """True when this exact client order id was created by this bot."""
        rows = self._query(
            "SELECT 1 FROM orders WHERE client_order_id = ?", (client_order_id,)
        )
        return bool(rows)

    def bot_net_filled_qty(self, symbol: str) -> float:
        """Net bot-owned broker fills for a symbol across the full database.

        This is used only as an ownership proof during restart recovery. It is
        intentionally not a substitute for the broker's current position quantity.
        """

        rows = self._query(
            "SELECT COALESCE(SUM(CASE WHEN lower(side) = 'buy' THEN filled_qty "
            "ELSE -filled_qty END), 0) AS net_qty FROM orders "
            "WHERE symbol = ? AND client_order_id LIKE ?",
            (symbol, f"{CLIENT_ORDER_PREFIX}-%"),
        )
        return float(rows[0]["net_qty"]) if rows else 0.0

    # ------------------------------------------------------------------
    # Trades (fill events)
    # ------------------------------------------------------------------

    def record_trade(
        self,
        session_id: Optional[str],
        symbol: str,
        action: str,
        qty: float,
        fill_price: Optional[float],
        realized_pnl: Optional[float],
        reason: str,
        signal_snapshot: Optional[dict[str, Any]],
        timestamp: Optional[datetime] = None,
        event_key: Optional[str] = None,
        fees: Optional[float] = None,
    ) -> bool:
        """Record one fill event, idempotently when ``event_key`` is supplied.

        ``realized_pnl`` is NET of fees; ``fees`` carries the amount charged on
        this fill so the deduction stays auditable.
        """
        cursor = self._execute(
            "INSERT OR IGNORE INTO trades (event_key, session_id, symbol, action, qty, fill_price,"
            " realized_pnl, reason, signal_json, timestamp, fees) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_key,
                session_id,
                symbol,
                action,
                qty,
                fill_price,
                realized_pnl,
                reason,
                json.dumps(signal_snapshot) if signal_snapshot is not None else None,
                _to_iso(timestamp or _utc_now()),
                fees,
            ),
        )
        return cursor.rowcount == 1

    def record_execution(
        self,
        session_id: Optional[str],
        client_order_id: str,
        symbol: str,
        purpose: str,
        side: str,
        reference_price: Optional[float],
        fill_price: float,
        filled_qty: float,
        fees: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        event_key: Optional[str] = None,
    ) -> bool:
        """Record one fill's execution quality, idempotently by event key.

        ``slippage_bps`` is signed against the trade's own direction: positive
        always means the fill was worse than the reference price, for a buy and
        a sell alike, so the column can be averaged without sign bookkeeping.
        """

        slippage_bps: Optional[float] = None
        if (
            reference_price is not None
            and math.isfinite(reference_price)
            and reference_price > 0
            and math.isfinite(fill_price)
        ):
            raw = (fill_price - reference_price) / reference_price * 10_000.0
            slippage_bps = raw if side == "buy" else -raw
        cursor = self._execute(
            "INSERT OR IGNORE INTO executions (event_key, session_id, client_order_id,"
            " symbol, purpose, side, reference_price, fill_price, filled_qty,"
            " slippage_bps, fees, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_key,
                session_id,
                client_order_id,
                symbol,
                purpose,
                side,
                reference_price,
                fill_price,
                filled_qty,
                slippage_bps,
                fees,
                _to_iso(timestamp or _utc_now()),
            ),
        )
        return cursor.rowcount == 1

    def execution_quality(self, session_id: Optional[str] = None) -> dict[str, Any]:
        """Realized slippage versus the modeled assumption, in bps.

        Returns ``samples == 0`` when nothing has filled yet; callers must not
        present an average drawn from an empty set as a measurement.
        """

        clause = " WHERE session_id = ?" if session_id else ""
        params: tuple = (session_id,) if session_id else ()
        row = self._query(
            "SELECT COUNT(slippage_bps) AS samples, AVG(slippage_bps) AS avg_bps,"
            " MAX(slippage_bps) AS worst_bps, SUM(fees) AS total_fees,"
            " SUM(filled_qty * fill_price) AS notional"
            f" FROM executions{clause}",
            params,
        )[0]
        return {
            "samples": int(row["samples"] or 0),
            "avg_slippage_bps": row["avg_bps"],
            "worst_slippage_bps": row["worst_bps"],
            "total_fees": row["total_fees"] or 0.0,
            "total_notional": row["notional"] or 0.0,
        }

    def get_executions(
        self, session_id: Optional[str] = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM executions WHERE session_id = ?"
                " ORDER BY timestamp DESC, id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM executions ORDER BY timestamp DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    def get_trades(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM trades WHERE session_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM trades ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    def session_realized_pnl(self, session_id: str) -> float:
        rows = self._query(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM trades WHERE session_id = ?",
            (session_id,),
        )
        return float(rows[0]["pnl"]) if rows else 0.0

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def record_equity(
        self,
        session_id: Optional[str],
        timestamp: datetime,
        equity: float,
        buying_power: Optional[float] = None,
    ) -> None:
        self._execute(
            "INSERT INTO equity_snapshots (session_id, timestamp, equity, buying_power)"
            " VALUES (?, ?, ?, ?)",
            (session_id, _to_iso(timestamp), equity, buying_power),
        )

    def get_equity_curve(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT timestamp, equity, buying_power FROM equity_snapshots"
            " WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["timestamp"] = _from_iso(d["timestamp"])
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def log(
        self,
        severity: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._execute(
            "INSERT INTO logs (session_id, timestamp, severity, message, context)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                _to_iso(_utc_now()),
                severity.upper(),
                message,
                json.dumps(context) if context else None,
            ),
        )

    def get_logs(
        self, limit: int = 100, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = self._query("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Key/value (kill switch, engine status snapshot, bot position)
    # ------------------------------------------------------------------

    def set_kv(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_kv(self, key: str) -> Optional[str]:
        rows = self._query("SELECT value FROM kv WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def delete_kv(self, key: str) -> None:
        self._execute("DELETE FROM kv WHERE key = ?", (key,))

    # ------------------------------------------------------------------
    # Bot-managed position bookkeeping (position ownership boundary)
    # ------------------------------------------------------------------

    def set_bot_position(self, symbol: str, data: dict[str, Any]) -> None:
        """Persist the bot-managed open position for ``symbol``.

        ``data`` holds qty, entry_price, atr_at_entry, stop_price,
        stop_client_order_id, session_id, entry_time (ISO string).
        """
        self.set_kv(f"bot_position:{symbol}", json.dumps(data))

    def get_bot_position(self, symbol: str) -> Optional[dict[str, Any]]:
        raw = self.get_kv(f"bot_position:{symbol}")
        return json.loads(raw) if raw else None

    def clear_bot_position(self, symbol: str) -> None:
        self.delete_kv(f"bot_position:{symbol}")
