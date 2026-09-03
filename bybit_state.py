"""Durable Bybit execution ownership and burn-in evidence store.

This database is intentionally separate from ``StateStore``'s schema-versioned
engine database.  It records venue identity/ownership and burn-in evidence while the
existing StateStore remains authoritative for engine intent/protection state.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


BYBIT_STATE_SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: Optional[datetime] = None) -> str:
    value = ts or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class OwnedInventory:
    strategy_symbol: str
    base_asset: str
    qty: float
    cost_basis_quote: float
    realized_pnl_quote: float
    updated_at: str

    @property
    def average_entry_price(self) -> float:
        return self.cost_basis_quote / self.qty if self.qty > 0 else 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS venue_orders (
    internal_intent_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    order_link_id TEXT NOT NULL UNIQUE,
    order_id TEXT,
    strategy_symbol TEXT NOT NULL,
    venue_symbol TEXT NOT NULL,
    purpose TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    requested_qty REAL NOT NULL DEFAULT 0,
    filled_qty REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'intent',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_venue_orders_order_id
    ON venue_orders(order_id) WHERE order_id IS NOT NULL AND order_id != '';
CREATE INDEX IF NOT EXISTS idx_venue_orders_symbol ON venue_orders(strategy_symbol, created_at);

CREATE TABLE IF NOT EXISTS venue_executions (
    execution_id TEXT PRIMARY KEY,
    internal_intent_id TEXT NOT NULL,
    order_link_id TEXT NOT NULL,
    order_id TEXT,
    strategy_symbol TEXT NOT NULL,
    venue_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    fee_asset TEXT,
    execution_at TEXT NOT NULL,
    raw_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_venue_exec_symbol ON venue_executions(strategy_symbol, execution_at);
CREATE INDEX IF NOT EXISTS idx_venue_exec_intent ON venue_executions(internal_intent_id);

CREATE TABLE IF NOT EXISTS owned_inventory (
    strategy_symbol TEXT PRIMARY KEY,
    base_asset TEXT NOT NULL,
    qty REAL NOT NULL DEFAULT 0,
    cost_basis_quote REAL NOT NULL DEFAULT 0,
    realized_pnl_quote REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS management_state (
    strategy_symbol TEXT PRIMARY KEY,
    original_qty REAL NOT NULL,
    remaining_fraction REAL NOT NULL,
    bars_held INTEGER NOT NULL DEFAULT 0,
    phase TEXT NOT NULL DEFAULT 'NORMAL_POSITION',
    partial_taken INTEGER NOT NULL DEFAULT 0,
    target_taken INTEGER NOT NULL DEFAULT 0,
    signal_reason TEXT NOT NULL DEFAULT '',
    entry_price REAL NOT NULL,
    initial_stop_price REAL NOT NULL,
    highest_price REAL NOT NULL,
    lowest_price REAL NOT NULL,
    threshold_025_at TEXT,
    threshold_050_at TEXT,
    threshold_075_at TEXT,
    threshold_100_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burnin_runs (
    run_id TEXT PRIMARY KEY,
    git_sha TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    universe_json TEXT NOT NULL,
    start_time TEXT NOT NULL,
    planned_end_time TEXT NOT NULL,
    environment TEXT NOT NULL,
    status TEXT NOT NULL,
    restart_count INTEGER NOT NULL DEFAULT 0,
    last_heartbeat TEXT,
    stopped_at TEXT,
    stop_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_burnin_active ON burnin_runs(status, start_time);

CREATE TABLE IF NOT EXISTS burnin_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_key TEXT UNIQUE,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT,
    strategy TEXT,
    decision_id TEXT,
    intent_id TEXT,
    order_link_id TEXT,
    order_id TEXT,
    execution_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_burnin_events_run ON burnin_events(run_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_burnin_events_type ON burnin_events(run_id, event_type, timestamp);

CREATE TABLE IF NOT EXISTS health_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    health TEXT NOT NULL,
    public_ws_up INTEGER NOT NULL DEFAULT 0,
    private_ws_up INTEGER NOT NULL DEFAULT 0,
    public_ws_reconnects INTEGER NOT NULL DEFAULT 0,
    private_ws_reconnects INTEGER NOT NULL DEFAULT 0,
    rest_failures INTEGER NOT NULL DEFAULT 0,
    rate_limit_events INTEGER NOT NULL DEFAULT 0,
    clock_skew_ms REAL,
    quote_age_ms REAL,
    spread_bps REAL,
    pending_intents INTEGER NOT NULL DEFAULT 0,
    unprotected_qty REAL NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_health_run ON health_samples(run_id, timestamp);
"""


class BybitStateStore:
    def __init__(self, db_path: str = "state/bybit_state.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
            found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if found > BYBIT_STATE_SCHEMA_VERSION:
                self._conn.close()
                raise RuntimeError(
                    f"Bybit state database schema {found} is newer than supported "
                    f"{BYBIT_STATE_SCHEMA_VERSION}."
                )
            self._conn.execute(f"PRAGMA user_version={BYBIT_STATE_SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------- order ownership --------------------------
    def record_order_intent(
        self,
        *,
        internal_intent_id: str,
        order_link_id: str,
        strategy_symbol: str,
        venue_symbol: str,
        purpose: str,
        attempt: int,
        requested_qty: float,
    ) -> None:
        now = iso()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM venue_orders WHERE internal_intent_id=?",
                (internal_intent_id,),
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["order_link_id"], existing["strategy_symbol"],
                    existing["venue_symbol"], existing["purpose"], int(existing["attempt"]),
                )
                wanted = (order_link_id, strategy_symbol, venue_symbol, purpose, int(attempt))
                if immutable != wanted:
                    raise RuntimeError("Economic intent ID was reused with different Bybit semantics")
                return
            self._conn.execute(
                """INSERT INTO venue_orders
                (internal_intent_id,venue,order_link_id,strategy_symbol,venue_symbol,purpose,
                 attempt,requested_qty,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    internal_intent_id, "bybit", order_link_id, strategy_symbol,
                    venue_symbol, purpose, int(attempt), float(requested_qty), now, now,
                ),
            )

    def update_order(
        self,
        internal_intent_id: str,
        *,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        filled_qty: Optional[float] = None,
    ) -> None:
        assignments = ["updated_at=?"]
        values: list[Any] = [iso()]
        if order_id is not None:
            assignments.append("order_id=?")
            values.append(str(order_id))
        if status is not None:
            assignments.append("status=?")
            values.append(str(status))
        if filled_qty is not None:
            assignments.append("filled_qty=?")
            values.append(float(filled_qty))
        values.append(internal_intent_id)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE venue_orders SET {', '.join(assignments)} WHERE internal_intent_id=?",
                tuple(values),
            )

    def order_by_intent(self, internal_intent_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM venue_orders WHERE internal_intent_id=?", (internal_intent_id,)
            ).fetchone()
        return dict(row) if row else None

    def order_by_link(self, order_link_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM venue_orders WHERE order_link_id=?", (order_link_id,)
            ).fetchone()
        return dict(row) if row else None

    def order_by_order_id(self, order_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM venue_orders WHERE order_id=?", (str(order_id),)
            ).fetchone()
        return dict(row) if row else None

    def bot_orders(self, strategy_symbol: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM venue_orders"
        params: tuple[Any, ...] = ()
        if strategy_symbol:
            sql += " WHERE strategy_symbol=?"
            params = (strategy_symbol,)
        sql += " ORDER BY created_at"
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def count_recent_entries(self, *, hours: int = 24) -> int:
        cutoff = iso(utc_now() - timedelta(hours=hours))
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS n FROM venue_orders
                   WHERE purpose='entry' AND created_at>=?""",
                (cutoff,),
            ).fetchone()
        return int(row["n"] or 0)

    # ---------------------------- executions ----------------------------
    def record_execution(
        self,
        *,
        execution_id: str,
        internal_intent_id: str,
        order_link_id: str,
        order_id: str,
        strategy_symbol: str,
        venue_symbol: str,
        base_asset: str,
        side: str,
        qty: float,
        price: float,
        fee: float,
        fee_asset: Optional[str],
        execution_at: datetime,
        raw: Any,
    ) -> bool:
        """Idempotently record one venue execution and update bot-owned inventory."""
        if qty <= 0 or price <= 0:
            raise ValueError("Execution qty and price must be positive")
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("Execution side must be buy or sell")
        raw_hash = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM venue_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if exists:
                return False
            self._conn.execute(
                """INSERT INTO venue_executions
                (execution_id,internal_intent_id,order_link_id,order_id,strategy_symbol,
                 venue_symbol,side,qty,price,fee,fee_asset,execution_at,raw_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    execution_id, internal_intent_id, order_link_id, order_id,
                    strategy_symbol, venue_symbol, normalized_side, float(qty), float(price),
                    max(0.0, float(fee or 0.0)), fee_asset, iso(execution_at), raw_hash,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM owned_inventory WHERE strategy_symbol=?", (strategy_symbol,)
            ).fetchone()
            owned_qty = float(row["qty"] if row else 0.0)
            cost_basis = float(row["cost_basis_quote"] if row else 0.0)
            realized = float(row["realized_pnl_quote"] if row else 0.0)
            fee_value = max(0.0, float(fee or 0.0))
            fee_asset_norm = (fee_asset or "").upper()
            base_norm = base_asset.upper()
            quote_fee = fee_value if fee_asset_norm and fee_asset_norm != base_norm else 0.0
            base_fee = fee_value if fee_asset_norm == base_norm else 0.0
            if normalized_side == "buy":
                acquired = max(0.0, float(qty) - base_fee)
                owned_qty += acquired
                cost_basis += float(qty) * float(price) + quote_fee
            else:
                sell_qty = float(qty) + base_fee
                if sell_qty > owned_qty + 1e-10:
                    raise RuntimeError(
                        f"Execution would make bot-owned {strategy_symbol} inventory negative"
                    )
                avg_cost = cost_basis / owned_qty if owned_qty > 0 else 0.0
                removed_cost = min(cost_basis, avg_cost * sell_qty)
                realized += float(qty) * float(price) - removed_cost - quote_fee
                owned_qty = max(0.0, owned_qty - sell_qty)
                cost_basis = max(0.0, cost_basis - removed_cost)
                if owned_qty <= 1e-12:
                    owned_qty = 0.0
                    cost_basis = 0.0
            self._conn.execute(
                """INSERT INTO owned_inventory
                (strategy_symbol,base_asset,qty,cost_basis_quote,realized_pnl_quote,updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(strategy_symbol) DO UPDATE SET
                  base_asset=excluded.base_asset, qty=excluded.qty,
                  cost_basis_quote=excluded.cost_basis_quote,
                  realized_pnl_quote=excluded.realized_pnl_quote,
                  updated_at=excluded.updated_at""",
                (strategy_symbol, base_norm, owned_qty, cost_basis, realized, iso(execution_at)),
            )
            self._conn.execute(
                "UPDATE venue_orders SET filled_qty=filled_qty+?, updated_at=? WHERE internal_intent_id=?",
                (float(qty), iso(execution_at), internal_intent_id),
            )
        return True

    def inventory(self, strategy_symbol: str) -> OwnedInventory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM owned_inventory WHERE strategy_symbol=?", (strategy_symbol,)
            ).fetchone()
        if not row:
            return OwnedInventory(strategy_symbol, strategy_symbol.split("/")[0], 0.0, 0.0, 0.0, iso())
        return OwnedInventory(
            strategy_symbol=row["strategy_symbol"], base_asset=row["base_asset"],
            qty=float(row["qty"]), cost_basis_quote=float(row["cost_basis_quote"]),
            realized_pnl_quote=float(row["realized_pnl_quote"]), updated_at=row["updated_at"],
        )

    def all_inventory(self) -> list[OwnedInventory]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM owned_inventory ORDER BY strategy_symbol").fetchall()
        return [
            OwnedInventory(
                r["strategy_symbol"], r["base_asset"], float(r["qty"]),
                float(r["cost_basis_quote"]), float(r["realized_pnl_quote"]), r["updated_at"]
            ) for r in rows
        ]

    # ------------------------ management / excursion --------------------
    def get_management_state(self, strategy_symbol: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM management_state WHERE strategy_symbol=?", (strategy_symbol,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_management_state(self, strategy_symbol: str, **values: Any) -> None:
        existing = self.get_management_state(strategy_symbol)
        base = existing or {
            "strategy_symbol": strategy_symbol,
            "original_qty": float(values.get("original_qty", 0)),
            "remaining_fraction": 1.0,
            "bars_held": 0,
            "phase": "NORMAL_POSITION",
            "partial_taken": 0,
            "target_taken": 0,
            "signal_reason": str(values.get("signal_reason", "")),
            "entry_price": float(values.get("entry_price", 0)),
            "initial_stop_price": float(values.get("initial_stop_price", 0)),
            "highest_price": float(values.get("entry_price", 0)),
            "lowest_price": float(values.get("entry_price", 0)),
            "threshold_025_at": None,
            "threshold_050_at": None,
            "threshold_075_at": None,
            "threshold_100_at": None,
            "updated_at": iso(),
        }
        base.update(values)
        base["updated_at"] = iso()
        columns = [
            "strategy_symbol","original_qty","remaining_fraction","bars_held","phase",
            "partial_taken","target_taken","signal_reason","entry_price","initial_stop_price",
            "highest_price","lowest_price","threshold_025_at","threshold_050_at",
            "threshold_075_at","threshold_100_at","updated_at",
        ]
        params = tuple(base.get(c) for c in columns)
        with self._lock, self._conn:
            self._conn.execute(
                f"""INSERT INTO management_state ({','.join(columns)})
                VALUES({','.join('?' for _ in columns)})
                ON CONFLICT(strategy_symbol) DO UPDATE SET
                {','.join(f'{c}=excluded.{c}' for c in columns if c != 'strategy_symbol')}""",
                params,
            )

    def delete_management_state(self, strategy_symbol: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM management_state WHERE strategy_symbol=?", (strategy_symbol,))

    # ------------------------------- burn-in ----------------------------
    def create_or_resume_run(
        self,
        *,
        run_id: str,
        git_sha: str,
        strategy_version: str,
        config_hash: str,
        universe: Iterable[str],
        start_time: datetime,
        planned_end_time: datetime,
        environment: str,
    ) -> bool:
        """Create run or resume an exactly matching frozen identity. Returns resumed."""
        universe_json = json.dumps(list(universe), sort_keys=True)
        with self._lock, self._conn:
            row = self._conn.execute("SELECT * FROM burnin_runs WHERE run_id=?", (run_id,)).fetchone()
            if row:
                frozen = (row["git_sha"], row["strategy_version"], row["config_hash"], row["universe_json"], row["environment"])
                wanted = (git_sha, strategy_version, config_hash, universe_json, environment)
                if frozen != wanted:
                    raise RuntimeError("Burn-in run ID cannot be resumed with changed code/config/universe")
                self._conn.execute(
                    "UPDATE burnin_runs SET restart_count=restart_count+1,status='running',last_heartbeat=? WHERE run_id=?",
                    (iso(), run_id),
                )
                return True
            self._conn.execute(
                """INSERT INTO burnin_runs
                (run_id,git_sha,strategy_version,config_hash,universe_json,start_time,
                 planned_end_time,environment,status,last_heartbeat)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (run_id, git_sha, strategy_version, config_hash, universe_json, iso(start_time),
                 iso(planned_end_time), environment, "running", iso()),
            )
        return False

    def active_run(self) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM burnin_runs WHERE status='running' ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def heartbeat_run(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE burnin_runs SET last_heartbeat=? WHERE run_id=?", (iso(), run_id))

    def stop_run(self, run_id: str, reason: str, *, status: str = "stopped") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE burnin_runs SET status=?,stopped_at=?,stop_reason=? WHERE run_id=?",
                (status, iso(), reason, run_id),
            )

    def record_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_key: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        decision_id: Optional[str] = None,
        intent_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    """INSERT INTO burnin_events
                    (run_id,event_key,event_type,timestamp,symbol,strategy,decision_id,intent_id,
                     order_link_id,order_id,execution_id,payload_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id,event_key,event_type,iso(timestamp),symbol,strategy,decision_id,intent_id,
                     order_link_id,order_id,execution_id,json.dumps(payload,sort_keys=True,default=str)),
                )
            return True
        except sqlite3.IntegrityError:
            if event_key:
                return False
            raise

    def events(self, run_id: str, event_type: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM burnin_events WHERE run_id=?"
        params: list[Any] = [run_id]
        if event_type:
            sql += " AND event_type=?"
            params.append(event_type)
        sql += " ORDER BY timestamp,id"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except Exception:
                item["payload"] = {}
            result.append(item)
        return result

    def record_health(self, run_id: str, sample: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO health_samples
                (run_id,timestamp,provider,health,public_ws_up,private_ws_up,
                 public_ws_reconnects,private_ws_reconnects,rest_failures,rate_limit_events,
                 clock_skew_ms,quote_age_ms,spread_bps,pending_intents,unprotected_qty,payload_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, iso(), str(sample.get("provider", "bybit")), str(sample.get("health", "DEGRADED")),
                    int(bool(sample.get("public_ws_up"))), int(bool(sample.get("private_ws_up"))),
                    int(sample.get("public_ws_reconnects", 0)), int(sample.get("private_ws_reconnects", 0)),
                    int(sample.get("rest_failures", 0)), int(sample.get("rate_limit_events", 0)),
                    sample.get("clock_skew_ms"), sample.get("quote_age_ms"), sample.get("spread_bps"),
                    int(sample.get("pending_intents", 0)), float(sample.get("unprotected_qty", 0.0)),
                    json.dumps(sample, sort_keys=True, default=str),
                ),
            )

    def health_samples(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM health_samples WHERE run_id=? ORDER BY timestamp", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]
