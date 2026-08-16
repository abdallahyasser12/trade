"""Durable SQLite store for the quant desk layer.

Deliberately a SEPARATE database file from the trading ``StateStore``: the
execution engines' crash-safety and recovery guarantees were audited
against that schema, and the quant layer must be able to evolve its own
schema without ever putting order/position recovery at risk.  The same
conventions apply: one connection guarded by an RLock, WAL journal mode,
ISO-8601 UTC timestamps, and explicit ``user_version`` gating so a newer
database refuses to run under an older build.

What lives here:

* every decision the desk makes, including rejections and abstentions;
* post-trade attribution records (real and shadow);
* strategy lifecycle state and full promotion/demotion history;
* research-run registry with per-gate outcomes and trial counts
  (multiple-testing bookkeeping);
* point-in-time feature snapshots;
* model registry with calibration measurements;
* drift metrics and desk risk events;
* shadow-executor virtual positions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from quant.contracts import (
    AttributionRecord,
    Market,
    OpportunityDecision,
    StrategyState,
)

DEFAULT_QUANT_DB_PATH = str(Path("data") / "quant_desk.db")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at              TEXT NOT NULL,
    signal_timestamp        TEXT,
    symbol                  TEXT NOT NULL,
    market                  TEXT NOT NULL,
    timeframe               TEXT NOT NULL,
    strategy_id             TEXT NOT NULL,
    strategy_version        TEXT NOT NULL,
    action                  TEXT NOT NULL,
    outcome                 TEXT NOT NULL,
    rank                    INTEGER,
    confidence              REAL,
    expected_gross_alpha_bps REAL,
    expected_cost_bps       REAL,
    expected_net_alpha_bps  REAL,
    opportunity_score       REAL,
    risk_allocation_pct     REAL,
    reason                  TEXT,
    rejection_reasons       TEXT,
    decision_json           TEXT NOT NULL,
    session_id              TEXT
);

CREATE TABLE IF NOT EXISTS attributions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at             TEXT NOT NULL,
    strategy_id             TEXT NOT NULL,
    strategy_version        TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    market                  TEXT NOT NULL,
    timeframe               TEXT NOT NULL,
    session_id              TEXT,
    signal_timestamp        TEXT,
    entry_timestamp         TEXT,
    exit_timestamp          TEXT,
    qty                     REAL NOT NULL,
    entry_price             REAL NOT NULL,
    exit_price              REAL,
    initial_stop_price      REAL,
    fees                    REAL NOT NULL DEFAULT 0,
    expected_cost_bps       REAL,
    realized_slippage_bps   REAL,
    gross_pnl               REAL,
    net_pnl                 REAL,
    r_multiple              REAL,
    mfe_pct                 REAL,
    mae_pct                 REAL,
    entry_reason            TEXT,
    exit_reason             TEXT,
    regime_at_entry         TEXT,
    opportunity_score       REAL,
    shadow                  INTEGER NOT NULL DEFAULT 0,
    features_json           TEXT
);

CREATE TABLE IF NOT EXISTS lifecycle (
    strategy_id  TEXT PRIMARY KEY,
    state        TEXT NOT NULL,
    since        TEXT NOT NULL,
    reason       TEXT,
    operator_ack INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    from_state   TEXT NOT NULL,
    to_state     TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    reason       TEXT NOT NULL,
    evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS research_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at           TEXT NOT NULL,
    strategy_id      TEXT NOT NULL,
    hypothesis_id    TEXT,
    params_json      TEXT NOT NULL,
    universe_json    TEXT NOT NULL,
    data_window      TEXT NOT NULL,
    candidate_count  INTEGER NOT NULL,
    trial_count      INTEGER NOT NULL,
    passed           INTEGER NOT NULL,
    gates_json       TEXT NOT NULL,
    cost_assumptions_json TEXT,
    git_commit       TEXT,
    seed             INTEGER,
    result_json      TEXT
);

CREATE TABLE IF NOT EXISTS features (
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    market          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    source          TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    available_at    TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    PRIMARY KEY (timestamp, symbol, market, timeframe, source, feature_version)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id     TEXT NOT NULL,
    version      TEXT NOT NULL,
    trained_at   TEXT NOT NULL,
    data_window  TEXT,
    feature_version TEXT,
    params_json  TEXT,
    metrics_json TEXT,
    calibration_json TEXT,
    PRIMARY KEY (model_id, version)
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at TEXT NOT NULL,
    subject     TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL,
    threshold   REAL,
    breached    INTEGER NOT NULL DEFAULT 0,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS risk_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    source      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    context_json TEXT
);

CREATE TABLE IF NOT EXISTS shadow_positions (
    strategy_id  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    qty          REAL NOT NULL,
    entry_price  REAL NOT NULL,
    entry_time   TEXT NOT NULL,
    stop_price   REAL NOT NULL,
    initial_stop_price REAL NOT NULL,
    highest_close REAL NOT NULL,
    lowest_close REAL NOT NULL,
    entry_fees   REAL NOT NULL DEFAULT 0,
    signal_json  TEXT,
    PRIMARY KEY (strategy_id, symbol)
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (decided_at);
CREATE INDEX IF NOT EXISTS idx_decisions_strategy ON decisions (strategy_id, decided_at);
CREATE INDEX IF NOT EXISTS idx_attr_strategy ON attributions (strategy_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features (symbol, market, timeframe, source, feature_version, timestamp);
CREATE INDEX IF NOT EXISTS idx_research_strategy ON research_runs (strategy_id, run_at);
CREATE INDEX IF NOT EXISTS idx_drift_subject ON drift_metrics (subject, computed_at);
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


class QuantStore:
    """Thread-safe persistent store for the quant desk layer."""

    def __init__(self, db_path: str = DEFAULT_QUANT_DB_PATH):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if found > SCHEMA_VERSION:
                self._conn.close()
                raise RuntimeError(
                    f"Quant database {self.db_path!r} was written by a newer build "
                    f"(schema v{found}; this build understands v{SCHEMA_VERSION}). "
                    "Upgrade the application rather than downgrading the database."
                )
            self._conn.executescript(_SCHEMA)
            # Migrations for future versions belong here, gated on ``found``,
            # exactly like StateStore: additive ALTERs keyed off PRAGMA
            # table_info so existing databases upgrade in place.
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
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
    # Decisions (the journal: every verdict, including rejections)
    # ------------------------------------------------------------------

    def record_decision(
        self, decision: OpportunityDecision, session_id: Optional[str] = None
    ) -> int:
        candidate = decision.candidate
        cur = self._execute(
            "INSERT INTO decisions (decided_at, signal_timestamp, symbol, market,"
            " timeframe, strategy_id, strategy_version, action, outcome, rank,"
            " confidence, expected_gross_alpha_bps, expected_cost_bps,"
            " expected_net_alpha_bps, opportunity_score, risk_allocation_pct,"
            " reason, rejection_reasons, decision_json, session_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _to_iso(decision.decided_at or _utc_now()),
                _to_iso(candidate.timestamp),
                candidate.symbol,
                candidate.market.value,
                candidate.timeframe,
                candidate.strategy_id,
                candidate.strategy_version,
                candidate.action.value,
                decision.outcome.value,
                decision.rank,
                candidate.confidence,
                candidate.expected_gross_alpha_bps,
                decision.expected_cost_bps,
                decision.expected_net_alpha_bps,
                decision.opportunity_score,
                decision.risk_allocation_pct,
                candidate.signal_reason,
                json.dumps(decision.rejection_reasons),
                json.dumps(decision.to_dict()),
                session_id,
            ),
        )
        return int(cur.lastrowid)

    def get_decisions(
        self,
        limit: int = 200,
        strategy_id: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM decisions"
        clauses, params = [], []
        if strategy_id:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._query(sql, tuple(params))]

    def decision_counts_since(self, since: datetime) -> dict[str, int]:
        rows = self._query(
            "SELECT outcome, COUNT(*) AS n FROM decisions WHERE decided_at >= ?"
            " GROUP BY outcome",
            (_to_iso(since),),
        )
        return {row["outcome"]: int(row["n"]) for row in rows}

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def record_attribution(self, record: AttributionRecord) -> int:
        cur = self._execute(
            "INSERT INTO attributions (recorded_at, strategy_id, strategy_version,"
            " symbol, market, timeframe, session_id, signal_timestamp,"
            " entry_timestamp, exit_timestamp, qty, entry_price, exit_price,"
            " initial_stop_price, fees, expected_cost_bps, realized_slippage_bps,"
            " gross_pnl, net_pnl, r_multiple, mfe_pct, mae_pct, entry_reason,"
            " exit_reason, regime_at_entry, opportunity_score, shadow, features_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _to_iso(_utc_now()),
                record.strategy_id,
                record.strategy_version,
                record.symbol,
                record.market.value,
                record.timeframe,
                record.session_id,
                _to_iso(record.signal_timestamp),
                _to_iso(record.entry_timestamp),
                _to_iso(record.exit_timestamp),
                record.qty,
                record.entry_price,
                record.exit_price,
                record.initial_stop_price,
                record.fees,
                record.expected_cost_bps,
                record.realized_slippage_bps,
                record.gross_pnl,
                record.net_pnl,
                record.r_multiple,
                record.max_favorable_excursion_pct,
                record.max_adverse_excursion_pct,
                record.entry_reason,
                record.exit_reason,
                record.regime_at_entry,
                record.opportunity_score,
                1 if record.shadow else 0,
                json.dumps(record.features_at_entry),
            ),
        )
        return int(cur.lastrowid)

    def get_attributions(
        self,
        strategy_id: Optional[str] = None,
        include_shadow: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM attributions"
        clauses, params = [], []
        if strategy_id:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        if not include_shadow:
            clauses.append("shadow = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._query(sql, tuple(params))]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_strategy_state(self, strategy_id: str) -> Optional[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM lifecycle WHERE strategy_id = ?", (strategy_id,)
        )
        if not rows:
            return None
        d = dict(rows[0])
        d["operator_ack"] = bool(d["operator_ack"])
        return d

    def get_all_strategy_states(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM lifecycle ORDER BY strategy_id")
        out = []
        for row in rows:
            d = dict(row)
            d["operator_ack"] = bool(d["operator_ack"])
            out.append(d)
        return out

    def set_strategy_state(
        self,
        strategy_id: str,
        from_state: StrategyState,
        to_state: StrategyState,
        reason: str,
        evidence: Optional[dict[str, Any]] = None,
        operator_ack: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a lifecycle transition and append it to promotion history.

        This is the storage layer only — legality of the transition is the
        lifecycle manager's job.  Both writes happen in one transaction so
        history can never disagree with current state.
        """
        now_iso = _to_iso(timestamp or _utc_now())
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO lifecycle (strategy_id, state, since, reason,"
                    " operator_ack, updated_at) VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(strategy_id) DO UPDATE SET"
                    " state = excluded.state, since = excluded.since,"
                    " reason = excluded.reason, operator_ack = excluded.operator_ack,"
                    " updated_at = excluded.updated_at",
                    (
                        strategy_id,
                        to_state.value,
                        now_iso,
                        reason,
                        1 if operator_ack else 0,
                        now_iso,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO promotion_history (strategy_id, from_state, to_state,"
                    " timestamp, reason, evidence_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        strategy_id,
                        from_state.value,
                        to_state.value,
                        now_iso,
                        reason,
                        json.dumps(evidence) if evidence is not None else None,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_promotion_history(
        self, strategy_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self._query(
                "SELECT * FROM promotion_history WHERE strategy_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (strategy_id, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM promotion_history ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Research runs (multiple-testing bookkeeping lives here)
    # ------------------------------------------------------------------

    def record_research_run(
        self,
        *,
        strategy_id: str,
        params: dict[str, Any],
        universe: list[str],
        data_window: str,
        candidate_count: int,
        trial_count: int,
        passed: bool,
        gates: dict[str, Any],
        cost_assumptions: Optional[dict[str, Any]] = None,
        hypothesis_id: Optional[str] = None,
        git_commit: Optional[str] = None,
        seed: Optional[int] = None,
        result: Optional[dict[str, Any]] = None,
        run_at: Optional[datetime] = None,
    ) -> int:
        cur = self._execute(
            "INSERT INTO research_runs (run_at, strategy_id, hypothesis_id,"
            " params_json, universe_json, data_window, candidate_count,"
            " trial_count, passed, gates_json, cost_assumptions_json,"
            " git_commit, seed, result_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _to_iso(run_at or _utc_now()),
                strategy_id,
                hypothesis_id,
                json.dumps(params, default=str),
                json.dumps(universe),
                data_window,
                candidate_count,
                trial_count,
                1 if passed else 0,
                json.dumps(gates, default=str),
                json.dumps(cost_assumptions, default=str) if cost_assumptions else None,
                git_commit,
                seed,
                json.dumps(result, default=str) if result else None,
            ),
        )
        return int(cur.lastrowid)

    def get_research_runs(
        self, strategy_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self._query(
                "SELECT * FROM research_runs WHERE strategy_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (strategy_id, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM research_runs ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in rows]

    def total_trials(self, strategy_id: Optional[str] = None) -> int:
        """Cumulative number of evaluated candidates, for deflated-Sharpe
        style multiple-testing corrections.  Counts every recorded run's
        trial_count so 'the best of 10,000 experiments' stays visible."""
        if strategy_id:
            rows = self._query(
                "SELECT COALESCE(SUM(trial_count), 0) AS n FROM research_runs"
                " WHERE strategy_id = ?",
                (strategy_id,),
            )
        else:
            rows = self._query(
                "SELECT COALESCE(SUM(trial_count), 0) AS n FROM research_runs"
            )
        return int(rows[0]["n"])

    # ------------------------------------------------------------------
    # Point-in-time features
    # ------------------------------------------------------------------

    def record_features(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        market: Market,
        timeframe: str,
        source: str,
        feature_version: str,
        payload: dict[str, Any],
        available_at: Optional[datetime] = None,
    ) -> None:
        """Store one feature snapshot.  ``available_at`` is when the data
        became knowable (publication time); it defaults to ``timestamp``
        and must never be earlier than it."""
        available = available_at or timestamp
        if _to_iso(available) < _to_iso(timestamp):  # type: ignore[operator]
            raise ValueError("available_at cannot precede the observation timestamp.")
        self._execute(
            "INSERT INTO features (timestamp, symbol, market, timeframe, source,"
            " feature_version, available_at, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(timestamp, symbol, market, timeframe, source, feature_version)"
            " DO UPDATE SET payload_json = excluded.payload_json,"
            " available_at = excluded.available_at",
            (
                _to_iso(timestamp),
                symbol,
                market.value,
                timeframe,
                source,
                feature_version,
                _to_iso(available),
                json.dumps(payload, default=str),
            ),
        )

    def get_features_asof(
        self,
        *,
        symbol: str,
        market: Market,
        timeframe: str,
        source: str,
        feature_version: str,
        asof: datetime,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Latest feature snapshots whose AVAILABILITY time is <= ``asof``.

        Filtering on ``available_at`` (not ``timestamp``) is the point-in-
        time guarantee: a snapshot published late is invisible until its
        publication moment, so historical research cannot peek.
        """
        rows = self._query(
            "SELECT * FROM features WHERE symbol = ? AND market = ? AND timeframe = ?"
            " AND source = ? AND feature_version = ? AND available_at <= ?"
            " ORDER BY timestamp DESC LIMIT ?",
            (
                symbol,
                market.value,
                timeframe,
                source,
                feature_version,
                _to_iso(asof),
                limit,
            ),
        )
        out = []
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["timestamp"] = _from_iso(d["timestamp"])
            d["available_at"] = _from_iso(d["available_at"])
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_id: str,
        version: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        calibration: Optional[dict[str, Any]] = None,
        data_window: Optional[str] = None,
        feature_version: Optional[str] = None,
        trained_at: Optional[datetime] = None,
    ) -> None:
        self._execute(
            "INSERT INTO model_registry (model_id, version, trained_at, data_window,"
            " feature_version, params_json, metrics_json, calibration_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(model_id, version) DO UPDATE SET"
            " trained_at = excluded.trained_at, data_window = excluded.data_window,"
            " feature_version = excluded.feature_version,"
            " params_json = excluded.params_json, metrics_json = excluded.metrics_json,"
            " calibration_json = excluded.calibration_json",
            (
                model_id,
                version,
                _to_iso(trained_at or _utc_now()),
                data_window,
                feature_version,
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                json.dumps(calibration, default=str) if calibration else None,
            ),
        )

    def get_models(self, model_id: Optional[str] = None) -> list[dict[str, Any]]:
        if model_id:
            rows = self._query(
                "SELECT * FROM model_registry WHERE model_id = ? ORDER BY trained_at DESC",
                (model_id,),
            )
        else:
            rows = self._query("SELECT * FROM model_registry ORDER BY trained_at DESC")
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Drift metrics and risk events
    # ------------------------------------------------------------------

    def record_drift(
        self,
        *,
        subject: str,
        metric: str,
        value: Optional[float],
        threshold: Optional[float],
        breached: bool,
        details: Optional[dict[str, Any]] = None,
        computed_at: Optional[datetime] = None,
    ) -> None:
        self._execute(
            "INSERT INTO drift_metrics (computed_at, subject, metric, value,"
            " threshold, breached, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _to_iso(computed_at or _utc_now()),
                subject,
                metric,
                value,
                threshold,
                1 if breached else 0,
                json.dumps(details, default=str) if details else None,
            ),
        )

    def get_drift(
        self, subject: Optional[str] = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if subject:
            rows = self._query(
                "SELECT * FROM drift_metrics WHERE subject = ? ORDER BY id DESC LIMIT ?",
                (subject, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM drift_metrics ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in rows]

    def record_risk_event(
        self,
        *,
        source: str,
        kind: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._execute(
            "INSERT INTO risk_events (timestamp, source, kind, message, context_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                _to_iso(timestamp or _utc_now()),
                source,
                kind,
                message,
                json.dumps(context, default=str) if context else None,
            ),
        )

    def get_risk_events(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM risk_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Shadow positions
    # ------------------------------------------------------------------

    def upsert_shadow_position(self, data: dict[str, Any]) -> None:
        required = {
            "strategy_id",
            "symbol",
            "qty",
            "entry_price",
            "entry_time",
            "stop_price",
            "initial_stop_price",
            "highest_close",
            "lowest_close",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Shadow position missing fields: {sorted(missing)}")
        self._execute(
            "INSERT INTO shadow_positions (strategy_id, symbol, qty, entry_price,"
            " entry_time, stop_price, initial_stop_price, highest_close,"
            " lowest_close, entry_fees, signal_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(strategy_id, symbol) DO UPDATE SET"
            " qty = excluded.qty, entry_price = excluded.entry_price,"
            " entry_time = excluded.entry_time, stop_price = excluded.stop_price,"
            " initial_stop_price = excluded.initial_stop_price,"
            " highest_close = excluded.highest_close,"
            " lowest_close = excluded.lowest_close,"
            " entry_fees = excluded.entry_fees, signal_json = excluded.signal_json",
            (
                data["strategy_id"],
                data["symbol"],
                data["qty"],
                data["entry_price"],
                data["entry_time"],
                data["stop_price"],
                data["initial_stop_price"],
                data["highest_close"],
                data["lowest_close"],
                data.get("entry_fees", 0.0),
                json.dumps(data.get("signal", {}), default=str),
            ),
        )

    def get_shadow_positions(
        self, strategy_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self._query(
                "SELECT * FROM shadow_positions WHERE strategy_id = ? ORDER BY symbol",
                (strategy_id,),
            )
        else:
            rows = self._query("SELECT * FROM shadow_positions ORDER BY strategy_id, symbol")
        return [dict(row) for row in rows]

    def delete_shadow_position(self, strategy_id: str, symbol: str) -> None:
        self._execute(
            "DELETE FROM shadow_positions WHERE strategy_id = ? AND symbol = ?",
            (strategy_id, symbol),
        )

    # ------------------------------------------------------------------
    # Key/value
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
