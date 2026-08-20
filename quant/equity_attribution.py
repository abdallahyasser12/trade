"""Crash-idempotent real-fill attribution for the market-wide equity desk.

``StateStore`` and ``QuantStore`` deliberately use separate SQLite databases, so a
broker fill cannot be committed atomically to both.  The execution state therefore
remains the recovery authority, while this adapter makes replay into QuantStore
idempotent: one quant transaction inserts the attribution and a durable event marker.

The helper is intentionally small and lives inside the quant package.  It uses
QuantStore's guarded SQLite connection so the attribution row and marker share one
transaction without widening the public QuantStore API for a single desk-specific
recovery concern.
"""

from __future__ import annotations

import json

from quant.contracts import AttributionRecord
from quant.store import QuantStore, _to_iso, _utc_now

_EVENT_PREFIX = "marketwide_equity.attribution_event.v1:"


def record_equity_attribution_once(
    store: QuantStore,
    record: AttributionRecord,
    *,
    event_key: str,
) -> bool:
    """Insert one real attribution exactly once for a stable broker-fill event key.

    Returns ``True`` when this call inserted the row and ``False`` when a prior crash
    or reconciliation pass already committed the same event.
    """
    if not isinstance(event_key, str) or not event_key.strip():
        raise ValueError("Equity attribution event_key must be a non-empty string.")
    if record.shadow:
        raise ValueError("Real equity fill attribution cannot be marked as shadow.")

    marker = _EVENT_PREFIX + event_key.strip()
    with store._lock:  # package-internal transaction over the QuantStore connection
        try:
            existing = store._conn.execute(
                "SELECT 1 FROM kv WHERE key = ?",
                (marker,),
            ).fetchone()
            if existing is not None:
                return False

            now_iso = _to_iso(_utc_now())
            cursor = store._conn.execute(
                "INSERT INTO attributions (recorded_at, strategy_id, strategy_version,"
                " symbol, market, timeframe, session_id, signal_timestamp,"
                " entry_timestamp, exit_timestamp, qty, entry_price, exit_price,"
                " initial_stop_price, fees, expected_cost_bps, realized_slippage_bps,"
                " gross_pnl, net_pnl, r_multiple, mfe_pct, mae_pct, entry_reason,"
                " exit_reason, regime_at_entry, opportunity_score, shadow, features_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                " ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now_iso,
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
                    0,
                    json.dumps(record.features_at_entry),
                ),
            )
            store._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)",
                (marker, str(int(cursor.lastrowid))),
            )
            store._conn.commit()
            return True
        except Exception:
            store._conn.rollback()
            raise
