"""Attaching the desk journal to a live engine.

The journal adapter and both engines' journal call sites already existed;
nothing ever constructed a journal, so every one of those call sites sat
behind a permanently-None guard. These tests lock the constructor and,
more importantly, the two properties that make attaching it safe: it
cannot break trading, and it cannot double-count a fill.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from quant.attach import attach_journal, reset_shared_stores, shared_quant_store
from quant.contracts import AttributionRecord, Market
from quant.journal import DecisionJournal
from quant.store import SCHEMA_VERSION, QuantStore


class _Strategy:
    """Stands in for a live strategy instance."""


class TrendStrategy(_Strategy):
    pass


class CrossSectionalMomentumStrategy(_Strategy):
    pass


@pytest.fixture(autouse=True)
def _clean_shared_stores():
    reset_shared_stores()
    yield
    reset_shared_stores()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_attach_builds_a_journal_tagged_with_the_registry_strategy_id(tmp_path):
    journal = attach_journal(
        market=Market.US_EQUITY,
        timeframe="15Min",
        strategy=TrendStrategy(),
        db_path=str(tmp_path / "desk.db"),
    )
    assert isinstance(journal, DecisionJournal)
    # Registry id, not the class name — journal rows have to join against
    # the strategy registry, marketplace and lifecycle ladder.
    assert journal.strategy_id == "stock_trend"
    assert journal.market is Market.US_EQUITY
    assert journal.timeframe == "15Min"


def test_attach_maps_the_crypto_decision_layers_too(tmp_path):
    journal = attach_journal(
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy=CrossSectionalMomentumStrategy(),
        db_path=str(tmp_path / "desk.db"),
    )
    assert journal.strategy_id == "crypto_cross_sectional"


def test_an_unknown_strategy_class_still_journals_under_its_class_name(tmp_path):
    class SomethingNew(_Strategy):
        pass

    journal = attach_journal(
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy=SomethingNew(),
        db_path=str(tmp_path / "desk.db"),
    )
    # A naming gap must cost a join, never a dropped row.
    assert journal.strategy_id == "SomethingNew"


def test_the_store_is_shared_per_path_so_reruns_do_not_leak_connections(tmp_path):
    path = str(tmp_path / "desk.db")
    first = shared_quant_store(path)
    second = shared_quant_store(path)
    assert first is second


def test_a_different_path_gets_a_different_store(tmp_path):
    assert shared_quant_store(str(tmp_path / "a.db")) is not shared_quant_store(
        str(tmp_path / "b.db")
    )


# ---------------------------------------------------------------------------
# The observer can never break the observed
# ---------------------------------------------------------------------------


def test_attach_returns_none_when_the_database_cannot_be_opened(tmp_path):
    # A path whose parent is a FILE cannot be created as a directory.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    journal = attach_journal(
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy=TrendStrategy(),
        db_path=str(blocker / "desk.db"),
    )
    assert journal is None, "an unopenable desk DB must disable journaling, not raise"


def test_attach_returns_none_for_a_database_from_a_newer_build(tmp_path):
    path = tmp_path / "desk.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.commit()
    conn.close()

    assert attach_journal(
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy=TrendStrategy(),
        db_path=str(path),
    ) is None


def test_a_failing_store_never_propagates_out_of_the_journal(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    warnings: list[str] = []
    journal = DecisionJournal(
        store,
        market=Market.US_EQUITY,
        timeframe="15Min",
        strategy_id="stock_trend",
        warn=warnings.append,
    )
    store.close()  # every subsequent write raises ProgrammingError

    journal.record_stock_exit_fill(
        symbol="SPY",
        session_id="s1",
        qty=10,
        exit_price=101.0,
        exit_reason="exit",
        entry_price=100.0,
    )
    assert warnings, "a failed attribution write must be reported, not silent"


# ---------------------------------------------------------------------------
# Idempotency: a re-booked fill must not become two trades
# ---------------------------------------------------------------------------


def _record(store, event_key, **overrides):
    payload = dict(
        strategy_id="stock_trend",
        strategy_version="1.0.0",
        symbol="SPY",
        market=Market.US_EQUITY,
        timeframe="15Min",
        session_id="s1",
        signal_timestamp=None,
        entry_timestamp=None,
        exit_timestamp=datetime(2024, 6, 12, 15, 0, tzinfo=timezone.utc),
        qty=10.0,
        entry_price=100.0,
        exit_price=101.0,
        initial_stop_price=98.0,
        fees=0.0,
        expected_cost_bps=None,
        realized_slippage_bps=None,
        gross_pnl=10.0,
        net_pnl=10.0,
        r_multiple=0.5,
        max_favorable_excursion_pct=1.0,
        max_adverse_excursion_pct=0.5,
        event_key=event_key,
    )
    payload.update(overrides)
    return store.record_attribution(AttributionRecord(**payload))


def test_the_same_event_key_books_one_attribution_row(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    first = _record(store, "fill:abc:sell")
    second = _record(store, "fill:abc:sell")
    assert first == second
    assert len(store.get_attributions()) == 1
    store.close()


def test_different_event_keys_book_separate_rows(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    _record(store, "fill:abc:sell")
    _record(store, "fill:def:sell")
    assert len(store.get_attributions()) == 2
    store.close()


def test_unkeyed_rows_are_never_deduplicated(tmp_path):
    # Shadow fills and backfills carry no key and must stay independent.
    store = QuantStore(str(tmp_path / "desk.db"))
    _record(store, None)
    _record(store, None)
    assert len(store.get_attributions()) == 2
    store.close()


def test_a_partial_close_books_one_row_per_distinct_fill(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    _record(store, "fill:o1:sell", qty=4.0)
    _record(store, "fill:o2:sell", qty=6.0)
    rows = store.get_attributions()
    assert sum(row["qty"] for row in rows) == pytest.approx(10.0)
    store.close()


# ---------------------------------------------------------------------------
# Schema migration: an existing user database must upgrade in place
# ---------------------------------------------------------------------------


_V1_ATTRIBUTIONS = """
CREATE TABLE attributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    session_id TEXT,
    signal_timestamp TEXT,
    entry_timestamp TEXT,
    exit_timestamp TEXT,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    initial_stop_price REAL,
    fees REAL NOT NULL DEFAULT 0,
    expected_cost_bps REAL,
    realized_slippage_bps REAL,
    gross_pnl REAL,
    net_pnl REAL,
    r_multiple REAL,
    mfe_pct REAL,
    mae_pct REAL,
    entry_reason TEXT,
    exit_reason TEXT,
    regime_at_entry TEXT,
    opportunity_score REAL,
    shadow INTEGER NOT NULL DEFAULT 0,
    features_json TEXT
);
"""


def _make_v1_database(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_V1_ATTRIBUTIONS)
    conn.execute(
        "INSERT INTO attributions (recorded_at, strategy_id, strategy_version,"
        " symbol, market, timeframe, qty, entry_price)"
        " VALUES ('2024-01-01T00:00:00+00:00', 'crypto_cross_sectional', '1.0.0',"
        " 'BTC/USD', 'crypto', '1Hour', 1.0, 40000.0)"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def test_a_v1_database_gains_the_event_key_column_in_place(tmp_path):
    path = tmp_path / "desk.db"
    _make_v1_database(path)

    store = QuantStore(str(path))
    columns = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(attributions)").fetchall()
    }
    assert "event_key" in columns
    assert int(store._conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    store.close()


def test_migration_preserves_the_existing_rows(tmp_path):
    path = tmp_path / "desk.db"
    _make_v1_database(path)

    store = QuantStore(str(path))
    rows = store.get_attributions()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC/USD"
    store.close()


def test_a_migrated_database_enforces_the_new_uniqueness(tmp_path):
    path = tmp_path / "desk.db"
    _make_v1_database(path)

    store = QuantStore(str(path))
    _record(store, "fill:xyz:sell")
    _record(store, "fill:xyz:sell")
    keyed = [r for r in store.get_attributions() if r["event_key"] == "fill:xyz:sell"]
    assert len(keyed) == 1
    store.close()


def test_migration_is_idempotent_across_reopens(tmp_path):
    path = tmp_path / "desk.db"
    _make_v1_database(path)
    for _ in range(3):
        store = QuantStore(str(path))
        store.close()
    store = QuantStore(str(path))
    assert len(store.get_attributions()) == 1
    store.close()


def test_a_partially_migrated_database_converges(tmp_path):
    # Crash between the ALTER and the version bump: column present, version
    # still 1. Reopening must succeed rather than trying to ALTER twice.
    path = tmp_path / "desk.db"
    _make_v1_database(path)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE attributions ADD COLUMN event_key TEXT")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = QuantStore(str(path))
    assert int(store._conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    store.close()
