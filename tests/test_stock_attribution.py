"""The stock engine's post-trade attribution path.

Before this, ``trade_engine`` journaled its *decisions* but never what
those decisions earned: ``DecisionJournal`` had ``record_crypto_exit_fill``
and no stock equivalent, so the marketplace could only ever see crypto and
every equity strategy showed zero trades forever.

These tests drive the real engine through real exits with a real (temp)
desk database and assert on the rows that come out. They reuse the
existing fake broker/feed harness, so no network and no credentials.
"""

from __future__ import annotations

import pytest

from broker import make_client_order_id
from models import OrderRecord
from quant.attach import attach_journal, reset_shared_stores
from quant.contracts import Market
from quant.store import QuantStore
from tests.test_trade_engine import (
    NOW,
    SYMBOL,
    build_harness,
    preset_bot_position,
)


@pytest.fixture(autouse=True)
def _clean_shared_stores():
    reset_shared_stores()
    yield
    reset_shared_stores()


def _with_journal(h, tmp_path):
    """Attach a desk journal to an already-built harness engine."""
    journal = attach_journal(
        market=Market.US_EQUITY,
        timeframe=h.strategy_cfg.timeframe,
        strategy=h.engine.strategy,
        db_path=str(tmp_path / "desk.db"),
    )
    assert journal is not None
    h.engine.quant_journal = journal
    return journal


def _attributions(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    try:
        return store.get_attributions()
    finally:
        store.close()


def _fill_exit(h, qty=10.0, price=105.0, reason="bearish_crossover"):
    """Put a fully-filled exit sell in front of startup reconciliation."""
    pos = h.store.get_bot_position(SYMBOL)
    stop_cid = pos["stop_client_order_id"]
    h.broker.orders[stop_cid].status = "canceled"
    exit_cid = make_client_order_id(h.engine.session_id, "exit", NOW, symbol=SYMBOL)
    h.broker.orders[exit_cid] = OrderRecord(
        "exit-1", exit_cid, SYMBOL, "exit", "sell", qty, qty,
        "filled", price, submitted_at=NOW, updated_at=NOW,
    )
    pos.update(
        stop_client_order_id=None,
        stop_price=None,
        exit_client_order_id=exit_cid,
        exit_pending_reason=reason,
    )
    h.store.set_bot_position(SYMBOL, pos)
    del h.broker.positions[SYMBOL]
    return exit_cid


# ---------------------------------------------------------------------------
# The row exists at all
# ---------------------------------------------------------------------------


def test_a_stock_exit_produces_an_attribution_row(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=105.0)

    h.engine._startup()

    rows = _attributions(tmp_path)
    assert len(rows) == 1, "the stock engine must now attribute its exits"
    row = rows[0]
    assert row["symbol"] == SYMBOL
    assert row["market"] == Market.US_EQUITY.value
    assert row["strategy_id"] == "stock_trend"


def test_the_attribution_matches_the_trade_ledger(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=105.0)

    h.engine._startup()

    sells = [t for t in h.store.get_trades() if t["action"] == "SELL"]
    rows = _attributions(tmp_path)
    assert len(sells) == len(rows) == 1
    assert rows[0]["qty"] == pytest.approx(sells[0]["qty"])
    assert rows[0]["gross_pnl"] == pytest.approx(sells[0]["realized_pnl"])


def test_pnl_and_r_multiple_are_computed_from_the_entry_and_initial_stop(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    pos = h.store.get_bot_position(SYMBOL)
    pos["initial_stop_price"] = 95.0
    h.store.set_bot_position(SYMBOL, pos)
    _fill_exit(h, qty=10.0, price=105.0)

    h.engine._startup()

    row = _attributions(tmp_path)[0]
    assert row["gross_pnl"] == pytest.approx(50.0)
    # Alpaca equities are commission-free: net equals gross, not an
    # invented fee. Spread/slippage live in the executions ledger.
    assert row["net_pnl"] == pytest.approx(50.0)
    assert row["fees"] == pytest.approx(0.0)
    # Risk at stop is (100 - 95) * 10 = 50, so a +50 net is exactly 1R.
    assert row["r_multiple"] == pytest.approx(1.0)


def test_the_exit_reason_is_carried_through(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=105.0, reason="trailing_close_breach")

    h.engine._startup()

    assert _attributions(tmp_path)[0]["exit_reason"] == "trailing_close_breach"


def test_a_losing_exit_is_attributed_with_a_negative_pnl(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=97.0)

    h.engine._startup()

    row = _attributions(tmp_path)[0]
    assert row["gross_pnl"] == pytest.approx(-30.0)
    assert row["net_pnl"] < 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_re_reconciling_the_same_fill_does_not_double_count_it(tmp_path):
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=105.0)

    h.engine._startup()
    h.engine._book_pending_sell_fills(
        h.store.get_bot_position(SYMBOL) or {}, NOW
    )

    assert len(_attributions(tmp_path)) == 1


# ---------------------------------------------------------------------------
# The observer contract
# ---------------------------------------------------------------------------


def test_trading_is_unchanged_when_no_journal_is_attached(tmp_path):
    """The whole point: attaching an observer must not alter behavior."""
    without = build_harness(tmp_path / "a")
    preset_bot_position(without, qty=10, entry_price=100.0)
    _fill_exit(without, qty=10.0, price=105.0)
    without.engine._startup()
    baseline = [
        (t["action"], t["qty"], t["fill_price"], t["realized_pnl"])
        for t in without.store.get_trades()
    ]

    with_journal = build_harness(tmp_path / "b")
    _with_journal(with_journal, tmp_path)
    preset_bot_position(with_journal, qty=10, entry_price=100.0)
    _fill_exit(with_journal, qty=10.0, price=105.0)
    with_journal.engine._startup()
    observed = [
        (t["action"], t["qty"], t["fill_price"], t["realized_pnl"])
        for t in with_journal.store.get_trades()
    ]

    assert observed == baseline
    assert with_journal.store.get_bot_position(SYMBOL) is None


def test_a_broken_journal_never_stops_the_exit_from_being_booked(tmp_path):
    h = build_harness(tmp_path)
    journal = _with_journal(h, tmp_path)
    journal.store.close()  # every write from here on raises

    preset_bot_position(h, qty=10, entry_price=100.0)
    _fill_exit(h, qty=10.0, price=105.0)

    h.engine._startup()  # must not raise

    sells = [t for t in h.store.get_trades() if t["action"] == "SELL"]
    assert len(sells) == 1, "the trade ledger is authoritative and unaffected"
    assert h.store.get_bot_position(SYMBOL) is None


def test_no_attribution_is_invented_without_a_local_position_record(tmp_path):
    """An exit booked after the position was reconciled away has no entry
    price. A row with a fabricated entry would be worse than no row."""
    h = build_harness(tmp_path)
    _with_journal(h, tmp_path)
    exit_cid = make_client_order_id(h.engine.session_id, "exit", NOW, symbol=SYMBOL)
    h.broker.orders[exit_cid] = OrderRecord(
        "orphan-exit", exit_cid, SYMBOL, "exit", "sell", 10.0, 10.0,
        "filled", 105.0, submitted_at=NOW, updated_at=NOW,
    )
    h.engine._pending_exit = {"client_order_id": exit_cid, "reason": "exit"}

    h.engine._reconcile_exit_order(NOW)

    assert _attributions(tmp_path) == []
