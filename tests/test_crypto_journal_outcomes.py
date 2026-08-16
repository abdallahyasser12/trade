"""A crypto BUY is journaled with the outcome it actually had.

crypto_engine journaled every decision immediately after
evaluate_prepared, and quant/journal.py maps a BUY with no block reason to
DecisionOutcome.EXECUTE. Every entry later rejected by quote safety,
correlation, cost-to-risk, sizing, exposure or heat was therefore recorded
as EXECUTED — in the same decisions table the marketplace and lifecycle
read. Nobody noticed because no journal was ever attached.
"""

from __future__ import annotations

import pytest

from quant.attach import attach_journal, reset_shared_stores
from quant.contracts import Market
from quant.store import QuantStore
from tests.test_crypto_engine import make_engine


@pytest.fixture(autouse=True)
def _clean_shared_stores():
    reset_shared_stores()
    yield
    reset_shared_stores()


def _attach(engine, tmp_path):
    journal = attach_journal(
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy=engine.strategy,
        db_path=str(tmp_path / "desk.db"),
    )
    assert journal is not None
    engine.quant_journal = journal
    return journal


def _decisions(tmp_path):
    store = QuantStore(str(tmp_path / "desk.db"))
    try:
        return store.get_decisions()
    finally:
        store.close()


def test_an_executed_buy_is_journaled_exactly_once_as_execute(tmp_path):
    engine, broker, store, _ = make_engine(tmp_path)
    _attach(engine, tmp_path)
    engine._startup()
    engine.run_iteration()

    buys = [call for call in broker.market_calls if call[2] == "buy"]
    assert len(buys) == 1, "harness must actually take an entry for this to mean anything"

    rows = [r for r in _decisions(tmp_path) if r["action"] == "BUY"]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "EXECUTE"


def test_a_rejected_buy_is_never_journaled_as_executed(tmp_path):
    """The bug: a blocked entry recorded as EXECUTE.

    Cost-to-risk rejection is the cheapest gate to trip deterministically.
    Whatever the engine does with the entry, it must not claim it executed.
    """
    from crypto_config import CryptoEngineConfig

    engine, broker, store, _ = make_engine(
        tmp_path,
        # An implausible spread makes round-trip cost eat the risk unit.
        engine_config=CryptoEngineConfig(
            poll_interval_seconds=5, expected_slippage_bps=5_000.0
        ),
    )
    _attach(engine, tmp_path)
    engine._startup()
    engine.run_iteration()

    buys = [call for call in broker.market_calls if call[2] == "buy"]
    assert buys == [], "this test needs the entry to be rejected"

    executed = [
        r for r in _decisions(tmp_path)
        if r["action"] == "BUY" and r["outcome"] == "EXECUTE"
    ]
    assert executed == [], (
        "a BUY that never reached the broker must not be journaled as EXECUTE"
    )


def test_exit_and_hold_decisions_are_still_journaled_at_signal_time(tmp_path):
    engine, broker, store, _ = make_engine(tmp_path)
    _attach(engine, tmp_path)
    engine._startup()
    engine.run_iteration()

    rows = _decisions(tmp_path)
    assert rows, "non-BUY decisions must still be recorded"
