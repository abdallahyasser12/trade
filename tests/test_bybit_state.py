from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bybit_state import BybitStateStore


def record_intent(store, intent="intent-1", link="tb-e-1234567890-a1", purpose="entry", qty=1.0):
    store.record_order_intent(
        internal_intent_id=intent,
        order_link_id=link,
        strategy_symbol="BTC/USD",
        venue_symbol="BTCUSDT",
        purpose=purpose,
        attempt=1,
        requested_qty=qty,
    )


def test_execution_idempotency_and_partial_exit_inventory(tmp_path):
    path = str(tmp_path / "venue.db")
    store = BybitStateStore(path)
    now = datetime.now(timezone.utc)
    record_intent(store)
    assert store.record_execution(
        execution_id="buy-1",
        internal_intent_id="intent-1",
        order_link_id="tb-e-1234567890-a1",
        order_id="o1",
        strategy_symbol="BTC/USD",
        venue_symbol="BTCUSDT",
        base_asset="BTC",
        side="buy",
        qty=1.0,
        price=100.0,
        fee=0.0,
        fee_asset="USDT",
        execution_at=now,
        raw={"execId": "buy-1"},
    ) is True
    assert store.record_execution(
        execution_id="buy-1",
        internal_intent_id="intent-1",
        order_link_id="tb-e-1234567890-a1",
        order_id="o1",
        strategy_symbol="BTC/USD",
        venue_symbol="BTCUSDT",
        base_asset="BTC",
        side="buy",
        qty=1.0,
        price=100.0,
        fee=0.0,
        fee_asset="USDT",
        execution_at=now,
        raw={"execId": "buy-1"},
    ) is False

    record_intent(store, intent="intent-2", link="tb-p-1234567890-a1", purpose="partial", qty=0.4)
    store.record_execution(
        execution_id="sell-1",
        internal_intent_id="intent-2",
        order_link_id="tb-p-1234567890-a1",
        order_id="o2",
        strategy_symbol="BTC/USD",
        venue_symbol="BTCUSDT",
        base_asset="BTC",
        side="sell",
        qty=0.4,
        price=110.0,
        fee=0.0,
        fee_asset="USDT",
        execution_at=now + timedelta(seconds=1),
        raw={"execId": "sell-1"},
    )
    inv = store.inventory("BTC/USD")
    assert inv.qty == pytest.approx(0.6)
    assert inv.cost_basis_quote == pytest.approx(60.0)
    assert inv.realized_pnl_quote == pytest.approx(4.0)
    store.close()

    reopened = BybitStateStore(path)
    try:
        inv = reopened.inventory("BTC/USD")
        assert inv.qty == pytest.approx(0.6)
        assert reopened.order_by_intent("intent-1")["order_link_id"] == "tb-e-1234567890-a1"
    finally:
        reopened.close()


def test_negative_inventory_is_impossible(tmp_path):
    store = BybitStateStore(str(tmp_path / "venue.db"))
    record_intent(store, intent="exit", link="tb-x-1234567890-a1", purpose="exit", qty=1.0)
    with pytest.raises(RuntimeError, match="negative"):
        store.record_execution(
            execution_id="sell",
            internal_intent_id="exit",
            order_link_id="tb-x-1234567890-a1",
            order_id="o",
            strategy_symbol="BTC/USD",
            venue_symbol="BTCUSDT",
            base_asset="BTC",
            side="sell",
            qty=1.0,
            price=100.0,
            fee=0.0,
            fee_asset="USDT",
            execution_at=datetime.now(timezone.utc),
            raw={},
        )
    store.close()


def test_burnin_identity_refuses_configuration_mutation(tmp_path):
    store = BybitStateStore(str(tmp_path / "venue.db"))
    start = datetime.now(timezone.utc)
    planned = start + timedelta(hours=72)
    assert store.create_or_resume_run(
        run_id="run-1",
        git_sha="abc",
        strategy_version="2.0.0",
        config_hash="hash-a",
        universe=("BTC/USD", "ETH/USD"),
        start_time=start,
        planned_end_time=planned,
        environment="BYBIT_DEMO",
    ) is False
    assert store.create_or_resume_run(
        run_id="run-1",
        git_sha="abc",
        strategy_version="2.0.0",
        config_hash="hash-a",
        universe=("BTC/USD", "ETH/USD"),
        start_time=start,
        planned_end_time=planned,
        environment="BYBIT_DEMO",
    ) is True
    with pytest.raises(RuntimeError, match="changed"):
        store.create_or_resume_run(
            run_id="run-1",
            git_sha="abc",
            strategy_version="2.0.0",
            config_hash="hash-b",
            universe=("BTC/USD", "ETH/USD"),
            start_time=start,
            planned_end_time=planned,
            environment="BYBIT_DEMO",
        )
    store.close()
