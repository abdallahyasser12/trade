from __future__ import annotations

import json
import sqlite3

from crypto_engine import CRYPTO_ENGINE_STATUS_KEY, CRYPTO_KILL_SWITCH_KEY
from mobile_push import (
    MAX_PUSH_TOKENS,
    MobilePushNotifier,
    events_between,
    get_push_tokens,
    operator_notifications_enabled,
    register_push_token,
    snapshot,
    unregister_push_token,
    valid_expo_push_token,
)
from state_store import StateStore
from trade_engine import ENGINE_STATUS_KEY, KILL_SWITCH_KEY


def _token(index: int) -> str:
    return f"ExpoPushToken[{index:02d}-" + ("x" * 32) + "]"


def test_push_token_validation_and_bounded_registration():
    store = StateStore(":memory:")
    try:
        assert valid_expo_push_token(_token(1))
        assert not valid_expo_push_token("plain-token")
        for index in range(MAX_PUSH_TOKENS + 3):
            register_push_token(store, _token(index))
        tokens = get_push_tokens(store)
        assert len(tokens) == MAX_PUSH_TOKENS
        assert tokens[-1] == _token(MAX_PUSH_TOKENS + 2)
        unregister_push_token(store, tokens[0])
        assert len(get_push_tokens(store)) == MAX_PUSH_TOKENS - 1
    finally:
        store.close()


def test_snapshot_reads_durable_engine_and_trade_state():
    store = StateStore(":memory:")
    try:
        store.set_kv(ENGINE_STATUS_KEY, json.dumps({"state": "RUNNING"}))
        store.set_kv(CRYPTO_ENGINE_STATUS_KEY, json.dumps({"state": "STOPPED"}))
        store.set_kv(KILL_SWITCH_KEY, json.dumps({"armed": False}))
        store.set_kv(CRYPTO_KILL_SWITCH_KEY, json.dumps({"armed": True}))
        store.record_trade(
            session_id="s",
            symbol="BTC/USD",
            action="BUY",
            qty=0.1,
            fill_price=100.0,
            realized_pnl=None,
            reason="test",
            signal_snapshot=None,
            event_key="mobile-push-fill",
        )
        value = snapshot(store)
        assert value["stock_status"]["state"] == "RUNNING"
        assert value["crypto_kill"]["armed"] is True
        assert value["latest_trade"]["symbol"] == "BTC/USD"
    finally:
        store.close()


def test_empty_ledger_baseline_notifies_first_real_fill():
    store = StateStore(":memory:")
    try:
        before = snapshot(store)
        assert before["latest_trade"]["id"] == 0
        store.record_trade(
            session_id="s",
            symbol="BTC/USD",
            action="BUY",
            qty=0.01,
            fill_price=100.0,
            realized_pnl=None,
            reason="first-fill",
            signal_snapshot=None,
            event_key="first-real-fill",
        )
        after = snapshot(store)
        fill_events = [
            event for event in events_between(before, after)
            if event.data["kind"] == "trade_fill"
        ]
        assert len(fill_events) == 1
        assert fill_events[0].data["trade_id"] == after["latest_trade"]["id"]
        assert fill_events[0].data["symbol"] == "BTC/USD"
    finally:
        store.close()


def test_restart_baseline_does_not_replay_existing_fill():
    store = StateStore(":memory:")
    try:
        store.record_trade(
            session_id="s",
            symbol="ETH/USD",
            action="BUY",
            qty=0.1,
            fill_price=200.0,
            realized_pnl=None,
            reason="historical",
            signal_snapshot=None,
            event_key="historical-fill",
        )
        baseline = snapshot(store)
        immediately_after_restart = snapshot(store)
        assert all(
            event.data["kind"] != "trade_fill"
            for event in events_between(baseline, immediately_after_restart)
        )
    finally:
        store.close()


def test_events_cover_engine_transition_safety_lock_and_new_fill():
    previous = {
        "stock_status": {"state": "STOPPED", "account_mode": "paper"},
        "crypto_status": {"state": "RUNNING", "account_mode": "paper"},
        "stock_kill": {"armed": False},
        "crypto_kill": {"armed": False},
        "latest_trade": {"id": 10},
    }
    current = {
        "stock_status": {"state": "RUNNING", "account_mode": "paper"},
        "crypto_status": {
            "state": "HALTED",
            "account_mode": "paper",
            "halt_reason": "loss limit",
        },
        "stock_kill": {"armed": True, "reason": "emergency"},
        "crypto_kill": {"armed": False},
        "latest_trade": {
            "id": 11,
            "action": "SELL",
            "symbol": "BTC/USD",
            "qty": 0.1,
            "fill_price": 101.25,
        },
    }
    events = events_between(previous, current)
    kinds = [event.data["kind"] for event in events]
    assert kinds.count("engine_state") == 2
    assert "safety_lock" in kinds
    assert "trade_fill" in kinds
    assert any("loss limit" in event.body for event in events)


def test_manual_none_baseline_remains_non_replaying():
    previous = {
        "stock_status": {},
        "crypto_status": {},
        "stock_kill": {},
        "crypto_kill": {},
        "latest_trade": None,
    }
    current = {
        **previous,
        "latest_trade": {"id": 1, "action": "BUY", "symbol": "SPY"},
    }
    assert all(event.data["kind"] != "trade_fill" for event in events_between(previous, current))


def test_notifier_can_observe_bybit_store_while_tokens_stay_in_mobile_store(tmp_path):
    source = StateStore(":memory:")
    tokens = StateStore(":memory:")
    try:
        notifier = MobilePushNotifier(source, 10, token_store=tokens)
        assert notifier.store is source
        assert notifier.token_store is tokens
        register_push_token(tokens, _token(1))
        assert get_push_tokens(source) == []
        assert get_push_tokens(tokens) == [_token(1)]
    finally:
        source.close()
        tokens.close()


def test_dashboard_notification_preference_controls_delivery_switch(tmp_path):
    db = tmp_path / "operator.db"
    connection = sqlite3.connect(db)
    connection.execute(
        """CREATE TABLE operator_preferences(
        id INTEGER PRIMARY KEY, notifications_enabled INTEGER NOT NULL, updated_at TEXT NOT NULL
        )"""
    )
    connection.execute("INSERT INTO operator_preferences VALUES(1,0,'now')")
    connection.commit()
    connection.close()
    assert operator_notifications_enabled(str(db)) is False

    connection = sqlite3.connect(db)
    connection.execute("UPDATE operator_preferences SET notifications_enabled=1 WHERE id=1")
    connection.commit()
    connection.close()
    assert operator_notifications_enabled(str(db)) is True
