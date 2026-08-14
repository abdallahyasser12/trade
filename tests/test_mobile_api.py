from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from crypto_engine import (
    CRYPTO_ENGINE_STATUS_KEY,
    CRYPTO_KILL_SWITCH_KEY,
    CRYPTO_LAST_SIGNALS_KEY,
)
from mobile_api import create_mobile_api
from mobile_push import get_push_tokens
from state_store import StateStore
from trade_engine import ENGINE_STATUS_KEY, KILL_SWITCH_KEY, LAST_SIGNAL_KEY


TOKEN = "mobile-test-token-" + ("x" * 32)


@pytest.fixture
def store():
    value = StateStore(":memory:")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def client(store):
    application = create_mobile_api(
        store=store,
        token=TOKEN,
        dashboard_url="https://dashboard.example.test",
        enable_push_notifier=False,
    )
    with TestClient(application) as test_client:
        yield test_client


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_is_public_and_contains_no_trading_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "trade-mobile-api",
        "version": "2",
        "server_background_runtime": True,
    }


def test_protected_routes_fail_closed_without_token(client):
    response = client.get("/v1/overview")
    assert response.status_code == 401

    response = client.get(
        "/v1/overview",
        headers={"Authorization": "Bearer wrong-" + ("x" * 40)},
    )
    assert response.status_code == 401


def test_api_refuses_short_configured_token(store):
    with pytest.raises(ValueError, match="at least 32"):
        create_mobile_api(store=store, token="too-short")


def test_overview_uses_persisted_engine_state_only(client, store):
    store.set_kv(
        ENGINE_STATUS_KEY,
        json.dumps(
            {
                "state": "RUNNING",
                "session_id": "stock-session",
                "symbol": "SPY",
                "timeframe": "1Hour",
            }
        ),
    )
    store.set_kv(
        LAST_SIGNAL_KEY,
        json.dumps({"action": "HOLD", "reason": "no_cross"}),
    )
    store.set_kv(
        CRYPTO_ENGINE_STATUS_KEY,
        json.dumps(
            {
                "state": "RUNNING",
                "session_id": "crypto-session",
                "account_mode": "paper",
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframe": "1Hour",
            }
        ),
    )
    store.set_kv(
        CRYPTO_LAST_SIGNALS_KEY,
        json.dumps(
            {
                "BTC/USD": {"action": "BUY", "reason": "breakout"},
                "ETH/USD": {"action": "HOLD", "reason": "no_entry"},
            }
        ),
    )
    store.set_kv(KILL_SWITCH_KEY, json.dumps({"reason": "TEST_STOCK_LOCK"}))
    store.set_kv(
        CRYPTO_KILL_SWITCH_KEY,
        json.dumps({"armed": True, "reason": "TEST_CRYPTO_LOCK"}),
    )

    response = client.get("/v1/overview", headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["stock"]["status"]["state"] == "RUNNING"
    assert body["stock"]["last_signal"]["reason"] == "no_cross"
    assert body["crypto"]["status"]["account_mode"] == "paper"
    assert body["crypto"]["last_signals"]["BTC/USD"]["action"] == "BUY"
    assert body["stock"]["kill_switch"]["reason"] == "TEST_STOCK_LOCK"
    assert body["crypto"]["kill_switch"]["armed"] is True
    assert body["dashboard_url"] == "https://dashboard.example.test"
    assert body["capabilities"]["server_background_runtime"] is True
    assert body["capabilities"]["trading_commands"] is False


def test_trades_and_logs_are_bounded(client, store):
    for index in range(4):
        store.record_trade(
            session_id="session",
            symbol="SPY",
            action="BUY",
            qty=1,
            fill_price=100 + index,
            realized_pnl=None,
            reason=f"test-{index}",
            signal_snapshot=None,
            event_key=f"mobile-test-{index}",
        )
        store.log("INFO", f"log-{index}", session_id="session")

    trades = client.get("/v1/trades?limit=2", headers=_auth())
    logs = client.get("/v1/logs?limit=3", headers=_auth())
    assert trades.status_code == 200
    assert logs.status_code == 200
    assert len(trades.json()["items"]) == 2
    assert len(logs.json()["items"]) == 3

    invalid = client.get("/v1/logs?limit=101", headers=_auth())
    assert invalid.status_code == 422


def test_equity_returns_latest_session_window(client, store):
    from datetime import datetime, timedelta, timezone

    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.create_session(
        "session",
        "SPY",
        "1Hour",
        started,
        None,
        100_000,
        "{}",
    )
    for index in range(5):
        store.record_equity(
            "session",
            started + timedelta(hours=index),
            100_000 + index * 100,
            50_000,
        )

    response = client.get("/v1/equity?limit=3", headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session"
    assert body["count"] == 3
    assert body["items"][0]["equity"] == 100_200
    assert body["items"][-1]["equity"] == 100_400


def test_push_registration_is_authenticated_bounded_and_revocable(client, store):
    push_token = "ExpoPushToken[" + ("a" * 40) + "]"

    denied = client.post(
        "/v1/devices/push",
        json={"token": push_token, "platform": "ios"},
    )
    assert denied.status_code == 401

    response = client.post(
        "/v1/devices/push",
        headers=_auth(),
        json={"token": push_token, "platform": "ios"},
    )
    assert response.status_code == 200
    assert response.json()["registered"] is True
    assert get_push_tokens(store) == [push_token]

    response = client.get("/v1/devices/push", headers=_auth())
    assert response.status_code == 200
    assert response.json()["device_count"] == 1

    response = client.request(
        "DELETE",
        "/v1/devices/push",
        headers={**_auth(), "Content-Type": "application/json"},
        json={"token": push_token},
    )
    assert response.status_code == 200
    assert response.json()["registered"] is False
    assert get_push_tokens(store) == []


def test_push_registration_rejects_non_ios_and_malformed_tokens(client):
    bad_platform = client.post(
        "/v1/devices/push",
        headers=_auth(),
        json={"token": "ExpoPushToken[" + ("a" * 40) + "]", "platform": "android"},
    )
    assert bad_platform.status_code == 422

    bad_token = client.post(
        "/v1/devices/push",
        headers=_auth(),
        json={"token": "not-a-push-token", "platform": "ios"},
    )
    assert bad_token.status_code == 422
