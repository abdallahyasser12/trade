"""Expo push registration and server-side trading-state notifications.

The iOS app is not the trading runtime. This module lets the always-on backend
notify a closed/suspended phone when the server-side engine changes state or a
new fill is persisted. Push failures never affect trading or reconciliation.

The notifier may observe a different StateStore from the one that owns Expo tokens.
That lets the production mobile API keep its existing token/auth database while the
notification observer watches the dedicated Bybit Demo engine database. A separate
operator-preference SQLite file can disable delivery without restarting services.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from crypto_engine import CRYPTO_ENGINE_STATUS_KEY, CRYPTO_KILL_SWITCH_KEY
from state_store import StateStore
from trade_engine import ENGINE_STATUS_KEY, KILL_SWITCH_KEY

PUSH_TOKENS_KEY = "mobile:expo_push_tokens:v1"
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
MAX_PUSH_TOKENS = 8


def _json(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def valid_expo_push_token(token: str) -> bool:
    token = token.strip()
    return (
        20 <= len(token) <= 256
        and (token.startswith("ExpoPushToken[") or token.startswith("ExponentPushToken["))
        and token.endswith("]")
    )


def get_push_tokens(store: StateStore) -> list[str]:
    raw = store.get_kv(PUSH_TOKENS_KEY)
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if valid_expo_push_token(str(value))][:MAX_PUSH_TOKENS]


def register_push_token(store: StateStore, token: str) -> list[str]:
    token = token.strip()
    if not valid_expo_push_token(token):
        raise ValueError("Invalid Expo push token.")
    tokens = get_push_tokens(store)
    if token not in tokens:
        tokens.append(token)
    tokens = tokens[-MAX_PUSH_TOKENS:]
    store.set_kv(PUSH_TOKENS_KEY, json.dumps(tokens))
    return tokens


def unregister_push_token(store: StateStore, token: str) -> list[str]:
    token = token.strip()
    tokens = [value for value in get_push_tokens(store) if value != token]
    store.set_kv(PUSH_TOKENS_KEY, json.dumps(tokens))
    return tokens


@dataclass(frozen=True)
class PushEvent:
    title: str
    body: str
    data: dict[str, Any]


def _state_name(status: dict[str, Any]) -> str:
    return str(status.get("state") or "STOPPED").upper()


def _is_armed(value: dict[str, Any]) -> bool:
    return bool(value.get("armed") or value.get("reason"))


def snapshot(store: StateStore) -> dict[str, Any]:
    trades = store.get_trades(limit=1)
    # A zero-id sentinel has two important properties:
    # 1) if the notifier starts with no historical trades, the first real fill
    #    (SQLite id >= 1) is observable as a transition and produces a notification;
    # 2) if the notifier restarts after historical fills exist, the baseline is the
    #    latest real id, so that old fill is not replayed as new.
    latest_trade = trades[0] if trades else {"id": 0}
    return {
        "stock_status": _json(store.get_kv(ENGINE_STATUS_KEY)),
        "crypto_status": _json(store.get_kv(CRYPTO_ENGINE_STATUS_KEY)),
        "stock_kill": _json(store.get_kv(KILL_SWITCH_KEY)),
        "crypto_kill": _json(store.get_kv(CRYPTO_KILL_SWITCH_KEY)),
        "latest_trade": latest_trade,
    }


def events_between(previous: dict[str, Any], current: dict[str, Any]) -> list[PushEvent]:
    events: list[PushEvent] = []
    for key, label in (("stock_status", "Stocks"), ("crypto_status", "Crypto")):
        before = previous.get(key) or {}
        after = current.get(key) or {}
        before_state = _state_name(before)
        after_state = _state_name(after)
        if before_state != after_state:
            mode = str(after.get("account_mode") or "paper").upper()
            if after_state == "RUNNING":
                title = f"{label} engine running"
                body = f"{mode} engine is active on the backend."
            elif after_state in {"HALTED", "ERROR"}:
                title = f"{label} engine {after_state.lower()}"
                body = str(after.get("halt_reason") or after.get("error") or "Review the app.")
            else:
                title = f"{label} engine {after_state.lower()}"
                body = "Server-side engine state changed."
            events.append(
                PushEvent(
                    title=title,
                    body=body,
                    data={"kind": "engine_state", "engine": label.lower(), "state": after_state},
                )
            )

    for key, label in (("stock_kill", "Stocks"), ("crypto_kill", "Crypto")):
        before = previous.get(key) or {}
        after = current.get(key) or {}
        if not _is_armed(before) and _is_armed(after):
            events.append(
                PushEvent(
                    title=f"{label} safety lock armed",
                    body=str(after.get("message") or after.get("reason") or "Open Trade to review."),
                    data={"kind": "safety_lock", "engine": label.lower()},
                )
            )

    old_trade = previous.get("latest_trade") or {}
    new_trade = current.get("latest_trade") or {}
    old_id = old_trade.get("id")
    new_id = new_trade.get("id")
    if new_id is not None and old_id is not None and new_id != old_id:
        action = str(new_trade.get("action") or "FILL")
        symbol = str(new_trade.get("symbol") or "")
        qty = new_trade.get("qty")
        price = new_trade.get("fill_price")
        suffix = ""
        if price is not None:
            try:
                suffix = f" @ ${float(price):,.2f}"
            except (TypeError, ValueError):
                suffix = ""
        events.append(
            PushEvent(
                title=f"{action} {symbol}".strip(),
                body=f"Filled {qty} {symbol}{suffix}".strip(),
                data={"kind": "trade_fill", "trade_id": new_id, "symbol": symbol},
            )
        )
    return events


async def send_push_event(store: StateStore, event: PushEvent) -> None:
    tokens = get_push_tokens(store)
    if not tokens:
        return
    payload = [
        {"to": token, "title": event.title, "body": event.body, "sound": "default", "data": event.data}
        for token in tokens
    ]
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    access_token = os.environ.get("EXPO_ACCESS_TOKEN", "").strip()
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(EXPO_PUSH_URL, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except Exception:
        return
    tickets = body.get("data") if isinstance(body, dict) else None
    if not isinstance(tickets, list):
        return
    invalid: set[str] = set()
    for token, ticket in zip(tokens, tickets):
        if not isinstance(ticket, dict):
            continue
        details = ticket.get("details")
        if isinstance(details, dict) and details.get("error") == "DeviceNotRegistered":
            invalid.add(token)
    if invalid:
        store.set_kv(PUSH_TOKENS_KEY, json.dumps([token for token in tokens if token not in invalid]))


def operator_notifications_enabled(path: Optional[str]) -> bool:
    """Read the dashboard preference fail-open for notifications, never for trading."""
    if not path or not os.path.exists(path):
        return True
    connection = sqlite3.connect(path, timeout=1.0)
    try:
        row = connection.execute(
            "SELECT notifications_enabled FROM operator_preferences WHERE id=1"
        ).fetchone()
        return bool(row[0]) if row else True
    except sqlite3.Error:
        return True
    finally:
        connection.close()


class MobilePushNotifier:
    """Low-frequency observer of durable engine state.

    ``store`` is the state being observed. ``token_store`` owns Expo device tokens and
    defaults to the same store for backward compatibility. A preference database may
    disable delivery. None of these paths can mutate trading/protection state.
    """

    def __init__(
        self,
        store: StateStore,
        interval_seconds: float = 10.0,
        *,
        token_store: Optional[StateStore] = None,
        preference_db_path: Optional[str] = None,
    ):
        self.store = store
        self.token_store = token_store or store
        self.preference_db_path = preference_db_path
        self.interval_seconds = max(5.0, float(interval_seconds))
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        previous = snapshot(self.store)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                break
            except asyncio.TimeoutError:
                pass
            current = snapshot(self.store)
            events = events_between(previous, current)
            if operator_notifications_enabled(self.preference_db_path):
                for event in events:
                    await send_push_event(self.token_store, event)
            previous = current
