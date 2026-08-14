"""Authenticated API for the standalone Trade mobile app.

Trading stays on the server. The phone can be suspended or closed without
stopping the Streamlit/engine process. This service exposes durable dashboard
state and registers Expo push tokens for server-side fill/safety notifications.
It intentionally does not add a second order-entry or live-unlock path.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import APP_NAME, DEFAULT_DB_PATH
from crypto_engine import (
    CRYPTO_ENGINE_STATUS_KEY,
    CRYPTO_KILL_SWITCH_KEY,
    CRYPTO_LAST_SIGNALS_KEY,
)
from mobile_push import (
    MobilePushNotifier,
    get_push_tokens,
    register_push_token,
    unregister_push_token,
)
from state_store import StateStore
from trade_engine import ENGINE_STATUS_KEY, KILL_SWITCH_KEY, LAST_SIGNAL_KEY


API_VERSION = "2"
MIN_TOKEN_LENGTH = 32
DEFAULT_MAX_EQUITY_POINTS = 500

_bearer = HTTPBearer(auto_error=False)


class PushRegistration(BaseModel):
    token: str
    platform: str = "ios"


def _json_value(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"unparsed": True}


def _serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _allowed_origins_from_env() -> list[str]:
    raw = os.environ.get("MOBILE_ALLOWED_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _configured_token(explicit: Optional[str]) -> str:
    token = explicit if explicit is not None else os.environ.get("MOBILE_API_TOKEN", "")
    token = token.strip()
    if token and len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"MOBILE_API_TOKEN must contain at least {MIN_TOKEN_LENGTH} characters."
        )
    return token


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_mobile_api(
    *,
    store: Optional[StateStore] = None,
    token: Optional[str] = None,
    dashboard_url: Optional[str] = None,
    enable_push_notifier: Optional[bool] = None,
) -> FastAPI:
    """Build the mobile API with injectable persistence/auth for tests.

    If ``store`` is omitted the service owns a StateStore instance and closes it
    on shutdown. If no API token is configured, protected endpoints fail closed.
    The optional push observer reads durable state only and never calls Alpaca.
    """

    configured_token = _configured_token(token)
    owns_store = store is None
    resolved_store = store or StateStore(
        os.environ.get("MOBILE_API_DB_PATH", DEFAULT_DB_PATH)
    )
    resolved_dashboard_url = (
        dashboard_url
        if dashboard_url is not None
        else os.environ.get("MOBILE_DASHBOARD_URL", "").strip()
    )
    push_enabled = (
        _env_bool("MOBILE_PUSH_ENABLED", True)
        if enable_push_notifier is None
        else enable_push_notifier
    )
    push_interval = float(os.environ.get("MOBILE_PUSH_POLL_SECONDS", "10"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        notifier: Optional[MobilePushNotifier] = None
        task: Optional[asyncio.Task] = None
        if push_enabled:
            notifier = MobilePushNotifier(resolved_store, push_interval)
            task = asyncio.create_task(notifier.run(), name="mobile-push-notifier")
        try:
            yield
        finally:
            if notifier is not None:
                notifier.stop()
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
            if owns_store:
                resolved_store.close()

    app = FastAPI(
        title=f"{APP_NAME} Mobile API",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.store = resolved_store
    app.state.mobile_api_token = configured_token
    app.state.dashboard_url = resolved_dashboard_url
    app.state.push_enabled = push_enabled

    origins = _allowed_origins_from_env()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    async def require_token(
        credentials: Annotated[
            Optional[HTTPAuthorizationCredentials], Depends(_bearer)
        ],
    ) -> None:
        expected = app.state.mobile_api_token
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mobile API authentication is not configured.",
            )
        presented = (
            credentials.credentials
            if credentials is not None
            and credentials.scheme.lower() == "bearer"
            else ""
        )
        if not presented or not secrets.compare_digest(presented, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid mobile API token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = Depends(require_token)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "trade-mobile-api",
            "version": API_VERSION,
            "server_background_runtime": True,
        }

    @app.get("/v1/overview", dependencies=[auth])
    async def overview() -> dict[str, Any]:
        state = app.state.store
        latest_session = state.get_latest_session()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stock": {
                "status": _json_value(state.get_kv(ENGINE_STATUS_KEY)),
                "last_signal": _json_value(state.get_kv(LAST_SIGNAL_KEY)),
                "kill_switch": _json_value(state.get_kv(KILL_SWITCH_KEY)),
            },
            "crypto": {
                "status": _json_value(state.get_kv(CRYPTO_ENGINE_STATUS_KEY)),
                "last_signals": _json_value(state.get_kv(CRYPTO_LAST_SIGNALS_KEY)),
                "kill_switch": _json_value(state.get_kv(CRYPTO_KILL_SWITCH_KEY)),
            },
            "latest_session": _serializable(latest_session),
            "dashboard_url": app.state.dashboard_url or None,
            "capabilities": {
                "monitoring": True,
                "recent_activity": True,
                "equity_history": True,
                "push_notifications": app.state.push_enabled,
                "server_background_runtime": True,
                "trading_commands": False,
            },
        }

    @app.get("/v1/trades", dependencies=[auth])
    async def trades(
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> dict[str, Any]:
        rows = app.state.store.get_trades(limit=limit)
        return {
            "items": _serializable(rows),
            "count": len(rows),
            "limit": limit,
        }

    @app.get("/v1/logs", dependencies=[auth])
    async def logs(
        limit: Annotated[int, Query(ge=1, le=100)] = 40,
    ) -> dict[str, Any]:
        rows = app.state.store.get_logs(limit=limit)
        return {
            "items": _serializable(rows),
            "count": len(rows),
            "limit": limit,
        }

    @app.get("/v1/equity", dependencies=[auth])
    async def equity(
        limit: Annotated[int, Query(ge=2, le=DEFAULT_MAX_EQUITY_POINTS)] = 120,
    ) -> dict[str, Any]:
        latest = app.state.store.get_latest_session()
        if latest is None:
            return {"session_id": None, "items": [], "count": 0}
        points = app.state.store.get_equity_curve(latest["session_id"])
        selected = points[-limit:]
        return {
            "session_id": latest["session_id"],
            "items": _serializable(selected),
            "count": len(selected),
        }

    @app.post("/v1/devices/push", dependencies=[auth])
    async def add_push_device(payload: PushRegistration) -> dict[str, Any]:
        if payload.platform.lower() != "ios":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="This release registers iOS push devices only.",
            )
        try:
            tokens = register_push_token(app.state.store, payload.token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return {"registered": True, "device_count": len(tokens)}

    @app.delete("/v1/devices/push", dependencies=[auth])
    async def remove_push_device(
        token_value: Annotated[str, Body(embed=True, alias="token")],
    ) -> dict[str, Any]:
        tokens = unregister_push_token(app.state.store, token_value)
        return {"registered": False, "device_count": len(tokens)}

    @app.get("/v1/devices/push", dependencies=[auth])
    async def push_device_status() -> dict[str, Any]:
        return {
            "enabled": app.state.push_enabled,
            "device_count": len(get_push_tokens(app.state.store)),
        }

    return app


app = create_mobile_api()
