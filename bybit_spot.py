"""Production-oriented Bybit V5 Spot broker and instrument adapter.

Safety is fail-closed and structural:
* only ``category=spot`` is accepted by the economic request boundary;
* every Spot create request explicitly carries ``isLeverage=0``;
* no margin-toggle, leverage, derivative, borrow, short, or position endpoint exists here;
* Demo and Live credentials are separate and Live construction requires two gates;
* REST order acknowledgement is not a fill; executions are reconciled by orderLinkId;
* wallet holdings are never adopted: ``list_positions`` exposes only execution-ledger
  inventory that this bot can prove it owns.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Callable, Optional

from broker import BrokerError
from bybit_state import BybitStateStore, OwnedInventory
from config import AccountMode
from crypto_venue import (
    CanonicalInstrument,
    CryptoExecutionEnvironment,
    DEFAULT_BYBIT_SPOT_UNIVERSE,
    InstrumentMetadata,
    bybit_instrument,
)
from models import AccountInfo, AssetInfo, BrokerPosition, OrderRecord

logger = logging.getLogger(__name__)

BYBIT_DEMO_REST_HOST = "https://api-demo.bybit.com"
BYBIT_LIVE_REST_HOST = "https://api.bybit.com"
BYBIT_PUBLIC_SPOT_WS_HOST = "wss://stream.bybit.com/v5/public/spot"
BYBIT_DEMO_PRIVATE_WS_HOST = "wss://stream-demo.bybit.com/v5/private"
BYBIT_LIVE_PRIVATE_WS_HOST = "wss://stream.bybit.com/v5/private"
BYBIT_ORDER_LINK_ID_MAX = 36
BYBIT_ORDER_LINK_PREFIX = "tb-"
SPOT_CATEGORY = "spot"
NO_LEVERAGE = 0

_DERIVATIVE_ONLY_FIELDS = frozenset(
    {"positionIdx", "reduceOnly", "closeOnTrigger", "tpslMode", "orderIv", "mmp"}
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_ -]?key|api[_ -]?secret|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)x-bapi-(api-key|sign)\s*[:=]\s*[^\s,;]+"),
)


def _safe_text(exc: Exception, limit: int = 320) -> str:
    text = str(exc)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1=<redacted>", text)
    return text[:limit] + ("…" if len(text) > limit else "")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_from_ms(value: Any) -> Optional[datetime]:
    try:
        ms = int(float(value))
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def decimal_floor(value: float, step: float) -> float:
    if not math.isfinite(value) or value <= 0 or not math.isfinite(step) or step <= 0:
        raise BrokerError("Quantity/price and increment must be finite and positive")
    try:
        raw = Decimal(str(value))
        inc = Decimal(str(step))
        units = (raw / inc).to_integral_value(rounding=ROUND_FLOOR)
        normalized = units * inc
    except InvalidOperation as exc:
        raise BrokerError("Could not normalize value to Bybit increment") from exc
    if normalized <= 0:
        raise BrokerError("Increment normalization produced zero")
    return float(normalized)


def decimal_ceil(value: float, step: float) -> float:
    if not math.isfinite(value) or value <= 0 or not math.isfinite(step) or step <= 0:
        raise BrokerError("Quantity/price and increment must be finite and positive")
    try:
        raw = Decimal(str(value))
        inc = Decimal(str(step))
        units = (raw / inc).to_integral_value(rounding=ROUND_CEILING)
        normalized = units * inc
    except InvalidOperation as exc:
        raise BrokerError("Could not normalize value to Bybit increment") from exc
    return float(normalized)


def as_decimal_string(value: float) -> str:
    result = format(Decimal(str(value)).normalize(), "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def assert_spot_only_payload(payload: dict[str, Any], *, require_leverage: bool = True) -> None:
    """Raise before network I/O if an economic payload escapes the Spot boundary."""
    if payload.get("category") != SPOT_CATEGORY:
        raise BrokerError("Bybit economic requests are restricted to category=spot")
    if require_leverage and payload.get("isLeverage") != NO_LEVERAGE:
        raise BrokerError("Bybit Spot requests must explicitly set isLeverage=0")
    if payload.get("isLeverage") == 1:
        raise BrokerError("Margin borrowing is forbidden")
    forbidden = _DERIVATIVE_ONLY_FIELDS.intersection(payload)
    if forbidden:
        raise BrokerError(
            "Derivative-only fields are forbidden in Bybit Spot execution: "
            + ", ".join(sorted(forbidden))
        )
    if payload.get("side") not in (None, "Buy", "Sell"):
        raise BrokerError("Unsupported Bybit Spot side")


def _purpose_token(purpose: str) -> str:
    mapping = {
        "entry": "e",
        "exit": "x",
        "stop": "s",
        "guard": "g",
        "partial": "p",
        "target": "t",
        "replacement": "r",
        "emergency": "m",
    }
    return mapping.get(purpose.lower(), "u")


def infer_intent_purpose(internal_intent_id: str, fallback: str) -> str:
    parts = str(internal_intent_id).split("-")
    if len(parts) >= 3:
        candidate = parts[-2].lower()
        if candidate in {"entry", "exit", "stop", "guard", "partial", "target", "replacement", "emergency"}:
            return candidate
    return fallback


def make_bybit_order_link_id(
    internal_intent_id: str,
    purpose: str,
    *,
    attempt: int = 1,
) -> str:
    """Deterministic recognizable Bybit ID, always <= the current 36-char contract."""
    if not internal_intent_id:
        raise ValueError("internal_intent_id is required")
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    digest = hashlib.blake2s(
        f"{internal_intent_id}|{purpose}|{attempt}".encode("utf-8"), digest_size=10
    ).hexdigest()
    value = f"{BYBIT_ORDER_LINK_PREFIX}{_purpose_token(purpose)}-{digest}-a{attempt:x}"
    if len(value) > BYBIT_ORDER_LINK_ID_MAX:
        raise ValueError("Generated Bybit orderLinkId exceeds 36 characters")
    return value


def is_bot_order_link_id(value: Optional[str]) -> bool:
    return bool(value) and str(value).startswith(BYBIT_ORDER_LINK_PREFIX)


@dataclass
class _Budget:
    limit: int
    window_seconds: float


class RequestThrottle:
    """Conservative local budgets intentionally below provider edge limits."""
    def __init__(self) -> None:
        self._budgets = {
            "economic": _Budget(3, 1.0),
            "private": _Budget(5, 1.0),
            "market": _Budget(8, 1.0),
        }
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(self, group: str, *, max_wait_seconds: float = 2.0) -> None:
        budget = self._budgets[group]
        deadline = time.monotonic() + max_wait_seconds
        while True:
            now = time.monotonic()
            with self._lock:
                queue = self._events[group]
                while queue and now - queue[0] >= budget.window_seconds:
                    queue.popleft()
                if len(queue) < budget.limit:
                    queue.append(now)
                    return
                wait = max(0.01, budget.window_seconds - (now - queue[0]))
            if now + wait > deadline:
                raise BrokerError(f"Local Bybit {group} request budget exhausted; failing closed")
            time.sleep(min(wait, 0.25))


class BybitInstrumentProvider:
    def __init__(self, http_client, *, cache_ttl_seconds: float = 900.0) -> None:
        self.http = http_client
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self._cache: dict[str, tuple[float, InstrumentMetadata]] = {}
        self._lock = threading.RLock()

    def get_instrument_metadata(self, symbol: str, *, force_refresh: bool = False) -> InstrumentMetadata:
        instrument = bybit_instrument(symbol)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(instrument.strategy_symbol)
            if cached and not force_refresh and now - cached[0] <= self.cache_ttl_seconds:
                return cached[1]
        response = self.http.get_instruments_info(category=SPOT_CATEGORY, symbol=instrument.venue_symbol)
        _require_ok(response, "instrument metadata")
        items = (((response or {}).get("result") or {}).get("list") or [])
        if len(items) != 1:
            raise BrokerError(f"Bybit returned no unambiguous Spot metadata for {instrument.venue_symbol}")
        item = items[0]
        lot = item.get("lotSizeFilter") or {}
        price = item.get("priceFilter") or {}
        status = str(item.get("status") or "").lower()
        if status not in {"trading", "available"}:
            raise BrokerError(f"Bybit Spot instrument {instrument.venue_symbol} status is {status!r}")
        metadata = InstrumentMetadata(
            instrument=instrument,
            tick_size=_num(price.get("tickSize")),
            quantity_step=_num(lot.get("basePrecision") or lot.get("qtyStep")),
            min_order_qty=_num(lot.get("minOrderQty")),
            min_order_amount=_num(lot.get("minOrderAmt")),
            max_order_qty=_num(lot.get("maxOrderQty")),
            max_market_order_qty=_num(lot.get("maxMarketOrderQty") or lot.get("maxOrderQty")),
            fetched_at_epoch_ms=int((response or {}).get("time") or time.time() * 1000),
        )
        errors = metadata.validate()
        if errors:
            raise BrokerError(
                f"Untrusted Bybit metadata for {instrument.venue_symbol}: " + "; ".join(errors)
            )
        with self._lock:
            self._cache[instrument.strategy_symbol] = (now, metadata)
        return metadata


class BybitSpotBroker:
    """Bybit Unified V5 Spot-only broker compatible with the existing crypto engine."""

    account_mode = AccountMode.PAPER  # Bybit Demo maps to existing non-live lifecycle semantics.

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        environment: CryptoExecutionEnvironment = CryptoExecutionEnvironment.BYBIT_DEMO,
        state: Optional[BybitStateStore] = None,
        strategy_capital_budget_usdt: float = 1_000.0,
        permitted_symbols: tuple[str, ...] = DEFAULT_BYBIT_SPOT_UNIVERSE,
        max_entries_rolling_24h: int = 5,
        max_clock_skew_ms: float = 1_500.0,
        recv_window_ms: int = 5_000,
        allow_live: bool = False,
        http_client=None,
    ) -> None:
        if not api_key or not api_secret:
            raise BrokerError("Bybit API key and secret are required")
        self.environment = CryptoExecutionEnvironment(environment)
        if self.environment not in {
            CryptoExecutionEnvironment.BYBIT_DEMO,
            CryptoExecutionEnvironment.BYBIT_LIVE,
        }:
            raise BrokerError("BybitSpotBroker only accepts BYBIT_DEMO or BYBIT_LIVE")
        if self.environment is CryptoExecutionEnvironment.BYBIT_LIVE:
            if not allow_live or not _truthy(os.getenv("ENABLE_BYBIT_LIVE_TRADING")):
                raise BrokerError(
                    "Bybit LIVE is disabled. Both allow_live=True and "
                    "ENABLE_BYBIT_LIVE_TRADING=true are required."
                )
            self.account_mode = AccountMode.LIVE
        if not math.isfinite(strategy_capital_budget_usdt) or strategy_capital_budget_usdt <= 0:
            raise BrokerError("strategy_capital_budget_usdt must be positive")
        if not 1 <= int(max_entries_rolling_24h) <= 50:
            raise BrokerError("max_entries_rolling_24h must be in [1, 50]")
        if not math.isfinite(max_clock_skew_ms) or max_clock_skew_ms <= 0:
            raise BrokerError("max_clock_skew_ms must be positive")
        if recv_window_ms < 1_000 or recv_window_ms > 10_000:
            raise BrokerError("recv_window_ms must be within [1000, 10000]")
        self._api_key = api_key
        self._api_secret = api_secret
        self.state = state or BybitStateStore()
        self._owns_state = state is None
        self.strategy_capital_budget_usdt = float(strategy_capital_budget_usdt)
        self.max_entries_rolling_24h = int(max_entries_rolling_24h)
        self.max_clock_skew_ms = float(max_clock_skew_ms)
        self.recv_window_ms = int(recv_window_ms)
        self.permitted_symbols = tuple(bybit_instrument(s).strategy_symbol for s in permitted_symbols)
        if len(set(self.permitted_symbols)) != len(self.permitted_symbols):
            raise BrokerError("Permitted Bybit universe contains duplicates")
        self._throttle = RequestThrottle()
        self._rest_failures = 0
        self._rate_limit_events = 0
        self._clock_skew_ms: Optional[float] = None
        self._circuit_failures = 0
        self._circuit_open_until = 0.0
        if http_client is None:
            from pybit.unified_trading import HTTP
            http_client = HTTP(
                testnet=False,
                demo=self.environment is CryptoExecutionEnvironment.BYBIT_DEMO,
                api_key=api_key,
                api_secret=api_secret,
                recv_window=self.recv_window_ms,
            )
        self.http = http_client
        self.instrument_provider = BybitInstrumentProvider(self.http)
        self._verify_sdk_endpoint_if_exposed()

    @property
    def expected_rest_host(self) -> str:
        return (
            BYBIT_DEMO_REST_HOST
            if self.environment is CryptoExecutionEnvironment.BYBIT_DEMO
            else BYBIT_LIVE_REST_HOST
        )

    def close(self) -> None:
        if self._owns_state:
            self.state.close()

    def _verify_sdk_endpoint_if_exposed(self) -> None:
        raw = None
        for name in ("endpoint", "_endpoint", "base_url", "_base_url"):
            candidate = getattr(self.http, name, None)
            if candidate:
                raw = str(getattr(candidate, "value", candidate)).rstrip("/")
                break
        if raw and raw.startswith("http") and raw != self.expected_rest_host:
            raise BrokerError(
                f"Bybit client endpoint {raw!r} does not match required {self.expected_rest_host!r}"
            )

    def _request(
        self,
        group: str,
        operation: str,
        fn: Callable[[], Any],
        *,
        economic: bool = False,
    ) -> Any:
        if time.monotonic() < self._circuit_open_until:
            raise BrokerError("Bybit provider circuit breaker is open")
        attempts = 1 if economic else 3
        last: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            self._throttle.acquire(group)
            try:
                result = fn()
                _require_ok(result, operation)
                self._circuit_failures = 0
                return result
            except BrokerError as exc:
                last = exc
                self._note_failure(exc)
            except Exception as exc:  # SDK/network boundary
                last = BrokerError(_safe_text(exc))
                self._note_failure(exc)
            if economic or attempt >= attempts:
                break
            time.sleep(0.20 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.12))
        raise BrokerError(f"Bybit {operation} failed: {_safe_text(last or Exception('unknown'))}") from last

    def _note_failure(self, exc: Exception) -> None:
        self._rest_failures += 1
        text = str(exc).lower()
        if "429" in text or "10006" in text or "rate limit" in text:
            self._rate_limit_events += 1
        self._circuit_failures += 1
        if self._circuit_failures >= 5:
            self._circuit_open_until = time.monotonic() + 30.0

    # --------------------------- environment/account --------------------
    def measure_clock_skew_ms(self) -> float:
        before = time.time() * 1000.0
        response = self._request("market", "server time", lambda: self.http.get_server_time())
        after = time.time() * 1000.0
        result = (response or {}).get("result") or {}
        server_ms = _num((response or {}).get("time"), 0.0)
        if server_ms <= 0 and result.get("timeSecond") is not None:
            server_ms = _num(result.get("timeSecond")) * 1000.0
        if server_ms <= 0 and result.get("timeNano") is not None:
            server_ms = _num(result.get("timeNano")) / 1_000_000.0
        if server_ms <= 0:
            raise BrokerError("Bybit server-time response is missing a usable timestamp")
        midpoint = (before + after) / 2.0
        self._clock_skew_ms = server_ms - midpoint
        return self._clock_skew_ms

    def _assert_clock_safe(self) -> None:
        skew = self.measure_clock_skew_ms()
        if abs(skew) > self.max_clock_skew_ms:
            raise BrokerError(
                f"Clock skew {skew:.0f}ms exceeds safe {self.max_clock_skew_ms:.0f}ms bound"
            )

    def verify_environment(self) -> dict[str, Any]:
        self._verify_sdk_endpoint_if_exposed()
        skew = self.measure_clock_skew_ms()
        if abs(skew) > self.max_clock_skew_ms:
            raise BrokerError("Local clock is too far from Bybit server time")
        # Private wallet query proves the API key belongs to the selected environment.
        self._wallet_snapshot()
        return {
            "provider": "bybit",
            "environment": self.environment.value,
            "rest_host": self.expected_rest_host,
            "clock_skew_ms": skew,
            "spot_only": True,
            "isLeverage": 0,
        }

    def verify_account(self) -> AccountInfo:
        self.verify_environment()
        self.verify_crypto_assets(self.permitted_symbols)
        account = self.get_account()
        if not account.can_trade:
            raise BrokerError("Bybit strategy account snapshot is not tradable")
        return account

    def _wallet_snapshot(self) -> dict[str, dict[str, float]]:
        response = self._request(
            "private",
            "wallet balance",
            lambda: self.http.get_wallet_balance(accountType="UNIFIED"),
        )
        accounts = (((response or {}).get("result") or {}).get("list") or [])
        if not accounts:
            raise BrokerError("Bybit returned no Unified wallet account")
        coins = accounts[0].get("coin") or []
        result: dict[str, dict[str, float]] = {}
        for item in coins:
            coin = str(item.get("coin") or "").upper()
            if not coin:
                continue
            wallet = max(0.0, _num(item.get("walletBalance")) - _num(item.get("spotBorrow")))
            locked = max(0.0, _num(item.get("locked")))
            available = max(0.0, wallet - locked)
            explicit_available = _num(item.get("availableToWithdraw"), -1.0)
            if explicit_available >= 0:
                available = min(available, explicit_available)
            result[coin] = {"wallet": wallet, "locked": locked, "available": available}
        return result

    def _latest_price(self, symbol: str) -> float:
        instrument = bybit_instrument(symbol)
        response = self._request(
            "market",
            "spot ticker",
            lambda: self.http.get_tickers(category=SPOT_CATEGORY, symbol=instrument.venue_symbol),
        )
        items = (((response or {}).get("result") or {}).get("list") or [])
        if not items:
            raise BrokerError(f"No Bybit ticker for {instrument.venue_symbol}")
        price = _num(items[0].get("lastPrice"))
        if price <= 0:
            raise BrokerError(f"Invalid Bybit price for {instrument.venue_symbol}")
        return price

    def get_account(self) -> AccountInfo:
        self.sync_bot_executions()
        wallet = self._wallet_snapshot()
        usdt_available = (wallet.get("USDT") or {}).get("available", 0.0)
        inventories = [inv for inv in self.state.all_inventory() if inv.strategy_symbol in self.permitted_symbols]
        realized = sum(inv.realized_pnl_quote for inv in inventories)
        cost_basis = sum(inv.cost_basis_quote for inv in inventories)
        market_value = 0.0
        for inv in inventories:
            if inv.qty > 0:
                market_value += inv.qty * self._latest_price(inv.strategy_symbol)
        strategy_cash = max(0.0, self.strategy_capital_budget_usdt + realized - cost_basis)
        strategy_cash = min(strategy_cash, usdt_available)
        equity = strategy_cash + market_value
        # If no bot inventory exists, Demo may hold a huge virtual wallet. We still
        # expose exactly the configured strategy allocation, capped by available USDT.
        if not inventories or (cost_basis <= 0 and market_value <= 0):
            equity = min(self.strategy_capital_budget_usdt + realized, usdt_available)
            strategy_cash = equity
        return AccountInfo(
            equity=max(0.0, equity),
            cash=max(0.0, strategy_cash),
            buying_power=max(0.0, strategy_cash),
            status="ACTIVE",
            currency="USDT",
            pattern_day_trader=False,
            daytrade_count=0,
            trading_blocked=False,
        )

    # ----------------------------- instruments -------------------------
    def get_asset(self, symbol: str) -> AssetInfo:
        if bybit_instrument(symbol).strategy_symbol not in self.permitted_symbols:
            raise BrokerError(f"{symbol} is outside the permitted Bybit Spot universe")
        return self.instrument_provider.get_instrument_metadata(symbol).as_asset_info()

    def verify_crypto_assets(self, symbols: list[str] | tuple[str, ...]) -> dict[str, AssetInfo]:
        result: dict[str, AssetInfo] = {}
        for symbol in symbols:
            canonical = bybit_instrument(symbol).strategy_symbol
            if canonical not in self.permitted_symbols:
                raise BrokerError(f"{canonical} is outside the configured Bybit Spot universe")
            metadata = self.instrument_provider.get_instrument_metadata(canonical, force_refresh=True)
            result[canonical] = metadata.as_asset_info()
        return result

    # ----------------------------- ownership ---------------------------
    def sync_bot_executions(self, symbol: Optional[str] = None) -> int:
        symbols = [bybit_instrument(symbol).strategy_symbol] if symbol else list(self.permitted_symbols)
        added = 0
        for strategy_symbol in symbols:
            instrument = bybit_instrument(strategy_symbol)
            response = self._request(
                "private",
                "execution history",
                lambda inst=instrument: self.http.get_executions(
                    category=SPOT_CATEGORY, symbol=inst.venue_symbol, limit=100
                ),
            )
            items = (((response or {}).get("result") or {}).get("list") or [])
            for item in items:
                order_link = str(item.get("orderLinkId") or "")
                if not is_bot_order_link_id(order_link):
                    continue
                mapping = self.state.order_by_link(order_link)
                if mapping is None:
                    # A recognizable namespace without a persisted pre-submit intent is
                    # not silently adopted after a database loss/copy.
                    continue
                execution_id = str(item.get("execId") or "")
                if not execution_id:
                    continue
                executed_at = _utc_from_ms(item.get("execTime")) or datetime.now(timezone.utc)
                if self.state.record_execution(
                    execution_id=execution_id,
                    internal_intent_id=mapping["internal_intent_id"],
                    order_link_id=order_link,
                    order_id=str(item.get("orderId") or mapping.get("order_id") or ""),
                    strategy_symbol=mapping["strategy_symbol"],
                    venue_symbol=mapping["venue_symbol"],
                    base_asset=instrument.base_asset,
                    side=str(item.get("side") or "").lower(),
                    qty=_num(item.get("execQty")),
                    price=_num(item.get("execPrice")),
                    fee=_num(item.get("execFee")),
                    fee_asset=str(item.get("feeCurrency") or "") or None,
                    execution_at=executed_at,
                    raw=item,
                ):
                    added += 1
        return added

    def _assert_wallet_not_deficit(self, instrument: CanonicalInstrument, owned: OwnedInventory) -> dict[str, float]:
        wallet = self._wallet_snapshot().get(instrument.base_asset) or {
            "wallet": 0.0, "available": 0.0, "locked": 0.0
        }
        if owned.qty > wallet["wallet"] + 1e-10:
            raise BrokerError(
                f"Bybit wallet deficit for {instrument.strategy_symbol}: proven bot-owned "
                f"{owned.qty:.12g} exceeds wallet {wallet['wallet']:.12g}; reconciliation required"
            )
        return wallet

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        instrument = bybit_instrument(symbol)
        self.sync_bot_executions(instrument.strategy_symbol)
        owned = self.state.inventory(instrument.strategy_symbol)
        self._assert_wallet_not_deficit(instrument, owned)
        if owned.qty <= 1e-12:
            return None
        price = self._latest_price(instrument.strategy_symbol)
        avg = owned.average_entry_price
        value = owned.qty * price
        pnl = value - owned.cost_basis_quote
        return BrokerPosition(
            symbol=instrument.strategy_symbol,
            qty=owned.qty,
            avg_entry_price=avg,
            current_price=price,
            market_value=value,
            unrealized_pl=pnl,
            unrealized_plpc=(pnl / owned.cost_basis_quote if owned.cost_basis_quote > 0 else 0.0),
            asset_class="crypto",
        )

    def list_positions(self) -> list[BrokerPosition]:
        result = []
        for symbol in self.permitted_symbols:
            position = self.get_position(symbol)
            if position is not None:
                result.append(position)
        return result

    def bot_owned_quantity(self, symbol: str) -> float:
        self.sync_bot_executions(symbol)
        return self.state.inventory(bybit_instrument(symbol).strategy_symbol).qty

    # ------------------------------- orders ----------------------------
    def _normalize_order_item(self, item: dict[str, Any], mapping: dict[str, Any]) -> OrderRecord:
        raw_status = str(item.get("orderStatus") or item.get("status") or "")
        status_map = {
            "New": "new", "Created": "new", "Untriggered": "accepted",
            "Triggered": "accepted", "PartiallyFilled": "partially_filled",
            "Filled": "filled", "Cancelled": "canceled", "Canceled": "canceled",
            "PartiallyFilledCanceled": "canceled", "Rejected": "rejected", "Deactivated": "canceled",
        }
        status = status_map.get(raw_status, raw_status.lower() or "accepted")
        filled = _num(item.get("cumExecQty"))
        self.state.update_order(
            mapping["internal_intent_id"],
            order_id=str(item.get("orderId") or mapping.get("order_id") or ""),
            status=status,
            filled_qty=filled,
        )
        self.sync_bot_executions(mapping["strategy_symbol"])
        return OrderRecord(
            broker_order_id=str(item.get("orderId") or mapping.get("order_id") or ""),
            client_order_id=mapping["internal_intent_id"],
            symbol=mapping["strategy_symbol"],
            purpose=mapping["purpose"],
            side=str(item.get("side") or "").lower(),
            requested_qty=_num(item.get("qty"), float(mapping.get("requested_qty") or 0.0)),
            filled_qty=filled,
            status=status,
            fill_price=(_num(item.get("avgPrice")) or None),
            stop_price=(_num(item.get("triggerPrice")) or None),
            limit_price=(_num(item.get("price")) or None),
            submitted_at=_utc_from_ms(item.get("createdTime")),
            updated_at=_utc_from_ms(item.get("updatedTime")),
        )

    def _lookup_order_item(self, mapping: dict[str, Any]) -> Optional[dict[str, Any]]:
        instrument = bybit_instrument(mapping["strategy_symbol"])
        kwargs = {"category": SPOT_CATEGORY, "symbol": instrument.venue_symbol, "orderLinkId": mapping["order_link_id"]}
        response = self._request("private", "open-order lookup", lambda: self.http.get_open_orders(**kwargs))
        items = (((response or {}).get("result") or {}).get("list") or [])
        if items:
            return items[0]
        response = self._request("private", "order-history lookup", lambda: self.http.get_order_history(**kwargs))
        items = (((response or {}).get("result") or {}).get("list") or [])
        return items[0] if items else None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]:
        mapping = self.state.order_by_intent(client_order_id)
        if mapping is None:
            return None
        item = self._lookup_order_item(mapping)
        if item is None:
            return OrderRecord(
                broker_order_id=str(mapping.get("order_id") or ""),
                client_order_id=client_order_id,
                symbol=mapping["strategy_symbol"], purpose=mapping["purpose"], side="",
                requested_qty=float(mapping["requested_qty"]), filled_qty=float(mapping["filled_qty"]),
                status=str(mapping["status"]),
            )
        return self._normalize_order_item(item, mapping)

    def get_order(self, broker_order_id: str) -> OrderRecord:
        mapping = self.state.order_by_order_id(str(broker_order_id))
        if mapping is None:
            raise BrokerError("Refusing to query an order not present in the Bybit bot ownership map")
        record = self.get_order_by_client_id(mapping["internal_intent_id"])
        if record is None:
            raise BrokerError("Bot-owned Bybit order disappeared from venue history")
        return record

    def list_open_orders(self, symbol: Optional[str] = None) -> list[OrderRecord]:
        response = self._request(
            "private",
            "open orders",
            lambda: self.http.get_open_orders(
                category=SPOT_CATEGORY,
                **({"symbol": bybit_instrument(symbol).venue_symbol} if symbol else {}),
            ),
        )
        records: list[OrderRecord] = []
        for item in (((response or {}).get("result") or {}).get("list") or []):
            link = str(item.get("orderLinkId") or "")
            if not is_bot_order_link_id(link):
                continue
            mapping = self.state.order_by_link(link)
            if mapping is None:
                continue
            records.append(self._normalize_order_item(item, mapping))
        return records

    def cancel_order(self, broker_order_id: str) -> None:
        mapping = self.state.order_by_order_id(str(broker_order_id))
        if mapping is None:
            raise BrokerError("Refusing to cancel an order the bot cannot prove it owns")
        instrument = bybit_instrument(mapping["strategy_symbol"])
        payload = {
            "category": SPOT_CATEGORY,
            "symbol": instrument.venue_symbol,
            "orderId": str(broker_order_id),
        }
        assert_spot_only_payload(payload, require_leverage=False)
        self._assert_clock_safe()
        self._request(
            "economic", "cancel order",
            lambda: self.http.cancel_order(**payload), economic=True,
        )

    def _prepare_intent(
        self,
        internal_intent_id: str,
        symbol: str,
        purpose: str,
        qty: float,
        *,
        attempt: int = 1,
    ) -> tuple[CanonicalInstrument, str]:
        instrument = bybit_instrument(symbol)
        if instrument.strategy_symbol not in self.permitted_symbols:
            raise BrokerError(f"{symbol} is outside the permitted Bybit Spot universe")
        link = make_bybit_order_link_id(internal_intent_id, purpose, attempt=attempt)
        self.state.record_order_intent(
            internal_intent_id=internal_intent_id,
            order_link_id=link,
            strategy_symbol=instrument.strategy_symbol,
            venue_symbol=instrument.venue_symbol,
            purpose=purpose,
            attempt=attempt,
            requested_qty=qty,
        )
        return instrument, link

    def _submit_once_reconcile_timeout(
        self,
        *,
        internal_intent_id: str,
        payload: dict[str, Any],
        operation: str,
    ) -> OrderRecord:
        assert_spot_only_payload(payload)
        self._assert_clock_safe()
        try:
            response = self._request(
                "economic", operation, lambda: self.http.place_order(**payload), economic=True
            )
            result = (response or {}).get("result") or {}
            order_id = str(result.get("orderId") or "")
            self.state.update_order(internal_intent_id, order_id=order_id, status="accepted")
        except BrokerError:
            # Never resend creation blindly. Query the deterministic orderLinkId.
            mapping = self.state.order_by_intent(internal_intent_id)
            if mapping:
                try:
                    item = self._lookup_order_item(mapping)
                except BrokerError:
                    item = None
                if item is not None:
                    return self._normalize_order_item(item, mapping)
            raise
        mapping = self.state.order_by_intent(internal_intent_id)
        if mapping is None:
            raise BrokerError("Persisted Bybit intent vanished after acknowledgement")
        item = self._lookup_order_item(mapping)
        if item is not None:
            return self._normalize_order_item(item, mapping)
        return OrderRecord(
            broker_order_id=str(mapping.get("order_id") or ""),
            client_order_id=internal_intent_id,
            symbol=mapping["strategy_symbol"], purpose=mapping["purpose"],
            side=str(payload.get("side") or "").lower(), requested_qty=float(mapping["requested_qty"]),
            filled_qty=0.0, status="accepted", submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def submit_crypto_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        *,
        trade_increment: float,
        min_order_size: float,
        reference_price: Optional[float] = None,
    ) -> OrderRecord:
        normalized_side = str(side).lower()
        if normalized_side not in {"buy", "sell"}:
            raise BrokerError("Bybit Spot market side must be buy or sell")
        metadata = self.instrument_provider.get_instrument_metadata(symbol)
        instrument = metadata.instrument
        requested = decimal_floor(float(qty), metadata.quantity_step)
        maximum = metadata.max_market_order_qty
        if requested > maximum + metadata.quantity_step / 10:
            raise BrokerError(f"Bybit market quantity {requested} exceeds dynamic maximum {maximum}")
        if requested + metadata.quantity_step / 10 < metadata.min_order_qty:
            raise BrokerError(f"Bybit quantity {requested} is below dynamic minimum {metadata.min_order_qty}")
        if reference_price is None or not math.isfinite(reference_price) or reference_price <= 0:
            reference_price = self._latest_price(symbol)
        if requested * reference_price + 1e-9 < metadata.min_order_amount:
            raise BrokerError(
                f"Bybit order amount {requested * reference_price:.8g} is below dynamic minimum "
                f"{metadata.min_order_amount:.8g}"
            )
        fallback = "entry" if normalized_side == "buy" else "exit"
        purpose = infer_intent_purpose(client_order_id, fallback)
        if normalized_side == "buy":
            if self.state.count_recent_entries(hours=24) >= self.max_entries_rolling_24h:
                raise BrokerError(
                    f"Rolling 24h high-quality entry cap ({self.max_entries_rolling_24h}) reached; zero additional trades required"
                )
            account = self.get_account()
            if requested * reference_price > account.buying_power + 1e-9:
                raise BrokerError("Bybit buy exceeds remaining strategy capital budget")
        else:
            self.sync_bot_executions(symbol)
            owned = self.state.inventory(instrument.strategy_symbol)
            wallet = self._assert_wallet_not_deficit(instrument, owned)
            allowed = min(owned.qty, wallet["available"])
            if requested > allowed + metadata.quantity_step / 10:
                raise BrokerError(
                    f"Refusing Bybit sell {requested}: safe bot-owned/available bound is {allowed}"
                )
            requested = decimal_floor(min(requested, allowed), metadata.quantity_step)
        instrument, link = self._prepare_intent(client_order_id, symbol, purpose, requested)
        payload = {
            "category": SPOT_CATEGORY,
            "symbol": instrument.venue_symbol,
            "side": "Buy" if normalized_side == "buy" else "Sell",
            "orderType": "Market",
            "qty": as_decimal_string(requested),
            "marketUnit": "baseCoin",
            "timeInForce": "IOC",
            "orderLinkId": link,
            "isLeverage": NO_LEVERAGE,
            "orderFilter": "Order",
        }
        return self._submit_once_reconcile_timeout(
            internal_intent_id=client_order_id, payload=payload, operation=f"{purpose} market order"
        )

    def submit_crypto_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_price: float,
        client_order_id: str,
        *,
        trade_increment: float,
        min_order_size: float,
        price_increment: float,
    ) -> OrderRecord:
        metadata = self.instrument_provider.get_instrument_metadata(symbol)
        instrument = metadata.instrument
        requested = decimal_floor(float(qty), metadata.quantity_step)
        stop = decimal_ceil(float(stop_price), metadata.tick_size)
        limit = decimal_floor(float(limit_price), metadata.tick_size)
        if limit >= stop:
            limit = decimal_floor(stop - metadata.tick_size, metadata.tick_size)
        if requested < metadata.min_order_qty or requested > metadata.max_order_qty:
            raise BrokerError("Protective quantity violates current Bybit instrument limits")
        if limit <= 0 or stop <= 0 or limit >= stop:
            raise BrokerError("Protective Bybit sell requires 0 < limit < trigger price")
        self.sync_bot_executions(symbol)
        owned = self.state.inventory(instrument.strategy_symbol)
        self._assert_wallet_not_deficit(instrument, owned)
        if requested > owned.qty + metadata.quantity_step / 10:
            raise BrokerError("Protective stop cannot exceed proven bot-owned quantity")
        instrument, link = self._prepare_intent(client_order_id, symbol, "stop", requested)
        payload = {
            "category": SPOT_CATEGORY,
            "symbol": instrument.venue_symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": as_decimal_string(requested),
            "price": as_decimal_string(limit),
            "triggerPrice": as_decimal_string(stop),
            "timeInForce": "GTC",
            "orderLinkId": link,
            "isLeverage": NO_LEVERAGE,
            # tpslOrder occupies Spot assets before trigger, preventing manual or
            # bot reuse of inventory that is supposed to remain protected.
            "orderFilter": "tpslOrder",
        }
        return self._submit_once_reconcile_timeout(
            internal_intent_id=client_order_id, payload=payload, operation="protective stop"
        )

    def amend_protection(
        self,
        client_order_id: str,
        *,
        qty: float,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
    ) -> OrderRecord:
        mapping = self.state.order_by_intent(client_order_id)
        if mapping is None or mapping.get("purpose") != "stop":
            raise BrokerError("Only a persisted bot-owned stop may be amended")
        metadata = self.instrument_provider.get_instrument_metadata(mapping["strategy_symbol"])
        normalized_qty = decimal_floor(qty, metadata.quantity_step)
        if normalized_qty < metadata.min_order_qty:
            raise BrokerError("Amended protective quantity is below Bybit minimum")
        payload: dict[str, Any] = {
            "category": SPOT_CATEGORY,
            "symbol": mapping["venue_symbol"],
            "orderLinkId": mapping["order_link_id"],
            "qty": as_decimal_string(normalized_qty),
        }
        if stop_price is not None:
            payload["triggerPrice"] = as_decimal_string(decimal_ceil(stop_price, metadata.tick_size))
        if limit_price is not None:
            payload["price"] = as_decimal_string(decimal_floor(limit_price, metadata.tick_size))
        assert_spot_only_payload(payload, require_leverage=False)
        self._assert_clock_safe()
        self._request(
            "economic", "amend protection", lambda: self.http.amend_order(**payload), economic=True
        )
        mapping = self.state.order_by_intent(client_order_id) or mapping
        item = self._lookup_order_item(mapping)
        if item is None:
            raise BrokerError("Protection amendment acknowledgement could not be reconciled")
        return self._normalize_order_item(item, mapping)

    def broker_health(self) -> dict[str, Any]:
        health = "HEALTHY"
        if time.monotonic() < self._circuit_open_until or self._circuit_failures >= 3:
            health = "DEGRADED"
        if self._clock_skew_ms is not None and abs(self._clock_skew_ms) > self.max_clock_skew_ms:
            health = "HALTED"
        pending = sum(
            1 for order in self.state.bot_orders() if order["status"] not in {"filled", "canceled", "rejected", "expired"}
        )
        return {
            "provider": "bybit",
            "environment": self.environment.value,
            "health": health,
            "rest_host": self.expected_rest_host,
            "clock_skew_ms": self._clock_skew_ms,
            "rest_failures": self._rest_failures,
            "rate_limit_events": self._rate_limit_events,
            "pending_intents": pending,
            "strategy_capital_budget_usdt": self.strategy_capital_budget_usdt,
            "spot_only": True,
            "isLeverage": 0,
            "live_enabled": self.environment is CryptoExecutionEnvironment.BYBIT_LIVE,
        }


def _require_ok(response: Any, operation: str) -> None:
    if not isinstance(response, dict):
        raise BrokerError(f"Bybit {operation} returned a non-object response")
    code = response.get("retCode", 0)
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        code_int = -1
    if code_int != 0:
        message = str(response.get("retMsg") or "request failed")[:180]
        raise BrokerError(f"Bybit {operation} rejected ({code_int}): {message}")
