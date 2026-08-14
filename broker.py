"""Alpaca trading operations isolated behind an environment-verified boundary.

Safety properties enforced here, by construction:

* The trading client receives an explicit paper/live enum and no URL override.
  ``verify_account`` requires exact equality with Alpaca's official endpoint.
* Live construction is disabled unless the process-level live feature flag is
  explicitly enabled. Separate live credentials are loaded by ``config``.
* Every order the bot creates carries a ``trendbot-`` client_order_id
  namespace; the broker only ever lists/cancels orders in that namespace.
* ``submit_order`` returning is never treated as a fill — callers must
  confirm fills through ``get_order``/``get_order_by_client_id``.

The strategy never calls Alpaca directly, and the backtester does not
import this module.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import math
import re
from typing import Optional

from config import AccountMode, CLIENT_ORDER_PREFIX, live_trading_enabled
from models import AccountInfo, AssetInfo, BrokerPosition, MarketClock, OrderRecord

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
PAPER_HOST_FRAGMENT = "paper-api.alpaca.markets"  # backwards-compatible import
_CLIENT_SYMBOL_RE = re.compile(r"[^A-Z0-9]")


class BrokerError(Exception):
    """A broker operation failed or a safety invariant could not be verified."""


def make_client_order_id(
    session_id: str,
    purpose: str,
    bar_timestamp: datetime,
    symbol: str = "",
    attempt: int = 1,
) -> str:
    """Deterministic, recognizable client order id.

    Format: ``trendbot-<session8>-<SYMBOL>-<purpose>-<barEpoch>`` (symbol
    segment omitted when empty).  Deterministic per (session, symbol,
    purpose, bar) so a resubmission attempt after a timeout maps to the
    same id and can be detected instead of duplicated.
    """
    if bar_timestamp.tzinfo is None:
        bar_timestamp = bar_timestamp.replace(tzinfo=timezone.utc)
    epoch = int(bar_timestamp.timestamp())
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    symbol_token = _CLIENT_SYMBOL_RE.sub("", symbol.upper())[:10]
    middle = f"-{symbol_token}" if symbol_token else ""
    attempt_token = ""
    value = attempt
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    while value:
        value, remainder = divmod(value, len(alphabet))
        attempt_token = alphabet[remainder] + attempt_token
    retry = f"-a{attempt_token}" if attempt > 1 else ""
    # Keep the total bounded even after many stop re-arms, while retaining
    # a recognizable session namespace.
    session_chars = max(4, 8 - max(0, len(attempt_token) - 1))
    client_id = (
        f"{CLIENT_ORDER_PREFIX}-{session_id[:session_chars]}{middle}{retry}-{purpose}-{epoch}"
    )
    if len(client_id) > 48:
        raise ValueError("Generated client_order_id exceeds Alpaca's 48-character limit")
    return client_id


def is_bot_order_id(client_order_id: Optional[str]) -> bool:
    """Whether an order belongs to this bot's namespace."""
    return bool(client_order_id) and client_order_id.startswith(f"{CLIENT_ORDER_PREFIX}-")


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_sell_stop_price(stop_price: float) -> float:
    """Normalize a protective sell stop to Alpaca's valid price increment.

    Alpaca accepts at most two decimals at/above $1 and four decimals below
    $1.  Rounding a sell stop upward is conservative for a long position and
    avoids silently increasing the configured loss distance.
    """
    if not math.isfinite(stop_price) or stop_price <= 0:
        raise BrokerError(f"Refusing to submit stop with price {stop_price}")
    places = Decimal("0.01") if stop_price >= 1.0 else Decimal("0.0001")
    try:
        normalized = Decimal(str(stop_price)).quantize(places, rounding=ROUND_CEILING)
    except InvalidOperation as exc:
        raise BrokerError(f"Invalid protective stop price {stop_price}") from exc
    return float(normalized)


def normalize_to_increment(value: float, increment: float, *, round_up: bool) -> float:
    """Normalize a positive finite value to an asset's dynamic increment."""

    if not math.isfinite(value) or value <= 0:
        raise BrokerError(f"Value must be finite and positive; received {value!r}.")
    if not math.isfinite(increment) or increment <= 0:
        raise BrokerError(f"Increment must be finite and positive; received {increment!r}.")
    try:
        raw = Decimal(str(value))
        step = Decimal(str(increment))
        units = (raw / step).to_integral_value(
            rounding=ROUND_CEILING if round_up else ROUND_FLOOR
        )
        normalized = units * step
    except InvalidOperation as exc:
        raise BrokerError("Could not normalize value to Alpaca asset increment.") from exc
    if normalized <= 0:
        raise BrokerError("Increment normalization produced a zero value.")
    return float(normalized)


def order_to_record(order) -> OrderRecord:
    """Convert an alpaca-py Order model to the internal OrderRecord."""
    client_id = order.client_order_id or ""
    purpose = "unknown"
    if is_bot_order_id(client_id):
        parts = client_id.split("-")
        if len(parts) >= 4:
            # Format: trendbot-<sess8>[-<symbol>]-<purpose>-<epoch>;
            # purpose is always the second-to-last dash segment.
            purpose = parts[-2]
    status = order.status.value if hasattr(order.status, "value") else str(order.status)
    side = order.side.value if hasattr(order.side, "value") else str(order.side)
    return OrderRecord(
        broker_order_id=str(order.id),
        client_order_id=client_id,
        symbol=order.symbol,
        purpose=purpose,
        side=side,
        requested_qty=_float_or_none(order.qty) or 0.0,
        filled_qty=_float_or_none(order.filled_qty) or 0.0,
        status=status,
        fill_price=_float_or_none(order.filled_avg_price),
        stop_price=_float_or_none(order.stop_price),
        limit_price=_float_or_none(getattr(order, "limit_price", None)),
        submitted_at=order.submitted_at,
        updated_at=order.updated_at,
    )


class AlpacaBroker:
    """Alpaca boundary for one immutable, verified account environment."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        account_mode: AccountMode | str = AccountMode.PAPER,
        *,
        max_live_order_notional: float = 1_000.0,
    ):
        from alpaca.trading.client import TradingClient

        if not api_key or not secret_key:
            raise BrokerError("Alpaca API key and secret key are required.")
        self._account_mode = AccountMode(account_mode)
        if self._account_mode is AccountMode.LIVE:
            if not live_trading_enabled():
                raise BrokerError(
                    "Live trading is disabled. Set ENABLE_LIVE_TRADING=true and restart "
                    "before constructing a live broker."
                )
        if (
            not math.isfinite(max_live_order_notional)
            or max_live_order_notional <= 0
            or max_live_order_notional > 10_000
        ):
            raise BrokerError("Live order notional cap must be in (0, $10,000].")
        self._max_live_order_notional = float(max_live_order_notional)
        # No URL override is accepted anywhere in this boundary.
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=self._account_mode is AccountMode.PAPER,
        )

    @property
    def account_mode(self) -> AccountMode:
        return self._account_mode

    # ------------------------------------------------------------------
    # Account / environment
    # ------------------------------------------------------------------

    def verify_account(self) -> AccountInfo:
        """Verify credentials, account health, and the exact SDK environment.

        Raises BrokerError when any check fails; live execution must stay
        disabled in that case.
        """
        raw_base_url = getattr(self._client, "_base_url", "")
        # alpaca-py 0.44 stores this as a BaseURL enum whose ``str()`` is
        # "BaseURL.TRADING_PAPER", not the endpoint value.
        base_url = str(getattr(raw_base_url, "value", raw_base_url))
        expected = (
            PAPER_BASE_URL
            if self._account_mode is AccountMode.PAPER
            else LIVE_BASE_URL
        )
        if base_url.rstrip("/") != expected:
            raise BrokerError(
                f"Trading client endpoint is {base_url!r}, not the required {expected!r}; "
                "refusing to trade."
            )
        try:
            account = self.get_account()
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        if account.status != "ACTIVE":
            raise BrokerError(f"Alpaca account status is {account.status}, not ACTIVE.")
        if account.trading_blocked:
            raise BrokerError("Alpaca reports trading is blocked on this account.")
        return account

    def verify_paper_account(self) -> AccountInfo:
        """Backwards-compatible paper-only verification used by the stock engine."""

        if self._account_mode is not AccountMode.PAPER:
            raise BrokerError("Paper-account verification cannot run in live mode.")
        return self.verify_account()

    def get_account(self) -> AccountInfo:
        try:
            acct = self._client.get_account()
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        status = acct.status.value if hasattr(acct.status, "value") else str(acct.status)
        return AccountInfo(
            equity=_float_or_none(acct.equity) or 0.0,
            cash=_float_or_none(acct.cash) or 0.0,
            buying_power=_float_or_none(acct.buying_power) or 0.0,
            status=status,
            currency=str(acct.currency or "USD"),
            pattern_day_trader=bool(acct.pattern_day_trader),
            daytrade_count=int(acct.daytrade_count or 0),
            trading_blocked=bool(acct.trading_blocked),
        )

    def get_clock(self) -> MarketClock:
        try:
            clock = self._client.get_clock()
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return MarketClock(
            timestamp=clock.timestamp,
            is_open=bool(clock.is_open),
            next_open=clock.next_open,
            next_close=clock.next_close,
        )

    def get_asset(self, symbol: str) -> AssetInfo:
        try:
            asset = self._client.get_asset(symbol)
        except Exception as exc:
            raise BrokerError(
                f"Alpaca does not recognize symbol '{symbol}' or the asset "
                f"lookup failed: {_sanitized_broker_message(exc)}"
            ) from exc
        asset_class = (
            asset.asset_class.value if hasattr(asset.asset_class, "value") else str(asset.asset_class)
        )
        status = asset.status.value if hasattr(asset.status, "value") else str(asset.status)
        return AssetInfo(
            symbol=asset.symbol,
            name=asset.name or asset.symbol,
            tradable=bool(asset.tradable),
            active=(status == "active"),
            fractionable=bool(asset.fractionable),
            asset_class=asset_class,
            min_order_size=_float_or_none(getattr(asset, "min_order_size", None)),
            min_trade_increment=_float_or_none(
                getattr(asset, "min_trade_increment", None)
            ),
            price_increment=_float_or_none(getattr(asset, "price_increment", None)),
            marginable=bool(getattr(asset, "marginable", False)),
            shortable=bool(getattr(asset, "shortable", False)),
        )

    def verify_crypto_assets(self, symbols: list[str] | tuple[str, ...]) -> dict[str, AssetInfo]:
        """Resolve and validate every configured long-only spot asset."""

        verified: dict[str, AssetInfo] = {}
        for symbol in symbols:
            asset = self.get_asset(symbol)
            if not asset.is_supported_crypto:
                raise BrokerError(
                    f"{symbol} is not an active, tradable, fractionable, non-marginable "
                    "Alpaca crypto asset."
                )
            if not asset.min_order_size or not asset.min_trade_increment or not asset.price_increment:
                raise BrokerError(f"{symbol} is missing Alpaca's dynamic crypto increments.")
            verified[symbol] = asset
        return verified

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Current position for ``symbol``, or None when flat."""
        from alpaca.common.exceptions import APIError

        try:
            pos = self._client.get_open_position(symbol)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return self._position_to_model(pos)

    def list_positions(self) -> list[BrokerPosition]:
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return [self._position_to_model(p) for p in positions]

    @staticmethod
    def _position_to_model(pos) -> BrokerPosition:
        return BrokerPosition(
            symbol=pos.symbol,
            qty=_float_or_none(pos.qty) or 0.0,
            avg_entry_price=_float_or_none(pos.avg_entry_price) or 0.0,
            current_price=_float_or_none(pos.current_price),
            market_value=_float_or_none(pos.market_value),
            unrealized_pl=_float_or_none(pos.unrealized_pl),
            unrealized_plpc=_float_or_none(pos.unrealized_plpc),
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_market_order(
        self, symbol: str, qty: int, side: str, client_order_id: str
    ) -> OrderRecord:
        """Submit a whole-share DAY market order in the bot's namespace."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        self._require_bot_id(client_order_id)
        if getattr(self, "_account_mode", AccountMode.PAPER) is AccountMode.LIVE:
            raise BrokerError(
                "Live equity orders are not enabled by this extension; use the "
                "guarded crypto workspace or the audited stock paper engine."
            )
        if qty <= 0:
            raise BrokerError(f"Refusing to submit order with qty {qty}")
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell"}:
            raise BrokerError(f"Unsupported order side {side!r}; expected 'buy' or 'sell'.")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if normalized_side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

    def submit_stop_order(
        self, symbol: str, qty: int, stop_price: float, client_order_id: str
    ) -> OrderRecord:
        """Submit a bot-owned protective SELL stop (GTC) for a long position."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        self._require_bot_id(client_order_id)
        if qty <= 0:
            raise BrokerError(f"Refusing to submit stop with qty {qty}")
        normalized_stop = normalize_sell_stop_price(stop_price)
        request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=normalized_stop,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

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
        """Submit a fractional GTC crypto market order after boundary checks.

        Buy notional is rechecked here in live mode. Risk-reducing sell exits are
        deliberately not blocked by an order-notional cap.
        """

        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        from crypto_config import validate_crypto_symbol

        self._require_bot_id(client_order_id)
        error = validate_crypto_symbol(symbol)
        if error:
            raise BrokerError(error)
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell"}:
            raise BrokerError(f"Unsupported order side {side!r}; expected 'buy' or 'sell'.")
        normalized_qty = normalize_to_increment(qty, trade_increment, round_up=False)
        if normalized_qty + (trade_increment / 10.0) < min_order_size:
            raise BrokerError(
                f"Crypto quantity {normalized_qty} is below Alpaca minimum {min_order_size}."
            )
        mode = getattr(self, "_account_mode", AccountMode.PAPER)
        if mode is AccountMode.LIVE and normalized_side == "buy":
            if reference_price is None or not math.isfinite(reference_price) or reference_price <= 0:
                raise BrokerError("A positive reference price is required for a live crypto buy.")
            notional = normalized_qty * reference_price
            cap = getattr(self, "_max_live_order_notional", 0.0)
            if notional > cap + 1e-9:
                raise BrokerError(
                    f"Live crypto buy notional ${notional:,.2f} exceeds hard cap ${cap:,.2f}."
                )
        request = MarketOrderRequest(
            symbol=symbol,
            qty=normalized_qty,
            side=OrderSide.BUY if normalized_side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

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
        """Submit a fractional GTC protective SELL stop-limit for spot crypto."""

        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import StopLimitOrderRequest
        from crypto_config import validate_crypto_symbol

        self._require_bot_id(client_order_id)
        error = validate_crypto_symbol(symbol)
        if error:
            raise BrokerError(error)
        normalized_qty = normalize_to_increment(qty, trade_increment, round_up=False)
        if normalized_qty + (trade_increment / 10.0) < min_order_size:
            raise BrokerError(
                f"Protective quantity {normalized_qty} is below Alpaca minimum {min_order_size}."
            )
        normalized_stop = normalize_to_increment(stop_price, price_increment, round_up=True)
        normalized_limit = normalize_to_increment(limit_price, price_increment, round_up=False)
        if normalized_limit >= normalized_stop:
            normalized_limit = normalize_to_increment(
                normalized_stop - price_increment, price_increment, round_up=False
            )
        if normalized_limit <= 0 or normalized_limit >= normalized_stop:
            raise BrokerError("Protective sell stop-limit requires 0 < limit price < stop price.")
        request = StopLimitOrderRequest(
            symbol=symbol,
            qty=normalized_qty,
            side=OrderSide.SELL,
            type=OrderType.STOP_LIMIT,
            time_in_force=TimeInForce.GTC,
            stop_price=normalized_stop,
            limit_price=normalized_limit,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

    def get_order(self, broker_order_id: str) -> OrderRecord:
        try:
            order = self._client.get_order_by_id(broker_order_id)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]:
        from alpaca.common.exceptions import APIError

        try:
            order = self._client.get_order_by_client_id(client_order_id)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return order_to_record(order)

    def list_open_orders(self, symbol: Optional[str] = None) -> list[OrderRecord]:
        """Open orders in the bot's namespace only.

        Orders that do not carry the ``trendbot-`` prefix are invisible to
        the bot: it must never manage orders it cannot prove it owns.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
            limit=200,
        )
        try:
            orders = self._client.get_orders(request)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        return [order_to_record(o) for o in orders if is_bot_order_id(o.client_order_id)]

    def cancel_order(self, broker_order_id: str) -> None:
        """Cancel one order — only after proving it is bot-owned."""
        record = self.get_order(broker_order_id)
        if not is_bot_order_id(record.client_order_id):
            raise BrokerError(
                "Refusing to cancel an order that does not belong to this bot "
                f"(client id {record.client_order_id!r})."
            )
        try:
            self._client.cancel_order_by_id(broker_order_id)
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc

    @staticmethod
    def _require_bot_id(client_order_id: str) -> None:
        if not is_bot_order_id(client_order_id):
            raise BrokerError(
                f"Client order id {client_order_id!r} is outside the bot namespace."
            )
        if len(client_order_id) > 48:
            raise BrokerError("Client order id exceeds Alpaca's 48-character limit.")


class AlpacaPaperBroker(AlpacaBroker):
    """Compatibility-safe paper broker used by the audited stock engine."""

    def __init__(self, api_key: str, secret_key: str):
        super().__init__(api_key, secret_key, AccountMode.PAPER)


class AlpacaLiveBroker(AlpacaBroker):
    """Explicitly gated live broker used only by the crypto workspace."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        max_live_order_notional: float = 1_000.0,
    ):
        super().__init__(
            api_key,
            secret_key,
            AccountMode.LIVE,
            max_live_order_notional=max_live_order_notional,
        )


def _sanitized_broker_message(exc: Exception) -> str:
    """Broker error text safe for the UI/logs: no headers, no secrets."""
    try:
        from alpaca.common.exceptions import APIError

        if isinstance(exc, APIError):
            code = exc.status_code
            try:
                message = exc.message
            except Exception:
                message = "request failed"
            if code == 401:
                return "Alpaca rejected the API credentials (401). Check your .env keys."
            if code == 403:
                return f"Alpaca denied the request (403): {message}"
            if code == 429:
                return "Alpaca rate limit reached (429). The bot will back off."
            return f"Alpaca API error ({code}): {message}"
    except Exception:
        # Sanitizing must never itself raise; fall back to the generic text.
        return _truncated(str(exc))
    return _truncated(str(exc))


def _truncated(text: str, limit: int = 300) -> str:
    return text if len(text) < limit else text[:limit] + "…"
