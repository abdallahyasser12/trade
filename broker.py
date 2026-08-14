"""Alpaca PAPER trading operations, isolated behind one class.

Safety properties enforced here, by construction:

* The trading client is created with ``paper=True`` and no URL override,
  and ``verify_paper_account`` additionally asserts the client's base URL
  is Alpaca's paper endpoint.  There is no live-trading switch anywhere.
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
from typing import Optional

from config import CLIENT_ORDER_PREFIX
from models import AccountInfo, AssetInfo, BrokerPosition, MarketClock, OrderRecord

logger = logging.getLogger(__name__)

PAPER_HOST_FRAGMENT = "paper-api.alpaca.markets"


class BrokerError(Exception):
    """A broker operation failed or a safety invariant could not be verified."""


def make_client_order_id(
    session_id: str, purpose: str, bar_timestamp: datetime, symbol: str = ""
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
    middle = f"-{symbol}" if symbol else ""
    return f"{CLIENT_ORDER_PREFIX}-{session_id[:8]}{middle}-{purpose}-{epoch}"


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
        submitted_at=order.submitted_at,
        updated_at=order.updated_at,
    )


class AlpacaPaperBroker:
    """All Alpaca trading-API access for the bot.  Paper-only by design."""

    def __init__(self, api_key: str, secret_key: str):
        from alpaca.trading.client import TradingClient

        if not api_key or not secret_key:
            raise BrokerError("Alpaca API key and secret key are required.")
        # paper=True is hardcoded on purpose; there is no live mode.
        self._client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    # ------------------------------------------------------------------
    # Account / environment
    # ------------------------------------------------------------------

    def verify_paper_account(self) -> AccountInfo:
        """Verify credentials, account health and the PAPER environment.

        Raises BrokerError when any check fails; live execution must stay
        disabled in that case.
        """
        base_url = str(getattr(self._client, "_base_url", ""))
        if PAPER_HOST_FRAGMENT not in base_url:
            raise BrokerError(
                "Trading client is not pointed at the Alpaca paper endpoint — refusing to trade."
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
        )

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
        if qty <= 0:
            raise BrokerError(f"Refusing to submit order with qty {qty}")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
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
        if stop_price <= 0:
            raise BrokerError(f"Refusing to submit stop with price {stop_price}")
        request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
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
