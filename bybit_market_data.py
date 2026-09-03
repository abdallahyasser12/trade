"""Bybit Spot REST + supervised WebSocket market/private streams.

Public Spot market data always uses mainnet public streams, including Demo trading.
Demo private account/order/execution events use Bybit's Demo private WebSocket.
Economic order entry is deliberately absent from this module and from WebSockets.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd

from broker import BrokerError
from bybit_spot import SPOT_CATEGORY, _num, _require_ok
from config import timeframe_to_timedelta
from crypto_data_feed import is_crypto_data_stale
from crypto_models import CryptoQuote
from crypto_venue import CryptoExecutionEnvironment, DEFAULT_BYBIT_SPOT_UNIVERSE, bybit_instrument
from data_feed import DataFeedError, drop_incomplete_bars

logger = logging.getLogger(__name__)

_TIMEFRAME_TO_BYBIT = {
    "1Min": "1",
    "5Min": "5",
    "15Min": "15",
    "1Hour": "60",
    "4Hour": "240",
    "1Day": "D",
}


def bybit_interval(timeframe: str) -> str:
    try:
        return _TIMEFRAME_TO_BYBIT[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported Bybit Spot timeframe {timeframe!r}") from exc


def _utc_ms(ms: Any) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


@dataclass(frozen=True)
class MarketContextSnapshot:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    spread_bps: float
    bid_depth: float
    ask_depth: float
    book_imbalance: float
    signed_trade_flow: float
    quote_age_ms: float
    orderbook_sequence: Optional[int]
    gap_detected: bool

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "spread_bps": self.spread_bps,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "book_imbalance": self.book_imbalance,
            "signed_trade_flow": self.signed_trade_flow,
            "quote_age_ms": self.quote_age_ms,
            "orderbook_sequence": self.orderbook_sequence,
            "gap_detected": self.gap_detected,
        }


class _OrderBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.update_id: Optional[int] = None
        self.sequence: Optional[int] = None
        self.timestamp: Optional[datetime] = None
        self.gap_detected = False

    @staticmethod
    def _apply(side: dict[float, float], updates: list[list[str]]) -> None:
        for price_raw, size_raw, *_ in updates:
            price = _num(price_raw)
            size = _num(size_raw)
            if price <= 0:
                continue
            if size <= 0:
                side.pop(price, None)
            else:
                side[price] = size

    def ingest(self, message: dict[str, Any]) -> bool:
        data = message.get("data") or {}
        msg_type = str(message.get("type") or "")
        update_id = int(data.get("u")) if data.get("u") is not None else None
        sequence = int(data.get("seq")) if data.get("seq") is not None else None
        if msg_type == "snapshot" or update_id == 1:
            self.bids.clear()
            self.asks.clear()
            self._apply(self.bids, data.get("b") or [])
            self._apply(self.asks, data.get("a") or [])
            self.gap_detected = False
        else:
            if update_id is not None and self.update_id is not None:
                if update_id <= self.update_id:
                    return False  # duplicate/out of order
                if update_id > self.update_id + 1:
                    self.gap_detected = True
            if sequence is not None and self.sequence is not None and sequence < self.sequence:
                return False
            self._apply(self.bids, data.get("b") or [])
            self._apply(self.asks, data.get("a") or [])
        if update_id is not None:
            self.update_id = update_id
        if sequence is not None:
            self.sequence = sequence
        ts = message.get("cts") or message.get("ts")
        self.timestamp = _utc_ms(ts) if ts is not None else datetime.now(timezone.utc)
        return True

    def load_rest(self, result: dict[str, Any]) -> None:
        self.bids.clear()
        self.asks.clear()
        self._apply(self.bids, result.get("b") or [])
        self._apply(self.asks, result.get("a") or [])
        self.update_id = int(result.get("u")) if result.get("u") is not None else None
        self.sequence = int(result.get("seq")) if result.get("seq") is not None else None
        ts = result.get("ts") or int(time.time() * 1000)
        self.timestamp = _utc_ms(ts)
        self.gap_detected = False

    def best(self) -> tuple[float, float, float, float]:
        if not self.bids or not self.asks:
            raise DataFeedError("Bybit order book has no two-sided market")
        bid = max(self.bids)
        ask = min(self.asks)
        if bid <= 0 or ask <= 0 or ask < bid:
            raise DataFeedError("Bybit order book is crossed or invalid")
        return bid, ask, self.bids[bid], self.asks[ask]

    def depth(self, levels: int = 10) -> tuple[float, float]:
        bids = sorted(self.bids.items(), reverse=True)[:levels]
        asks = sorted(self.asks.items())[:levels]
        return sum(qty for _, qty in bids), sum(qty for _, qty in asks)


class BybitPublicWebSocketManager:
    """One supervised public Spot socket; reconnect never creates a trading engine."""
    def __init__(
        self,
        http_client,
        *,
        symbols: tuple[str, ...] = DEFAULT_BYBIT_SPOT_UNIVERSE,
        timeframe: str = "5Min",
        depth: int = 50,
        websocket_factory=None,
        stale_after_seconds: float = 30.0,
    ) -> None:
        if depth not in {1, 50, 200, 1000}:
            raise ValueError("Bybit Spot order-book depth must be 1, 50, 200 or 1000")
        self.http = http_client
        self.symbols = tuple(bybit_instrument(s).strategy_symbol for s in symbols)
        self.timeframe = timeframe
        self.depth = depth
        self.stale_after_seconds = float(stale_after_seconds)
        self._factory = websocket_factory
        self._socket = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._supervisor: Optional[threading.Thread] = None
        self._last_message_monotonic: Optional[float] = None
        self._reconnects = 0
        self._gaps = 0
        self._duplicates = 0
        self._books = {s: _OrderBook() for s in self.symbols}
        self._closed_klines: dict[str, deque[dict[str, Any]]] = {
            s: deque(maxlen=1500) for s in self.symbols
        }
        self._trades: dict[str, deque[dict[str, Any]]] = {
            s: deque(maxlen=1000) for s in self.symbols
        }
        self._seen_trade_ids: dict[str, set[str]] = defaultdict(set)
        self._seen_trade_order: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=2000))

    def _new_socket(self):
        if self._factory is not None:
            return self._factory(testnet=False, channel_type="spot")
        from pybit.unified_trading import WebSocket
        return WebSocket(testnet=False, channel_type="spot")

    def start(self) -> None:
        with self._lock:
            if self._socket is not None:
                return
            self._stop.clear()
            self._connect_locked()
            self._supervisor = threading.Thread(
                target=self._supervise, name="bybit-public-ws-supervisor", daemon=True
            )
            self._supervisor.start()

    def _connect_locked(self) -> None:
        ws = self._new_socket()
        interval = bybit_interval(self.timeframe)
        for strategy_symbol in self.symbols:
            venue = bybit_instrument(strategy_symbol).venue_symbol
            ws.orderbook_stream(depth=self.depth, symbol=venue, callback=self._on_orderbook)
            ws.trade_stream(symbol=venue, callback=self._on_trade)
            ws.kline_stream(interval=interval, symbol=venue, callback=self._on_kline)
        self._socket = ws
        self._last_message_monotonic = time.monotonic()

    def _strategy_symbol_from_venue(self, venue_symbol: str) -> str:
        return bybit_instrument(venue_symbol).strategy_symbol

    def _touch(self) -> None:
        self._last_message_monotonic = time.monotonic()

    def _on_orderbook(self, message: dict[str, Any]) -> None:
        try:
            data = message.get("data") or {}
            symbol = self._strategy_symbol_from_venue(str(data.get("s") or ""))
            with self._lock:
                before_gap = self._books[symbol].gap_detected
                accepted = self._books[symbol].ingest(message)
                if not accepted:
                    self._duplicates += 1
                if self._books[symbol].gap_detected and not before_gap:
                    self._gaps += 1
                self._touch()
        except Exception:
            logger.exception("Bybit public order-book callback failed")

    def _on_trade(self, message: dict[str, Any]) -> None:
        try:
            rows = message.get("data") or []
            with self._lock:
                for row in rows:
                    symbol = self._strategy_symbol_from_venue(str(row.get("s") or ""))
                    trade_id = str(row.get("i") or "")
                    if trade_id and trade_id in self._seen_trade_ids[symbol]:
                        self._duplicates += 1
                        continue
                    if trade_id:
                        order = self._seen_trade_order[symbol]
                        if len(order) == order.maxlen and order:
                            expired = order[0]
                            self._seen_trade_ids[symbol].discard(expired)
                        order.append(trade_id)
                        self._seen_trade_ids[symbol].add(trade_id)
                    self._trades[symbol].append(dict(row))
                self._touch()
        except Exception:
            logger.exception("Bybit public trade callback failed")

    def _on_kline(self, message: dict[str, Any]) -> None:
        try:
            topic = str(message.get("topic") or "")
            venue = topic.rsplit(".", 1)[-1]
            symbol = self._strategy_symbol_from_venue(venue)
            with self._lock:
                for row in message.get("data") or []:
                    if not bool(row.get("confirm")):
                        continue
                    start = int(row.get("start"))
                    if self._closed_klines[symbol] and int(self._closed_klines[symbol][-1]["start"]) == start:
                        self._closed_klines[symbol][-1] = dict(row)
                    else:
                        self._closed_klines[symbol].append(dict(row))
                self._touch()
        except Exception:
            logger.exception("Bybit public kline callback failed")

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.wait(2.0):
            last = self._last_message_monotonic
            if last is not None and time.monotonic() - last <= self.stale_after_seconds:
                backoff = 1.0
                continue
            try:
                with self._lock:
                    old = self._socket
                    self._socket = None
                    if old is not None:
                        try:
                            old.exit()
                        except Exception:
                            pass
                    self._connect_locked()
                    self._reconnects += 1
                backoff = 1.0
            except Exception:
                logger.exception("Bybit public WebSocket reconnect failed")
                self._stop.wait(backoff + random.uniform(0, min(1.0, backoff / 4)))
                backoff = min(30.0, backoff * 2.0)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            ws = self._socket
            self._socket = None
        if ws is not None:
            try:
                ws.exit()
            except Exception:
                logger.exception("Could not close Bybit public WebSocket cleanly")
        thread = self._supervisor
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _rest_resync_book(self, symbol: str) -> None:
        venue = bybit_instrument(symbol).venue_symbol
        response = self.http.get_orderbook(category=SPOT_CATEGORY, symbol=venue, limit=self.depth)
        _require_ok(response, "order book resync")
        result = (response or {}).get("result") or {}
        with self._lock:
            self._books[symbol].load_rest(result)

    def snapshot(self, symbol: str, *, max_age_seconds: float = 5.0) -> MarketContextSnapshot:
        strategy_symbol = bybit_instrument(symbol).strategy_symbol
        with self._lock:
            book = self._books[strategy_symbol]
            need_resync = book.gap_detected or book.timestamp is None
        if need_resync:
            self._rest_resync_book(strategy_symbol)
        with self._lock:
            book = self._books[strategy_symbol]
            bid, ask, _, _ = book.best()
            bid_depth, ask_depth = book.depth(10)
            timestamp = book.timestamp or datetime.now(timezone.utc)
            age_ms = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() * 1000.0)
            if age_ms > max_age_seconds * 1000.0:
                raise DataFeedError(f"Bybit order book for {strategy_symbol} is stale ({age_ms:.0f}ms)")
            denom = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / denom if denom > 0 else 0.0
            signed_flow = 0.0
            for row in list(self._trades[strategy_symbol])[-100:]:
                qty = _num(row.get("v"))
                signed_flow += qty if str(row.get("S")) == "Buy" else -qty
            midpoint = (bid + ask) / 2.0
            spread_bps = (ask - bid) / midpoint * 10_000.0
            return MarketContextSnapshot(
                symbol=strategy_symbol,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                book_imbalance=imbalance,
                signed_trade_flow=signed_flow,
                quote_age_ms=age_ms,
                orderbook_sequence=book.sequence,
                gap_detected=book.gap_detected,
            )

    def closed_klines(self, symbol: str) -> list[dict[str, Any]]:
        strategy_symbol = bybit_instrument(symbol).strategy_symbol
        with self._lock:
            return list(self._closed_klines[strategy_symbol])

    def health(self) -> dict[str, Any]:
        last = self._last_message_monotonic
        age = time.monotonic() - last if last is not None else math.inf
        return {
            "public_ws_up": self._socket is not None and age <= self.stale_after_seconds,
            "public_ws_message_age_seconds": age,
            "public_ws_reconnects": self._reconnects,
            "public_ws_gaps": self._gaps,
            "public_ws_duplicates": self._duplicates,
        }


class BybitPrivateWebSocketManager:
    """Supervised private Demo/Live event stream. It never submits orders."""
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        environment: CryptoExecutionEnvironment,
        on_order: Optional[Callable[[dict[str, Any]], None]] = None,
        on_execution: Optional[Callable[[dict[str, Any]], None]] = None,
        on_wallet: Optional[Callable[[dict[str, Any]], None]] = None,
        websocket_factory=None,
        stale_after_seconds: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.environment = CryptoExecutionEnvironment(environment)
        if self.environment not in {
            CryptoExecutionEnvironment.BYBIT_DEMO,
            CryptoExecutionEnvironment.BYBIT_LIVE,
        }:
            raise ValueError("Private Bybit stream requires Demo or Live environment")
        self.on_order = on_order
        self.on_execution = on_execution
        self.on_wallet = on_wallet
        self._factory = websocket_factory
        self.stale_after_seconds = float(stale_after_seconds)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._socket = None
        self._supervisor: Optional[threading.Thread] = None
        self._last_message_monotonic: Optional[float] = None
        self._reconnects = 0
        self._duplicates = 0
        self._seen_exec: deque[str] = deque(maxlen=5000)
        self._seen_exec_set: set[str] = set()

    def _new_socket(self):
        kwargs = {
            "testnet": False,
            "channel_type": "private",
            "demo": self.environment is CryptoExecutionEnvironment.BYBIT_DEMO,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
        }
        if self._factory is not None:
            return self._factory(**kwargs)
        from pybit.unified_trading import WebSocket
        return WebSocket(**kwargs)

    def start(self) -> None:
        with self._lock:
            if self._socket is not None:
                return
            self._stop.clear()
            self._connect_locked()
            self._supervisor = threading.Thread(
                target=self._supervise, name="bybit-private-ws-supervisor", daemon=True
            )
            self._supervisor.start()

    def _connect_locked(self) -> None:
        ws = self._new_socket()
        ws.order_stream(callback=self._handle_order)
        ws.execution_stream(callback=self._handle_execution)
        ws.wallet_stream(callback=self._handle_wallet)
        self._socket = ws
        self._last_message_monotonic = time.monotonic()

    def _touch(self) -> None:
        self._last_message_monotonic = time.monotonic()

    def _handle_order(self, message: dict[str, Any]) -> None:
        self._touch()
        if self.on_order:
            self.on_order(message)

    def _handle_execution(self, message: dict[str, Any]) -> None:
        self._touch()
        data = message.get("data") or []
        unique_rows = []
        with self._lock:
            for row in data:
                exec_id = str(row.get("execId") or "")
                if exec_id and exec_id in self._seen_exec_set:
                    self._duplicates += 1
                    continue
                if exec_id:
                    if len(self._seen_exec) == self._seen_exec.maxlen and self._seen_exec:
                        self._seen_exec_set.discard(self._seen_exec[0])
                    self._seen_exec.append(exec_id)
                    self._seen_exec_set.add(exec_id)
                unique_rows.append(row)
        if unique_rows and self.on_execution:
            forwarded = dict(message)
            forwarded["data"] = unique_rows
            self.on_execution(forwarded)

    def _handle_wallet(self, message: dict[str, Any]) -> None:
        self._touch()
        if self.on_wallet:
            self.on_wallet(message)

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.wait(5.0):
            last = self._last_message_monotonic
            if last is not None and time.monotonic() - last <= self.stale_after_seconds:
                backoff = 1.0
                continue
            try:
                with self._lock:
                    old = self._socket
                    self._socket = None
                    if old is not None:
                        try:
                            old.exit()
                        except Exception:
                            pass
                    self._connect_locked()
                    self._reconnects += 1
                backoff = 1.0
            except Exception:
                logger.exception("Bybit private WebSocket reconnect failed")
                self._stop.wait(backoff + random.uniform(0, min(1.0, backoff / 4)))
                backoff = min(30.0, backoff * 2.0)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            ws = self._socket
            self._socket = None
        if ws is not None:
            try:
                ws.exit()
            except Exception:
                logger.exception("Could not close Bybit private WebSocket cleanly")
        thread = self._supervisor
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def health(self) -> dict[str, Any]:
        last = self._last_message_monotonic
        age = time.monotonic() - last if last is not None else math.inf
        return {
            "private_ws_up": self._socket is not None and age <= self.stale_after_seconds,
            "private_ws_message_age_seconds": age,
            "private_ws_reconnects": self._reconnects,
            "private_ws_duplicates": self._duplicates,
        }


class BybitSpotDataFeed:
    """Canonical completed bars/quotes plus microstructure from public Spot WS."""
    def __init__(
        self,
        http_client,
        *,
        symbols: tuple[str, ...] = DEFAULT_BYBIT_SPOT_UNIVERSE,
        timeframe: str = "5Min",
        websocket_manager: Optional[BybitPublicWebSocketManager] = None,
    ) -> None:
        self.http = http_client
        self.symbols = tuple(bybit_instrument(s).strategy_symbol for s in symbols)
        self.timeframe = timeframe
        self.ws = websocket_manager or BybitPublicWebSocketManager(
            http_client, symbols=self.symbols, timeframe=timeframe
        )
        self._interrupt_event = None
        self._rate_limit_events = 0

    def start(self) -> None:
        self.ws.start()

    def stop(self) -> None:
        self.ws.stop()

    def set_interrupt_event(self, event) -> None:
        self._interrupt_event = event

    @property
    def rate_limit_events(self) -> int:
        return self._rate_limit_events

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        instrument = bybit_instrument(symbol)
        if instrument.strategy_symbol not in self.symbols:
            raise DataFeedError(f"{symbol} is outside the configured Bybit market-data universe")
        if start.tzinfo is None or (end is not None and end.tzinfo is None):
            raise ValueError("Bybit bar timestamps must be timezone-aware")
        response = self.http.get_kline(
            category=SPOT_CATEGORY,
            symbol=instrument.venue_symbol,
            interval=bybit_interval(timeframe),
            start=int(start.timestamp() * 1000),
            end=int((end or datetime.now(timezone.utc)).timestamp() * 1000),
            limit=min(1000, max(1, int(limit or 1000))),
        )
        _require_ok(response, "historical Spot bars")
        rows = (((response or {}).get("result") or {}).get("list") or [])
        parsed = []
        for row in rows:
            if len(row) < 6:
                continue
            parsed.append(
                {
                    "timestamp": pd.Timestamp(int(row[0]), unit="ms", tz="UTC"),
                    "open": _num(row[1]), "high": _num(row[2]), "low": _num(row[3]),
                    "close": _num(row[4]), "volume": _num(row[5]),
                }
            )
        if not parsed:
            raise DataFeedError(f"Bybit returned no bars for {instrument.venue_symbol}")
        frame = pd.DataFrame(parsed).set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
            raise DataFeedError("Bybit returned invalid OHLCV values")
        return frame

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        *,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
    ) -> pd.DataFrame:
        if lookback_bars < 1:
            raise ValueError("lookback_bars must be positive")
        now = now or datetime.now(timezone.utc)
        duration = timeframe_to_timedelta(timeframe)
        start = now - duration * max(lookback_bars + 8, int(lookback_bars * 1.2))
        frame = self.get_bars(symbol, timeframe, start, now, limit=min(1000, lookback_bars + 20))
        frame = drop_incomplete_bars(frame, timeframe, now, tolerance_seconds)
        if len(frame) and is_crypto_data_stale(frame.index[-1].to_pydatetime(), timeframe, now, 3):
            raise DataFeedError(f"Bybit completed bars for {symbol} are stale")
        return frame.iloc[-lookback_bars:]

    def _rest_quote(self, symbol: str) -> CryptoQuote:
        instrument = bybit_instrument(symbol)
        response = self.http.get_orderbook(category=SPOT_CATEGORY, symbol=instrument.venue_symbol, limit=1)
        _require_ok(response, "top of book")
        result = (response or {}).get("result") or {}
        bids = result.get("b") or []
        asks = result.get("a") or []
        if not bids or not asks:
            raise DataFeedError("Bybit top of book is empty")
        bid, bid_size = _num(bids[0][0]), _num(bids[0][1])
        ask, ask_size = _num(asks[0][0]), _num(asks[0][1])
        if bid <= 0 or ask <= 0 or ask < bid:
            raise DataFeedError("Bybit top of book is invalid")
        ts = result.get("ts") or (response or {}).get("time") or int(time.time() * 1000)
        return CryptoQuote(
            symbol=instrument.strategy_symbol,
            timestamp=_utc_ms(ts), bid_price=bid, ask_price=ask,
            bid_size=bid_size, ask_size=ask_size,
        )

    def get_latest_quotes(self, symbols: tuple[str, ...] | list[str]) -> dict[str, CryptoQuote]:
        result: dict[str, CryptoQuote] = {}
        for symbol in symbols:
            strategy_symbol = bybit_instrument(symbol).strategy_symbol
            try:
                snapshot = self.ws.snapshot(strategy_symbol, max_age_seconds=5.0)
                result[strategy_symbol] = CryptoQuote(
                    symbol=strategy_symbol,
                    timestamp=snapshot.timestamp,
                    bid_price=snapshot.bid,
                    ask_price=snapshot.ask,
                    bid_size=None,
                    ask_size=None,
                )
            except Exception:
                result[strategy_symbol] = self._rest_quote(strategy_symbol)
        if not result:
            raise DataFeedError("Bybit returned no valid Spot quotes")
        return result

    def market_context(self, symbol: str) -> MarketContextSnapshot:
        return self.ws.snapshot(symbol)

    def recent_trades(self, symbol: str, limit: int = 60) -> list[dict[str, Any]]:
        instrument = bybit_instrument(symbol)
        response = self.http.get_public_trade_history(
            category=SPOT_CATEGORY, symbol=instrument.venue_symbol, limit=min(60, max(1, limit))
        )
        _require_ok(response, "recent public trades")
        return list((((response or {}).get("result") or {}).get("list") or []))

    def market_health(self) -> dict[str, Any]:
        return self.ws.health()
