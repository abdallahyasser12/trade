"""Manifest-enforced deployment wrapper for the market-wide equity engine.

The base engine owns crash safety and broker execution.  This wrapper adds the
research/deployment invariant: once any strategy advances beyond RESEARCH, the runtime
must be the exact researched git/config identity and may only trade symbols from the
locked universe that still pass the current hard executability screen.

It also closes two recovery edges at the production boundary:
* non-terminal orders created by a prior market-wide session are reconciled after a
  process restart before new economic work is allowed;
* an entry that gaps through its persisted signal stop is adopted as bot-owned first,
  then flattened immediately.  If the exit cannot be submitted, the base engine can
  still place the conservative emergency stop on the now-owned position.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from data_feed import DataFeedError
from equity_marketwide_engine import (
    EQUITY_MARKETWIDE_SESSION_SYMBOL,
    MarketWideEquityTradeEngine,
)
from quant.contracts import Market, StrategyState
from quant.equity_manifest import EQUITY_STRATEGY_IDS, require_common_locked_universe

_PENDING_ENTRY_PREFIX = "equity_marketwide_pending_entry:"


class ManifestLockedMarketWideEquityTradeEngine(MarketWideEquityTradeEngine):
    """Production entry point for SHADOW/PAPER market-wide equities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locked_research_universe: tuple[str, ...] = ()

    def _forward_strategy_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for strategy_id in sorted(EQUITY_STRATEGY_IDS):
            state = self.lifecycle.state(strategy_id)
            if state not in {StrategyState.RESEARCH, StrategyState.QUARANTINE}:
                ids.append(strategy_id)
        return tuple(ids)

    def _startup(self) -> None:
        forward = self._forward_strategy_ids()
        if forward:
            self._locked_research_universe = require_common_locked_universe(
                self.quant_store,
                forward,
                runtime_config=self.config,
            )
        else:
            # RESEARCH-only sessions may inspect discovery but cannot receive broker
            # allocation because lifecycle risk multipliers remain zero.
            self._locked_research_universe = ()
        super()._startup()

    def _refresh_universe_if_due(self, now) -> None:
        super()._refresh_universe_if_due(now)
        forward = self._forward_strategy_ids()
        if not forward:
            return
        if not self._locked_research_universe:
            raise DataFeedError("Forward equity strategy has no locked research universe.")
        selected_now = tuple(self._universe.selected if self._universe else ())
        selected_set = set(selected_now)
        allowed = tuple(
            symbol for symbol in self._locked_research_universe if symbol in selected_set
        )
        if len(allowed) < 3:
            raise DataFeedError(
                "Fewer than three researched equity symbols remain currently executable; "
                "new entries are blocked while existing protective stops remain managed."
            )
        self._universe.selected = allowed
        with self._lock:
            self._status.selected_symbols = len(allowed)
            self._persist_status()

    def _reconcile_orders(self, now) -> None:
        """Reconcile this session plus non-terminal orders from prior desk sessions.

        ``StateStore`` is shared only by this market-wide runner in the production CLI,
        but the ownership check still verifies the originating session marker before
        touching a historical order.  Other bot engines' orders are ignored.
        """
        for row in self.store.get_orders(limit=5000):
            client_id = str(row.get("client_order_id") or "")
            if not client_id:
                continue
            local = self.store.get_order_by_client_id(client_id)
            if local is None or local.is_terminal:
                continue

            row_session_id = str(row.get("session_id") or "")
            owned_session = row_session_id == self.session_id
            if not owned_session and row_session_id:
                session = self.store.get_session(row_session_id)
                owned_session = bool(
                    session
                    and str(session.get("symbol") or "")
                    == EQUITY_MARKETWIDE_SESSION_SYMBOL
                )
            if not owned_session:
                continue

            broker_order = self.broker.get_order_by_client_id(client_id)
            if broker_order is None:
                # A durable intent may legitimately precede a crash before broker
                # acceptance.  Deterministic client ids make the next submission
                # idempotent, so absence at the broker is not treated as a fill.
                continue
            self._sync_order(broker_order, local, now)

    def _adopt_entry_fill(self, symbol, order, now) -> None:
        """Adopt a fill before any attempt to manage or flatten its downside risk."""
        raw = self.store.get_kv(_PENDING_ENTRY_PREFIX + symbol)
        if not raw:
            return super()._adopt_entry_fill(symbol, order, now)
        try:
            context = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return super()._adopt_entry_fill(symbol, order, now)

        position = self.broker.get_position(symbol)
        if position is None or position.qty <= 0:
            return
        try:
            signal_stop = float(context.get("stop_price"))
            entry_price = float(position.avg_entry_price)
        except (TypeError, ValueError):
            return super()._adopt_entry_fill(symbol, order, now)

        if not (
            math.isfinite(signal_stop)
            and math.isfinite(entry_price)
            and signal_stop > 0
            and entry_price > 0
        ):
            return super()._adopt_entry_fill(symbol, order, now)
        if signal_stop < entry_price:
            return super()._adopt_entry_fill(symbol, order, now)

        strategy_id = str(context.get("strategy_id") or "")
        if strategy_id not in EQUITY_STRATEGY_IDS:
            raise DataFeedError(
                f"Cannot recover {symbol} fill: persisted strategy ownership is invalid."
            )

        # A fill at/below the planned stop invalidates the original setup.  Establish
        # explicit ownership before attempting the market exit so a submission failure
        # can never strand the position outside the normal protective-stop machinery.
        emergency_stop = max(0.01, entry_price * 0.99)
        if emergency_stop >= entry_price:
            emergency_stop = max(0.01, entry_price - 0.01)
        record = {
            "market": Market.US_EQUITY.value,
            "strategy_id": strategy_id,
            "strategy_version": str(context.get("strategy_version") or "1.0.0"),
            "qty": float(position.qty),
            "entry_price": entry_price,
            "entry_time": (order.updated_at or now).isoformat(),
            "entry_client_order_id": order.client_order_id,
            "entry_reason": context.get("entry_reason"),
            "stop_price": emergency_stop,
            "initial_stop_price": emergency_stop,
            "highest_close": entry_price,
            "stop_client_order_id": None,
            "stop_attempt": 0,
            "pending_exit_client_order_id": None,
            "pending_exit_reason": None,
            "recovery_reason": "entry_filled_through_signal_stop",
        }
        self.store.set_bot_position(symbol, record)
        if order.is_terminal:
            self.store.delete_kv(_PENDING_ENTRY_PREFIX + symbol)

        timestamp = pd.Timestamp(order.updated_at or now)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        self._initiate_exit(
            symbol,
            position,
            strategy_id,
            "entry_filled_through_signal_stop",
            timestamp,
            now,
        )

    def _execute_ranked_entries(self, decisions, quotes, bar_timestamp, account, broker_positions, now):
        """Re-check live quote freshness/spread immediately before broker submission."""
        safe = []
        now_stamp = pd.Timestamp(now)
        if now_stamp.tzinfo is None:
            raise DataFeedError("Runtime clock must be timezone-aware before entry gating.")
        now_stamp = now_stamp.tz_convert("UTC")
        for decision in decisions:
            quote = quotes.get(decision.candidate.symbol)
            if quote is None:
                continue
            try:
                bid = float(quote.bid_price)
                ask = float(quote.ask_price)
                stamp = pd.Timestamp(quote.timestamp)
            except (AttributeError, TypeError, ValueError):
                continue
            if stamp.tzinfo is None:
                continue
            if not all(math.isfinite(value) for value in (bid, ask)) or bid <= 0 or ask <= bid:
                continue
            age = max(0.0, (now_stamp - stamp.tz_convert("UTC")).total_seconds())
            midpoint = (bid + ask) / 2.0
            spread_bps = (ask - bid) / midpoint * 10_000.0
            if age > self.config.max_quote_age_seconds:
                continue
            if spread_bps > self.config.max_spread_bps:
                continue
            if (
                self._locked_research_universe
                and decision.candidate.symbol not in self._locked_research_universe
            ):
                continue
            safe.append(decision)
        return super()._execute_ranked_entries(
            safe,
            quotes,
            bar_timestamp,
            account,
            broker_positions,
            now,
        )
