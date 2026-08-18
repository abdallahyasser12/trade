"""Manifest-enforced deployment wrapper for the market-wide equity engine.

The base engine owns crash safety and broker execution. This wrapper adds the
research/deployment invariant: once any strategy advances beyond RESEARCH, the runtime
must be the exact researched git/config identity and may only trade symbols from the
locked universe that still pass the current hard executability screen.

It also closes production recovery/accounting edges:
* non-terminal orders created by prior market-wide sessions are reconciled after a
  process restart before new economic work is allowed;
* an entry that gaps through its persisted signal stop is adopted as bot-owned first,
  then flattened immediately, with fallback stop protection if the exit cannot submit;
* real sell fills are written to QuantStore attribution exactly once across crashes so
  lifecycle health reflects broker-backed PAPER outcomes rather than only decisions;
* incremental partial-fill price is reconstructed from cumulative broker averages, so
  P&L and slippage do not double-use the latest cumulative average.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data_feed import DataFeedError
from equity_marketwide_engine import (
    EQUITY_MARKETWIDE_SESSION_SYMBOL,
    MarketWideEquityTradeEngine,
)
from models import OrderRecord
from quant.contracts import AttributionRecord, Market, StrategyState
from quant.equity_attribution import record_equity_attribution_once
from quant.equity_manifest import EQUITY_STRATEGY_IDS, require_common_locked_universe

_PENDING_ENTRY_PREFIX = "equity_marketwide_pending_entry:"
_QTY_EPS = 1e-9


def _parse_timestamp(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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

    def _order_session_id(self, client_order_id: str) -> str:
        for row in self.store.get_orders(limit=5000):
            if str(row.get("client_order_id") or "") == client_order_id:
                return str(row.get("session_id") or self.session_id)
        return self.session_id

    def _reconcile_orders(self, now) -> None:
        """Reconcile this session plus non-terminal orders from prior desk sessions."""
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
                # acceptance. Deterministic client ids make the next submission
                # idempotent, so absence at the broker is not treated as a fill.
                continue
            self._sync_order(broker_order, local, now)

    @staticmethod
    def _incremental_fill_price(order: OrderRecord, previous: Optional[OrderRecord]) -> Optional[float]:
        cumulative_qty = _finite(order.filled_qty)
        cumulative_average = _finite(order.fill_price)
        if cumulative_qty is None or cumulative_average is None or cumulative_qty <= 0:
            return None
        previous_qty = _finite(previous.filled_qty) if previous is not None else 0.0
        previous_qty = max(0.0, previous_qty or 0.0)
        delta = cumulative_qty - previous_qty
        if delta <= _QTY_EPS:
            return None
        previous_average = _finite(previous.fill_price) if previous is not None else None
        if previous_qty > _QTY_EPS and previous_average is not None and previous_average > 0:
            incremental_notional = (
                cumulative_qty * cumulative_average - previous_qty * previous_average
            )
            incremental = incremental_notional / delta
            if math.isfinite(incremental) and incremental > 0:
                return incremental
        return cumulative_average if cumulative_average > 0 else None

    def _record_real_exit_attribution(
        self,
        order: OrderRecord,
        previous: Optional[OrderRecord],
        *,
        delta: float,
        fill_price: float,
        event_key: str,
        now: datetime,
    ) -> None:
        local = self.store.get_bot_position(order.symbol)
        if not self._is_ours(local):
            return
        strategy_id = str(local.get("strategy_id") or "")
        if strategy_id not in EQUITY_STRATEGY_IDS:
            return
        entry_price = _finite(local.get("entry_price"))
        if entry_price is None or entry_price <= 0:
            raise DataFeedError(
                f"Cannot attribute {order.symbol} exit: bot entry price is missing."
            )
        initial_stop = _finite(local.get("initial_stop_price"))
        gross = (fill_price - entry_price) * delta
        fees = 0.0
        net = gross - fees
        r_multiple = None
        if initial_stop is not None and 0 < initial_stop < entry_price:
            initial_risk = (entry_price - initial_stop) * delta
            if initial_risk > 0:
                r_multiple = net / initial_risk

        highest = _finite(local.get("highest_close"))
        lowest = _finite(local.get("lowest_close"))
        mfe = (
            max(0.0, (highest - entry_price) / entry_price * 100.0)
            if highest is not None
            else None
        )
        mae = (
            max(0.0, (entry_price - lowest) / entry_price * 100.0)
            if lowest is not None
            else None
        )
        reference = _finite(previous.reference_price) if previous is not None else None
        realized_slippage = None
        if reference is not None and reference > 0:
            # Positive always means worse execution. For sells, a lower fill is worse.
            realized_slippage = -(fill_price - reference) / reference * 10_000.0

        reason = (
            str(local.get("pending_exit_reason") or "strategy_exit")
            if order.purpose == "exit"
            else "protective_stop_fill"
        )
        entry_time = _parse_timestamp(local.get("entry_time"))
        signal_time = _parse_timestamp(local.get("signal_timestamp")) or entry_time
        record = AttributionRecord(
            strategy_id=strategy_id,
            strategy_version=str(local.get("strategy_version") or "1.0.0"),
            symbol=order.symbol,
            market=Market.US_EQUITY,
            timeframe=self.config.timeframe,
            session_id=self._order_session_id(order.client_order_id),
            signal_timestamp=signal_time,
            entry_timestamp=entry_time,
            exit_timestamp=order.updated_at or now,
            qty=delta,
            entry_price=entry_price,
            exit_price=fill_price,
            initial_stop_price=initial_stop,
            fees=fees,
            expected_cost_bps=None,
            realized_slippage_bps=realized_slippage,
            gross_pnl=gross,
            net_pnl=net,
            r_multiple=r_multiple,
            max_favorable_excursion_pct=mfe,
            max_adverse_excursion_pct=mae,
            entry_reason=str(local.get("entry_reason") or ""),
            exit_reason=reason,
            regime_at_entry=(
                str(local.get("regime_at_entry")) if local.get("regime_at_entry") else None
            ),
            opportunity_score=_finite(local.get("opportunity_score")),
            shadow=False,
            features_at_entry={},
        )
        record_equity_attribution_once(
            self.quant_store,
            record,
            event_key=event_key,
        )

    def _sync_order(self, order: OrderRecord, previous: Optional[OrderRecord], now: datetime) -> None:
        """Synchronize cumulative broker state with incremental, crash-safe accounting."""
        previous_filled = float(previous.filled_qty if previous else 0.0)
        reference = previous.reference_price if previous else None
        if reference is not None and order.reference_price is None:
            order.reference_price = reference
        delta = max(0.0, float(order.filled_qty) - previous_filled)
        event_key = f"{order.client_order_id}:{float(order.filled_qty):.10g}"
        incremental_price = self._incremental_fill_price(order, previous)

        # The quant attribution is committed idempotently BEFORE advancing the local
        # cumulative fill marker. If the process dies between databases, replay sees
        # the same event key, skips the quant duplicate, then completes StateStore.
        if (
            delta > _QTY_EPS
            and incremental_price is not None
            and str(order.side).lower() == "sell"
        ):
            self._record_real_exit_attribution(
                order,
                previous,
                delta=delta,
                fill_price=incremental_price,
                event_key=event_key,
                now=now,
            )

        self.store.record_order(order, self.session_id)
        if delta <= _QTY_EPS or incremental_price is None:
            return

        self.store.record_execution(
            self._order_session_id(order.client_order_id),
            order.client_order_id,
            order.symbol,
            order.purpose,
            str(order.side).lower(),
            reference,
            incremental_price,
            delta,
            timestamp=order.updated_at or now,
            event_key=event_key,
        )
        side = str(order.side).lower()
        if side == "buy":
            self.store.record_trade(
                self._order_session_id(order.client_order_id),
                order.symbol,
                "BUY",
                delta,
                incremental_price,
                None,
                "marketwide_entry_fill",
                None,
                order.updated_at or now,
                event_key=event_key,
            )
            self._adopt_entry_fill(order.symbol, order, now)
        elif side == "sell":
            local = self.store.get_bot_position(order.symbol)
            entry_price = _finite(local.get("entry_price")) if self._is_ours(local) else None
            realized = (
                (incremental_price - entry_price) * delta
                if entry_price is not None and entry_price > 0
                else None
            )
            reason = (
                str(local.get("pending_exit_reason") or "strategy_exit")
                if order.purpose == "exit" and local
                else "protective_stop_fill"
            )
            self.store.record_trade(
                self._order_session_id(order.client_order_id),
                order.symbol,
                "SELL",
                delta,
                incremental_price,
                realized,
                reason,
                None,
                order.updated_at or now,
                event_key=event_key,
            )

    def _adopt_entry_fill(self, symbol, order, now) -> None:
        """Adopt a fill before any attempt to manage or flatten its downside risk."""
        existing_before = self.store.get_bot_position(symbol)
        previous_entry_time = (
            existing_before.get("entry_time") if self._is_ours(existing_before) else None
        )
        previous_high = (
            _finite(existing_before.get("highest_close"))
            if self._is_ours(existing_before)
            else None
        )
        previous_low = (
            _finite(existing_before.get("lowest_close"))
            if self._is_ours(existing_before)
            else None
        )

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
        signal_stop = _finite(context.get("stop_price"))
        entry_price = _finite(position.avg_entry_price)
        if signal_stop is None or entry_price is None or signal_stop <= 0 or entry_price <= 0:
            return super()._adopt_entry_fill(symbol, order, now)

        if signal_stop < entry_price:
            super()._adopt_entry_fill(symbol, order, now)
            local = self.store.get_bot_position(symbol)
            if self._is_ours(local):
                if previous_entry_time:
                    local["entry_time"] = previous_entry_time
                local["signal_timestamp"] = context.get("signal_timestamp")
                local["highest_close"] = max(
                    entry_price,
                    previous_high if previous_high is not None else entry_price,
                )
                local["lowest_close"] = min(
                    entry_price,
                    previous_low if previous_low is not None else entry_price,
                )
                self.store.set_bot_position(symbol, local)
            return

        strategy_id = str(context.get("strategy_id") or "")
        if strategy_id not in EQUITY_STRATEGY_IDS:
            raise DataFeedError(
                f"Cannot recover {symbol} fill: persisted strategy ownership is invalid."
            )

        # A fill at/below the planned stop invalidates the original setup. Establish
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
            "entry_time": previous_entry_time or (order.updated_at or now).isoformat(),
            "signal_timestamp": context.get("signal_timestamp"),
            "entry_client_order_id": order.client_order_id,
            "entry_reason": context.get("entry_reason"),
            "stop_price": emergency_stop,
            "initial_stop_price": emergency_stop,
            "highest_close": max(
                entry_price,
                previous_high if previous_high is not None else entry_price,
            ),
            "lowest_close": min(
                entry_price,
                previous_low if previous_low is not None else entry_price,
            ),
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

    def _ratchet_stops(self, strategies, bars, timestamp, positions, now):
        super()._ratchet_stops(strategies, bars, timestamp, positions, now)
        for symbol, local in self._owned_positions(positions).items():
            frame = bars.get(symbol)
            if frame is None or timestamp not in frame.index:
                continue
            close = _finite(frame.loc[timestamp].get("close"))
            if close is None or close <= 0:
                continue
            local = self.store.get_bot_position(symbol)
            if not self._is_ours(local):
                continue
            entry_price = _finite(local.get("entry_price")) or close
            prior_low = _finite(local.get("lowest_close"))
            local["lowest_close"] = min(
                close,
                prior_low if prior_low is not None else entry_price,
            )
            self.store.set_bot_position(symbol, local)

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
