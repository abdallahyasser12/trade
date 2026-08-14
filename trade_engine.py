"""Background live-trading engine and its process-level controller.

The engine runs in a single daemon thread, uses interruptible waits so
Stop/Emergency Stop are responsive, persists everything important to the
SQLite StateStore, and treats broker state as the source of truth for
account facts.  All collaborators (broker, data feed, strategy, risk
manager, state store) are injected, so tests drive it with fakes.

Key safety behaviors implemented here:
* one engine per process (EngineController singleton; Start is idempotent);
* completed-bars-only, each bar processed at most once (persisted);
* deterministic client order ids => idempotent order creation;
* fills confirmed from broker state before any dependent action;
* protective stop created from the ACTUAL fill price and ATR-at-entry;
* exits cancel the bot's stop first so an exit can never double-sell;
* position ownership boundary: a pre-existing position the bot did not
  open blocks the session;
* PDT, near-close, staleness, clock-skew and daily-loss protections;
* unclean stops persist a kill switch the UI must acknowledge.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from broker import BrokerError, make_client_order_id
from config import (
    EngineConfig,
    INTRADAY_TIMEFRAMES,
    RiskConfig,
    StrategyConfig,
)
from data_feed import (
    DataFeedError,
    DataSubscriptionError,
    is_data_stale,
)
from models import (
    BotAction,
    EngineState,
    EngineStatus,
    OrderPurpose,
    OrderRecord,
    SignalDecision,
    StopReason,
    UNCLEAN_STOP_REASONS,
)
from risk_manager import RiskManager
from state_store import StateStore
from strategy import TrendStrategy

logger = logging.getLogger(__name__)

KILL_SWITCH_KEY = "kill_switch"
ENGINE_STATUS_KEY = "engine_status"
LAST_SIGNAL_KEY = "last_signal"
POSITION_SNAPSHOT_KEY_PREFIX = "position_snapshot:"

# Minimum equity to be exempt from FINRA's pattern-day-trading restriction.
PDT_EQUITY_THRESHOLD = 25_000.0
# Day trades allowed in 5 business days for accounts under the threshold.
PDT_MAX_DAY_TRADES = 3

_EQUITY_SNAPSHOT_INTERVAL = timedelta(seconds=60)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TradeEngine:
    """Live paper-trading engine (one symbol, one session)."""

    def __init__(
        self,
        broker,
        data_feed,
        strategy: TrendStrategy,
        risk_manager: RiskManager,
        state_store: StateStore,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        engine_config: EngineConfig,
        session_id: Optional[str] = None,
        now_fn: Callable[[], datetime] = _utc_now,
    ):
        self.broker = broker
        self.data_feed = data_feed
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.store = state_store
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.engine_config = engine_config
        self.session_id = session_id or uuid.uuid4().hex
        self.now_fn = now_fn

        self.symbol = strategy_config.symbol
        self.timeframe = strategy_config.timeframe
        self.poll_interval = engine_config.resolved_poll_interval(self.timeframe)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._stop_reason: Optional[StopReason] = None
        self._status = EngineStatus(state=EngineState.STOPPED)
        self._session_start_equity: Optional[float] = None
        self._started_at: Optional[datetime] = None
        self._intended_stop_at: Optional[datetime] = None
        self._last_equity_snapshot: Optional[datetime] = None
        self._pending_entry: Optional[dict] = None  # {client_order_id, atr, signal}
        self._pending_exit: Optional[dict] = None  # {client_order_id, reason}
        self._clock_skew_blocked = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def status(self) -> EngineStatus:
        with self._lock:
            snap = EngineStatus(**{**self._status.__dict__})
            snap.warnings = list(self._status.warnings)
            return snap

    def start(self) -> None:
        """Start the engine thread.  Idempotent — a running engine is a no-op."""
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._stop_reason = None
            self._status = EngineStatus(
                state=EngineState.STARTING,
                session_id=self.session_id,
                symbol=self.symbol,
                timeframe=self.timeframe,
            )
            self._thread = threading.Thread(
                target=self._run, name="TradeEngineThread", daemon=True
            )
            self._thread.start()

    def stop(self, reason: StopReason = StopReason.USER_STOP) -> None:
        """Request a graceful shutdown."""
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
            if self._status.state in (EngineState.RUNNING, EngineState.STARTING):
                self._status.state = EngineState.STOPPING
        self._stop_event.set()

    def emergency_stop(self) -> None:
        """Signal immediate termination.  Bot-owned protective stops are
        deliberately LEFT IN PLACE at the broker so an open position keeps
        its downside protection after the engine dies (documented)."""
        self.stop(StopReason.EMERGENCY_STOP)

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._startup()
        except Exception as exc:
            self._fail_fatal(f"Engine failed to start: {exc}")
            self._finalize()
            return

        try:
            while not self._stop_event.is_set():
                try:
                    self.run_iteration()
                except (BrokerError, DataSubscriptionError, DataFeedError) as exc:
                    self._log("WARNING", f"Transient failure, skipping cycle: {exc}")
                except Exception as exc:
                    logger.error("Fatal engine error:\n%s", traceback.format_exc())
                    self._fail_fatal(f"Unexpected engine error: {exc}")
                    break
                self._stop_event.wait(self.poll_interval)
        finally:
            self._finalize()

    def _startup(self) -> None:
        """Verify environment, reconcile prior state, open the session."""
        account = self.broker.verify_paper_account()

        asset = self.broker.get_asset(self.symbol)
        if not asset.is_supported:
            raise BrokerError(
                f"{self.symbol} is not an active, tradable US equity on Alpaca — "
                "refusing to start."
            )

        self._reconcile_startup_position()
        self._adopt_open_orders()

        now = self.now_fn()
        self._started_at = now
        if self.engine_config.max_runtime_seconds:
            self._intended_stop_at = now + timedelta(
                seconds=self.engine_config.max_runtime_seconds
            )
        self._session_start_equity = account.equity

        config_json = json.dumps(
            {
                "strategy": self.strategy_config.to_dict(),
                "risk": self.risk_config.to_dict(),
                "engine": self.engine_config.to_dict(),
            }
        )
        self.store.create_session(
            session_id=self.session_id,
            symbol=self.symbol,
            timeframe=self.timeframe,
            started_at=now,
            intended_stop_at=self._intended_stop_at,
            starting_equity=account.equity,
            config_json=config_json,
        )
        self.store.record_equity(self.session_id, now, account.equity, account.buying_power)
        self._last_equity_snapshot = now
        self._log(
            "INFO",
            f"Session {self.session_id[:8]} started: {self.symbol} {self.timeframe}, "
            f"equity ${account.equity:,.2f}",
        )
        with self._lock:
            self._status.state = EngineState.RUNNING
            self._status.started_at = self._started_at
            self._status.intended_stop_at = self._intended_stop_at
        self._persist_status()

    def _reconcile_startup_position(self) -> None:
        """Enforce the position ownership boundary before trading.

        A position in the configured symbol that the bot did not open
        blocks the session; stale local records pointing at positions the
        broker no longer holds are cleared with a warning.
        """
        broker_pos = self.broker.get_position(self.symbol)
        bot_pos = self.store.get_bot_position(self.symbol)

        if broker_pos is not None and bot_pos is None:
            raise BrokerError(
                f"A {broker_pos.qty:g}-share position in {self.symbol} already exists "
                "that this bot did not open. The bot only manages positions it created. "
                "Close or move that position manually, then start the bot again."
            )
        if broker_pos is None and bot_pos is not None:
            self._log(
                "WARNING",
                f"Local state shows a bot position in {self.symbol} but the broker has "
                "none — the position was closed outside a session (e.g. the protective "
                "stop filled). Clearing local record; broker state wins.",
            )
            self._cancel_orphan_stop(bot_pos)
            self.store.clear_bot_position(self.symbol)
        elif broker_pos is not None and bot_pos is not None:
            if abs(broker_pos.qty - float(bot_pos.get("qty", 0))) > 1e-9:
                raise BrokerError(
                    f"Broker holds {broker_pos.qty:g} {self.symbol} but the bot's records "
                    f"say {bot_pos.get('qty')} — the position was changed outside the bot. "
                    "Resolve it manually before restarting."
                )
            self._log(
                "INFO",
                f"Resuming management of existing bot position: {bot_pos.get('qty')} "
                f"{self.symbol} @ {bot_pos.get('entry_price')}",
            )

    def _cancel_orphan_stop(self, bot_pos: dict) -> None:
        """Cancel a bot stop order left behind after its position vanished."""
        stop_id = bot_pos.get("stop_client_order_id")
        if not stop_id:
            return
        try:
            order = self.broker.get_order_by_client_id(stop_id)
            if order is not None and order.is_open:
                self.broker.cancel_order(order.broker_order_id)
                self._log("INFO", f"Canceled orphan protective stop {stop_id}")
        except BrokerError as exc:
            self._log("WARNING", f"Could not cancel orphan stop {stop_id}: {exc}")

    def _adopt_open_orders(self) -> None:
        """Adopt bot-owned open orders from a previous process so their
        fills are tracked instead of forgotten."""
        try:
            open_orders = self.broker.list_open_orders(self.symbol)
        except BrokerError as exc:
            self._log("WARNING", f"Could not list open orders during startup: {exc}")
            return
        for order in open_orders:
            self.store.record_order(order, self.session_id)
            if order.purpose == OrderPurpose.ENTRY.value:
                self._pending_entry = {
                    "client_order_id": order.client_order_id,
                    "atr": None,
                    "signal": None,
                }
                self._log("WARNING", f"Adopted pending entry order {order.client_order_id}")
            elif order.purpose == OrderPurpose.EXIT.value:
                self._pending_exit = {
                    "client_order_id": order.client_order_id,
                    "reason": "adopted_exit",
                }
                self._log("WARNING", f"Adopted pending exit order {order.client_order_id}")
            elif order.purpose == OrderPurpose.STOP.value:
                # Link an already-resting protective stop to the bot position
                # so _ensure_protective_stop never arms a SECOND stop (two
                # resting sells could turn the account short).
                bot_pos = self.store.get_bot_position(self.symbol)
                if bot_pos is not None and not bot_pos.get("stop_client_order_id"):
                    bot_pos["stop_client_order_id"] = order.client_order_id
                    if order.stop_price is not None:
                        bot_pos["stop_price"] = order.stop_price
                    self.store.set_bot_position(self.symbol, bot_pos)
                    self._log(
                        "WARNING", f"Adopted resting protective stop {order.client_order_id}"
                    )

    def _finalize(self) -> None:
        """Persist final state on the way out; never raises."""
        reason = self._stop_reason or StopReason.USER_STOP
        now = self.now_fn()
        try:
            try:
                account = self.broker.get_account()
                self.store.record_equity(self.session_id, now, account.equity, account.buying_power)
            except Exception:
                logger.warning("Could not record final equity snapshot")
            if self.store.get_session(self.session_id) is not None:
                self.store.end_session(self.session_id, now, reason.value)
            if reason in UNCLEAN_STOP_REASONS:
                self.store.set_kv(
                    KILL_SWITCH_KEY,
                    json.dumps(
                        {
                            "reason": reason.value,
                            "session_id": self.session_id,
                            "timestamp": now.isoformat(),
                            "message": self._status.error or self._status.halt_reason or "",
                        }
                    ),
                )
            self._log("INFO", f"Engine stopped ({reason.value})")
        except Exception:
            logger.error("Error finalizing engine:\n%s", traceback.format_exc())
        finally:
            with self._lock:
                if self._status.state != EngineState.ERROR:
                    self._status.state = (
                        EngineState.HALTED
                        if reason == StopReason.DAILY_LOSS_HALT
                        else EngineState.STOPPED
                    )
                self._status.stop_reason = reason
            self._persist_status()

    def _fail_fatal(self, message: str) -> None:
        self._log("ERROR", message)
        with self._lock:
            self._status.state = EngineState.ERROR
            self._status.error = message
        if self._stop_reason is None:
            self._stop_reason = StopReason.FATAL_ERROR
        self._stop_event.set()

    # ------------------------------------------------------------------
    # One engine iteration (public for tests)
    # ------------------------------------------------------------------

    def run_iteration(self) -> None:
        """One full engine cycle.  Raises transient errors; the loop
        catches them and tries again next cycle."""
        now = self.now_fn()
        with self._lock:
            self._status.last_heartbeat = now

        if self._runtime_expired(now):
            self.stop(StopReason.RUNTIME_EXPIRED)
            return

        clock = self.broker.get_clock()
        with self._lock:
            self._status.market_open = clock.is_open
        self._check_clock_skew(clock, now)

        account = self.broker.get_account()
        self._maybe_snapshot_equity(now, account)

        if self._daily_loss_breached(account):
            return

        self._reconcile_pending_orders(now)
        self._ensure_protective_stop(now)
        self._snapshot_position(now)

        if not clock.is_open:
            self._persist_status()
            return

        if self._clock_skew_blocked:
            self._persist_status()
            return

        self._maybe_flatten_before_close(clock, now)

        bars = self._fetch_bars(now)
        if bars is None:
            self._persist_status()
            return

        last_bar_ts = bars.index[-1].to_pydatetime()

        if is_data_stale(last_bar_ts, self.timeframe, now, self.engine_config.staleness_bars):
            self._log(
                "WARNING",
                f"Market data is stale (last completed bar {last_bar_ts.isoformat()}); "
                "skipping trading decision.",
                context={"reason": "stale_data"},
            )
            self._persist_status()
            return

        if self.store.is_bar_processed(self.symbol, self.timeframe, last_bar_ts):
            self._persist_status()
            return
        if not self.store.mark_bar_processed(
            self.symbol, self.timeframe, last_bar_ts, self.session_id
        ):
            self._persist_status()
            return

        decision = self.strategy.generate_signal(bars)
        with self._lock:
            self._status.last_bar_time = last_bar_ts
            self._status.last_signal = decision
        self.store.set_kv(LAST_SIGNAL_KEY, json.dumps(decision.to_dict()))
        self._log(
            "INFO",
            f"Signal on {last_bar_ts.isoformat()}: {decision.action.value} ({decision.reason})",
            context={"signal": decision.to_dict()},
        )

        if decision.action == BotAction.EXIT:
            self._handle_exit_signal(decision, last_bar_ts)
        elif decision.action == BotAction.BUY:
            self._handle_buy_signal(decision, last_bar_ts, clock, account, now)

        self._persist_status()

    # ------------------------------------------------------------------
    # Iteration pieces
    # ------------------------------------------------------------------

    def _runtime_expired(self, now: datetime) -> bool:
        return self._intended_stop_at is not None and now >= self._intended_stop_at

    def _check_clock_skew(self, clock, now: datetime) -> None:
        skew = abs((clock.timestamp - now).total_seconds())
        blocked = skew > self.engine_config.clock_skew_threshold_seconds
        if blocked and not self._clock_skew_blocked:
            self._log(
                "WARNING",
                f"Local clock differs from Alpaca's by {skew:.0f}s — trading blocked "
                "until the clocks agree (bar-completion logic depends on local time).",
                context={"reason": "clock_skew_detected", "skew_seconds": skew},
            )
        self._clock_skew_blocked = blocked
        with self._lock:
            warn = "clock_skew_detected"
            if blocked and warn not in self._status.warnings:
                self._status.warnings.append(warn)
            elif not blocked and warn in self._status.warnings:
                self._status.warnings.remove(warn)

    def _maybe_snapshot_equity(self, now: datetime, account) -> None:
        if (
            self._last_equity_snapshot is None
            or now - self._last_equity_snapshot >= _EQUITY_SNAPSHOT_INTERVAL
        ):
            self.store.record_equity(self.session_id, now, account.equity, account.buying_power)
            self._last_equity_snapshot = now

    def _daily_loss_breached(self, account) -> bool:
        """Halt the session when the loss limit fires.  No auto-liquidation:
        the position (and its broker-side protective stop) are left as-is."""
        if self._session_start_equity is None:
            return False
        if self.risk_manager.check_daily_loss_limit(self._session_start_equity, account.equity):
            loss_pct = self.risk_manager.session_loss_pct(
                self._session_start_equity, account.equity
            )
            reason = (
                f"Session loss {loss_pct:.2f}% reached the configured maximum "
                f"{self.risk_manager.config.max_daily_loss_pct:.2f}% — halting new entries."
            )
            self.store.set_session_halted(self.session_id, reason)
            with self._lock:
                self._status.halt_reason = reason
            self._log("ERROR", reason, context={"reason": "daily_loss_halt"})
            self.stop(StopReason.DAILY_LOSS_HALT)
            return True
        return False

    def _reconcile_pending_orders(self, now: datetime) -> None:
        """Confirm fills from broker state and act on transitions."""
        if self._pending_entry is not None:
            self._reconcile_entry_order(now)
        if self._pending_exit is not None:
            self._reconcile_exit_order(now)
        self._reconcile_stop_order(now)

    def _refresh_order(self, client_order_id: str) -> Optional[OrderRecord]:
        order = self.broker.get_order_by_client_id(client_order_id)
        if order is not None:
            self.store.record_order(order, self.session_id)
            with self._lock:
                self._status.last_order_status = f"{order.purpose}:{order.status}"
        return order

    def _reconcile_entry_order(self, now: datetime) -> None:
        pending = self._pending_entry
        order = self._refresh_order(pending["client_order_id"])
        if order is None:
            self._log(
                "WARNING",
                f"Pending entry {pending['client_order_id']} not found at broker; clearing.",
            )
            self._pending_entry = None
            return
        if not order.is_terminal:
            return  # still working (accepted / partially_filled) — wait.
        self._pending_entry = None
        if order.filled_qty > 0:
            atr_at_entry = pending.get("atr")
            signal = pending.get("signal")
            self.store.set_bot_position(
                self.symbol,
                {
                    "qty": order.filled_qty,
                    "entry_price": order.fill_price,
                    "atr_at_entry": atr_at_entry,
                    "stop_price": None,
                    "stop_client_order_id": None,
                    "session_id": self.session_id,
                    "entry_time": (order.updated_at or now).isoformat(),
                },
            )
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="BUY",
                qty=order.filled_qty,
                fill_price=order.fill_price,
                realized_pnl=None,
                reason=(signal or {}).get("reason", "entry_fill")
                if isinstance(signal, dict)
                else "entry_fill",
                signal_snapshot=signal if isinstance(signal, dict) else None,
                timestamp=now,
            )
            status_word = "filled" if order.is_filled else f"partially filled ({order.status})"
            self._log(
                "INFO",
                f"Entry {status_word}: {order.filled_qty:g} {self.symbol} @ {order.fill_price}",
            )
            self._ensure_protective_stop(now)
        else:
            self._log(
                "WARNING",
                f"Entry order {order.client_order_id} ended {order.status} with no fill.",
                context={"order_status": order.status},
            )

    def _reconcile_exit_order(self, now: datetime) -> None:
        pending = self._pending_exit
        order = self._refresh_order(pending["client_order_id"])
        if order is None:
            self._log(
                "WARNING",
                f"Pending exit {pending['client_order_id']} not found at broker; clearing.",
            )
            self._pending_exit = None
            return
        if not order.is_terminal:
            return
        self._pending_exit = None
        bot_pos = self.store.get_bot_position(self.symbol)
        if order.filled_qty > 0 and bot_pos is not None:
            entry_price = float(bot_pos.get("entry_price") or 0.0)
            pnl = (float(order.fill_price or 0.0) - entry_price) * order.filled_qty
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="SELL",
                qty=order.filled_qty,
                fill_price=order.fill_price,
                realized_pnl=pnl,
                reason=pending.get("reason", "exit"),
                signal_snapshot=None,
                timestamp=now,
            )
            remaining = float(bot_pos.get("qty", 0)) - order.filled_qty
            if remaining > 1e-9:
                bot_pos["qty"] = remaining
                bot_pos["stop_client_order_id"] = None
                bot_pos["stop_price"] = None
                self.store.set_bot_position(self.symbol, bot_pos)
                self._log(
                    "WARNING",
                    f"Exit partially filled; {remaining:g} shares remain — "
                    "re-arming protective stop.",
                )
                self._ensure_protective_stop(now)
            else:
                self.store.clear_bot_position(self.symbol)
                self._log(
                    "INFO",
                    f"Exit filled: {order.filled_qty:g} {self.symbol} @ {order.fill_price} "
                    f"(realized P&L ${pnl:,.2f})",
                )
        elif order.filled_qty <= 0:
            self._log(
                "WARNING",
                f"Exit order {order.client_order_id} ended {order.status} with no fill — "
                "position still open; protective stop will be re-armed.",
            )
            self._ensure_protective_stop(now)

    def _reconcile_stop_order(self, now: datetime) -> None:
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            return
        stop_id = bot_pos.get("stop_client_order_id")
        if not stop_id:
            return
        order = self._refresh_order(stop_id)
        if order is None or not order.is_terminal:
            return
        if order.filled_qty > 0:
            entry_price = float(bot_pos.get("entry_price") or 0.0)
            pnl = (float(order.fill_price or 0.0) - entry_price) * order.filled_qty
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="SELL",
                qty=order.filled_qty,
                fill_price=order.fill_price,
                realized_pnl=pnl,
                reason="atr_stop_triggered",
                signal_snapshot=None,
                timestamp=now,
            )
            remaining = float(bot_pos.get("qty", 0)) - order.filled_qty
            if remaining > 1e-9:
                bot_pos["qty"] = remaining
                bot_pos["stop_client_order_id"] = None
                bot_pos["stop_price"] = None
                self.store.set_bot_position(self.symbol, bot_pos)
                self._ensure_protective_stop(now)
            else:
                self.store.clear_bot_position(self.symbol)
            self._log(
                "WARNING",
                f"Protective stop triggered: sold {order.filled_qty:g} {self.symbol} "
                f"@ {order.fill_price} (realized P&L ${pnl:,.2f})",
            )
        else:
            # Stop ended without a fill (canceled/expired/rejected).
            bot_pos["stop_client_order_id"] = None
            bot_pos["stop_price"] = None
            self.store.set_bot_position(self.symbol, bot_pos)
            if self._pending_exit is None:
                self._log(
                    "WARNING",
                    f"Protective stop ended {order.status} while position open — re-arming.",
                )
                self._ensure_protective_stop(now)

    def _ensure_protective_stop(self, now: datetime) -> None:
        """Create the ATR stop when a bot position has none working.

        Stop price = actual fill price − ATR_at_entry × multiplier, fixed
        from entry (no trailing).  Stop quantity always matches the bot
        position quantity.
        """
        if self._pending_exit is not None:
            return  # an exit is in flight; do not re-arm a stop under it.
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None or float(bot_pos.get("qty", 0)) <= 0:
            return
        if bot_pos.get("stop_client_order_id"):
            return

        entry_price = float(bot_pos.get("entry_price") or 0.0)
        atr_at_entry = bot_pos.get("atr_at_entry")
        stop_price = bot_pos.get("stop_price")
        if stop_price is None:
            if not atr_at_entry or entry_price <= 0:
                self._log(
                    "WARNING",
                    "Cannot compute protective stop (missing ATR-at-entry or entry price); "
                    "position is UNPROTECTED — manual attention recommended.",
                )
                return
            try:
                stop_price = self.risk_manager.calculate_stop_price(
                    entry_price, float(atr_at_entry), self.strategy_config.atr_multiplier
                )
            except ValueError as exc:
                self._log("WARNING", f"Cannot compute protective stop: {exc}")
                return

        qty = int(float(bot_pos["qty"]))
        client_id = make_client_order_id(
            self.session_id, OrderPurpose.STOP.value, now, symbol=self.symbol
        )
        existing = self.store.get_order_by_client_id(client_id)
        if existing is not None:
            return  # already created this cycle (idempotency guard)
        order = self.broker.submit_stop_order(self.symbol, qty, stop_price, client_id)
        self.store.record_order(order, self.session_id)
        bot_pos["stop_client_order_id"] = client_id
        bot_pos["stop_price"] = stop_price
        self.store.set_bot_position(self.symbol, bot_pos)
        self._log(
            "INFO",
            f"Protective stop placed: sell {qty} {self.symbol} @ stop {stop_price:.2f}",
        )

    def _snapshot_position(self, now: datetime) -> None:
        """Persist a broker-truth snapshot of the bot position so the
        dashboard can show current price / market value / unrealized P&L
        without ever calling Alpaca itself."""
        key = POSITION_SNAPSHOT_KEY_PREFIX + self.symbol
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            self.store.delete_kv(key)
            return
        try:
            broker_pos = self.broker.get_position(self.symbol)
        except BrokerError as exc:
            self._log("WARNING", f"Could not refresh position snapshot: {exc}")
            return
        if broker_pos is None:
            return  # reconciliation handles a vanished position elsewhere.
        self.store.set_kv(
            key,
            json.dumps(
                {
                    "symbol": self.symbol,
                    "qty": broker_pos.qty,
                    "avg_entry_price": broker_pos.avg_entry_price,
                    "current_price": broker_pos.current_price,
                    "market_value": broker_pos.market_value,
                    "unrealized_pl": broker_pos.unrealized_pl,
                    "unrealized_plpc": broker_pos.unrealized_plpc,
                    "stop_price": bot_pos.get("stop_price"),
                    "timestamp": now.isoformat(),
                }
            ),
        )

    def _fetch_bars(self, now: datetime):
        lookback = self.strategy.required_bars() + 5
        try:
            bars = self.data_feed.get_recent_bars(
                self.symbol,
                self.timeframe,
                lookback_bars=lookback,
                now=now,
                tolerance_seconds=self.engine_config.bar_completion_tolerance_seconds,
            )
        except DataSubscriptionError as exc:
            self._log("ERROR", str(exc), context={"reason": "data_subscription"})
            return None
        except DataFeedError as exc:
            self._log("WARNING", f"Data fetch failed (will retry next cycle): {exc}")
            return None
        if bars is None or len(bars) == 0:
            self._log("WARNING", "No completed bars returned; skipping cycle.")
            return None
        return bars

    def _maybe_flatten_before_close(self, clock, now: datetime) -> None:
        """Optional flat-before-close: exit the bot position before the close."""
        if not self.engine_config.flat_before_close:
            return
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None or self._pending_exit is not None:
            return
        seconds_to_close = (clock.next_close - now).total_seconds()
        if 0 <= seconds_to_close <= self.engine_config.flat_before_close_minutes * 60:
            self._log(
                "INFO",
                f"Flat-before-close: exiting {self.symbol} position "
                f"{seconds_to_close / 60:.0f} min before the close.",
            )
            self._initiate_exit("flat_before_close", now)

    def _handle_exit_signal(self, decision: SignalDecision, bar_ts: datetime) -> None:
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            self._log("INFO", "EXIT signal with no bot position — no action.")
            return
        if self._pending_exit is not None:
            self._log("INFO", "EXIT signal while an exit is already pending — no action.")
            return
        self._initiate_exit("bearish_crossover", bar_ts, signal=decision.to_dict())

    def _initiate_exit(
        self, reason: str, bar_ts: datetime, signal: Optional[dict] = None
    ) -> None:
        """Safely exit the bot position: cancel the bot's protective stop
        first (so the exit can never combine with the stop into a short),
        then submit a market sell for the confirmed broker quantity."""
        now = self.now_fn()
        broker_pos = self.broker.get_position(self.symbol)  # broker truth (§99)
        bot_pos = self.store.get_bot_position(self.symbol)
        if broker_pos is None or bot_pos is None:
            self._log(
                "WARNING",
                "Exit requested but no broker/bot position found — reconciling instead.",
            )
            if broker_pos is None and bot_pos is not None:
                self._cancel_orphan_stop(bot_pos)
                self.store.clear_bot_position(self.symbol)
            return

        stop_id = bot_pos.get("stop_client_order_id")
        if stop_id:
            stop_order = self._refresh_order(stop_id)
            if stop_order is not None and stop_order.is_open:
                try:
                    self.broker.cancel_order(stop_order.broker_order_id)
                except BrokerError as exc:
                    self._log("WARNING", f"Could not cancel protective stop yet: {exc}")
                    return
                # Confirm the cancel with a short interruptible wait.
                for _ in range(10):
                    refreshed = self._refresh_order(stop_id)
                    if refreshed is None or refreshed.is_terminal:
                        break
                    if self._stop_event.wait(0.5):
                        return
                refreshed = self.broker.get_order_by_client_id(stop_id)
                if refreshed is not None and not refreshed.is_terminal:
                    self._log(
                        "WARNING",
                        "Protective stop cancel not confirmed yet — deferring exit to "
                        "next cycle to avoid a double sell.",
                    )
                    return
                if refreshed is not None and refreshed.filled_qty > 0:
                    # The stop filled while we were canceling — the
                    # reconciler will book it; no exit order needed.
                    self._log("INFO", "Stop filled during cancel; skipping exit order.")
                    return
            bot_pos["stop_client_order_id"] = None
            self.store.set_bot_position(self.symbol, bot_pos)

        qty = int(min(float(bot_pos.get("qty", 0)), broker_pos.qty))
        if qty <= 0:
            self._log("WARNING", "Exit requested but computed quantity is 0 — no action.")
            return
        client_id = make_client_order_id(
            self.session_id, OrderPurpose.EXIT.value, bar_ts, symbol=self.symbol
        )
        if self.store.get_order_by_client_id(client_id) is not None:
            self._log("INFO", f"Exit order {client_id} already exists — not resubmitting.")
            return
        order = self.broker.submit_market_order(self.symbol, qty, "sell", client_id)
        self.store.record_order(order, self.session_id)
        self._pending_exit = {"client_order_id": client_id, "reason": reason}
        self._log(
            "INFO",
            f"Exit order submitted: sell {qty} {self.symbol} ({reason})",
            context={"signal": signal} if signal else None,
        )
        self._quick_poll_order(client_id)
        self._reconcile_pending_orders(self.now_fn())

    def _handle_buy_signal(
        self, decision: SignalDecision, bar_ts: datetime, clock, account, now: datetime
    ) -> None:
        block = self._entry_block_reason(decision, bar_ts, clock, account, now)
        if block is not None:
            decision.details["entry_block"] = block
            self.store.set_kv(LAST_SIGNAL_KEY, json.dumps(decision.to_dict()))
            self._log(
                "WARNING",
                f"BUY signal blocked: {block}",
                context={"reason": block, "bar": bar_ts.isoformat()},
            )
            return

        close_price = float(decision.details.get("close") or 0.0)
        try:
            stop_estimate = self.risk_manager.calculate_stop_price(
                close_price, float(decision.atr), self.strategy_config.atr_multiplier
            )
        except ValueError as exc:
            decision.details["entry_block"] = "stop_not_computable"
            self.store.set_kv(LAST_SIGNAL_KEY, json.dumps(decision.to_dict()))
            self._log("WARNING", f"BUY signal blocked: cannot compute a valid stop ({exc})")
            return
        sizing = self.risk_manager.calculate_position_size(
            equity=account.equity,
            buying_power=account.buying_power,
            entry_price=close_price,
            stop_price=stop_estimate,
        )
        if not sizing.approved:
            decision.details["entry_block"] = sizing.rejection_reason
            self.store.set_kv(LAST_SIGNAL_KEY, json.dumps(decision.to_dict()))
            self._log(
                "WARNING",
                f"BUY signal rejected by risk manager: {sizing.rejection_reason}",
                context={"sizing": sizing.__dict__},
            )
            return

        client_id = make_client_order_id(
            self.session_id, OrderPurpose.ENTRY.value, bar_ts, symbol=self.symbol
        )
        if (
            self.store.get_order_by_client_id(client_id) is not None
            or self.broker.get_order_by_client_id(client_id) is not None
        ):
            self._log("INFO", f"Entry order {client_id} already exists — not resubmitting.")
            return

        order = self.broker.submit_market_order(self.symbol, sizing.qty, "buy", client_id)
        self.store.record_order(order, self.session_id)
        self._pending_entry = {
            "client_order_id": client_id,
            "atr": float(decision.atr),
            "signal": decision.to_dict(),
        }
        self._log(
            "INFO",
            f"Entry order submitted: buy {sizing.qty} {self.symbol} "
            f"(risk ${sizing.risk_amount:,.2f}, est. stop {stop_estimate:.2f})",
            context={"signal": decision.to_dict()},
        )
        self._quick_poll_order(client_id)
        self._reconcile_pending_orders(self.now_fn())

    def _entry_block_reason(
        self, decision: SignalDecision, bar_ts: datetime, clock, account, now: datetime
    ) -> Optional[str]:
        """All pre-order entry guards; returns a block reason or None."""
        if self.store.get_bot_position(self.symbol) is not None:
            return "position_already_open"
        if self._pending_entry is not None:
            return "entry_order_pending"
        if self._pending_exit is not None:
            return "exit_order_pending"
        try:
            open_orders = self.broker.list_open_orders(self.symbol)
        except BrokerError as exc:
            self._log("WARNING", f"Could not verify open orders: {exc}")
            return "open_orders_unverified"
        if open_orders:
            return "bot_orders_already_open"
        broker_pos = self.broker.get_position(self.symbol)
        if broker_pos is not None:
            return "broker_position_exists"
        if not clock.is_open:
            return "market_closed"
        if self.engine_config.block_entries_near_close:
            seconds_to_close = (clock.next_close - now).total_seconds()
            if 0 <= seconds_to_close <= self.engine_config.close_buffer_minutes * 60:
                return "too_close_to_close"
        if self.timeframe in INTRADAY_TIMEFRAMES:
            if account.equity < PDT_EQUITY_THRESHOLD and (
                account.pattern_day_trader
                or account.daytrade_count >= PDT_MAX_DAY_TRADES
            ):
                return "pdt_restriction"
        if decision.atr is None or decision.atr <= 0:
            return "missing_atr"
        close_price = decision.details.get("close")
        if not close_price or close_price <= 0:
            return "missing_reference_price"
        try:
            asset = self.broker.get_asset(self.symbol)
        except BrokerError:
            return "asset_unverified"
        if not asset.is_supported:
            return "asset_not_tradable"
        return None

    def _quick_poll_order(self, client_order_id: str, attempts: int = 3) -> None:
        """Short interruptible poll so market-order fills land promptly."""
        for _ in range(attempts):
            if self._stop_event.wait(1.0):
                return
            order = self._refresh_order(client_order_id)
            if order is not None and order.is_terminal:
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, severity: str, message: str, context: Optional[dict] = None) -> None:
        getattr(logger, severity.lower(), logger.info)(message)
        try:
            self.store.log(severity, message, context=context, session_id=self.session_id)
        except Exception:
            logger.warning("Could not persist log entry")

    def _persist_status(self) -> None:
        try:
            self.store.set_kv(ENGINE_STATUS_KEY, json.dumps(self.status.to_dict()))
        except Exception:
            logger.warning("Could not persist engine status")


# ---------------------------------------------------------------------------
# Process-level singleton controller
# ---------------------------------------------------------------------------


class EngineController:
    """Guarantees at most one engine per Streamlit process.

    Streamlit reruns re-execute the script but the process (and this
    module) stay loaded, so a module-level singleton plus a lock is the
    correct guard against duplicate engines.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: Optional[TradeEngine] = None

    @property
    def engine(self) -> Optional[TradeEngine]:
        with self._lock:
            return self._engine

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._engine is not None and self._engine.is_running

    @property
    def status(self) -> Optional[EngineStatus]:
        with self._lock:
            return self._engine.status if self._engine is not None else None

    def start_engine(self, engine_factory: Callable[[], TradeEngine]) -> tuple[bool, str]:
        """Start a new engine unless one is already running (idempotent)."""
        with self._lock:
            if self._engine is not None and self._engine.is_running:
                return False, "The bot is already running — Start ignored."
            engine = engine_factory()
            self._engine = engine
            engine.start()
            return True, f"Bot started (session {engine.session_id[:8]})."

    def stop_engine(self, reason: StopReason = StopReason.USER_STOP) -> tuple[bool, str]:
        with self._lock:
            engine = self._engine
        if engine is None or not engine.is_running:
            return False, "The bot is not running."
        engine.stop(reason)
        return True, "Stop requested — the bot will shut down shortly."

    def emergency_stop(self) -> tuple[bool, str]:
        with self._lock:
            engine = self._engine
        if engine is None or not engine.is_running:
            return False, "The bot is not running."
        engine.emergency_stop()
        return True, "EMERGENCY STOP signaled."


_controller: Optional[EngineController] = None
_controller_lock = threading.Lock()


def get_controller() -> EngineController:
    """Return the process-wide controller, creating it once."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = EngineController()
        return _controller
