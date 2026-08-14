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

import copy
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
PENDING_ENTRY_KEY_PREFIX = "pending_entry:"

# Give up on a strategy exit after this many rejected/failed sell attempts
# and fall back to re-arming the protective stop instead.
_MAX_EXIT_ATTEMPTS = 3

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
        # Own deep-copied collaborators/configs so later Streamlit reruns or
        # caller-side mutation cannot alter a running session mid-flight.
        self.strategy_config = copy.deepcopy(strategy_config)
        self.risk_config = copy.deepcopy(risk_config)
        self.engine_config = copy.deepcopy(engine_config)
        self.strategy = copy.deepcopy(strategy)
        self.risk_manager = copy.deepcopy(risk_manager)
        self.strategy.config = self.strategy_config
        self.risk_manager.config = self.risk_config
        self.store = state_store
        self.session_id = session_id or uuid.uuid4().hex
        self.now_fn = now_fn

        self.symbol = self.strategy_config.symbol
        self.timeframe = self.strategy_config.timeframe
        self.poll_interval = self.engine_config.resolved_poll_interval(self.timeframe)

        self._stop_event = threading.Event()
        set_interrupt_event = getattr(self.data_feed, "set_interrupt_event", None)
        if callable(set_interrupt_event):
            set_interrupt_event(self._stop_event)
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
            # Deep-copy the signal: it holds mutable dicts the engine thread
            # may update later (entry_block), and the UI thread serializes
            # this snapshot concurrently.
            snap.last_signal = copy.deepcopy(self._status.last_signal)
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
                # Fill/cancel reconciliation is safety-critical; do not leave
                # a partially filled market order unattended for a full bar
                # polling interval.
                wait_seconds = (
                    min(1.0, self.poll_interval)
                    if self._pending_entry is not None or self._pending_exit is not None
                    else self.poll_interval
                )
                self._stop_event.wait(wait_seconds)
        finally:
            self._finalize()

    def _startup(self) -> None:
        """Verify environment, reconcile prior state, open the session."""
        if self.store.get_kv(KILL_SWITCH_KEY):
            raise BrokerError(
                "A previous session ended with an unclean stop (kill switch armed). "
                "Acknowledge it in the dashboard before starting the bot again."
            )
        if not self.store.acquire_engine_lock():
            raise BrokerError(
                "Another trading engine already holds the lock on this state "
                "database (a second app instance, or an orphaned engine from a "
                "source reload). Exactly one engine may trade per database."
            )
        try:
            self._startup_locked()
        except BaseException:
            self.store.release_engine_lock()
            raise

    def _startup_locked(self) -> None:
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
            if not self._adopt_position_from_own_fill(broker_pos):
                raise BrokerError(
                    f"A {broker_pos.qty:g}-share position in {self.symbol} already exists "
                    "that this bot did not open. The bot only manages positions it "
                    "created. Close or move that position manually, then start the "
                    "bot again."
                )
            return
        if broker_pos is None and bot_pos is not None:
            self._log(
                "WARNING",
                f"Local state shows a bot position in {self.symbol} but the broker has "
                "none — the position was closed outside a session (e.g. the protective "
                "stop filled). Clearing local record; broker state wins.",
            )
            if not self._cancel_orphan_stop(bot_pos):
                raise BrokerError(
                    "The broker position is gone but its protective stop could not be "
                    "confirmed canceled. Resolve that bot order manually before restart."
                )
            bot_pos = self._book_pending_sell_fills(bot_pos, self.now_fn())
            self.store.clear_bot_position(self.symbol)
        elif broker_pos is not None and bot_pos is not None:
            local_qty = float(bot_pos.get("qty", 0))
            if abs(broker_pos.qty - local_qty) > 1e-9:
                unbooked = self._unbooked_pending_sell_qty(bot_pos)
                if unbooked > 0 and abs(local_qty - unbooked - broker_pos.qty) <= 1e-9:
                    # The difference is exactly explained by fills on the
                    # bot's own pending sell orders (exit and/or stop) that
                    # happened while the process was down.  Resume; the
                    # first cycle's reconcilers book those fills.
                    self._log(
                        "WARNING",
                        f"Broker holds {broker_pos.qty:g} {self.symbol} vs local "
                        f"{local_qty:g}; the bot's own pending sell fills explain the "
                        "difference — resuming and reconciling.",
                    )
                else:
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

    def _adopt_position_from_own_fill(self, broker_pos) -> bool:
        """Recover a position created by one of OUR entry orders that filled
        while the process was down (crash window between submit and fill
        handling).  The orders table plus the broker's order state prove
        ownership.  Returns True when the position was adopted."""
        pending_ctx: dict = {}
        raw_ctx = self.store.get_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
        if raw_ctx:
            try:
                pending_ctx = json.loads(raw_ctx)
            except json.JSONDecodeError:
                pending_ctx = {}
        client_id = pending_ctx.get("client_order_id")
        if not client_id:
            return False
        # The pre-submit KV intent plus the broker's own record of the exact
        # bot-namespace client id prove ownership.  The local orders row is
        # NOT required: a crash between submit and record_order means the
        # row never existed, and recovery must still work.
        row = self.store.get_order_by_client_id(client_id)
        if row is not None and row.purpose != OrderPurpose.ENTRY.value:
            return False
        try:
            order = self.broker.get_order_by_client_id(client_id)
        except BrokerError:
            return False
        if (
            order is None
            or order.purpose != OrderPurpose.ENTRY.value
            or order.side != "buy"
            or order.filled_qty <= 0
            or abs(order.filled_qty - broker_pos.qty) > 1e-9
        ):
            return False
        self.store.record_order(order, self.session_id)
        atr_at_entry = pending_ctx.get("atr")
        self.store.set_bot_position(
            self.symbol,
            {
                "qty": order.filled_qty,
                "entry_price": order.fill_price or broker_pos.avg_entry_price,
                "atr_at_entry": atr_at_entry,
                "stop_price": None,
                "stop_client_order_id": None,
                "entry_client_order_id": order.client_order_id,
                "session_id": self.session_id,
                "entry_time": (order.updated_at or self.now_fn()).isoformat(),
            },
        )
        if order.is_terminal:
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="BUY",
                qty=order.filled_qty,
                fill_price=order.fill_price or broker_pos.avg_entry_price,
                realized_pnl=None,
                reason="entry_fill_recovered_at_restart",
                signal_snapshot=pending_ctx.get("signal"),
                timestamp=self.now_fn(),
                event_key=f"fill:{order.client_order_id}:buy",
            )
            self.store.delete_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
        else:
            self._pending_entry = pending_ctx
        self._log(
            "WARNING",
            f"Reconciliation: adopted {order.filled_qty:g} {self.symbol} filled by "
            f"the exact pending bot entry {order.client_order_id}. A protective stop "
            "will be armed.",
        )
        return True

    def _book_pending_sell_fills(self, bot_pos: dict, now: datetime) -> dict:
        """Book fills on the bot's own pending sell orders (strategy exit
        and protective stop) — idempotent via event keys.  Called before a
        vanished position's local record is cleared so realized trades are
        never lost from the ledger (e.g. crash between the fill and
        record_trade)."""
        for id_key, reason, accounted_key in (
            (
                "exit_client_order_id",
                bot_pos.get("exit_pending_reason") or "exit",
                "exit_accounted_filled_qty",
            ),
            ("stop_client_order_id", "atr_stop_triggered", "stop_accounted_filled_qty"),
        ):
            cid = bot_pos.get(id_key)
            if not cid:
                continue
            try:
                order = self.broker.get_order_by_client_id(cid)
            except BrokerError:
                continue
            if order is None or order.side != "sell" or order.filled_qty <= 0:
                continue
            bot_pos = self._apply_sell_fill(bot_pos, order, accounted_key)
            entry_price = float(bot_pos.get("entry_price") or 0.0)
            pnl = (float(order.fill_price or 0.0) - entry_price) * order.filled_qty
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="SELL",
                qty=order.filled_qty,
                fill_price=order.fill_price,
                realized_pnl=pnl,
                reason=reason,
                signal_snapshot=None,
                timestamp=now,
                event_key=f"fill:{order.client_order_id}:sell",
            )
        return bot_pos

    def _unbooked_pending_sell_qty(self, bot_pos: dict) -> float:
        """Filled-but-unaccounted quantity on the bot's own pending sell
        orders (strategy exit and protective stop).  Used at startup to
        recognize a position reduced by the bot's OWN fills while the
        process was down, as opposed to external interference."""
        total = 0.0
        for accounted_key, id_key in (
            ("exit_accounted_filled_qty", "exit_client_order_id"),
            ("stop_accounted_filled_qty", "stop_client_order_id"),
        ):
            cid = bot_pos.get(id_key)
            if not cid:
                continue
            try:
                order = self.broker.get_order_by_client_id(cid)
            except BrokerError:
                continue
            if order is None or order.side != "sell":
                continue
            accounted = float(bot_pos.get(f"{accounted_key}:{cid}") or 0.0)
            total += max(0.0, float(order.filled_qty) - accounted)
        return total

    def _cancel_orphan_stop(self, bot_pos: dict) -> bool:
        """Cancel a bot stop order left behind after its position vanished."""
        stop_id = bot_pos.get("stop_client_order_id")
        if not stop_id:
            return True
        try:
            order = self.broker.get_order_by_client_id(stop_id)
            if order is not None and order.is_open:
                if not self._cancel_and_confirm(order):
                    self._log("ERROR", f"Could not confirm cancellation of orphan stop {stop_id}")
                    return False
                self._log("INFO", f"Canceled and confirmed orphan protective stop {stop_id}")
            return True
        except BrokerError as exc:
            self._log("WARNING", f"Could not cancel orphan stop {stop_id}: {exc}")
            return False

    def _cancel_and_confirm(self, order: OrderRecord) -> bool:
        """Cancel one proven bot order and verify it no longer can fill."""
        if order.is_terminal:
            return True
        self.broker.cancel_order(order.broker_order_id)
        for attempt in range(6):
            refreshed = self._refresh_order(order.client_order_id)
            if refreshed is None or refreshed.is_terminal:
                return True
            if attempt < 5 and self._stop_event.wait(0.1):
                return False
        return False

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
                # Recover the ATR/signal context persisted when the entry
                # was submitted, so the eventual fill still gets its stop.
                ctx: dict = {}
                raw_ctx = self.store.get_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
                if raw_ctx:
                    try:
                        stored = json.loads(raw_ctx)
                        if stored.get("client_order_id") == order.client_order_id:
                            ctx = stored
                    except json.JSONDecodeError:
                        ctx = {}
                self._pending_entry = {
                    "client_order_id": order.client_order_id,
                    "atr": ctx.get("atr"),
                    "signal": ctx.get("signal"),
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
                remaining_qty = max(0.0, order.requested_qty - order.filled_qty)
                exact = (
                    bot_pos is not None
                    and order.side == "sell"
                    and abs(remaining_qty - float(bot_pos.get("qty", 0))) <= 1e-9
                )
                linked_id = bot_pos.get("stop_client_order_id") if bot_pos else None
                if exact and (not linked_id or linked_id == order.client_order_id):
                    bot_pos["stop_client_order_id"] = order.client_order_id
                    if order.stop_price is not None:
                        bot_pos["stop_price"] = order.stop_price
                    self.store.set_bot_position(self.symbol, bot_pos)
                    self._log(
                        "WARNING", f"Adopted resting protective stop {order.client_order_id}"
                    )
                elif not exact or linked_id != order.client_order_id:
                    try:
                        if not self._cancel_and_confirm(order):
                            raise BrokerError("cancel confirmation timed out")
                        self._log(
                            "WARNING",
                            f"Canceled mismatched or duplicate bot stop {order.client_order_id}",
                        )
                    except BrokerError as exc:
                        raise BrokerError(
                            f"Unsafe bot stop {order.client_order_id} could not be canceled: {exc}"
                        ) from exc

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
            self.store.release_engine_lock()
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

        self._reconcile_pending_orders(now)
        if not self._reconcile_position_truth(now):
            self._persist_status()
            return
        self._retry_exit_intent(now)
        self._ensure_protective_stop(now)
        self._snapshot_position(now)

        # Fills and stop protection are reconciled before the loss halt can
        # terminate the loop.  A safety circuit breaker must never strand a
        # fill that occurred since the previous cycle.
        if self._daily_loss_breached(account):
            return

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

    def _reconcile_position_truth(self, now: datetime) -> bool:
        """Reconcile broker quantity drift before any new economic action."""
        broker_pos = self.broker.get_position(self.symbol)
        bot_pos = self.store.get_bot_position(self.symbol)
        if broker_pos is None and bot_pos is None:
            return True
        if broker_pos is not None and bot_pos is None:
            if self._pending_entry is not None or self.store.get_kv(
                PENDING_ENTRY_KEY_PREFIX + self.symbol
            ):
                # An entry order is still in flight: its fill can become
                # visible as a position before the order object reports it.
                # Defer to order reconciliation next cycle instead of
                # declaring the position foreign; no economic action now.
                self._log(
                    "INFO",
                    "Broker position appeared while the bot entry order is in "
                    "flight — waiting for order reconciliation to adopt the fill.",
                )
                return False
            self._fail_fatal(
                f"Broker now holds {broker_pos.qty:g} {self.symbol} without bot ownership; "
                "session halted for manual reconciliation."
            )
            return False
        if broker_pos is None and bot_pos is not None:
            if not self._cancel_orphan_stop(bot_pos):
                self._fail_fatal(
                    "Broker position vanished and its bot sell stop could not be confirmed "
                    "canceled; manual reconciliation required."
                )
                return False
            # If the position vanished BECAUSE a bot sell order filled
            # (stop or exit), book those trades before dropping the record.
            bot_pos = self._book_pending_sell_fills(bot_pos, now)
            self.store.clear_bot_position(self.symbol)
            self.store.delete_kv(POSITION_SNAPSHOT_KEY_PREFIX + self.symbol)
            self._log("WARNING", "Broker position vanished; cleared reconciled local state.")
            return True

        local_qty = float(bot_pos.get("qty", 0))
        broker_qty = float(broker_pos.qty)
        if abs(local_qty - broker_qty) <= 1e-9:
            return True
        if broker_qty > local_qty:
            self._fail_fatal(
                f"Broker quantity {broker_qty:g} exceeds bot-owned quantity {local_qty:g}; "
                "session halted without adopting externally added shares."
            )
            return False

        # A manual/external reduction makes the old stop oversized.  Cancel
        # it first, then resize local ownership and re-arm an exact stop.
        stop_id = bot_pos.get("stop_client_order_id")
        if stop_id:
            stop = self._refresh_order(stop_id)
            if stop is not None and stop.is_open:
                try:
                    if not self._cancel_and_confirm(stop):
                        raise BrokerError("cancel confirmation timed out")
                except BrokerError as exc:
                    self._log(
                        "ERROR",
                        f"Position was externally reduced but oversized stop cancellation "
                        f"is unconfirmed: {exc}",
                    )
                    return False
            # The cancel may have raced a fill; book it and re-read broker
            # truth so the resized stop can never exceed the real position.
            bot_pos = self._book_unlinked_stop_fill(bot_pos, stop_id, now)
        fresh_pos = self.broker.get_position(self.symbol)
        broker_qty = float(fresh_pos.qty) if fresh_pos is not None else 0.0
        if broker_qty <= 0:
            self.store.clear_bot_position(self.symbol)
            self.store.delete_kv(POSITION_SNAPSHOT_KEY_PREFIX + self.symbol)
            self._log("WARNING", "Position fully gone after stop cancellation race.")
            return True
        bot_pos["qty"] = broker_qty
        bot_pos["stop_client_order_id"] = None
        bot_pos["stop_price"] = None
        self.store.set_bot_position(self.symbol, bot_pos)
        self._log(
            "WARNING",
            f"Broker truth reduced position from {local_qty:g} to {broker_qty:g}; "
            "protective stop will be resized.",
        )
        self._ensure_protective_stop(now)
        return True

    def _reconcile_pending_orders(self, now: datetime) -> None:
        """Confirm fills from broker state and act on transitions.

        In-flight order intent is persisted BEFORE each submit, so this
        also restores tracking that a crash or lost HTTP response wiped
        from memory — broker state is the truth being reconciled against.
        """
        if self._pending_entry is None:
            raw = self.store.get_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
            if raw:
                try:
                    self._pending_entry = json.loads(raw)
                except json.JSONDecodeError:
                    self.store.delete_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
        if self._pending_entry is not None:
            self._reconcile_entry_order(now)
        if self._pending_exit is None:
            bot_pos = self.store.get_bot_position(self.symbol)
            if bot_pos is not None and bot_pos.get("exit_client_order_id"):
                self._pending_exit = {
                    "client_order_id": bot_pos["exit_client_order_id"],
                    "reason": bot_pos.get("exit_pending_reason") or "exit",
                }
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

    @staticmethod
    def _remaining_order_qty(order: OrderRecord) -> float:
        return max(0.0, float(order.requested_qty) - float(order.filled_qty))

    def _sync_entry_fill(self, pending: dict, order: OrderRecord, now: datetime) -> dict:
        """Persist cumulative entry fill truth, including partial fills."""
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is not None:
            owner_id = bot_pos.get("entry_client_order_id")
            if owner_id and owner_id != order.client_order_id:
                raise BrokerError(
                    "A second entry fill conflicts with the bot-owned position; refusing adoption."
                )
            prior_qty = float(bot_pos.get("qty", 0))
            if abs(prior_qty - order.filled_qty) > 1e-9 and bot_pos.get(
                "stop_client_order_id"
            ):
                stop = self._refresh_order(bot_pos["stop_client_order_id"])
                if stop is not None and stop.is_open:
                    if not self._cancel_and_confirm(stop):
                        self._log(
                            "ERROR",
                            "Entry received another partial fill but the old protective stop "
                            "could not yet be canceled; reconciliation will retry immediately.",
                        )
                    else:
                        bot_pos["stop_client_order_id"] = None
                        bot_pos["stop_price"] = None
        else:
            bot_pos = {}
        signal = pending.get("signal")
        bot_pos.update(
            {
                "qty": float(order.filled_qty),
                "entry_price": order.fill_price,
                "atr_at_entry": pending.get("atr"),
                "entry_client_order_id": order.client_order_id,
                "session_id": self.session_id,
                "entry_time": (order.updated_at or now).isoformat(),
            }
        )
        bot_pos.setdefault("stop_price", None)
        bot_pos.setdefault("stop_client_order_id", None)
        self.store.set_bot_position(self.symbol, bot_pos)
        self._log(
            "INFO",
            f"Entry fill reconciled: {order.filled_qty:g} {self.symbol} @ {order.fill_price}",
            context={"signal": signal} if isinstance(signal, dict) else None,
        )
        return bot_pos

    def _reconcile_entry_order(self, now: datetime) -> None:
        pending = self._pending_entry
        order = self._refresh_order(pending["client_order_id"])
        if order is None:
            self._log(
                "WARNING",
                f"Pending entry {pending['client_order_id']} not found at broker — "
                "the submit never landed; entry dropped (conservative).",
            )
            self._pending_entry = None
            self.store.delete_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)
            return
        if order.filled_qty > 0:
            signal = pending.get("signal")
            self._sync_entry_fill(pending, order, now)
            self._ensure_protective_stop(now)
            if not order.is_terminal:
                # Freeze the economic quantity after the first partial fill.
                # The current fill is protected before canceling the
                # remainder.  Any open status qualifies (done_for_day,
                # pending_review, ... can also carry fills); pending_cancel
                # is already on its way out.
                if order.status != "pending_cancel":
                    try:
                        if self._cancel_and_confirm(order):
                            order = self._refresh_order(order.client_order_id) or order
                    except BrokerError as exc:
                        self._log(
                            "WARNING",
                            f"Could not cancel unfilled entry remainder yet: {exc}",
                        )
                if not order.is_terminal:
                    return
                # Capture any additional fill received while cancellation raced.
                if order.filled_qty > 0:
                    self._sync_entry_fill(pending, order, now)
                    self._ensure_protective_stop(now)
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
                event_key=f"fill:{order.client_order_id}:buy",
            )
            status_word = "filled" if order.is_filled else f"partially filled ({order.status})"
            self._log(
                "INFO",
                f"Entry {status_word}: {order.filled_qty:g} {self.symbol} @ {order.fill_price}",
            )
            self._ensure_protective_stop(now)
        else:
            if not order.is_terminal:
                return
            self._log(
                "WARNING",
                f"Entry order {order.client_order_id} ended {order.status} with no fill.",
                context={"order_status": order.status},
            )
        # Clear the crash-recovery intent only after durable position/trade
        # state has been written.
        self._pending_entry = None
        self.store.delete_kv(PENDING_ENTRY_KEY_PREFIX + self.symbol)

    def _apply_sell_fill(
        self, bot_pos: dict, order: OrderRecord, accounted_key: str
    ) -> dict:
        """Apply only the newly observed portion of a cumulative sell fill.

        Accounting is keyed per order id so a later stop/exit order
        generation can never inherit a previous order's accounted
        quantity (which would silently suppress its fills)."""
        key = f"{accounted_key}:{order.client_order_id}"
        accounted = float(bot_pos.get(key) or 0.0)
        cumulative = max(accounted, float(order.filled_qty))
        delta = cumulative - accounted
        if delta > 0:
            bot_pos["qty"] = max(0.0, float(bot_pos.get("qty", 0)) - delta)
            bot_pos[key] = cumulative
            self.store.set_bot_position(self.symbol, bot_pos)
        return bot_pos

    def _book_unlinked_stop_fill(self, bot_pos: dict, stop_id: str, now: datetime) -> dict:
        """After canceling (or losing) the linked stop, book any fill the
        cancellation raced against — idempotent via the trade event key —
        and drop the dead link WITHOUT re-arming from stale local state."""
        refreshed = self._refresh_order(stop_id)
        if refreshed is not None and refreshed.filled_qty > 0:
            bot_pos = self._apply_sell_fill(bot_pos, refreshed, "stop_accounted_filled_qty")
            entry_price = float(bot_pos.get("entry_price") or 0.0)
            pnl = (float(refreshed.fill_price or 0.0) - entry_price) * refreshed.filled_qty
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="SELL",
                qty=refreshed.filled_qty,
                fill_price=refreshed.fill_price,
                realized_pnl=pnl,
                reason="atr_stop_triggered",
                signal_snapshot=None,
                timestamp=now,
                event_key=f"fill:{refreshed.client_order_id}:sell",
            )
            bot_pos.pop(f"stop_accounted_filled_qty:{refreshed.client_order_id}", None)
        bot_pos["stop_client_order_id"] = None
        bot_pos["stop_price"] = None
        self.store.set_bot_position(self.symbol, bot_pos)
        return bot_pos

    def _reconcile_exit_order(self, now: datetime) -> None:
        pending = self._pending_exit
        order = self._refresh_order(pending["client_order_id"])
        bot_pos = self.store.get_bot_position(self.symbol)
        if order is None:
            # The submit never landed at the broker.  Keep the durable exit
            # INTENT so the exit is retried, but drop the dead order id.
            self._log(
                "WARNING",
                f"Pending exit {pending['client_order_id']} not found at broker — "
                "the submit never landed; the exit will be retried.",
            )
            self._pending_exit = None
            if bot_pos is not None and bot_pos.get("exit_client_order_id"):
                bot_pos["exit_client_order_id"] = None
                self.store.set_bot_position(self.symbol, bot_pos)
            return
        if order.filled_qty > 0 and bot_pos is None:
            # The local position record is already gone (e.g. a racing stop
            # fill was reconciled first).  Book the exit fill so the trade
            # ledger stays truthful, then release the pending intent — it
            # must never deadlock and block future entries.
            if not order.is_terminal:
                return
            self._pending_exit = None
            self.store.record_trade(
                self.session_id,
                self.symbol,
                action="SELL",
                qty=order.filled_qty,
                fill_price=order.fill_price,
                realized_pnl=None,
                reason=pending.get("reason", "exit"),
                signal_snapshot=None,
                timestamp=now,
                event_key=f"fill:{order.client_order_id}:sell",
            )
            self._log(
                "WARNING",
                f"Exit fill of {order.filled_qty:g} {self.symbol} booked without a local "
                "position record (position was reconciled away first).",
            )
            return
        if order.filled_qty > 0 and bot_pos is not None:
            bot_pos = self._apply_sell_fill(bot_pos, order, "exit_accounted_filled_qty")
            if not order.is_terminal:
                if order.status != "pending_cancel":
                    try:
                        if self._cancel_and_confirm(order):
                            order = self._refresh_order(order.client_order_id) or order
                            bot_pos = self._apply_sell_fill(
                                bot_pos, order, "exit_accounted_filled_qty"
                            )
                    except BrokerError as exc:
                        self._log("WARNING", f"Could not cancel partial exit remainder: {exc}")
                if not order.is_terminal:
                    return
            self._pending_exit = None
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
                event_key=f"fill:{order.client_order_id}:sell",
            )
            remaining = float(bot_pos.get("qty", 0))
            if remaining > 1e-9:
                bot_pos.pop(f"exit_accounted_filled_qty:{order.client_order_id}", None)
                bot_pos["stop_client_order_id"] = None
                bot_pos["stop_price"] = None
                self._clear_exit_intent(bot_pos)
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
            if not order.is_terminal:
                return
            self._pending_exit = None
            attempts = int(bot_pos.get("exit_attempts") or 0) if bot_pos else 0
            if bot_pos is not None:
                bot_pos["exit_client_order_id"] = None
                if attempts >= _MAX_EXIT_ATTEMPTS:
                    # Give up on the strategy exit; keep the position
                    # protected by its stop instead of looping forever.
                    self._clear_exit_intent(bot_pos)
                    self.store.set_bot_position(self.symbol, bot_pos)
                    self._log(
                        "ERROR",
                        f"Exit order failed {attempts} times ({order.status}) — giving up "
                        "on the strategy exit and re-arming the protective stop.",
                    )
                    self._ensure_protective_stop(now)
                    return
                self.store.set_bot_position(self.symbol, bot_pos)
            self._log(
                "WARNING",
                f"Exit order {order.client_order_id} ended {order.status} with no fill — "
                "the exit will be retried.",
            )

    @staticmethod
    def _clear_exit_intent(bot_pos: dict) -> None:
        bot_pos["exit_pending_reason"] = None
        bot_pos["exit_bar_epoch"] = None
        bot_pos["exit_client_order_id"] = None
        bot_pos["exit_attempts"] = 0

    def _reconcile_stop_order(self, now: datetime) -> None:
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            return
        stop_id = bot_pos.get("stop_client_order_id")
        if not stop_id:
            return
        order = self._refresh_order(stop_id)
        if order is None:
            # Intent was persisted but the broker never received the order
            # (submit failed before acceptance).  Clear the dead id so the
            # stop is re-armed with a fresh one.
            self._log(
                "WARNING",
                f"Protective stop {stop_id} not found at broker — re-arming.",
            )
            bot_pos["stop_client_order_id"] = None
            self.store.set_bot_position(self.symbol, bot_pos)
            self._ensure_protective_stop(now)
            return
        if order.filled_qty > 0:
            bot_pos = self._apply_sell_fill(bot_pos, order, "stop_accounted_filled_qty")
            if not order.is_terminal:
                # The remainder of this same sell stop exactly protects the
                # remaining shares; never stack another stop on a partial.
                return
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
                event_key=f"fill:{order.client_order_id}:sell",
            )
            remaining = float(bot_pos.get("qty", 0))
            if remaining > 1e-9:
                bot_pos.pop(f"stop_accounted_filled_qty:{order.client_order_id}", None)
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
            if not order.is_terminal:
                return
            # Stop ended without a fill (canceled/expired/rejected).
            bot_pos["stop_client_order_id"] = None
            bot_pos["stop_price"] = None
            self.store.set_bot_position(self.symbol, bot_pos)
            if self._pending_exit is None and not bot_pos.get("exit_pending_reason"):
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

        Duplicate-stop safety: the intended client id is persisted to the
        bot position BEFORE the submit, and any open bot stop already
        resting at the broker is adopted instead of stacking a second one
        — two resting sells could turn the account short.
        """
        if self._pending_exit is not None:
            return  # an exit is in flight; do not re-arm a stop under it.
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None or float(bot_pos.get("qty", 0)) <= 0:
            return
        if bot_pos.get("exit_pending_reason"):
            return  # a strategy exit is being pursued; don't arm under it.
        # The bot trades whole shares; an externally created fractional
        # remainder is protected at its floor (a whole-share stop cannot
        # cover dust, and chasing it would churn cancel/re-arm forever).
        target_qty = float(int(float(bot_pos.get("qty", 0))))
        if target_qty <= 0:
            self._log(
                "WARNING",
                f"Position of {float(bot_pos.get('qty', 0)):g} {self.symbol} is below one "
                "whole share — no protective stop can be armed for the fraction.",
            )
            return
        if target_qty != float(bot_pos.get("qty", 0)):
            self._log(
                "WARNING",
                f"Fractional position {float(bot_pos.get('qty', 0)):g} {self.symbol}: "
                f"protecting {target_qty:g} whole shares; the fraction is unprotected.",
            )
        linked_stop_id = bot_pos.get("stop_client_order_id")
        if linked_stop_id:
            linked = self._refresh_order(linked_stop_id)
            remaining = self._remaining_order_qty(linked) if linked is not None else 0.0
            exact = (
                linked is not None
                and linked.is_open
                and linked.side == "sell"
                and abs(remaining - target_qty) <= 1e-9
            )
            if exact:
                return
            if linked is not None and linked.is_open:
                try:
                    if not self._cancel_and_confirm(linked):
                        self._log("ERROR", "Mismatched protective stop cancel is not confirmed.")
                        return
                except BrokerError as exc:
                    self._log("ERROR", f"Could not cancel mismatched protective stop: {exc}")
                    return
            # Book any fill the cancel raced against before re-arming, and
            # recompute the whole-share target from the reduced quantity.
            bot_pos = self._book_unlinked_stop_fill(bot_pos, linked_stop_id, now)
            if float(bot_pos.get("qty", 0)) <= 0:
                return
            target_qty = float(int(float(bot_pos.get("qty", 0))))
            if target_qty <= 0:
                return

        # Adopt any bot stop already resting at the broker (e.g. a submit
        # whose HTTP response was lost) rather than submitting a second one.
        try:
            open_orders = self.broker.list_open_orders(self.symbol)
        except BrokerError as exc:
            self._log("WARNING", f"Cannot verify open orders before arming stop: {exc}")
            return
        stop_orders = [o for o in open_orders if o.purpose == OrderPurpose.STOP.value]
        exact_stop = None
        for existing_order in stop_orders:
            exact = (
                existing_order.side == "sell"
                and abs(self._remaining_order_qty(existing_order) - target_qty) <= 1e-9
            )
            if exact and exact_stop is None:
                exact_stop = existing_order
                continue
            try:
                if not self._cancel_and_confirm(existing_order):
                    self._log("ERROR", "Unsafe duplicate/mismatched stop cancel unconfirmed.")
                    return
            except BrokerError as exc:
                self._log("ERROR", f"Cannot cancel unsafe protective stop: {exc}")
                return
        if exact_stop is not None:
            bot_pos["stop_client_order_id"] = exact_stop.client_order_id
            if exact_stop.stop_price is not None:
                bot_pos["stop_price"] = exact_stop.stop_price
            self.store.set_bot_position(self.symbol, bot_pos)
            self.store.record_order(exact_stop, self.session_id)
            self._log(
                "WARNING",
                f"Adopted exact-quantity protective stop {exact_stop.client_order_id}.",
            )
            return

        entry_price = float(bot_pos.get("entry_price") or 0.0)
        atr_at_entry = bot_pos.get("atr_at_entry")
        stop_price = bot_pos.get("stop_price")
        if stop_price is None:
            if (not atr_at_entry) and self._status.last_signal is not None:
                # Recovery fallback (e.g. entry adopted after a restart with
                # no persisted context): protect with the latest computed ATR
                # rather than leaving the position unprotected.
                with self._lock:
                    fallback_atr = (
                        self._status.last_signal.atr if self._status.last_signal else None
                    )
                if fallback_atr:
                    atr_at_entry = fallback_atr
                    self._log(
                        "WARNING",
                        "ATR-at-entry unavailable; using the latest computed ATR "
                        f"({fallback_atr:.4f}) for the protective stop.",
                    )
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

        qty = int(target_qty)
        stop_attempt = int(bot_pos.get("stop_attempts") or 0) + 1
        client_id = make_client_order_id(
            self.session_id,
            OrderPurpose.STOP.value,
            now,
            symbol=self.symbol,
            attempt=stop_attempt,
        )
        if self.store.get_order_by_client_id(client_id) is not None:
            return  # already created this cycle (idempotency guard)
        # Persist the intent BEFORE submitting: if the submit's response is
        # lost, the next cycle reconciles this id against broker truth
        # instead of arming a second stop.
        bot_pos["stop_client_order_id"] = client_id
        bot_pos["stop_price"] = stop_price
        bot_pos["stop_attempts"] = stop_attempt
        self.store.set_bot_position(self.symbol, bot_pos)
        try:
            order = self.broker.submit_stop_order(self.symbol, qty, stop_price, client_id)
        except BrokerError as exc:
            self._log(
                "WARNING",
                f"Protective stop submit failed ({exc}); will reconcile and retry.",
            )
            return
        self.store.record_order(order, self.session_id)
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
        if (
            bot_pos is None
            or self._pending_exit is not None
            or bot_pos.get("exit_pending_reason")
        ):
            return  # nothing to do, or an exit is already being pursued.
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
        then submit a market sell for the confirmed broker quantity.

        The exit INTENT is persisted before any broker call, so a deferral
        (unconfirmed cancel), an exception, or a crash never silently drops
        the exit — every cycle retries it until it completes or exhausts
        its attempts.  The order id is likewise persisted before the
        submit, so a lost HTTP response is healed from broker truth
        instead of producing an untracked live sell.
        """
        now = self.now_fn()
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            self._log("WARNING", "Exit requested but no bot position found — no action.")
            return

        # Persist the durable exit intent first.
        if bot_pos.get("exit_pending_reason") != reason:
            bot_pos["exit_pending_reason"] = reason
            bot_pos["exit_bar_epoch"] = int(bar_ts.timestamp())
            bot_pos.setdefault("exit_attempts", 0)
            self.store.set_bot_position(self.symbol, bot_pos)

        # Heal a previously-submitted exit whose response was lost.
        exit_id = bot_pos.get("exit_client_order_id")
        if exit_id:
            order = self._refresh_order(exit_id)
            if order is not None:
                self._pending_exit = {"client_order_id": exit_id, "reason": reason}
                return  # tracked again; the reconciler takes it from here.
            bot_pos["exit_client_order_id"] = None
            self.store.set_bot_position(self.symbol, bot_pos)

        broker_pos = self.broker.get_position(self.symbol)  # broker truth (§99)
        if broker_pos is None:
            self._log(
                "WARNING",
                "Exit requested but the broker shows no position — reconciling instead.",
            )
            if self._cancel_orphan_stop(bot_pos):
                self.store.clear_bot_position(self.symbol)
            else:
                self._log(
                    "ERROR",
                    "Refusing to clear local ownership while an orphan sell stop may remain open.",
                )
            return

        stop_id = bot_pos.get("stop_client_order_id")
        if stop_id:
            stop_order = self._refresh_order(stop_id)
            if stop_order is not None and stop_order.is_open:
                try:
                    self.broker.cancel_order(stop_order.broker_order_id)
                except BrokerError as exc:
                    self._log(
                        "WARNING",
                        f"Could not cancel protective stop yet ({exc}) — the exit "
                        "will be retried next cycle.",
                    )
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
                        "Protective stop cancel not confirmed yet — the exit will be "
                        "retried next cycle to avoid a double sell.",
                    )
                    return
                if refreshed is not None and refreshed.filled_qty > 0:
                    # The stop filled while cancellation raced. Reconcile
                    # that cumulative fill now. If shares remain, preserve
                    # the strategy-exit intent and sell only broker truth.
                    self._log("INFO", "Stop filled during cancel; reconciling before exit.")
                    self._reconcile_stop_order(now)
                    bot_pos = self.store.get_bot_position(self.symbol)
                    broker_pos = self.broker.get_position(self.symbol)
                    if bot_pos is None or broker_pos is None:
                        return
            bot_pos["stop_client_order_id"] = None
            self.store.set_bot_position(self.symbol, bot_pos)

        qty = int(min(float(bot_pos.get("qty", 0)), broker_pos.qty))
        if qty <= 0:
            self._log("WARNING", "Exit requested but computed quantity is 0 — no action.")
            return
        attempt = int(bot_pos.get("exit_attempts") or 0) + 1
        client_id = make_client_order_id(
            self.session_id,
            OrderPurpose.EXIT.value,
            bar_ts,
            symbol=self.symbol,
            attempt=attempt,
        )
        known = self.store.get_order_by_client_id(client_id)
        at_broker = self.broker.get_order_by_client_id(client_id) if known is None else None
        if known is not None or at_broker is not None:
            bot_pos["exit_client_order_id"] = client_id
            self.store.set_bot_position(self.symbol, bot_pos)
            self._pending_exit = {"client_order_id": client_id, "reason": reason}
            self._log("INFO", f"Exit order {client_id} already exists — tracking it.")
            return
        # Persist the order-id intent BEFORE submitting (lost-response safety).
        bot_pos["exit_client_order_id"] = client_id
        bot_pos["exit_attempts"] = attempt
        self.store.set_bot_position(self.symbol, bot_pos)
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

    def _retry_exit_intent(self, now: datetime) -> None:
        """Re-drive a persisted exit intent that has no live order behind it
        (deferred cancel, failed submit, process restart)."""
        bot_pos = self.store.get_bot_position(self.symbol)
        if bot_pos is None:
            return
        reason = bot_pos.get("exit_pending_reason")
        if not reason:
            return
        if self._pending_exit is not None or bot_pos.get("exit_client_order_id"):
            return
        if int(bot_pos.get("exit_attempts") or 0) >= _MAX_EXIT_ATTEMPTS:
            self._clear_exit_intent(bot_pos)
            self.store.set_bot_position(self.symbol, bot_pos)
            self._log(
                "ERROR",
                f"Exit ({reason}) exhausted its attempts — abandoning the strategy exit; "
                "the protective stop will be re-armed.",
            )
            self._ensure_protective_stop(now)
            return
        epoch = bot_pos.get("exit_bar_epoch")
        bar_ts = (
            datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch else now
        )
        self._log("INFO", f"Retrying pending exit ({reason}).")
        self._initiate_exit(reason, bar_ts)

    def _handle_buy_signal(
        self, decision: SignalDecision, bar_ts: datetime, clock, account, now: datetime
    ) -> None:
        block = self._entry_block_reason(decision, bar_ts, clock, account, now)
        if block is not None:
            self._mark_entry_blocked(decision, block)
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
            self._mark_entry_blocked(decision, "stop_not_computable")
            self._log("WARNING", f"BUY signal blocked: cannot compute a valid stop ({exc})")
            return
        sizing = self.risk_manager.calculate_position_size(
            equity=account.equity,
            buying_power=account.buying_power,
            entry_price=close_price,
            stop_price=stop_estimate,
        )
        if not sizing.approved:
            self._mark_entry_blocked(decision, sizing.rejection_reason)
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

        # Persist the entry context BEFORE submitting: if the response is
        # lost or the process dies, the next cycle (or restart) reconciles
        # this id against broker truth, and the eventual fill still knows
        # its ATR-at-entry for the protective stop.
        self.store.set_kv(
            PENDING_ENTRY_KEY_PREFIX + self.symbol,
            json.dumps(
                {
                    "client_order_id": client_id,
                    "atr": float(decision.atr),
                    "signal": decision.to_dict(),
                }
            ),
        )
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

    def _mark_entry_blocked(self, decision: SignalDecision, block: Optional[str]) -> None:
        """Attach the block reason to the published signal (under the lock —
        the UI thread snapshots this object concurrently) and persist it."""
        with self._lock:
            decision.details["entry_block"] = block
        self.store.set_kv(LAST_SIGNAL_KEY, json.dumps(decision.to_dict()))

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
