"""24/7 Alpaca crypto engine with broker-truth reconciliation and stop watchdog."""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from broker import AlpacaBroker, BrokerError, is_bot_order_id, make_client_order_id
from config import AccountMode
from crypto_config import (
    CRYPTO_SESSION_SYMBOL,
    CryptoEngineConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)
from crypto_data_feed import AlpacaCryptoDataFeed, is_crypto_data_stale, quote_is_safe
from crypto_models import CryptoEngineStatus, CryptoPositionState, CryptoSignalDecision
from cost_model import cost_to_risk, round_trip_cost_bps
from crypto_risk import CryptoRiskManager, portfolio_market_value
from crypto_strategy import CryptoRegimeStrategy
from data_feed import DataFeedError
from models import BotAction, EngineState, OrderRecord, StopReason
from state_store import StateStore


logger = logging.getLogger(__name__)

CRYPTO_ENGINE_STATUS_KEY = "crypto_engine_status"
CRYPTO_KILL_SWITCH_KEY = "crypto_kill_switch"
CRYPTO_LAST_SIGNALS_KEY = "crypto_last_signals"
_QTY_EPSILON = 1e-12
# A guard exit that keeps coming back rejected is an unhandled broker state,
# not something to retry indefinitely against a live position.
_MAX_GUARD_ATTEMPTS = 5


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CryptoTradeEngine:
    """One immutable crypto portfolio session.

    Alpaca position and order responses are authoritative. Local state stores
    ownership, stop arithmetic, and idempotent intent; it never overrides a
    broker-reported quantity.
    """

    def __init__(
        self,
        broker: AlpacaBroker,
        data_feed: AlpacaCryptoDataFeed,
        store: StateStore,
        strategy_config: CryptoStrategyConfig,
        risk_config: CryptoRiskConfig,
        engine_config: CryptoEngineConfig,
        *,
        now_fn: Callable[[], datetime] = _utc_now,
    ):
        errors = (
            strategy_config.validate()
            + risk_config.validate()
            + engine_config.validate()
        )
        if errors:
            raise ValueError("Invalid crypto engine configuration: " + "; ".join(errors))
        self.broker = broker
        self.data_feed = data_feed
        self.store = store
        # Configs are frozen dataclasses; this is the session's immutable snapshot.
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.engine_config = engine_config
        self.strategy = CryptoRegimeStrategy(strategy_config)
        self.risk = CryptoRiskManager(risk_config)
        self._now_fn = now_fn
        self.session_id = uuid.uuid4().hex
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._assets = {}
        self._starting_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._started_at: Optional[datetime] = None
        self._fatal = False
        self._status = CryptoEngineStatus(
            state=EngineState.STOPPED,
            session_id=self.session_id,
            account_mode=AccountMode(engine_config.account_mode).value,
            symbols=list(strategy_config.symbols),
            timeframe=strategy_config.timeframe,
        )
        self.data_feed.set_interrupt_event(self._stop_event)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> CryptoEngineStatus:
        with self._lock:
            return CryptoEngineStatus(**self._status.__dict__)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._status.state = EngineState.STARTING
            self._persist_status()
            self._thread = threading.Thread(
                target=self._run,
                name=f"crypto-engine-{self.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, reason: StopReason = StopReason.USER_STOP) -> None:
        with self._lock:
            self._status.state = EngineState.STOPPING
            self._status.stop_reason = reason
            self._persist_status()
        self._stop_event.set()

    def emergency_stop(self) -> None:
        self.store.set_kv(
            CRYPTO_KILL_SWITCH_KEY,
            json.dumps({"armed": True, "reason": "EMERGENCY_STOP", "at": self._now_fn().isoformat()}),
        )
        self.stop(StopReason.EMERGENCY_STOP)

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self) -> None:
        reason = StopReason.USER_STOP
        try:
            self._startup()
            with self._lock:
                self._status.state = EngineState.RUNNING
                self._persist_status()
            idle_interval = self.engine_config.resolved_poll_interval(
                self.strategy_config.timeframe
            )
            exposed_interval = self.engine_config.exposed_poll_interval(
                self.strategy_config.timeframe
            )
            while not self._stop_event.is_set():
                if self._runtime_expired(self._now_fn()):
                    reason = StopReason.RUNTIME_EXPIRED
                    break
                try:
                    self.run_iteration()
                except (BrokerError, DataFeedError) as exc:
                    self._log("ERROR", str(exc))
                    with self._lock:
                        self._status.warnings = [str(exc)]
                        self._persist_status()
                # Watch far more often while something is at risk.  The bar
                # cadence governs decisions; protection is a quote-level job.
                interval = exposed_interval if self._has_exposure() else idle_interval
                if self._stop_event.wait(interval):
                    break
            with self._lock:
                reason = self._status.stop_reason or reason
        except Exception as exc:  # noqa: BLE001 - fatal engine boundary
            self._fatal = True
            reason = StopReason.FATAL_ERROR
            self.store.set_kv(
                CRYPTO_KILL_SWITCH_KEY,
                json.dumps({"armed": True, "reason": "FATAL_ERROR", "at": self._now_fn().isoformat()}),
            )
            with self._lock:
                self._status.state = EngineState.ERROR
                self._status.error = str(exc)
                self._status.stop_reason = reason
                self._persist_status()
            self._log("CRITICAL", f"Crypto engine stopped: {exc}")
        finally:
            self._finalize(reason)

    def _startup(self) -> None:
        mode = AccountMode(self.engine_config.account_mode)
        broker_mode = getattr(self.broker, "account_mode", None)
        if broker_mode != mode:
            raise BrokerError(
                f"Frozen engine mode is {mode.value}, but broker mode is "
                f"{getattr(broker_mode, 'value', broker_mode)!r}."
            )
        kill = self.store.get_kv(CRYPTO_KILL_SWITCH_KEY)
        if kill:
            try:
                if json.loads(kill).get("armed"):
                    raise BrokerError(
                        "Crypto kill switch is armed. Acknowledge it in the dashboard before Start."
                    )
            except json.JSONDecodeError as exc:
                raise BrokerError("Crypto kill-switch state is corrupt; refusing to start.") from exc
        account = self.broker.verify_account()
        if not account.can_trade:
            raise BrokerError("Alpaca account is not ACTIVE and tradable.")
        self._assets = self.broker.verify_crypto_assets(self.strategy_config.symbols)
        now = self._now_fn()
        self._started_at = now
        self._starting_equity = account.equity
        self._peak_equity = account.equity
        intended_stop = (
            now + timedelta(seconds=self.engine_config.max_runtime_seconds)
            if self.engine_config.max_runtime_seconds is not None
            else None
        )
        active = self.store.get_active_session()
        if active is not None and active["session_id"] != self.session_id:
            self.store.end_session(
                active["session_id"], now, "RECOVERED_AFTER_PROCESS_RESTART"
            )
        snapshot = {
            "account_mode": mode.value,
            "strategy": self.strategy_config.to_dict(),
            "risk": self.risk_config.to_dict(),
            "engine": self.engine_config.to_dict(),
        }
        self.store.create_session(
            self.session_id,
            CRYPTO_SESSION_SYMBOL,
            self.strategy_config.timeframe,
            now,
            intended_stop,
            account.equity,
            json.dumps(snapshot, sort_keys=True),
        )
        with self._lock:
            self._status.started_at = now
            self._status.intended_stop_at = intended_stop
        self._refresh_recoverable_orders()
        self._reconcile_position_truth(now, startup=True)
        self._ensure_all_protection(now)
        self._log(
            "INFO",
            f"Crypto {mode.value} session started for {', '.join(self.strategy_config.symbols)}.",
        )

    def _finalize(self, reason: StopReason) -> None:
        now = self._now_fn()
        try:
            session = self.store.get_session(self.session_id)
            if session is not None:
                self.store.end_session(self.session_id, now, reason.value)
        except Exception:
            logger.exception("Could not finalize crypto session")
        with self._lock:
            if not self._fatal:
                self._status.state = (
                    EngineState.HALTED
                    if reason is StopReason.DAILY_LOSS_HALT
                    else EngineState.STOPPED
                )
            self._status.stop_reason = reason
            self._status.last_heartbeat = now
            self._persist_status()

    def _runtime_expired(self, now: datetime) -> bool:
        return bool(
            self._started_at
            and self.engine_config.max_runtime_seconds is not None
            and (now - self._started_at).total_seconds()
            >= self.engine_config.max_runtime_seconds
        )

    def run_iteration(self) -> None:
        now = self._now_fn()
        account = self.broker.get_account()
        if not account.can_trade:
            raise BrokerError("Alpaca account became blocked or inactive; entries halted.")
        self._peak_equity = max(self._peak_equity or account.equity, account.equity)
        was_halted = self.risk.session_halted
        if self.risk.check_session_loss(self._starting_equity or account.equity, account.equity):
            self.store.set_session_halted(self.session_id, self.risk.halt_reason or "loss limit")
            if not was_halted:
                self.store.set_kv(
                    CRYPTO_KILL_SWITCH_KEY,
                    json.dumps(
                        {
                            "armed": True,
                            "reason": "DAILY_LOSS_HALT",
                            "at": now.isoformat(),
                            "message": self.risk.halt_reason,
                        }
                    ),
                )
            with self._lock:
                self._status.state = EngineState.HALTED
                self._status.halt_reason = self.risk.halt_reason
                self._status.stop_reason = StopReason.DAILY_LOSS_HALT

        self._refresh_recoverable_orders()
        self._cancel_stale_entry_orders(now)
        broker_positions = self._reconcile_position_truth(now)
        # Protection is independent of quote availability. A newly observed
        # partial fill must receive a stop even if the quote endpoint is down.
        self._ensure_all_protection(now, broker_positions)
        quotes = self.data_feed.get_latest_quotes(list(self.strategy_config.symbols))
        self._run_stop_watchdogs(now, quotes, broker_positions)
        broker_positions = self._reconcile_position_truth(now)
        self._ensure_all_protection(now, broker_positions)

        bars = {
            symbol: self.data_feed.get_recent_bars(
                symbol,
                self.strategy_config.timeframe,
                self.strategy.required_bars() + 5,
                now=now,
                tolerance_seconds=self.engine_config.bar_completion_tolerance_seconds,
            )
            for symbol in self.strategy_config.symbols
        }
        if any(len(frame) < self.strategy.required_bars() for frame in bars.values()):
            self._warn("Insufficient completed crypto bars; no new decisions.")
            return
        stale = [
            symbol
            for symbol, frame in bars.items()
            if is_crypto_data_stale(
                frame.index[-1].to_pydatetime(),
                self.strategy_config.timeframe,
                now,
                self.engine_config.staleness_bars,
            )
        ]
        if stale:
            self._warn(f"Stale crypto bars for {', '.join(stale)}; entries blocked.")
            return
        common = None
        for frame in bars.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            self._warn("Crypto symbols have no exactly aligned completed bar; entries blocked.")
            return
        bar_timestamp = common[-1]
        if all(
            self.store.is_bar_processed(symbol, self.strategy_config.timeframe, bar_timestamp)
            for symbol in self.strategy_config.symbols
        ):
            self._heartbeat(now, bar_timestamp)
            return

        prepared = self.strategy.prepare_universe(bars)
        held = {symbol for symbol, position in broker_positions.items() if position.qty > 0}
        decisions = self.strategy.evaluate_prepared(prepared, bar_timestamp, held)
        with self._lock:
            self._status.last_signals = {decision.symbol: decision for decision in decisions}
        self.store.set_kv(
            CRYPTO_LAST_SIGNALS_KEY,
            json.dumps({decision.symbol: decision.to_dict() for decision in decisions}),
        )

        # Exits are initiated before any entries. A strategy exit first cancels
        # and confirms the bot-owned stop; it never leaves a stale sell behind.
        for decision in decisions:
            if decision.action is BotAction.EXIT and decision.symbol in broker_positions:
                self._initiate_exit(decision.symbol, decision, bar_timestamp)

        broker_positions = self._reconcile_position_truth(now)
        for symbol, position in broker_positions.items():
            if symbol not in prepared or bar_timestamp not in prepared[symbol].index:
                continue
            local = self._load_position(symbol)
            if local is None or local.pending_exit_client_order_id:
                continue
            current_atr = float(prepared[symbol].loc[bar_timestamp, "atr"])
            current_close = float(prepared[symbol].loc[bar_timestamp, "close"])
            new_stop, highest = self.strategy.ratchet_stop(
                local.stop_price, local.highest_close, current_close, current_atr
            )
            new_stop = min(new_stop, current_close * (1 - 1e-12))
            if new_stop > local.stop_price + float(self._assets[symbol].price_increment or 0):
                local.stop_price = new_stop
                local.highest_close = highest
                self._save_position(local)
                self._ensure_protection(symbol, position, now)
            elif highest > local.highest_close:
                local.highest_close = highest
                self._save_position(local)

        if not self.risk.session_halted:
            self._handle_entries(
                decisions,
                prepared,
                bar_timestamp,
                quotes,
                account,
                broker_positions,
                now,
            )

        # Marker comes after economic actions. A crash before this point repeats
        # the bar with identical client IDs and reconciles rather than duplicating.
        for symbol in self.strategy_config.symbols:
            self.store.mark_bar_processed(
                symbol, self.strategy_config.timeframe, bar_timestamp, self.session_id
            )
        self._heartbeat(now, bar_timestamp)

    def _handle_entries(
        self,
        decisions: list[CryptoSignalDecision],
        prepared,
        bar_timestamp: pd.Timestamp,
        quotes,
        account,
        broker_positions,
        now: datetime,
    ) -> None:
        open_orders = self.broker.list_open_orders()
        pending_buy_symbols = {
            order.symbol for order in open_orders if order.side == "buy" and order.is_open
        }
        positions = dict(broker_positions)
        all_account_positions = self.broker.list_positions()
        # Broker truth includes every asset class on a shared account. Only
        # crypto counts toward the crypto exposure cap; an unknown class still
        # counts, so the gate can only fail conservative.
        exposure = portfolio_market_value(
            [
                position
                for position in all_account_positions
                if position.asset_class in (None, "crypto")
            ]
        )
        # Portfolio heat: the sum of distance-to-stop across held positions,
        # marked from the latest completed close. This is what the heat cap
        # in sizing measures new candidates against.
        open_risk = 0.0
        for held_symbol, held_position in positions.items():
            local = self._load_position(held_symbol)
            frame = prepared.get(held_symbol)
            if local is None or frame is None:
                continue
            marks = frame["close"].loc[:bar_timestamp]
            if marks.empty:
                continue
            mark = float(marks.iloc[-1])
            open_risk += max(0.0, mark - local.stop_price) * held_position.qty
        newly_submitted = 0
        remaining_cash = account.cash
        remaining_buying_power = account.buying_power
        held_returns = {
            symbol: prepared[symbol]["close"].pct_change().loc[:bar_timestamp]
            for symbol in positions
            if symbol in prepared
        }
        for decision in decisions:
            symbol = decision.symbol
            if decision.action is not BotAction.BUY:
                continue
            if symbol in positions or symbol in pending_buy_symbols:
                continue
            if len(positions) + newly_submitted >= self.risk_config.max_positions:
                break
            quote = quotes.get(symbol)
            if quote is None:
                self._log("WARNING", f"No quote for {symbol}; entry skipped.")
                continue
            quote_ok, quote_reason = quote_is_safe(
                quote,
                now,
                max_age_seconds=self.engine_config.quote_max_age_seconds,
                max_spread_bps=self.engine_config.max_spread_bps,
            )
            if not quote_ok:
                self._log("WARNING", f"{symbol} entry blocked: {quote_reason}")
                continue
            candidate_returns = prepared[symbol]["close"].pct_change().loc[:bar_timestamp]
            allowed, correlation_reason, _ = self.risk.correlation_gate(
                candidate_returns, held_returns
            )
            if not allowed:
                self._log("INFO", f"{symbol} entry blocked: {correlation_reason}")
                continue
            entry_price = quote.ask_price
            atr_value = float(decision.atr or 0.0)
            stop_price = entry_price - atr_value * self.strategy_config.atr_stop_multiplier
            # Friction measured against this trade's own risk unit, using the
            # live spread. A trade that pays most of 1R to get in and out needs
            # an implausible win rate; it is rejected before it is sized.
            trade_cost_to_risk = cost_to_risk(
                entry_price=entry_price,
                stop_price=stop_price,
                round_trip_bps=round_trip_cost_bps(
                    slippage_bps=self.engine_config.expected_slippage_bps,
                    spread_bps=quote.spread_bps,
                    taker_fee_bps=self.engine_config.taker_fee_bps,
                ),
            )
            if trade_cost_to_risk > self.risk_config.max_cost_to_risk:
                self._log(
                    "INFO",
                    f"{symbol} entry blocked: round-trip cost is "
                    f"{trade_cost_to_risk:.2f}R, above the "
                    f"{self.risk_config.max_cost_to_risk:.2f}R limit.",
                )
                continue
            asset = self._assets[symbol]
            size = self.risk.size_position(
                equity=account.equity,
                cash=remaining_cash,
                buying_power=remaining_buying_power,
                entry_price=entry_price,
                stop_price=stop_price,
                asset=asset,
                current_crypto_exposure=exposure,
                current_position_count=len(positions) + newly_submitted,
                current_open_risk=open_risk,
                peak_equity=self._peak_equity or account.equity,
                atr_percentile=(
                    decision.atr_percentile
                    if decision.atr_percentile is not None
                    else 100.0
                ),
                account_mode=self.engine_config.account_mode,
            )
            if not size.approved:
                self._log("INFO", f"{symbol} entry sizing rejected: {size.rejection_reason}")
                continue
            client_id = make_client_order_id(
                self.session_id, "entry", bar_timestamp.to_pydatetime(), symbol
            )
            intent = OrderRecord(
                broker_order_id="",
                client_order_id=client_id,
                symbol=symbol,
                purpose="entry",
                side="buy",
                requested_qty=size.qty,
                filled_qty=0.0,
                status="intent",
                submitted_at=now,
                updated_at=now,
            )
            self.store.record_order(intent, self.session_id)
            self.store.set_kv(
                self._entry_intent_key(symbol),
                json.dumps(
                    {
                        "client_order_id": client_id,
                        "atr": decision.atr,
                        "signal_timestamp": decision.timestamp.isoformat()
                        if decision.timestamp
                        else None,
                        "reason": decision.reason,
                    }
                ),
            )
            order = self._submit_idempotent(
                client_id,
                lambda: self.broker.submit_crypto_market_order(
                    symbol,
                    size.qty,
                    "buy",
                    client_id,
                    trade_increment=float(asset.min_trade_increment),
                    min_order_size=float(asset.min_order_size),
                    reference_price=entry_price,
                ),
            )
            self.store.record_order(order, self.session_id)
            self._record_fill_delta(intent, order, decision)
            pending_buy_symbols.add(symbol)
            exposure += size.position_notional
            open_risk += size.qty * size.risk_per_unit
            remaining_cash = max(0.0, remaining_cash - size.position_notional)
            remaining_buying_power = max(
                0.0, remaining_buying_power - size.position_notional
            )
            newly_submitted += 1
            held_returns[symbol] = candidate_returns
            if order.filled_qty > 0:
                current = self.broker.get_position(symbol)
                if current is not None:
                    actual_stop = (
                        current.avg_entry_price
                        - atr_value * self.strategy_config.atr_stop_multiplier
                    )
                    local = CryptoPositionState(
                        symbol=symbol,
                        qty=current.qty,
                        entry_price=current.avg_entry_price,
                        entry_time=now.isoformat(),
                        atr_at_entry=atr_value,
                        stop_price=max(
                            actual_stop, float(asset.price_increment or 0.0)
                        ),
                        highest_close=current.avg_entry_price,
                    )
                    self._save_position(local)
                    if actual_stop <= 0 or actual_stop >= current.avg_entry_price:
                        emergency = CryptoSignalDecision(
                            symbol=symbol,
                            action=BotAction.EXIT,
                            timestamp=now,
                            reason="invalid_actual_fill_stop",
                            close=current.avg_entry_price,
                        )
                        self._initiate_exit(symbol, emergency, bar_timestamp)
                    else:
                        self._ensure_protection(symbol, current, now, decision)

    def _initiate_exit(
        self,
        symbol: str,
        decision: CryptoSignalDecision,
        bar_timestamp: pd.Timestamp,
    ) -> None:
        position = self.broker.get_position(symbol)
        if position is None or position.qty <= 0:
            return
        local = self._load_position(symbol)
        if local is None:
            raise BrokerError(f"Refusing strategy exit for unowned {symbol} position.")
        if local.pending_exit_client_order_id:
            existing = self.broker.get_order_by_client_id(local.pending_exit_client_order_id)
            if existing is not None and not existing.is_terminal:
                return
        if local.stop_client_order_id:
            stop = self.broker.get_order_by_client_id(local.stop_client_order_id)
            if stop is not None and not stop.is_terminal and not self._cancel_and_confirm(stop):
                raise BrokerError(f"Could not confirm {symbol} protective-stop cancellation.")
            local.stop_client_order_id = None
            local.stop_broker_order_id = None
            local.stop_qty = 0.0
        # A stop can fill partially or completely during cancellation. Broker
        # truth must be re-read before sizing the market exit.
        position = self.broker.get_position(symbol)
        if position is None or position.qty <= _QTY_EPSILON:
            self.store.clear_bot_position(symbol)
            return
        client_id = make_client_order_id(
            self.session_id, "exit", bar_timestamp.to_pydatetime(), symbol
        )
        local.pending_exit_client_order_id = client_id
        self._save_position(local)
        asset = self._assets[symbol]
        intent = OrderRecord(
            "",
            client_id,
            symbol,
            "exit",
            "sell",
            position.qty,
            0.0,
            "intent",
            submitted_at=self._now_fn(),
            updated_at=self._now_fn(),
        )
        self.store.record_order(intent, self.session_id)
        order = self._submit_idempotent(
            client_id,
            lambda: self.broker.submit_crypto_market_order(
                symbol,
                position.qty,
                "sell",
                client_id,
                trade_increment=float(asset.min_trade_increment),
                min_order_size=float(asset.min_order_size),
                reference_price=decision.close,
            ),
        )
        self.store.record_order(order, self.session_id)
        self._record_fill_delta(intent, order, decision)

    def _submit_idempotent(self, client_id: str, submit) -> OrderRecord:
        existing = self.broker.get_order_by_client_id(client_id)
        if existing is not None:
            return existing
        try:
            return submit()
        except BrokerError:
            # A timeout can occur after acceptance. Reconcile the deterministic
            # ID before surfacing uncertainty; never generate a second ID.
            accepted = self.broker.get_order_by_client_id(client_id)
            if accepted is not None:
                return accepted
            raise

    def _refresh_recoverable_orders(self) -> None:
        rows = self.store.get_orders(limit=1_000)
        for row in rows:
            if row["symbol"] not in self.strategy_config.symbols:
                continue
            client_id = row["client_order_id"]
            if not is_bot_order_id(client_id):
                continue
            if row["status"] in {"filled", "canceled", "expired", "rejected", "replaced"}:
                continue
            try:
                current = self.broker.get_order_by_client_id(client_id)
            except BrokerError:
                continue
            if current is None:
                continue
            previous = self.store.get_order_by_client_id(client_id)
            self.store.record_order(current, row.get("session_id") or self.session_id)
            if previous is not None:
                self._record_fill_delta(previous, current, None)

    def _cancel_stale_entry_orders(self, now: datetime) -> None:
        """Prevent an old GTC market entry from filling long after its signal."""

        for order in self.broker.list_open_orders():
            if order.side != "buy" or order.purpose != "entry" or order.submitted_at is None:
                continue
            submitted = order.submitted_at
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=timezone.utc)
            age = (now - submitted.astimezone(timezone.utc)).total_seconds()
            if age <= self.engine_config.entry_order_timeout_seconds:
                continue
            if self._cancel_and_confirm(order):
                self._log(
                    "WARNING",
                    f"Canceled stale {order.symbol} entry after {age:.1f}s without completion.",
                )
            else:
                raise BrokerError(
                    f"Could not confirm cancellation of stale {order.symbol} entry."
                )

    def _record_fill_delta(
        self,
        previous: OrderRecord,
        current: OrderRecord,
        decision: Optional[CryptoSignalDecision],
    ) -> None:
        delta = current.filled_qty - previous.filled_qty
        if delta <= _QTY_EPSILON or current.fill_price is None:
            return
        if previous.filled_qty > 0 and previous.fill_price is not None:
            incremental_price = (
                current.filled_qty * current.fill_price
                - previous.filled_qty * previous.fill_price
            ) / delta
        else:
            incremental_price = current.fill_price
        local = self._load_position(current.symbol)
        realized = None
        if current.side == "sell" and local is not None:
            realized = (incremental_price - local.entry_price) * delta
        self.store.record_trade(
            self.session_id,
            current.symbol,
            "FILL_BUY" if current.side == "buy" else "FILL_SELL",
            delta,
            incremental_price,
            realized,
            current.purpose,
            decision.to_dict() if decision else None,
            event_key=f"{current.client_order_id}:{current.filled_qty:.12f}",
        )

    def _reconcile_position_truth(self, now: datetime, startup: bool = False):
        positions = {
            position.symbol: position
            for position in self.broker.list_positions()
            if position.symbol in self.strategy_config.symbols
        }
        for symbol in self.strategy_config.symbols:
            broker_position = positions.get(symbol)
            local = self._load_position(symbol)
            if broker_position is not None and broker_position.qty < -_QTY_EPSILON:
                raise BrokerError(f"Short crypto position detected for {symbol}; refusing to act.")
            if broker_position is None or broker_position.qty <= _QTY_EPSILON:
                if local is not None:
                    self._cancel_orphan_stop(local)
                    self.store.clear_bot_position(symbol)
                pending_buy = any(
                    order.side == "buy" and order.is_open
                    for order in self.broker.list_open_orders(symbol)
                )
                if not pending_buy:
                    self.store.delete_kv(self._entry_intent_key(symbol))
                continue
            increment = float(self._assets[symbol].min_trade_increment or _QTY_EPSILON)
            provable_qty = self.store.bot_net_filled_qty(symbol)
            if broker_position.qty - provable_qty > max(increment, _QTY_EPSILON) * 1.1:
                raise BrokerError(
                    f"Broker quantity for {symbol} exceeds bot-proven net fills; "
                    "manual or foreign exposure detected."
                )
            if local is None:
                local = self._adopt_owned_position(symbol, broker_position, now)
                if local is None:
                    raise BrokerError(
                        f"Broker reports {symbol} position but bot ownership cannot be proven."
                    )
            local.qty = broker_position.qty
            local.entry_price = broker_position.avg_entry_price
            if local.highest_close < broker_position.avg_entry_price:
                local.highest_close = broker_position.avg_entry_price
            self._save_position(local)
            if local.pending_exit_client_order_id:
                exit_order = self.broker.get_order_by_client_id(local.pending_exit_client_order_id)
                if exit_order is None:
                    exit_order = self._retry_pending_exit(local, broker_position, now)
                if exit_order is not None and exit_order.is_terminal:
                    previous = self.store.get_order_by_client_id(exit_order.client_order_id)
                    if previous is not None:
                        self._record_fill_delta(previous, exit_order, None)
                    self.store.record_order(exit_order, self.session_id)
                    refreshed = self.broker.get_position(symbol)
                    if exit_order.status == "filled":
                        if refreshed is None or refreshed.qty <= _QTY_EPSILON:
                            self.store.clear_bot_position(symbol)
                            self.store.delete_kv(self._entry_intent_key(symbol))
                            positions.pop(symbol, None)
                            continue
                        raise BrokerError(
                            f"{symbol} exit is FILLED but Alpaca still reports a long "
                            "position; waiting for consistent broker truth."
                        )
                    if refreshed is None or refreshed.qty <= _QTY_EPSILON:
                        self.store.clear_bot_position(symbol)
                        positions.pop(symbol, None)
                        continue
                    positions[symbol] = refreshed
                    local.qty = refreshed.qty
                    local.entry_price = refreshed.avg_entry_price
                    local.pending_exit_client_order_id = None
                    self._save_position(local)
        return positions

    def _retry_pending_exit(
        self,
        local: CryptoPositionState,
        broker_position,
        now: datetime,
    ) -> OrderRecord:
        """Resolve an exit whose first submit outcome was unknown.

        The same client ID is reused. A potentially accepted exit is never
        abandoned in favor of a concurrent stop, which could oversell the long.
        """

        client_id = local.pending_exit_client_order_id
        if not client_id:
            raise BrokerError("Pending exit intent is missing its deterministic client ID.")
        asset = self._assets[local.symbol]
        previous = self.store.get_order_by_client_id(client_id)
        if previous is None:
            previous = OrderRecord(
                "", client_id, local.symbol, "exit", "sell",
                broker_position.qty, 0.0, "intent", submitted_at=now, updated_at=now,
            )
            self.store.record_order(previous, self.session_id)
        order = self._submit_idempotent(
            client_id,
            lambda: self.broker.submit_crypto_market_order(
                local.symbol,
                broker_position.qty,
                "sell",
                client_id,
                trade_increment=float(asset.min_trade_increment),
                min_order_size=float(asset.min_order_size),
                reference_price=(broker_position.current_price or broker_position.avg_entry_price),
            ),
        )
        self._record_fill_delta(previous, order, None)
        self.store.record_order(order, self.session_id)
        return order

    def _adopt_owned_position(self, symbol, broker_position, now) -> Optional[CryptoPositionState]:
        # Nonterminal orders were refreshed immediately before reconciliation.
        # Terminal quantities already persisted in SQLite are immutable, so a
        # full SQL aggregate proves ownership without a 1,000-order retention cap.
        net = self.store.bot_net_filled_qty(symbol)
        increment = float(self._assets[symbol].min_trade_increment or _QTY_EPSILON)
        if abs(net - broker_position.qty) > max(increment, _QTY_EPSILON) * 1.1:
            return None
        intent_raw = self.store.get_kv(self._entry_intent_key(symbol))
        atr_value = 0.0
        latest_close = broker_position.avg_entry_price
        if intent_raw:
            try:
                intent = json.loads(intent_raw)
                atr_value = float(intent.get("atr") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                atr_value = 0.0
        if not math.isfinite(atr_value) or atr_value <= 0:
            bars = self.data_feed.get_recent_bars(
                symbol,
                self.strategy_config.timeframe,
                self.strategy.required_bars() + 2,
                now=now,
                tolerance_seconds=self.engine_config.bar_completion_tolerance_seconds,
            )
            prepared = self.strategy.prepare_symbol(bars)
            atr_value = float(prepared["atr"].iloc[-1])
            latest_close = float(prepared["close"].iloc[-1])
        entry = broker_position.avg_entry_price
        if not entry or not math.isfinite(atr_value) or atr_value <= 0:
            return None
        stop = entry - atr_value * self.strategy_config.atr_stop_multiplier
        if stop <= 0:
            return None
        local = CryptoPositionState(
            symbol=symbol,
            qty=broker_position.qty,
            entry_price=entry,
            entry_time=now.isoformat(),
            atr_at_entry=atr_value,
            stop_price=stop,
            highest_close=max(entry, latest_close),
        )
        self._save_position(local)
        self._log("WARNING", f"Recovered bot-owned {symbol} position from broker fills.")
        return local

    def _ensure_all_protection(self, now: datetime, positions=None) -> None:
        if positions is None:
            positions = {
                position.symbol: position
                for position in self.broker.list_positions()
                if position.symbol in self.strategy_config.symbols
            }
        for symbol, position in positions.items():
            local = self._load_position(symbol)
            if local is not None and not local.pending_exit_client_order_id:
                self._ensure_protection(symbol, position, now)

    def _ensure_protection(
        self,
        symbol,
        broker_position,
        now,
        decision: Optional[CryptoSignalDecision] = None,
    ) -> None:
        local = self._load_position(symbol)
        if local is None:
            atr_value = float(decision.atr or 0.0) if decision else 0.0
            if atr_value <= 0:
                raise BrokerError(f"Cannot establish {symbol} protection without a valid ATR.")
            stop = broker_position.avg_entry_price - atr_value * self.strategy_config.atr_stop_multiplier
            local = CryptoPositionState(
                symbol=symbol,
                qty=broker_position.qty,
                entry_price=broker_position.avg_entry_price,
                entry_time=now.isoformat(),
                atr_at_entry=atr_value,
                stop_price=stop,
                highest_close=broker_position.avg_entry_price,
            )
            self._save_position(local)
        if local.pending_exit_client_order_id:
            return
        asset = self._assets[symbol]
        increment = float(asset.min_trade_increment)
        existing = (
            self.broker.get_order_by_client_id(local.stop_client_order_id)
            if local.stop_client_order_id
            else None
        )
        if existing is not None and not existing.is_terminal:
            remaining = max(0.0, existing.requested_qty - existing.filled_qty)
            qty_matches = abs(remaining - broker_position.qty) <= increment * 1.1
            price_matches = (
                existing.stop_price is not None
                and abs(existing.stop_price - local.stop_price)
                <= float(asset.price_increment) * 1.1
            )
            if qty_matches and price_matches:
                local.stop_qty = remaining
                local.stop_broker_order_id = existing.broker_order_id
                self._save_position(local)
                return
            if not self._cancel_and_confirm(existing):
                raise BrokerError(f"Could not confirm replacement of {symbol} protective stop.")
            refreshed_position = self.broker.get_position(symbol)
            if refreshed_position is None or refreshed_position.qty <= _QTY_EPSILON:
                self.store.clear_bot_position(symbol)
                return
            broker_position = refreshed_position
            local.qty = broker_position.qty
            local.entry_price = broker_position.avg_entry_price
            local.stop_attempt += 1
            local.stop_client_order_id = None
            local.stop_broker_order_id = None
            local.stop_qty = 0.0
            self._save_position(local)
        elif existing is not None and existing.status == "filled":
            return
        elif existing is not None:
            local.stop_attempt += 1
            local.stop_client_order_id = None
            local.stop_broker_order_id = None
            local.stop_qty = 0.0

        client_id = local.stop_client_order_id or make_client_order_id(
            self.session_id, "stop", now, symbol, attempt=local.stop_attempt
        )
        limit_price = local.stop_price * (1.0 - self.engine_config.stop_limit_buffer_bps / 10_000.0)
        intent = OrderRecord(
            "",
            client_id,
            symbol,
            "stop",
            "sell",
            broker_position.qty,
            0.0,
            "intent",
            stop_price=local.stop_price,
            limit_price=limit_price,
            submitted_at=now,
            updated_at=now,
        )
        self.store.record_order(intent, self.session_id)
        local.stop_client_order_id = client_id
        local.stop_qty = broker_position.qty
        self._save_position(local)
        order = self._submit_idempotent(
            client_id,
            lambda: self.broker.submit_crypto_stop_limit_order(
                symbol,
                broker_position.qty,
                local.stop_price,
                limit_price,
                client_id,
                trade_increment=increment,
                min_order_size=float(asset.min_order_size),
                price_increment=float(asset.price_increment),
            ),
        )
        self.store.record_order(order, self.session_id)
        local.stop_broker_order_id = order.broker_order_id
        local.stop_client_order_id = order.client_order_id
        local.stop_qty = max(0.0, order.requested_qty - order.filled_qty)
        self._save_position(local)

    def _run_stop_watchdogs(self, now, quotes, positions) -> None:
        for symbol, position in positions.items():
            local = self._load_position(symbol)
            quote = quotes.get(symbol)
            if local is None or quote is None or local.pending_exit_client_order_id:
                continue
            watchdog_quote_ok, watchdog_reason = quote_is_safe(
                quote,
                now,
                max_age_seconds=self.engine_config.quote_max_age_seconds,
                max_spread_bps=10_000.0,
            )
            if not watchdog_quote_ok:
                self._log(
                    "ERROR",
                    f"{symbol} stop watchdog ignored unsafe quote: {watchdog_reason}",
                )
                continue
            # A breach must be observed, sustained, and still true at
            # escalation time.  Any quote back above the stop clears the latch:
            # without this un-latch a single wick permanently arms a market
            # liquidation that fires on the next poll even after a full price
            # recovery, which turns winning trends into scratches.
            if quote.bid_price > local.stop_price:
                if local.breached_at is not None or local.breach_confirmations:
                    local.breached_at = None
                    local.breach_confirmations = 0
                    self._save_position(local)
                    self._log(
                        "INFO",
                        f"{symbol} stop breach cleared; bid {quote.bid_price} "
                        f"recovered above stop {local.stop_price}.",
                    )
                continue
            if local.breached_at is None:
                local.breached_at = now.isoformat()
            local.breach_confirmations += 1
            self._save_position(local)
            breached = datetime.fromisoformat(local.breached_at)
            if (now - breached).total_seconds() < self.engine_config.stop_watchdog_grace_seconds:
                continue
            if local.breach_confirmations < self.engine_config.stop_watchdog_min_confirmations:
                continue
            stop = (
                self.broker.get_order_by_client_id(local.stop_client_order_id)
                if local.stop_client_order_id
                else None
            )
            if stop is not None and stop.status == "filled":
                self.store.record_order(stop, self.session_id)
                continue
            if stop is not None and not stop.is_terminal and not self._cancel_and_confirm(stop):
                self._log("ERROR", f"{symbol} stop watchdog could not confirm cancellation.")
                continue
            current = self.broker.get_position(symbol)
            if current is None or current.qty <= 0:
                self.store.clear_bot_position(symbol)
                continue
            timestamp = breached
            # The id stays deterministic per breach so a timeout after
            # acceptance maps back to the same order.  It advances only when
            # the previous guard is terminal WITHOUT a fill: otherwise a single
            # rejection freezes this id forever and every later escalation
            # resolves to that dead order, leaving the position unprotected.
            attempt = max(1, local.guard_attempt)
            client_id = make_client_order_id(
                self.session_id, "guard", timestamp, symbol, attempt=attempt
            )
            while attempt < _MAX_GUARD_ATTEMPTS:
                prior = self.broker.get_order_by_client_id(client_id)
                if prior is None or not prior.is_terminal or prior.filled_qty > 0:
                    break
                attempt += 1
                client_id = make_client_order_id(
                    self.session_id, "guard", timestamp, symbol, attempt=attempt
                )
            else:
                self._log(
                    "CRITICAL",
                    f"{symbol} guard exit was rejected {_MAX_GUARD_ATTEMPTS} times; "
                    "arming the kill switch instead of resubmitting.",
                )
                self._arm_kill_switch(f"{symbol} guard exit repeatedly rejected")
                continue
            local.guard_attempt = attempt
            local.pending_exit_client_order_id = client_id
            local.stop_client_order_id = None
            local.stop_broker_order_id = None
            local.stop_qty = 0.0
            self._save_position(local)
            asset = self._assets[symbol]
            intent = OrderRecord(
                "", client_id, symbol, "guard", "sell", current.qty, 0.0, "intent",
                submitted_at=now, updated_at=now,
            )
            self.store.record_order(intent, self.session_id)
            order = self._submit_idempotent(
                client_id,
                lambda: self.broker.submit_crypto_market_order(
                    symbol,
                    current.qty,
                    "sell",
                    client_id,
                    trade_increment=float(asset.min_trade_increment),
                    min_order_size=float(asset.min_order_size),
                    reference_price=quote.bid_price,
                ),
            )
            self.store.record_order(order, self.session_id)
            self._record_fill_delta(intent, order, None)
            self._log("CRITICAL", f"{symbol} stop-limit watchdog submitted market exit.")

    def _arm_kill_switch(self, message: str) -> None:
        """Latch the dashboard kill switch; Start stays blocked until acked."""

        self.store.set_kv(
            CRYPTO_KILL_SWITCH_KEY,
            json.dumps(
                {
                    "armed": True,
                    "reason": "GUARD_EXIT_FAILED",
                    "at": self._now_fn().isoformat(),
                    "message": message,
                }
            ),
        )

    def _cancel_and_confirm(self, order: OrderRecord) -> bool:
        if order.is_terminal:
            return True
        self.broker.cancel_order(order.broker_order_id)
        for _ in range(self.engine_config.fill_poll_attempts):
            current = self.broker.get_order_by_client_id(order.client_order_id)
            if current is None or current.is_terminal:
                if current is not None:
                    previous = self.store.get_order_by_client_id(current.client_order_id)
                    if previous is not None:
                        self._record_fill_delta(previous, current, None)
                    self.store.record_order(current, self.session_id)
                return True
            if self._stop_event.wait(0.25):
                break
        return False

    def _cancel_orphan_stop(self, local: CryptoPositionState) -> None:
        if not local.stop_client_order_id:
            return
        try:
            order = self.broker.get_order_by_client_id(local.stop_client_order_id)
            if order is not None and not order.is_terminal:
                self._cancel_and_confirm(order)
        except BrokerError as exc:
            self._log("ERROR", f"Could not cancel orphan {local.symbol} stop: {exc}")

    def _has_exposure(self) -> bool:
        """True when a bot position is open, from local state only.

        Deliberately does not call the broker: this only picks a sleep length
        and runs outside the iteration's error handling, so it must never
        raise and must never add a network round trip to the loop.
        """

        try:
            for symbol in self.strategy_config.symbols:
                if self.store.get_bot_position(symbol):
                    return True
        except Exception:  # noqa: BLE001 - cadence choice must never break the loop
            return True
        return False

    def _load_position(self, symbol: str) -> Optional[CryptoPositionState]:
        data = self.store.get_bot_position(symbol)
        return CryptoPositionState.from_dict(data) if data else None

    def _save_position(self, position: CryptoPositionState) -> None:
        self.store.set_bot_position(position.symbol, position.to_dict())

    @staticmethod
    def _entry_intent_key(symbol: str) -> str:
        return f"crypto_entry_intent:{symbol}"

    def _heartbeat(self, now: datetime, bar_timestamp) -> None:
        timestamp = pd.Timestamp(bar_timestamp)
        with self._lock:
            self._status.last_heartbeat = now
            self._status.last_bar_time = timestamp.to_pydatetime()
            if not self.risk.session_halted:
                self._status.state = EngineState.RUNNING
            self._status.warnings = []
            self._persist_status()
        account = self.broker.get_account()
        self.store.record_equity(
            self.session_id, now, account.equity, account.buying_power
        )

    def _warn(self, message: str) -> None:
        with self._lock:
            self._status.warnings = [message]
            self._status.last_heartbeat = self._now_fn()
            self._persist_status()
        self._log("WARNING", message)

    def _log(self, severity: str, message: str) -> None:
        self.store.log(severity, message, session_id=self.session_id)

    def _persist_status(self) -> None:
        try:
            self.store.set_kv(CRYPTO_ENGINE_STATUS_KEY, json.dumps(self._status.to_dict()))
        except Exception:
            logger.warning("Could not persist crypto engine status")
