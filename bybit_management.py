"""Production bridge from existing TradeManagementPolicy to Bybit execution.

This is deliberately an executor for ``quant.exits.TradeManagementPolicy`` rather
than a second exit policy.  The Bybit runtime wraps the entry strategy so its legacy
ATR ratchet is inert; all post-entry stop/partial/full-exit decisions then originate
from the existing policy object below.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from broker import BrokerError, make_client_order_id
from bybit_spot import BybitSpotBroker
from bybit_state import BybitStateStore, iso
from crypto_models import CryptoSignalDecision
from models import BotAction, OrderRecord
from quant.exit_intelligence import HoldIntelligenceConfig, assess_hold
from quant.exits import (
    ExitInstruction,
    ExitInstructionKind,
    ManagedPosition,
    PositionPhase,
    TradeManagementPolicy,
)

_QTY_EPS = 1e-12


class PolicyManagedStrategy:
    """Delegate every strategy behavior except the legacy production stop ratchet."""
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        # TradeManagementPolicy owns stop movement in the Bybit runtime.
        return current_stop, max(highest_completed_close, completed_close)


def _atr(frame: pd.DataFrame, period: int) -> float:
    if len(frame) < period + 1:
        return math.nan
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    value = float(tr.rolling(period, min_periods=period).mean().iloc[-1])
    return value if math.isfinite(value) and value > 0 else math.nan


def _efficiency(frame: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    close = frame["close"].astype(float).tail(lookback + 1)
    if len(close) < 3:
        return None
    travelled = float(close.diff().abs().dropna().sum())
    if travelled <= 0:
        return 0.0
    return max(0.0, min(1.0, abs(float(close.iloc[-1] - close.iloc[0])) / travelled))


class BybitTradeManagementExecutor:
    def __init__(
        self,
        engine,
        broker: BybitSpotBroker,
        venue_state: BybitStateStore,
        policy: TradeManagementPolicy,
        *,
        strategy_id: str,
        hold_config: HoldIntelligenceConfig = HoldIntelligenceConfig(),
    ) -> None:
        self.engine = engine
        self.broker = broker
        self.venue_state = venue_state
        self.policy = policy
        self.strategy_id = strategy_id
        self.hold_config = hold_config

    def _load_or_initialize(self, symbol: str, local, position, signal_reason: str) -> dict[str, Any]:
        state = self.venue_state.get_management_state(symbol)
        initial_stop = float(local.initial_stop_price or local.stop_price)
        if state is None:
            original_qty = max(float(position.qty), _QTY_EPS)
            state = {
                "original_qty": original_qty,
                "remaining_fraction": 1.0,
                "bars_held": 0,
                "phase": PositionPhase.NORMAL.value,
                "partial_taken": 0,
                "target_taken": 0,
                "signal_reason": signal_reason,
                "entry_price": float(local.entry_price),
                "initial_stop_price": initial_stop,
                "highest_price": float(local.entry_price),
                "lowest_price": float(local.entry_price),
            }
            self.venue_state.upsert_management_state(symbol, **state)
            state = self.venue_state.get_management_state(symbol) or state
        if float(state["original_qty"]) + _QTY_EPS < float(position.qty):
            # Quantity increased after the episode started: that would be averaging
            # into an open trade, which this mission explicitly forbids.
            raise BrokerError(f"{symbol} quantity increased after entry; averaging down/up is disabled")
        return state

    def _update_excursion(self, symbol: str, state: dict[str, Any], bar: pd.Series) -> dict[str, Any]:
        entry = float(state["entry_price"])
        highest = max(float(state["highest_price"]), float(bar["high"]))
        lowest = min(float(state["lowest_price"]), float(bar["low"]))
        values: dict[str, Any] = {"highest_price": highest, "lowest_price": lowest}
        now_iso = iso()
        for pct, key in (
            (0.0025, "threshold_025_at"),
            (0.0050, "threshold_050_at"),
            (0.0075, "threshold_075_at"),
            (0.0100, "threshold_100_at"),
        ):
            if not state.get(key) and highest >= entry * (1.0 + pct):
                values[key] = now_iso
        self.venue_state.upsert_management_state(symbol, **values)
        merged = dict(state)
        merged.update(values)
        return merged

    def manage_symbol(
        self,
        symbol: str,
        completed_frame: pd.DataFrame,
        bar_timestamp: pd.Timestamp,
        *,
        signal_reason: str = "intraday_adaptive",
    ) -> list[ExitInstruction]:
        position = self.broker.get_position(symbol)
        local = self.engine._load_position(symbol)
        if position is None or position.qty <= _QTY_EPS:
            self.venue_state.delete_management_state(symbol)
            return []
        if local is None:
            raise BrokerError(f"Cannot execute smart exits for {symbol}: engine position state missing")
        if bar_timestamp not in completed_frame.index:
            return []
        frame = completed_frame.loc[:bar_timestamp, ["open", "high", "low", "close", "volume"]].copy()
        state = self._load_or_initialize(symbol, local, position, signal_reason)
        state = self._update_excursion(symbol, state, frame.iloc[-1])
        atr_value = _atr(frame, int(self.engine.strategy_config.atr_period))
        if not math.isfinite(atr_value) or atr_value <= 0:
            return []
        assessment = assess_hold(
            frame,
            bar_timestamp,
            strategy_id=self.strategy_id,
            signal_reason=str(state.get("signal_reason") or signal_reason),
            config=self.hold_config,
        )
        original_qty = float(state["original_qty"])
        remaining_fraction = min(1.0, max(_QTY_EPS, float(position.qty) / original_qty))
        managed = ManagedPosition(
            entry_price=float(local.entry_price),
            initial_stop_price=float(state.get("initial_stop_price") or local.initial_stop_price or local.stop_price),
            current_stop_price=float(local.stop_price),
            highest_close=max(float(local.highest_close), float(state["highest_price"])),
            bars_held=int(state["bars_held"]),
            remaining_fraction=remaining_fraction,
            phase=PositionPhase(str(state["phase"])),
            partial_taken=bool(state["partial_taken"]),
            target_taken=bool(state["target_taken"]),
        )
        update = self.policy.on_completed_bar(
            managed,
            close=float(frame["close"].iloc[-1]),
            atr_value=atr_value,
            efficiency_ratio=_efficiency(frame),
            hold_score=assessment.hold_score if assessment.measurable else None,
        )
        executed: list[ExitInstruction] = []
        for instruction in update.instructions:
            if instruction.kind is ExitInstructionKind.FULL_EXIT:
                self._full_exit(symbol, instruction, bar_timestamp, float(frame["close"].iloc[-1]))
                executed.append(instruction)
                break
            if instruction.kind is ExitInstructionKind.PARTIAL_EXIT:
                self._partial_exit(symbol, instruction, bar_timestamp, float(frame["close"].iloc[-1]), original_qty)
                executed.append(instruction)
                # Broker truth changed; any following stop update is applied to the
                # freshly reconciled residual by _update_stop.
            elif instruction.kind is ExitInstructionKind.UPDATE_STOP:
                self._update_stop(symbol, float(instruction.new_stop or 0.0))
                executed.append(instruction)
        refreshed = self.broker.get_position(symbol)
        if refreshed is None or refreshed.qty <= _QTY_EPS:
            self.venue_state.delete_management_state(symbol)
        else:
            self.venue_state.upsert_management_state(
                symbol,
                remaining_fraction=min(1.0, max(_QTY_EPS, refreshed.qty / original_qty)),
                bars_held=update.position.bars_held,
                phase=update.position.phase.value,
                partial_taken=int(update.position.partial_taken),
                target_taken=int(update.position.target_taken),
                highest_price=max(float(state["highest_price"]), update.position.highest_close),
            )
        return executed

    def _update_stop(self, symbol: str, new_stop: float) -> None:
        if not math.isfinite(new_stop) or new_stop <= 0:
            raise BrokerError("TradeManagementPolicy produced an invalid stop")
        position = self.broker.get_position(symbol)
        local = self.engine._load_position(symbol)
        if position is None or local is None:
            return
        if new_stop <= local.stop_price:
            return
        # Keep stop below current market; a breached stop is handled by the existing
        # watchdog/emergency path rather than submitted as a malformed conditional.
        mark = float(position.current_price or position.avg_entry_price)
        new_stop = min(new_stop, mark * (1.0 - 1e-12))
        buffer_bps = float(self.engine.engine_config.stop_limit_buffer_bps)
        limit = new_stop * (1.0 - buffer_bps / 10_000.0)
        if local.stop_client_order_id:
            amended = self.broker.amend_protection(
                local.stop_client_order_id,
                qty=position.qty,
                stop_price=new_stop,
                limit_price=limit,
            )
            if amended.is_terminal:
                raise BrokerError(f"{symbol} protection became terminal during in-place stop ratchet")
            local.stop_price = new_stop
            local.stop_qty = max(0.0, amended.requested_qty - amended.filled_qty)
            local.stop_broker_order_id = amended.broker_order_id
            self.engine._save_position(local)
            self.engine.store.record_order(amended, self.engine.session_id)
            return
        # Missing protection is not a replacement; parent creates a fresh one and
        # its normal protection verification/reconciliation remains authoritative.
        local.stop_price = new_stop
        self.engine._save_position(local)
        self.engine._ensure_protection(symbol, position, self.engine._now_fn())

    def _partial_exit(
        self,
        symbol: str,
        instruction: ExitInstruction,
        bar_timestamp: pd.Timestamp,
        reference_price: float,
        original_qty: float,
    ) -> None:
        fraction = float(instruction.exit_fraction or 0.0)
        if fraction <= 0:
            return
        position = self.broker.get_position(symbol)
        local = self.engine._load_position(symbol)
        if position is None or local is None:
            return
        asset = self.engine._assets[symbol]
        increment = float(asset.min_trade_increment)
        min_qty = float(asset.min_order_size)
        desired = min(float(position.qty), original_qty * fraction)
        desired = math.floor((desired + 1e-15) / increment) * increment
        residual = max(0.0, float(position.qty) - desired)
        if desired < min_qty or residual < min_qty:
            self._full_exit(symbol, instruction, bar_timestamp, reference_price)
            return
        if not local.stop_client_order_id:
            raise BrokerError(f"Refusing {symbol} partial exit without established protection")

        # In-place quantity reduction keeps the post-partial residual protected while
        # releasing only the quantity intended for immediate market sale.
        buffer_bps = float(self.engine.engine_config.stop_limit_buffer_bps)
        amended = self.broker.amend_protection(
            local.stop_client_order_id,
            qty=residual,
            stop_price=local.stop_price,
            limit_price=local.stop_price * (1.0 - buffer_bps / 10_000.0),
        )
        if amended.is_terminal:
            raise BrokerError(f"{symbol} residual protection could not be established before partial exit")
        local.stop_qty = max(0.0, amended.requested_qty - amended.filled_qty)
        local.stop_broker_order_id = amended.broker_order_id
        self.engine._save_position(local)
        self.engine.store.record_order(amended, self.engine.session_id)

        client_id = make_client_order_id(
            self.engine.session_id, "partial", bar_timestamp.to_pydatetime(), symbol
        )
        intent = OrderRecord(
            broker_order_id="",
            client_order_id=client_id,
            symbol=symbol,
            purpose="partial",
            side="sell",
            requested_qty=desired,
            filled_qty=0.0,
            status="intent",
            submitted_at=self.engine._now_fn(),
            updated_at=self.engine._now_fn(),
            reference_price=reference_price,
        )
        self.engine.store.record_order(intent, self.engine.session_id)
        previous = intent
        try:
            current = self.engine._submit_idempotent(
                client_id,
                lambda: self.broker.submit_crypto_market_order(
                    symbol,
                    desired,
                    "sell",
                    client_id,
                    trade_increment=increment,
                    min_order_size=min_qty,
                    reference_price=reference_price,
                ),
            )
            for _ in range(6):
                self.engine._record_fill_delta(previous, current, None)
                self.engine.store.record_order(current, self.engine.session_id)
                if current.is_terminal:
                    break
                previous = current
                time.sleep(0.20)
                refreshed_order = self.broker.get_order_by_client_id(client_id)
                if refreshed_order is None:
                    break
                current = refreshed_order
        finally:
            # Regardless of partial/zero/full fill, immediately reconcile the actual
            # owned quantity and resize the already-live residual protection to it.
            actual = self.broker.get_position(symbol)
            if actual is None or actual.qty <= _QTY_EPS:
                self.engine.store.clear_bot_position(symbol)
                return
            local = self.engine._load_position(symbol)
            if local is None or not local.stop_client_order_id:
                self._emergency_exit(symbol, bar_timestamp, reference_price, "partial_exit_protection_state_lost")
                raise BrokerError(f"{symbol} residual protection state was lost after partial exit")
            try:
                restored = self.broker.amend_protection(
                    local.stop_client_order_id,
                    qty=actual.qty,
                    stop_price=local.stop_price,
                    limit_price=local.stop_price * (1.0 - buffer_bps / 10_000.0),
                )
                if restored.is_terminal:
                    raise BrokerError("residual stop terminal")
                local.qty = actual.qty
                local.stop_qty = max(0.0, restored.requested_qty - restored.filled_qty)
                local.stop_broker_order_id = restored.broker_order_id
                self.engine._save_position(local)
                self.engine.store.record_order(restored, self.engine.session_id)
            except Exception as exc:
                self._emergency_exit(symbol, bar_timestamp, reference_price, "partial_exit_protection_uncertain")
                raise BrokerError(
                    f"{symbol} residual protection uncertain after partial exit; emergency close initiated"
                ) from exc

    def _full_exit(
        self,
        symbol: str,
        instruction: ExitInstruction,
        bar_timestamp: pd.Timestamp,
        reference_price: float,
    ) -> None:
        decision = CryptoSignalDecision(
            symbol=symbol,
            action=BotAction.EXIT,
            timestamp=bar_timestamp.to_pydatetime(),
            reason=instruction.reason,
            close=reference_price,
        )
        self.engine._initiate_exit(symbol, decision, bar_timestamp)

    def _emergency_exit(
        self,
        symbol: str,
        bar_timestamp: pd.Timestamp,
        reference_price: float,
        reason: str,
    ) -> None:
        decision = CryptoSignalDecision(
            symbol=symbol,
            action=BotAction.EXIT,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            close=reference_price,
        )
        self.engine._initiate_exit(symbol, decision, bar_timestamp)
