"""Production bridge between the audited crypto execution engine and Quant Desk.

The parent ``CryptoTradeEngine`` remains authoritative for broker truth, order
idempotency, protective stops, watchdog exits and recovery. This subclass makes
strategy evidence authoritative for broker-backed ENTRY privilege and closes the
feedback loop from real fills back into the Quant Desk.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

from broker import BrokerError
from config import AccountMode
from crypto_config import _authorize_quant_live_engine_config
from crypto_engine import CryptoTradeEngine
from crypto_risk import CryptoRiskManager
from models import BotAction
from quant.contracts import Market, StrategyState
from quant.evidence_scanner import EvidenceQuantScanner
from quant.execution import ExecutionQualityEngine
from quant.journal import DecisionJournal
from quant.lifecycle import DegradationMonitor, LifecycleManager
from quant.marketplace import compute_strategy_health
from quant.portfolio import BookPosition, PortfolioBook
from quant.scanner import QuantScanner, ScanStrategy
from quant.store import QuantStore


_PAPER_EXECUTION_STATES = frozenset(
    {
        StrategyState.PAPER,
        StrategyState.LIVE_CANARY,
        StrategyState.LIVE,
        StrategyState.REDUCED_RISK,
    }
)
_LIVE_EXECUTION_STATES = frozenset(
    {StrategyState.LIVE_CANARY, StrategyState.LIVE, StrategyState.REDUCED_RISK}
)


class _AttributionOnlyJournal:
    """Let the scanner own decisions while parent-engine fills own attribution."""

    def __init__(self, delegate: DecisionJournal):
        self.delegate = delegate

    def record_crypto_decisions(self, _decisions, _session_id) -> None:
        return None

    def record_crypto_exit_fill(self, **kwargs) -> None:
        self.delegate.record_crypto_exit_fill(**kwargs)


class QuantControlledCryptoTradeEngine(CryptoTradeEngine):
    """CryptoTradeEngine with evidence/lifecycle/portfolio gates on entries."""

    def __init__(
        self,
        *args,
        strategy_id: str,
        strategy_version: str = "1.0.0",
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[QuantScanner] = None,
        degradation_monitor: Optional[DegradationMonitor] = None,
        **kwargs,
    ) -> None:
        if not strategy_id or not strategy_id.strip():
            raise ValueError("Quant-controlled execution requires a strategy_id.")
        if "quant_journal" in kwargs:
            raise ValueError(
                "QuantControlledCryptoTradeEngine owns quant_journal so fills and "
                "scanner decisions cannot be written to different stores."
            )

        if len(args) > 5:
            mutable_args = list(args)
            mutable_args[5] = _authorize_quant_live_engine_config(mutable_args[5])
            args = tuple(mutable_args)
        elif "engine_config" in kwargs:
            kwargs["engine_config"] = _authorize_quant_live_engine_config(
                kwargs["engine_config"]
            )
        else:
            raise ValueError("Quant-controlled execution requires engine_config.")

        desk_store = quant_store or QuantStore()
        identity = strategy_id.strip()
        journal = DecisionJournal(
            desk_store,
            market=Market.CRYPTO,
            timeframe=(
                args[3].timeframe
                if len(args) > 3
                else kwargs["strategy_config"].timeframe
            ),
            strategy_id=identity,
            strategy_version=strategy_version,
        )
        kwargs["quant_journal"] = _AttributionOnlyJournal(journal)
        super().__init__(*args, **kwargs)

        self.strategy_id = identity
        self.strategy_version = strategy_version
        self.quant_store = desk_store
        self._owns_quant_store = quant_store is None
        self.lifecycle = lifecycle or LifecycleManager(self.quant_store)
        self.execution_quality = ExecutionQualityEngine(state_store=self.store)
        self.quant_scanner = scanner or EvidenceQuantScanner(
            store=self.quant_store,
            lifecycle=self.lifecycle,
            execution_engine=self.execution_quality,
        )
        self.degradation_monitor = degradation_monitor or DegradationMonitor()

    def _has_historical_live_ack(self) -> bool:
        row = self.quant_store.get_strategy_state(self.strategy_id)
        if row and row.get("operator_ack"):
            return True
        for event in self.quant_store.get_promotion_history(self.strategy_id, limit=500):
            if event.get("to_state") not in {
                StrategyState.LIVE_CANARY.value,
                StrategyState.LIVE.value,
            }:
                continue
            raw = event.get("evidence_json")
            if not raw:
                continue
            try:
                evidence = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if evidence.get("operator_ack") is True:
                return True
        return False

    def _startup(self) -> None:
        state = self.lifecycle.state(self.strategy_id)
        mode = AccountMode(self.engine_config.account_mode)
        allowed = (
            _LIVE_EXECUTION_STATES
            if mode is AccountMode.LIVE
            else _PAPER_EXECUTION_STATES
        )
        if state not in allowed:
            required = "LIVE_CANARY/LIVE" if mode is AccountMode.LIVE else "PAPER"
            raise BrokerError(
                f"Strategy {self.strategy_id!r} is {state.value}; broker-backed "
                f"{mode.value} execution requires lifecycle state {required} or a "
                "risk-reduced state derived from it. Run research/shadow promotion "
                "instead of bypassing the evidence ladder."
            )
        if mode is AccountMode.LIVE and not self._has_historical_live_ack():
            raise BrokerError(
                f"Strategy {self.strategy_id!r} has no recorded operator-acknowledged "
                "LIVE_CANARY/LIVE promotion; refusing real-money startup."
            )
        super()._startup()

    def run_iteration(self) -> None:
        super().run_iteration()
        self._supervise_strategy_health()

    @staticmethod
    def _state_since(row) -> Optional[datetime]:
        if not row or not row.get("since"):
            return None
        try:
            timestamp = datetime.fromisoformat(str(row["since"]))
        except (TypeError, ValueError):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _supervise_strategy_health(self) -> None:
        state = self.lifecycle.state(self.strategy_id)
        if state in {
            StrategyState.RESEARCH,
            StrategyState.SHADOW,
            StrategyState.QUARANTINE,
        }:
            return
        state_row = self.quant_store.get_strategy_state(self.strategy_id)
        health = compute_strategy_health(
            self.quant_store,
            self.strategy_id,
            state,
            include_shadow=False,
            since=self._state_since(state_row),
        )
        verdict = self.degradation_monitor.evaluate(health)
        if verdict.action == "none":
            return
        reason = "automatic_degradation:" + ",".join(verdict.reasons)
        if verdict.action == "quarantine":
            self.lifecycle.demote(
                self.strategy_id,
                StrategyState.QUARANTINE,
                reason,
                evidence={"health": health.to_dict(), "verdict": verdict.action},
            )
            self._log(
                "CRITICAL", f"Quant Desk quarantined {self.strategy_id}: {reason}"
            )
            return
        if verdict.action == "reduce_risk":
            target = (
                StrategyState.REDUCED_RISK
                if state
                in {
                    StrategyState.LIVE_CANARY,
                    StrategyState.LIVE,
                    StrategyState.REDUCED_RISK,
                }
                else StrategyState.SHADOW
            )
            if state is not target:
                self.lifecycle.demote(
                    self.strategy_id,
                    target,
                    reason,
                    evidence={"health": health.to_dict(), "verdict": verdict.action},
                )
                self._log(
                    "WARNING",
                    f"Quant Desk demoted {self.strategy_id} to {target.value}: {reason}",
                )

    def _portfolio_book(self, broker_positions, prepared, equity: float) -> PortfolioBook:
        positions: list[BookPosition] = []
        for symbol, position in broker_positions.items():
            if position.qty <= 0:
                continue
            local = self._load_position(symbol)
            if local is None:
                raise BrokerError(
                    f"Cannot construct Quant Desk book for {symbol}: local stop state is missing."
                )
            price = getattr(position, "current_price", None)
            if price is None or price <= 0:
                price = float(prepared[symbol]["close"].iloc[-1])
            positions.append(
                BookPosition.from_stop(
                    symbol,
                    Market.CRYPTO,
                    self.strategy_id,
                    qty=float(position.qty),
                    price=float(price),
                    stop_price=float(local.stop_price),
                )
            )
        return PortfolioBook(equity=float(equity), positions=positions)

    def _handle_entries(
        self,
        decisions,
        prepared,
        bar_timestamp,
        quotes,
        account,
        broker_positions,
        now: datetime,
    ) -> None:
        buy_decisions = [
            decision for decision in decisions if decision.action is BotAction.BUY
        ]
        if not buy_decisions:
            return

        raw_bars = {
            symbol: frame[["open", "high", "low", "close", "volume"]].copy()
            for symbol, frame in prepared.items()
        }
        book = self._portfolio_book(broker_positions, prepared, account.equity)
        peak = self._peak_equity or account.equity
        drawdown_pct = (
            max(0.0, (peak - account.equity) / peak * 100.0) if peak > 0 else 0.0
        )
        report = self.quant_scanner.scan(
            strategies=[
                ScanStrategy(
                    strategy_id=self.strategy_id,
                    strategy=self.strategy,
                    market=Market.CRYPTO,
                    timeframe=self.strategy_config.timeframe,
                    version=self.strategy_version,
                    base_risk_pct=self.risk_config.risk_per_trade_pct,
                )
            ],
            bars_by_symbol=raw_bars,
            equity=account.equity,
            book=book,
            quotes=quotes,
            benchmark_symbol=self.strategy_config.benchmark_symbol,
            drawdown_pct=drawdown_pct,
            session_id=self.session_id,
            now=now,
        )
        grants = {
            decision.candidate.symbol: float(decision.risk_allocation_pct or 0.0)
            for decision in report.executable
            if decision.candidate.strategy_id == self.strategy_id
        }
        if not grants:
            return

        original_risk = self.risk
        original_config = self.risk_config
        try:
            for decision in buy_decisions:
                grant = min(
                    original_config.risk_per_trade_pct,
                    grants.get(decision.symbol, 0.0),
                )
                if grant <= 0:
                    continue
                scaled_config = replace(
                    original_config,
                    risk_per_trade_pct=grant,
                    max_portfolio_heat_pct=max(
                        grant, original_config.max_portfolio_heat_pct
                    ),
                )
                scaled_risk = CryptoRiskManager(scaled_config)
                scaled_risk._session_halted = original_risk.session_halted
                scaled_risk._halt_reason = original_risk.halt_reason
                self.risk_config = scaled_config
                self.risk = scaled_risk
                super()._handle_entries(
                    [decision],
                    prepared,
                    bar_timestamp,
                    quotes,
                    account,
                    broker_positions,
                    now,
                )
                broker_positions = self._reconcile_position_truth(now)
                account = self.broker.get_account()
        finally:
            self.risk = original_risk
            self.risk_config = original_config
