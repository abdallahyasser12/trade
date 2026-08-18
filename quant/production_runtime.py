"""Production bridge between the audited crypto execution engine and Quant Desk.

This module deliberately does *not* reimplement broker/order/stop/recovery logic.
It subclasses :class:`crypto_engine.CryptoTradeEngine` and makes the Quant Desk
an authoritative PRE-TRADE layer while the parent remains authoritative for
execution and position protection.

Guarantees
----------
* broker-backed PAPER requires lifecycle state PAPER or higher;
* LIVE requires LIVE_CANARY/LIVE/REDUCED_RISK;
* every BUY must also be approved by QuantScanner (measured expectancy,
  execution-quality, abstention, portfolio limits and lifecycle risk scaling);
* EXIT decisions remain owned by the parent engine, so a quant ranking failure
  can never block a risk-reducing exit;
* Quant Desk can only reduce per-trade risk. It can never raise the risk manager's
  configured risk_per_trade_pct;
* no synthetic or RESEARCH/SHADOW strategy can reach a broker order through this
  runtime.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Optional

from broker import BrokerError
from config import AccountMode
from crypto_engine import CryptoTradeEngine
from crypto_risk import CryptoRiskManager
from models import BotAction
from quant.contracts import Market, StrategyState
from quant.execution import ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
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


class QuantControlledCryptoTradeEngine(CryptoTradeEngine):
    """CryptoTradeEngine with evidence/lifecycle/portfolio gates on entries.

    Adapters may replace ``self.strategy`` after ``super().__init__`` exactly as
    the legacy cross-sectional/ensemble adapters already do. The scanner reads
    the current strategy at decision time, so that replacement remains safe.
    """

    def __init__(
        self,
        *args,
        strategy_id: str,
        strategy_version: str = "1.0.0",
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[QuantScanner] = None,
        **kwargs,
    ) -> None:
        if not strategy_id or not strategy_id.strip():
            raise ValueError("Quant-controlled execution requires a strategy_id.")
        super().__init__(*args, **kwargs)
        self.strategy_id = strategy_id.strip()
        self.strategy_version = strategy_version
        self.quant_store = quant_store or QuantStore()
        self._owns_quant_store = quant_store is None
        self.lifecycle = lifecycle or LifecycleManager(self.quant_store)
        self.quant_scanner = scanner or QuantScanner(
            store=self.quant_store,
            lifecycle=self.lifecycle,
            execution_engine=ExecutionQualityEngine(state_store=self.store),
        )

    def _startup(self) -> None:
        state = self.lifecycle.state(self.strategy_id)
        mode = AccountMode(self.engine_config.account_mode)
        allowed = _LIVE_EXECUTION_STATES if mode is AccountMode.LIVE else _PAPER_EXECUTION_STATES
        if state not in allowed:
            required = "LIVE_CANARY/LIVE" if mode is AccountMode.LIVE else "PAPER"
            raise BrokerError(
                f"Strategy {self.strategy_id!r} is {state.value}; broker-backed "
                f"{mode.value} execution requires lifecycle state {required} or a "
                "risk-reduced state derived from it. Run research/shadow promotion "
                "instead of bypassing the evidence ladder."
            )
        super()._startup()

    def _portfolio_book(self, broker_positions, prepared, equity: float) -> PortfolioBook:
        positions: list[BookPosition] = []
        for symbol, position in broker_positions.items():
            if position.qty <= 0:
                continue
            local = self._load_position(symbol)
            if local is None:
                # Parent reconciliation should make this impossible. Failing closed
                # here avoids understating portfolio heat if state is inconsistent.
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
        """Require QuantScanner approval, then delegate execution to the parent.

        The parent remains the final authority for quote validation, correlation,
        sizing, asset increments, cost-to-risk, idempotent order submission and
        protection. The scanner is therefore an additional gate, never a bypass.
        """

        buy_decisions = [decision for decision in decisions if decision.action is BotAction.BUY]
        if not buy_decisions:
            return

        # Scanner prepares strategies itself. Strip indicator columns so we do not
        # accidentally make a strategy depend on an earlier preparation pass.
        raw_bars = {
            symbol: frame[["open", "high", "low", "close", "volume"]].copy()
            for symbol, frame in prepared.items()
        }
        book = self._portfolio_book(broker_positions, prepared, account.equity)
        peak = self._peak_equity or account.equity
        drawdown_pct = max(0.0, (peak - account.equity) / peak * 100.0) if peak > 0 else 0.0
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

        # Execute one candidate at a time so each subsequent candidate sees fresh
        # broker exposure. Quant risk can only shrink the configured risk budget.
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
                    # Validation requires heat >= per-trade risk. Never increase
                    # the original heat ceiling; grant itself is already <= base risk.
                    max_portfolio_heat_pct=max(grant, original_config.max_portfolio_heat_pct),
                )
                scaled_risk = CryptoRiskManager(scaled_config)
                # Preserve the session-loss latch across the temporary sizing view.
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
                # Refresh before the next candidate so total exposure/heat cannot
                # be computed from a pre-fill snapshot.
                broker_positions = self._reconcile_position_truth(now)
                account = self.broker.get_account()
        finally:
            self.risk = original_risk
            self.risk_config = original_config
