"""Decision journal + attribution adapter between engines and the quant store.

The execution engines already explain every decision in their own typed
structures (``SignalDecision``, ``CryptoSignalDecision``).  This adapter
maps those onto the desk-wide journal schema so ONE store answers "what
did any strategy decide, and why" — including HOLDs with their failed
filters, which are rejections and must be as queryable as executions.

Journaling must never interfere with trading: every public method
swallows its own exceptions after logging through the engine-provided
callback.  A failed journal write costs an audit row, never an order.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from crypto_models import CryptoSignalDecision
from models import BotAction, SignalDecision
from quant.contracts import (
    AttributionRecord,
    DecisionOutcome,
    Market,
    OpportunityCandidate,
    OpportunityDecision,
    QuantAction,
)
from quant.store import QuantStore

# Engine strategy classes -> registry strategy ids.  The registry is the
# naming authority; unknown classes fall back to their class name so the
# journal never drops a row over a naming gap.
_STRATEGY_CLASS_IDS = {
    "CryptoRegimeStrategy": "crypto_regime_breakout",
    "CrossSectionalMomentumStrategy": "crypto_cross_sectional",
    "MeanReversionStrategy": "crypto_mean_reversion",
    "VolatilityRegimeStrategy": "crypto_volatility_regime",
    "RegimeAwareEnsembleStrategy": "crypto_regime_ensemble",
    "CouncilStrategy": "crypto_council",
    "TrendStrategy": "stock_trend",
    "DonchianTrendStrategy": "donchian_trend",
    "OpeningRangeBreakoutStrategy": "opening_range_breakout",
    "VwapReversionStrategy": "vwap_reversion",
    "VolatilitySqueezeStrategy": "volatility_squeeze",
}


def strategy_id_for(strategy: object) -> str:
    name = type(strategy).__name__
    return _STRATEGY_CLASS_IDS.get(name, name)


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


class DecisionJournal:
    """Best-effort desk journal bound to one engine's identity."""

    def __init__(
        self,
        store: QuantStore,
        *,
        market: Market,
        timeframe: str,
        strategy_id: str,
        strategy_version: str = "1.0.0",
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.store = store
        self.market = market
        self.timeframe = timeframe
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self._warn = warn or (lambda message: None)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def record_crypto_decisions(
        self,
        decisions: Iterable[CryptoSignalDecision],
        session_id: Optional[str],
    ) -> None:
        try:
            for decision in decisions:
                self.store.record_decision(
                    self._crypto_decision_to_journal(decision), session_id
                )
        except Exception as error:  # journaling must never break trading
            self._warn(f"Decision journal write failed: {error}")

    def record_stock_decision(
        self,
        symbol: str,
        decision: SignalDecision,
        session_id: Optional[str],
        *,
        blocked_reason: Optional[str] = None,
    ) -> None:
        try:
            candidate = OpportunityCandidate(
                symbol=symbol,
                market=Market.US_EQUITY,
                timeframe=self.timeframe,
                strategy_id=self.strategy_id,
                strategy_version=self.strategy_version,
                action=QuantAction(decision.action.value),
                timestamp=decision.timestamp,
                close=_finite(decision.details.get("close")),
                stop_price=None,
                atr=_finite(decision.atr),
                expected_gross_alpha_bps=0.0,
                confidence=0.0,
                signal_reason=decision.reason,
                filters=dict(decision.filters),
                details={"unmeasured_alpha": True},
            )
            outcome, reasons = self._outcome_for(
                decision.action, decision.filters, blocked_reason
            )
            self.store.record_decision(
                OpportunityDecision(
                    candidate=candidate,
                    outcome=outcome,
                    rank=None,
                    expected_cost_bps=None,
                    expected_net_alpha_bps=None,
                    opportunity_score=None,
                    risk_allocation_pct=None,
                    rejection_reasons=reasons,
                    decided_at=datetime.now(timezone.utc),
                ),
                session_id,
            )
        except Exception as error:
            self._warn(f"Decision journal write failed: {error}")

    def _crypto_decision_to_journal(
        self, decision: CryptoSignalDecision
    ) -> OpportunityDecision:
        candidate = OpportunityCandidate(
            symbol=decision.symbol,
            market=self.market,
            timeframe=self.timeframe,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            action=QuantAction(decision.action.value),
            timestamp=decision.timestamp,
            close=_finite(decision.close),
            stop_price=_finite(decision.stop_price),
            atr=_finite(decision.atr),
            expected_gross_alpha_bps=0.0,
            confidence=0.0,
            signal_reason=decision.reason,
            filters=dict(decision.filters),
            details={
                "regime": decision.regime.value,
                "score": _finite(decision.score),
                "unmeasured_alpha": True,
            },
        )
        outcome, reasons = self._outcome_for(
            decision.action,
            decision.filters,
            None,
            failed_filters=decision.details.get("failed_filters"),
        )
        return OpportunityDecision(
            candidate=candidate,
            outcome=outcome,
            rank=None,
            expected_cost_bps=None,
            expected_net_alpha_bps=None,
            opportunity_score=None,
            risk_allocation_pct=None,
            rejection_reasons=reasons,
            decided_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _outcome_for(
        action: BotAction,
        filters: dict[str, bool],
        blocked_reason: Optional[str],
        failed_filters: Optional[list[str]] = None,
    ) -> tuple[DecisionOutcome, list[str]]:
        if action in (BotAction.BUY, BotAction.EXIT):
            if blocked_reason:
                return DecisionOutcome.REJECT, [blocked_reason]
            return DecisionOutcome.EXECUTE, []
        reasons = list(failed_filters or [])
        if not reasons:
            reasons = sorted(name for name, ok in filters.items() if not ok)
        return DecisionOutcome.REJECT, reasons

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def record_crypto_exit_fill(
        self,
        *,
        symbol: str,
        session_id: Optional[str],
        qty: float,
        exit_price: float,
        total_fees: float,
        exit_reason: str,
        entry_price: float,
        entry_time_iso: Optional[str],
        initial_stop_price: Optional[float],
        highest_close: Optional[float],
        lowest_close: Optional[float],
        entry_regime: Optional[str],
        exit_timestamp: Optional[datetime] = None,
    ) -> None:
        """Attribute one sell fill against its position's entry.

        A round trip closed across several partial fills produces several
        rows whose quantities sum to the position; per-unit statistics
        (R-multiple, excursions) are consistent row by row, so marketplace
        aggregation over rows remains unbiased.

        Excursions here are CLOSE-based (the live position state tracks
        completed closes, not intrabar extremes) — systematically no wider
        than the intrabar truth, and labeled as such downstream.
        """
        try:
            entry_ts = (
                datetime.fromisoformat(entry_time_iso) if entry_time_iso else None
            )
            gross = (exit_price - entry_price) * qty
            net = gross - total_fees
            r_multiple = None
            stop = _finite(initial_stop_price)
            if stop is not None and 0 < stop < entry_price:
                risk = (entry_price - stop) * qty
                if risk > 0:
                    r_multiple = net / risk
            high = _finite(highest_close)
            low = _finite(lowest_close)
            mfe = (
                max(0.0, (high - entry_price) / entry_price * 100.0)
                if high is not None and entry_price > 0
                else None
            )
            mae = (
                max(0.0, (entry_price - low) / entry_price * 100.0)
                if low is not None and entry_price > 0
                else None
            )
            self.store.record_attribution(
                AttributionRecord(
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    symbol=symbol,
                    market=self.market,
                    timeframe=self.timeframe,
                    session_id=session_id,
                    signal_timestamp=entry_ts,
                    entry_timestamp=entry_ts,
                    exit_timestamp=exit_timestamp or datetime.now(timezone.utc),
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    initial_stop_price=stop,
                    fees=total_fees,
                    expected_cost_bps=None,
                    realized_slippage_bps=None,
                    gross_pnl=gross,
                    net_pnl=net,
                    r_multiple=r_multiple,
                    max_favorable_excursion_pct=mfe,
                    max_adverse_excursion_pct=mae,
                    entry_reason="",
                    exit_reason=exit_reason,
                    regime_at_entry=entry_regime,
                    shadow=False,
                )
            )
        except Exception as error:
            self._warn(f"Attribution write failed: {error}")
