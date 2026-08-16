"""The desk scanner: one cycle of the full decision pipeline (spec §75).

    bars -> strategies -> regime -> abstention -> global ranking ->
    portfolio heat -> execution gate -> journaled decisions

The scanner is a pure DECISION layer.  It places no orders: EXECUTE
decisions with granted risk are what an execution engine (or an operator)
acts on; SHADOW-state strategies flow to the shadow executor for virtual
fills.  Everything — executions, rejections, abstentions, shadow routing —
is journaled with reasons.

Expected alpha is MEASURED, never assumed: a candidate's gross alpha is
its strategy's marketplace expectancy (R per trade, from attributed real
or shadow fills) converted to bps through its own stop distance.  A
strategy with no measured expectancy cannot be ranked for execution —
its decisions run in shadow until the record exists.  That is the honest
cold-start: evidence first, capital second.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import pandas as pd

from cost_model import stop_distance_bps
from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.abstention import AbstentionPolicy
from quant.contracts import (
    DecisionOutcome,
    Market,
    OpportunityCandidate,
    OpportunityDecision,
    QuantAction,
    RegimeAssessment,
    StrategyState,
)
from quant.execution import ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
from quant.marketplace import MIN_SAMPLE, compute_strategy_health, regime_compatibility
from quant.opportunity import (
    GlobalOpportunityRanker,
    OpportunityScoringConfig,
    RankerInput,
)
from quant.portfolio import (
    PortfolioBook,
    PortfolioLimitsConfig,
    PortfolioRiskGovernor,
    sector_for,
)
from quant.regime import RegimeEngine
from quant.shadow import ShadowExecutor
from quant.store import QuantStore
from quant.strategies.base import UniverseStrategy


@dataclass
class ScanStrategy:
    """One strategy instance participating in a scan cycle."""

    strategy_id: str
    strategy: UniverseStrategy
    market: Market
    timeframe: str
    version: str = "1.0.0"


@dataclass
class ScanReport:
    timestamp: Optional[pd.Timestamp]
    regime: Optional[RegimeAssessment]
    decisions: list[OpportunityDecision] = field(default_factory=list)
    grants: dict[str, float] = field(default_factory=dict)
    shadow_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_strategies: dict[str, str] = field(default_factory=dict)

    @property
    def executable(self) -> list[OpportunityDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.outcome is DecisionOutcome.EXECUTE
            and (decision.risk_allocation_pct or 0.0) > 0.0
        ]


class QuantScanner:
    """Runs the decision pipeline over prepared strategy sets."""

    def __init__(
        self,
        *,
        store: QuantStore,
        lifecycle: LifecycleManager,
        execution_engine: ExecutionQualityEngine,
        regime_engine: Optional[RegimeEngine] = None,
        abstention_policy: Optional[AbstentionPolicy] = None,
        scoring_config: Optional[OpportunityScoringConfig] = None,
        portfolio_config: Optional[PortfolioLimitsConfig] = None,
        shadow_executor: Optional[ShadowExecutor] = None,
    ) -> None:
        self.store = store
        self.lifecycle = lifecycle
        self.execution_engine = execution_engine
        self.regime_engine = regime_engine or RegimeEngine()
        self.abstention_policy = abstention_policy or AbstentionPolicy()
        self.ranker = GlobalOpportunityRanker(
            scoring_config, self.abstention_policy, execution_engine
        )
        self.governor = PortfolioRiskGovernor(portfolio_config)
        self.shadow = shadow_executor or ShadowExecutor(store)

    def scan(
        self,
        *,
        strategies: list[ScanStrategy],
        bars_by_symbol: Mapping[str, pd.DataFrame],
        equity: float,
        book: PortfolioBook,
        quotes: Optional[Mapping[str, Any]] = None,
        benchmark_symbol: Optional[str] = None,
        drawdown_pct: float = 0.0,
        session_id: Optional[str] = None,
    ) -> ScanReport:
        """One full decision cycle over the latest aligned completed bar."""
        quotes = quotes or {}
        common: Optional[pd.DatetimeIndex] = None
        for frame in bars_by_symbol.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            return ScanReport(timestamp=None, regime=None)
        timestamp = common.sort_values()[-1]

        regime = None
        bench = benchmark_symbol or sorted(bars_by_symbol)[0]
        bench_bars = bars_by_symbol.get(bench)
        if bench_bars is not None and len(bench_bars) >= self.regime_engine.required_bars():
            member_closes = {
                symbol: frame["close"] for symbol, frame in bars_by_symbol.items()
            }
            regime = self.regime_engine.assess(
                bench_bars,
                market=strategies[0].market if strategies else Market.CRYPTO,
                member_closes=member_closes,
            )

        if regime is not None:
            # KV snapshot for the dashboard's market-state panel; decisions
            # embed their own regime copy, this is just the latest view.
            self.store.set_kv("last_regime", json.dumps(regime.to_dict()))
        report = ScanReport(timestamp=timestamp, regime=regime)
        ranker_inputs: list[RankerInput] = []
        candidate_context: dict[str, dict[str, Any]] = {}

        for entry in strategies:
            state = self.lifecycle.state(entry.strategy_id)
            if state in (StrategyState.RESEARCH, StrategyState.QUARANTINE):
                report.skipped_strategies[entry.strategy_id] = (
                    f"state_{state.value.lower()}_not_scanned"
                )
                continue
            universe = [
                symbol
                for symbol in getattr(entry.strategy.config, "symbols", [])
                if symbol in bars_by_symbol
            ]
            if not universe:
                report.skipped_strategies[entry.strategy_id] = "no_universe_data"
                continue
            try:
                prepared = entry.strategy.prepare_universe(
                    {symbol: bars_by_symbol[symbol] for symbol in universe}
                )
            except ValueError as error:
                report.skipped_strategies[entry.strategy_id] = f"prepare_failed: {error}"
                continue
            aligned = all(timestamp in frame.index for frame in prepared.values())
            if not aligned:
                report.skipped_strategies[entry.strategy_id] = "bars_not_aligned"
                continue

            if state is StrategyState.SHADOW:
                held = self.shadow.held_symbols(entry.strategy_id)
                decisions = entry.strategy.evaluate_prepared(prepared, timestamp, held)
                report.shadow_counts[entry.strategy_id] = self.shadow.process_cycle(
                    strategy_id=entry.strategy_id,
                    strategy_version=entry.version,
                    strategy=entry.strategy,
                    market=entry.market,
                    timeframe=entry.timeframe,
                    decisions=decisions,
                    prepared=prepared,
                    timestamp=timestamp,
                )
                for decision in decisions:
                    report.decisions.append(
                        self._journal_shadow_decision(
                            entry, decision, session_id=session_id
                        )
                    )
                continue

            held_symbols = frozenset(
                position.symbol for position in book.positions
            )
            decisions = entry.strategy.evaluate_prepared(prepared, timestamp, held_symbols)
            health = compute_strategy_health(
                self.store,
                entry.strategy_id,
                state,
                include_shadow=True,
            )
            for decision in decisions:
                if decision.action is not BotAction.BUY:
                    continue
                candidate, context = self._candidate_from_decision(
                    entry, decision, health, prepared, regime, drawdown_pct, quotes, equity
                )
                if candidate is None:
                    # No measured expectancy yet: journaled as a rejection
                    # with the honest reason, never given fabricated alpha.
                    report.decisions.append(
                        self._journal_unmeasured(entry, decision, session_id=session_id)
                    )
                    continue
                key = f"{entry.strategy_id}:{decision.symbol}"
                candidate_context[key] = context
                ranker_inputs.append(context["ranker_input"])

        ranked = self.ranker.rank(ranker_inputs)

        allocation_entries = []
        for decision in ranked:
            if decision.outcome is not DecisionOutcome.EXECUTE:
                continue
            key = f"{decision.candidate.strategy_id}:{decision.candidate.symbol}"
            context = candidate_context.get(key)
            if context is None:
                continue
            allocation_entries.append(
                {
                    "key": key,
                    "requested_risk_pct": context["requested_risk_pct"],
                    "notional": context["notional"],
                    "symbol": decision.candidate.symbol,
                    "market": decision.candidate.market.value,
                    "sector": sector_for(
                        decision.candidate.symbol, decision.candidate.market
                    ),
                    "strategy_id": decision.candidate.strategy_id,
                    "returns": context.get("returns"),
                }
            )
        grants = self.governor.allocate(
            PortfolioBook(equity=equity, positions=list(book.positions)),
            allocation_entries,
        )
        report.grants = grants

        for decision in ranked:
            key = f"{decision.candidate.strategy_id}:{decision.candidate.symbol}"
            if decision.outcome is DecisionOutcome.EXECUTE:
                granted = grants.get(key, 0.0)
                decision.risk_allocation_pct = granted
                if granted <= 0.0:
                    decision.outcome = DecisionOutcome.REJECT
                    decision.rejection_reasons = ["portfolio_risk_budget"]
            self.store.record_decision(decision, session_id)
            report.decisions.append(decision)
        return report

    # ------------------------------------------------------------------

    def _candidate_from_decision(
        self,
        entry: ScanStrategy,
        decision: CryptoSignalDecision,
        health,
        prepared: Mapping[str, pd.DataFrame],
        regime: Optional[RegimeAssessment],
        drawdown_pct: float,
        quotes: Mapping[str, Any],
        equity: float,
    ) -> tuple[Optional[OpportunityCandidate], dict[str, Any]]:
        close = decision.close
        stop = decision.stop_price
        if close is None or stop is None:
            return None, {}
        if health.expectancy_r is None or health.trades < MIN_SAMPLE:
            return None, {}
        distance_bps = stop_distance_bps(close, stop)
        if not math.isfinite(distance_bps):
            return None, {}
        gross_alpha_bps = health.expectancy_r * distance_bps
        confidence = health.hit_rate if health.hit_rate is not None else 0.0

        candidate = OpportunityCandidate(
            symbol=decision.symbol,
            market=entry.market,
            timeframe=entry.timeframe,
            strategy_id=entry.strategy_id,
            strategy_version=entry.version,
            action=QuantAction.BUY,
            timestamp=decision.timestamp,
            close=close,
            stop_price=stop,
            atr=decision.atr,
            expected_gross_alpha_bps=gross_alpha_bps,
            confidence=confidence,
            signal_reason=decision.reason,
            filters=dict(decision.filters),
            details={
                "expectancy_r": health.expectancy_r,
                "measured_trades": health.trades,
                "stop_distance_bps": distance_bps,
            },
        )

        quote = quotes.get(decision.symbol)
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        frame = prepared.get(decision.symbol)
        avg_dollar_volume = None
        returns = None
        if frame is not None:
            closes = frame["close"].astype(float)
            volumes = frame["volume"].astype(float)
            tail = (closes * volumes).tail(20)
            if len(tail) == 20:
                avg_dollar_volume = float(tail.mean())
            returns = closes.pct_change()
        risk_pct = self.lifecycle.risk_multiplier(entry.strategy_id) * 0.35
        notional = 0.0
        if equity > 0 and close > stop:
            qty = equity * risk_pct / 100.0 / (close - stop)
            notional = qty * close
        estimate = self.execution_engine.estimate(
            decision.symbol,
            entry.market,
            bid=bid,
            ask=ask,
            quote_age_seconds=0.0 if quote is not None else None,
            notional=notional if notional > 0 else None,
            avg_bar_dollar_volume=avg_dollar_volume,
        )
        compatibility = None
        if regime is not None:
            compatibility = regime_compatibility(
                health, regime.dominant_trend_label.value
            )
        reliability = None
        if health.profit_factor is not None and math.isfinite(health.profit_factor):
            reliability = health.profit_factor / (1.0 + health.profit_factor)
        ranker_input = RankerInput(
            candidate=candidate,
            execution_estimate=estimate,
            regime=regime,
            regime_compatibility=compatibility,
            strategy_reliability=reliability,
            liquidity_quality=1.0 if estimate.viable else 0.0,
            drawdown_pct=drawdown_pct,
            atr_percentile=decision.atr_percentile,
        )
        return candidate, {
            "ranker_input": ranker_input,
            "requested_risk_pct": risk_pct,
            "notional": notional,
            "returns": returns,
        }

    def _journal_shadow_decision(
        self,
        entry: ScanStrategy,
        decision: CryptoSignalDecision,
        *,
        session_id: Optional[str],
    ) -> OpportunityDecision:
        journal_decision = OpportunityDecision(
            candidate=OpportunityCandidate(
                symbol=decision.symbol,
                market=entry.market,
                timeframe=entry.timeframe,
                strategy_id=entry.strategy_id,
                strategy_version=entry.version,
                action=QuantAction(decision.action.value),
                timestamp=decision.timestamp,
                close=decision.close,
                stop_price=decision.stop_price,
                atr=decision.atr,
                expected_gross_alpha_bps=0.0,
                confidence=0.0,
                signal_reason=decision.reason,
                filters=dict(decision.filters),
                details={"shadow": True},
            ),
            outcome=DecisionOutcome.SHADOW,
            rank=None,
            expected_cost_bps=None,
            expected_net_alpha_bps=None,
            opportunity_score=None,
            risk_allocation_pct=None,
            rejection_reasons=[],
        )
        self.store.record_decision(journal_decision, session_id)
        return journal_decision

    def _journal_unmeasured(
        self,
        entry: ScanStrategy,
        decision: CryptoSignalDecision,
        *,
        session_id: Optional[str],
    ) -> OpportunityDecision:
        journal_decision = OpportunityDecision(
            candidate=OpportunityCandidate(
                symbol=decision.symbol,
                market=entry.market,
                timeframe=entry.timeframe,
                strategy_id=entry.strategy_id,
                strategy_version=entry.version,
                action=QuantAction(decision.action.value),
                timestamp=decision.timestamp,
                close=decision.close,
                stop_price=decision.stop_price,
                atr=decision.atr,
                expected_gross_alpha_bps=0.0,
                confidence=0.0,
                signal_reason=decision.reason,
                filters=dict(decision.filters),
                details={"unmeasured_alpha": True},
            ),
            outcome=DecisionOutcome.REJECT,
            rank=None,
            expected_cost_bps=None,
            expected_net_alpha_bps=None,
            opportunity_score=None,
            risk_allocation_pct=None,
            rejection_reasons=["no_measured_expectancy"],
        )
        self.store.record_decision(journal_decision, session_id)
        return journal_decision
