"""Bybit-only production entry/target intelligence layered on the existing Quant Desk.

This module does not create alpha and does not modify the adaptive strategy version.
It is a negative-only venue layer:

* the existing :class:`EvidenceQuantScanner` remains responsible for measured edge,
  regime, costs, setup intelligence, global ranking and portfolio risk;
* Bybit multi-level order-book state may only demote/reject a candidate;
* a causal target plan is frozen only after a candidate survives global ranking and
  receives a positive executable risk grant;
* post-entry execution feeds that frozen target into the existing
  ``TradeManagementPolicy`` rather than creating a second exit system.

The target planner is deliberately structural instead of a hard-coded +0.5/+1% order:
it combines initial R, ATR, recent realized movement, nearby completed-bar resistance
and current measured/assumed execution costs. The 0.5/0.75/1.0 families are recorded
for burn-in research only; they are not a quota and are not selected merely because a
return target was requested.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import pandas as pd

from bybit_management import BybitTradeManagementExecutor
from quant.edge_intelligence import edge_profile
from quant.evidence_scanner import EvidenceQuantScanner
from quant.exits import TradeManagementPolicy

_EPS = 1e-12
_TARGET_KEY_PREFIX = "bybit_target_plan:"


@dataclass(frozen=True)
class BybitEntryQualityConfig:
    max_context_age_ms: float = 5_000.0
    min_ask_depth_quote: float = 5_000.0
    min_depth_multiple: float = 5.0
    max_impact_proxy_bps: float = 12.0
    resistance_lookback: int = 20
    structure_buffer: float = 0.90
    target_atr_multiple: float = 1.25
    target_r_multiple: float = 1.50
    expected_move_bars: int = 3
    min_target_r: float = 1.0
    min_net_target_r_after_costs: float = 0.75

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in (
            "max_context_age_ms",
            "min_ask_depth_quote",
            "min_depth_multiple",
            "max_impact_proxy_bps",
            "structure_buffer",
            "target_atr_multiple",
            "target_r_multiple",
            "min_target_r",
            "min_net_target_r_after_costs",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{name} must be finite and positive")
        if not (0.0 < self.structure_buffer <= 1.0):
            errors.append("structure_buffer must be in (0, 1]")
        if self.resistance_lookback < 5:
            errors.append("resistance_lookback must be at least 5 bars")
        if self.expected_move_bars < 1:
            errors.append("expected_move_bars must be positive")
        return errors


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _target_plan(
    frame: pd.DataFrame,
    timestamp,
    *,
    entry_price: float,
    stop_price: float,
    atr: float,
    signal_reason: str,
    round_trip_cost_bps: float,
    config: BybitEntryQualityConfig,
) -> dict[str, Any]:
    risk = entry_price - stop_price
    if not math.isfinite(risk) or risk <= 0:
        return {"tradeable": False, "reason": "invalid_initial_risk"}
    ts = pd.Timestamp(timestamp)
    if ts not in frame.index:
        return {"tradeable": False, "reason": "target_timestamp_missing"}
    history = frame.loc[:ts, ["high", "low", "close"]].copy()
    history = history.replace([math.inf, -math.inf], pd.NA).dropna()
    if history.empty or history.index[-1] != ts:
        return {"tradeable": False, "reason": "target_history_invalid"}

    profile = edge_profile("crypto_intraday_adaptive", signal_reason)
    prior = history.iloc[-(config.resistance_lookback + 1) : -1]
    prior_high = float(prior["high"].max()) if not prior.empty else math.nan
    prior_low = float(prior["low"].min()) if not prior.empty else math.nan
    overhead_room = (
        prior_high - entry_price
        if math.isfinite(prior_high) and prior_high > entry_price
        else math.inf
    )

    returns = history["close"].astype(float).pct_change().dropna().tail(32)
    sigma = float(returns.std(ddof=1)) if len(returns) >= 8 else math.nan
    expected_move = (
        entry_price * sigma * math.sqrt(config.expected_move_bars)
        if math.isfinite(sigma) and sigma > 0
        else math.inf
    )

    candidates = {
        "atr": max(_EPS, atr * config.target_atr_multiple),
        "r": risk * config.target_r_multiple,
    }
    if math.isfinite(expected_move) and expected_move > 0:
        candidates["realized_move"] = expected_move
    # A clean breakout above the completed prior range has no measured overhead
    # resistance to cap it. Pullback/reversion/trend entries below a prior swing do.
    if math.isfinite(overhead_room) and profile != "breakout":
        candidates["structure"] = overhead_room * config.structure_buffer

    positive = {
        name: value
        for name, value in candidates.items()
        if value > 0 and math.isfinite(value)
    }
    if not positive:
        return {"tradeable": False, "reason": "no_positive_target_candidate"}
    selected_family, selected_distance = min(positive.items(), key=lambda item: item[1])
    cost_distance = max(0.0, round_trip_cost_bps) / 10_000.0 * entry_price
    target_r = selected_distance / risk
    net_target_r = (selected_distance - cost_distance) / risk
    tradeable = (
        target_r >= config.min_target_r
        and net_target_r >= config.min_net_target_r_after_costs
    )
    reason = "target_room_passed" if tradeable else "insufficient_target_room_after_costs"
    target_fraction = 1.0 if profile in {"mean_reversion", "scalp"} else 0.5
    return {
        "tradeable": tradeable,
        "reason": reason,
        "profile": profile,
        "entry_price": entry_price,
        "initial_stop_price": stop_price,
        "initial_risk": risk,
        "atr": atr,
        "prior_resistance": prior_high if math.isfinite(prior_high) else None,
        "prior_support": prior_low if math.isfinite(prior_low) else None,
        "overhead_room": overhead_room if math.isfinite(overhead_room) else None,
        "round_trip_cost_bps": round_trip_cost_bps,
        "cost_distance": cost_distance,
        "candidate_distances": positive,
        "selected_family": selected_family,
        "selected_distance": selected_distance,
        "target_price": entry_price + selected_distance,
        "target_r": target_r,
        "net_target_r_after_costs": net_target_r,
        "target_fraction": target_fraction,
        "research_family_prices": {
            "0.50pct": entry_price * 1.005,
            "0.75pct": entry_price * 1.0075,
            "1.00pct": entry_price * 1.010,
        },
    }


class BybitEvidenceQuantScanner(EvidenceQuantScanner):
    """Evidence scanner with Bybit-only depth/impact/target-room fail-closed gates."""

    def __init__(
        self,
        *args,
        market_context_provider: Callable[[str], Any],
        target_plan_store,
        bybit_quality_config: BybitEntryQualityConfig = BybitEntryQualityConfig(),
        **kwargs,
    ) -> None:
        errors = bybit_quality_config.validate()
        if errors:
            raise ValueError("Invalid Bybit entry-quality config: " + "; ".join(errors))
        super().__init__(*args, **kwargs)
        self.market_context_provider = market_context_provider
        self.target_plan_store = target_plan_store
        self.bybit_quality_config = bybit_quality_config

    def _reject_execution(self, context: dict[str, Any], reason: str) -> None:
        ranker_input = context.get("ranker_input")
        if ranker_input is None:
            return
        estimate = ranker_input.execution_estimate
        ranker_input.execution_estimate = replace(
            estimate,
            viable=False,
            reason=reason,
        )
        ranker_input.liquidity_quality = 0.0
        ranker_input.setup_quality = 0.0

    def _candidate_from_decision(
        self,
        entry,
        decision,
        health,
        prepared,
        regime,
        drawdown_pct,
        quotes,
        equity,
        now,
    ):
        candidate, context = super()._candidate_from_decision(
            entry,
            decision,
            health,
            prepared,
            regime,
            drawdown_pct,
            quotes,
            equity,
            now,
        )
        if candidate is None:
            return candidate, context
        ranker_input = context.get("ranker_input")
        if ranker_input is None:
            return candidate, context

        reasons: list[str] = []
        market_context = None
        try:
            market_context = self.market_context_provider(decision.symbol)
        except Exception as exc:
            reasons.append("bybit_market_context_unavailable")
            candidate.details["bybit_market_context_error"] = str(exc)[:180]

        notional = max(0.0, float(context.get("notional") or 0.0))
        if market_context is not None:
            snapshot = (
                market_context.to_dict()
                if hasattr(market_context, "to_dict")
                else dict(market_context)
            )
            candidate.details["bybit_microstructure"] = snapshot
            quote_age_ms = _finite(snapshot.get("quote_age_ms"))
            spread_bps = _finite(snapshot.get("spread_bps"))
            ask = _finite(snapshot.get("ask"))
            ask_depth = _finite(snapshot.get("ask_depth"))
            gap = bool(snapshot.get("gap_detected"))
            if gap:
                reasons.append("bybit_orderbook_gap")
            if quote_age_ms is None or quote_age_ms > self.bybit_quality_config.max_context_age_ms:
                reasons.append("bybit_microstructure_stale")
            ask_depth_quote = (
                ask * ask_depth
                if ask is not None and ask_depth is not None and ask > 0 and ask_depth > 0
                else 0.0
            )
            depth_multiple = (
                ask_depth_quote / max(notional, _EPS) if notional > 0 else math.inf
            )
            participation = notional / max(ask_depth_quote, _EPS) if notional > 0 else 0.0
            impact_proxy_bps = (spread_bps or 0.0) * 0.5 + participation * 50.0
            candidate.details["bybit_depth_quote"] = ask_depth_quote
            candidate.details["bybit_depth_multiple"] = depth_multiple
            candidate.details["bybit_impact_proxy_bps"] = impact_proxy_bps
            if ask_depth_quote < self.bybit_quality_config.min_ask_depth_quote:
                reasons.append("bybit_depth_below_absolute_minimum")
            if depth_multiple < self.bybit_quality_config.min_depth_multiple:
                reasons.append("bybit_depth_below_order_multiple")
            if impact_proxy_bps > self.bybit_quality_config.max_impact_proxy_bps:
                reasons.append("bybit_impact_proxy_above_maximum")
            # Positive/neutral order flow never boosts alpha; adverse flow only
            # reduces the existing liquidity multiplier.
            flow = _finite(snapshot.get("signed_trade_flow"))
            imbalance = _finite(snapshot.get("book_imbalance"))
            adverse = 0.0
            if flow is not None and flow < 0:
                adverse += min(1.0, abs(flow)) * 0.20
            if imbalance is not None and imbalance < 0:
                adverse += min(1.0, abs(imbalance)) * 0.20
            ranker_input.liquidity_quality = min(
                float(ranker_input.liquidity_quality),
                max(0.50, 1.0 - adverse),
            )

        frame = prepared.get(decision.symbol)
        entry_price = _finite(candidate.close)
        stop_price = _finite(candidate.stop_price)
        atr = _finite(candidate.atr)
        estimate = ranker_input.execution_estimate
        if frame is None or entry_price is None or stop_price is None or atr is None or atr <= 0:
            plan = {"tradeable": False, "reason": "target_inputs_unavailable"}
        else:
            plan = _target_plan(
                frame,
                decision.timestamp,
                entry_price=entry_price,
                stop_price=stop_price,
                atr=atr,
                signal_reason=candidate.signal_reason,
                round_trip_cost_bps=float(estimate.round_trip_bps),
                config=self.bybit_quality_config,
            )
        candidate.details["bybit_target_plan"] = plan
        if not plan.get("tradeable"):
            reasons.append("bybit_" + str(plan.get("reason") or "target_plan_rejected"))

        if reasons:
            candidate.details["bybit_entry_rejection_reasons"] = sorted(set(reasons))
            self._reject_execution(context, sorted(set(reasons))[0])
        return candidate, context

    def scan(self, *args, **kwargs):
        """Persist target plans only for candidates that actually receive risk.

        Candidate construction happens before global ranking and portfolio allocation.
        Persisting there would make a rejected candidate look like an accepted trade
        plan in the operator UI.  We therefore let the normal scanner complete first,
        then freeze the target only for ``report.executable`` decisions. Rejected
        current candidates clear stale pre-entry plans; held positions are unaffected
        because held symbols do not create BUY candidates in this strategy.
        """
        report = super().scan(*args, **kwargs)
        executable_symbols: set[str] = set()
        for decision in report.executable:
            plan = decision.candidate.details.get("bybit_target_plan")
            if not isinstance(plan, dict) or not plan.get("tradeable"):
                continue
            frozen = dict(plan)
            frozen.update(
                {
                    "frozen_for_execution": True,
                    "decision_timestamp": str(decision.candidate.timestamp),
                    "strategy_id": decision.candidate.strategy_id,
                    "signal_reason": decision.candidate.signal_reason,
                    "risk_allocation_pct": decision.risk_allocation_pct,
                }
            )
            self.target_plan_store.set_kv(
                _TARGET_KEY_PREFIX + decision.candidate.symbol,
                json.dumps(frozen, sort_keys=True, default=str),
            )
            executable_symbols.add(decision.candidate.symbol)

        for decision in report.decisions:
            details = getattr(decision.candidate, "details", {}) or {}
            if "bybit_target_plan" not in details:
                continue
            symbol = decision.candidate.symbol
            if symbol not in executable_symbols:
                self.target_plan_store.delete_kv(_TARGET_KEY_PREFIX + symbol)
        return report


class TargetAwareBybitTradeManagementExecutor(BybitTradeManagementExecutor):
    """Feeds the frozen Bybit target plan into the existing exit policy."""

    def manage_symbol(self, symbol: str, frame: pd.DataFrame, timestamp: pd.Timestamp):
        raw = self.engine.store.get_kv(_TARGET_KEY_PREFIX + symbol)
        original_policy = self.policy
        try:
            if raw:
                try:
                    plan = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    plan = None
                if (
                    isinstance(plan, dict)
                    and plan.get("tradeable")
                    and plan.get("frozen_for_execution")
                ):
                    target_r = _finite(plan.get("target_r"))
                    target_fraction = _finite(plan.get("target_fraction"))
                    if (
                        target_r is not None
                        and target_r > 0
                        and target_fraction is not None
                        and 0 < target_fraction <= 1
                    ):
                        config = replace(
                            original_policy.config,
                            enable_profit_target=True,
                            profit_target_r=target_r,
                            profit_target_fraction=target_fraction,
                        )
                        self.policy = TradeManagementPolicy(config)
            return super().manage_symbol(symbol, frame, timestamp)
        finally:
            self.policy = original_policy


def wire_bybit_quality_layers(service) -> None:
    """Production composition hook called before the Bybit engine starts."""
    scanner = BybitEvidenceQuantScanner(
        store=service.quant_store,
        lifecycle=service.lifecycle,
        execution_engine=service.engine.execution_quality,
        market_context_provider=service.feed.market_context,
        target_plan_store=service.engine_store,
    )
    service.engine.quant_scanner = scanner
    previous = service.engine.trade_manager
    service.engine.trade_manager = TargetAwareBybitTradeManagementExecutor(
        service.engine,
        service.broker,
        service.venue_state,
        previous.policy,
        strategy_id=previous.strategy_id,
    )
