"""Deterministic regime-aware multi-strategy ensemble for Alpaca spot crypto.

The ensemble is deliberately transparent: it combines the existing audited
breakout strategy with causal pullback and mean-reversion sleeves, then gates
new exposure by a cross-sectional market-regime classifier. It never creates
short exposure, leverage, averaging-down, or looser exits.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional

import pandas as pd

from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from crypto_strategy import CryptoRegimeStrategy
from models import BotAction


class MarketRegime(str, Enum):
    """Cross-sectional market state used to route strategy sleeves."""

    BULL_EXPANSION = "BULL_EXPANSION"
    BULL_NORMAL = "BULL_NORMAL"
    RANGE = "RANGE"
    RISK_OFF = "RISK_OFF"
    STRESS = "STRESS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EnsembleConfig:
    """Immutable controls for the second-generation decision layer."""

    min_buy_score: float = 0.52
    min_regime_confidence: float = 0.45
    stress_atr_percentile: float = 93.0
    min_breadth_for_trend: float = 0.50
    pullback_max_distance_atr: float = 0.90
    mean_reversion_max_rsi: float = 44.0
    mean_reversion_max_distance_atr: float = 1.20
    require_positive_benchmark_momentum: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        finite = {
            "min_buy_score": self.min_buy_score,
            "min_regime_confidence": self.min_regime_confidence,
            "stress_atr_percentile": self.stress_atr_percentile,
            "min_breadth_for_trend": self.min_breadth_for_trend,
            "pullback_max_distance_atr": self.pullback_max_distance_atr,
            "mean_reversion_max_rsi": self.mean_reversion_max_rsi,
            "mean_reversion_max_distance_atr": self.mean_reversion_max_distance_atr,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                errors.append(f"{name} must be finite.")
        if not 0 < self.min_buy_score <= 1:
            errors.append("min_buy_score must be in (0, 1].")
        if not 0 <= self.min_regime_confidence <= 1:
            errors.append("min_regime_confidence must be in [0, 1].")
        if not 0 < self.stress_atr_percentile <= 100:
            errors.append("stress_atr_percentile must be in (0, 100].")
        if not 0 <= self.min_breadth_for_trend <= 1:
            errors.append("min_breadth_for_trend must be in [0, 1].")
        if self.pullback_max_distance_atr <= 0:
            errors.append("pullback_max_distance_atr must be positive.")
        if not 0 < self.mean_reversion_max_rsi < 100:
            errors.append("mean_reversion_max_rsi must be in (0, 100).")
        if self.mean_reversion_max_distance_atr <= 0:
            errors.append("mean_reversion_max_distance_atr must be positive.")
        return errors


@dataclass(frozen=True)
class RegimeSnapshot:
    timestamp: Optional[datetime]
    regime: MarketRegime
    confidence: float
    breadth: float
    benchmark_bull: bool
    benchmark_momentum: float
    benchmark_adx: float
    benchmark_efficiency: float
    benchmark_atr_percentile: float
    dispersion: float

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "regime": self.regime.value,
            "confidence": self.confidence,
            "breadth": self.breadth,
            "benchmark_bull": self.benchmark_bull,
            "benchmark_momentum": self.benchmark_momentum,
            "benchmark_adx": self.benchmark_adx,
            "benchmark_efficiency": self.benchmark_efficiency,
            "benchmark_atr_percentile": self.benchmark_atr_percentile,
            "dispersion": self.dispersion,
        }


def _finite(value, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _triangle(value: float, low: float, center: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == center:
        return 1.0
    if value < center:
        return _clip((value - low) / max(center - low, 1e-12))
    return _clip((high - value) / max(high - center, 1e-12))


class RegimeAwareEnsembleStrategy:
    """Drop-in strategy implementing regime routing and multi-sleeve scoring.

    The public methods intentionally mirror ``CryptoRegimeStrategy`` so the same
    live engine and audited backtester can use this object without alternate
    execution logic.
    """

    def __init__(
        self,
        strategy_config: CryptoStrategyConfig,
        ensemble_config: Optional[EnsembleConfig] = None,
    ):
        self.strategy_config = strategy_config
        self.base = CryptoRegimeStrategy(strategy_config)
        self.ensemble_config = ensemble_config or EnsembleConfig()
        errors = self.ensemble_config.validate()
        if errors:
            raise ValueError("Invalid ensemble configuration: " + "; ".join(errors))

    def required_bars(self) -> int:
        return self.base.required_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.base.prepare_symbol(bars)

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        return self.base.prepare_universe(bars_by_symbol)

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        # Exit/protective-stop behavior remains exactly the audited base logic.
        return self.base.ratchet_stop(
            current_stop,
            highest_completed_close,
            completed_close,
            current_atr,
        )

    def evaluate(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        prepared = self.prepare_universe(bars_by_symbol)
        common: Optional[pd.DatetimeIndex] = None
        for frame in prepared.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            return self.base.evaluate(bars_by_symbol, held_symbols)
        return self.evaluate_prepared(prepared, common[-1], held_symbols)

    def classify_regime(
        self,
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp | datetime,
    ) -> RegimeSnapshot:
        ts = pd.Timestamp(timestamp)
        benchmark_symbol = self.strategy_config.benchmark_symbol
        benchmark = prepared.get(benchmark_symbol)
        if benchmark is None or ts not in benchmark.index:
            return RegimeSnapshot(
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else None,
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                breadth=0.0,
                benchmark_bull=False,
                benchmark_momentum=0.0,
                benchmark_adx=0.0,
                benchmark_efficiency=0.0,
                benchmark_atr_percentile=100.0,
                dispersion=0.0,
            )

        row = benchmark.loc[ts]
        close = _finite(row.get("close"))
        fast = _finite(row.get("fast_ma"))
        slow = _finite(row.get("slow_ma"))
        regime_ma = _finite(row.get("regime_ma"))
        slope = _finite(row.get("regime_slope"))
        momentum = _finite(row.get("momentum"))
        adx = _finite(row.get("adx"))
        efficiency = _finite(row.get("efficiency_ratio"))
        atr_pct = _finite(row.get("atr_percentile"), 100.0)
        benchmark_flag = row.get("benchmark_bull")
        benchmark_bull = bool(benchmark_flag) if not pd.isna(benchmark_flag) else False

        breadth_members = 0
        breadth_bull = 0
        momenta: list[float] = []
        for frame in prepared.values():
            if ts not in frame.index:
                continue
            member = frame.loc[ts]
            member_close = _finite(member.get("close"))
            member_fast = _finite(member.get("fast_ma"))
            member_slow = _finite(member.get("slow_ma"))
            member_momentum = _finite(member.get("momentum"))
            if member_close <= 0 or member_fast <= 0 or member_slow <= 0:
                continue
            breadth_members += 1
            if member_close > member_slow and member_fast > member_slow:
                breadth_bull += 1
            momenta.append(member_momentum)
        breadth = breadth_bull / breadth_members if breadth_members else 0.0
        dispersion = (
            float(pd.Series(momenta, dtype=float).std(ddof=0))
            if len(momenta) > 1
            else 0.0
        )

        structural_bull = (
            benchmark_bull
            and close > 0
            and regime_ma > 0
            and close > regime_ma
            and slope > 0
        )
        trend_stack = close > fast > slow > 0
        stress = atr_pct >= self.ensemble_config.stress_atr_percentile
        momentum_ok = (
            momentum > 0
            if self.ensemble_config.require_positive_benchmark_momentum
            else True
        )

        if stress:
            regime = MarketRegime.STRESS
        elif not structural_bull or not momentum_ok:
            regime = MarketRegime.RISK_OFF
        elif (
            trend_stack
            and adx >= 25.0
            and efficiency >= 0.30
            and breadth >= self.ensemble_config.min_breadth_for_trend
        ):
            regime = MarketRegime.BULL_EXPANSION
        elif benchmark_bull and breadth >= self.ensemble_config.min_breadth_for_trend:
            regime = MarketRegime.BULL_NORMAL
        else:
            regime = MarketRegime.RANGE

        trend_strength = _clip(adx / 40.0)
        efficiency_strength = _clip(efficiency / 0.50)
        breadth_strength = _clip(breadth)
        slope_strength = _clip(max(slope, 0.0) / 0.02)
        vol_quality = 1.0 - _clip(atr_pct / 100.0)

        if regime in {MarketRegime.BULL_EXPANSION, MarketRegime.BULL_NORMAL}:
            confidence = (
                0.30 * trend_strength
                + 0.25 * efficiency_strength
                + 0.25 * breadth_strength
                + 0.10 * slope_strength
                + 0.10 * vol_quality
            )
        elif regime is MarketRegime.RANGE:
            range_adx = 1.0 - _clip(adx / 35.0)
            confidence = (
                0.35 * range_adx
                + 0.25 * breadth_strength
                + 0.20 * efficiency_strength
                + 0.20 * vol_quality
            )
        elif regime in {MarketRegime.RISK_OFF, MarketRegime.STRESS}:
            confidence = max(
                _clip(1.0 - breadth),
                _clip(atr_pct / 100.0),
                _clip(abs(min(slope, 0.0)) / 0.02),
            )
        else:
            confidence = 0.0

        return RegimeSnapshot(
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else None,
            regime=regime,
            confidence=_clip(confidence),
            breadth=_clip(breadth),
            benchmark_bull=benchmark_bull,
            benchmark_momentum=momentum,
            benchmark_adx=adx,
            benchmark_efficiency=efficiency,
            benchmark_atr_percentile=atr_pct,
            dispersion=max(0.0, dispersion),
        )

    def evaluate_prepared(
        self,
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp | datetime,
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        ts = pd.Timestamp(timestamp)
        base_decisions = self.base.evaluate_prepared(prepared, ts, held_symbols)
        base_by_symbol = {decision.symbol: decision for decision in base_decisions}
        snapshot = self.classify_regime(prepared, ts)

        decisions: list[CryptoSignalDecision] = []
        for symbol in self.strategy_config.symbols:
            base = base_by_symbol.get(symbol)
            if base is None:
                base = CryptoSignalDecision(
                    symbol=symbol,
                    action=BotAction.HOLD,
                    timestamp=ts.to_pydatetime(),
                    reason="base_decision_missing",
                )

            # Never suppress an audited exit. Defensive actions dominate entries.
            if symbol in held_symbols or base.action is BotAction.EXIT:
                decisions.append(self._annotate(base, snapshot, {}))
                continue

            if snapshot.regime in {
                MarketRegime.UNKNOWN,
                MarketRegime.RISK_OFF,
                MarketRegime.STRESS,
            }:
                blocked = copy.deepcopy(base)
                blocked.action = BotAction.HOLD
                blocked.reason = "ensemble_regime_blocks_entry"
                blocked.score = 0.0
                decisions.append(self._annotate(blocked, snapshot, {
                    "breakout": 0.0,
                    "pullback": 0.0,
                    "mean_reversion": 0.0,
                    "combined": 0.0,
                }))
                continue

            frame = prepared.get(symbol)
            if frame is None or ts not in frame.index:
                decisions.append(self._annotate(base, snapshot, {}))
                continue

            breakout = self._breakout_score(base)
            pullback = self._pullback_score(frame, ts)
            mean_reversion = self._mean_reversion_score(frame, ts, snapshot)
            weights = self._weights(snapshot.regime)
            combined = _clip(
                weights["breakout"] * breakout
                + weights["pullback"] * pullback
                + weights["mean_reversion"] * mean_reversion
            )
            sleeve_scores = {
                "breakout": breakout,
                "pullback": pullback,
                "mean_reversion": mean_reversion,
                "combined": combined,
            }

            if (
                combined >= self.ensemble_config.min_buy_score
                and snapshot.confidence >= self.ensemble_config.min_regime_confidence
            ):
                row = frame.loc[ts]
                close = _finite(row.get("close"))
                atr_value = _finite(row.get("atr"))
                if close > 0 and atr_value > 0:
                    decision = copy.deepcopy(base)
                    decision.action = BotAction.BUY
                    decision.reason = self._entry_reason(sleeve_scores)
                    decision.score = combined
                    decision.close = close
                    decision.atr = atr_value
                    decision.stop_price = close - (
                        atr_value * self.strategy_config.atr_stop_multiplier
                    )
                    decision.regime = (
                        CryptoRegime.BULL_TREND
                        if snapshot.regime
                        in {MarketRegime.BULL_EXPANSION, MarketRegime.BULL_NORMAL}
                        else CryptoRegime.RANGE
                    )
                    decisions.append(self._annotate(decision, snapshot, sleeve_scores))
                    continue

            hold = copy.deepcopy(base)
            hold.action = BotAction.HOLD
            hold.reason = "ensemble_score_below_entry_threshold"
            hold.score = combined
            decisions.append(self._annotate(hold, snapshot, sleeve_scores))

        buys = sorted(
            (decision for decision in decisions if decision.action is BotAction.BUY),
            key=lambda decision: (-(decision.score or -math.inf), decision.symbol),
        )
        non_buys = [
            decision for decision in decisions if decision.action is not BotAction.BUY
        ]
        return buys + non_buys

    def _breakout_score(self, base: CryptoSignalDecision) -> float:
        if base.action is BotAction.BUY:
            return _clip(0.85 + 0.15 * _clip(_finite(base.score)))
        if not base.filters.get("new_breakout", False):
            return 0.0
        if not base.filters:
            return 0.0
        quality = sum(bool(value) for value in base.filters.values()) / len(base.filters)
        return _clip(0.25 + 0.60 * quality)

    def _pullback_score(self, frame: pd.DataFrame, ts: pd.Timestamp) -> float:
        row = frame.loc[ts]
        close = _finite(row.get("close"))
        fast = _finite(row.get("fast_ma"))
        slow = _finite(row.get("slow_ma"))
        atr_value = _finite(row.get("atr"))
        rsi_value = _finite(row.get("rsi"))
        adx_value = _finite(row.get("adx"))
        efficiency = _finite(row.get("efficiency_ratio"))
        volume = _finite(row.get("volume_ratio"))
        momentum = _finite(row.get("momentum"))
        benchmark_flag = row.get("benchmark_bull")
        benchmark_bull = bool(benchmark_flag) if not pd.isna(benchmark_flag) else False

        if (
            min(close, fast, slow, atr_value) <= 0
            or not benchmark_bull
            or fast <= slow
            or close <= slow
            or momentum < -0.01
        ):
            return 0.0
        distance_atr = abs(close - fast) / atr_value
        if distance_atr > self.ensemble_config.pullback_max_distance_atr:
            return 0.0
        distance_quality = 1.0 - _clip(
            distance_atr / self.ensemble_config.pullback_max_distance_atr
        )
        rsi_quality = _triangle(rsi_value, 42.0, 54.0, 68.0)
        adx_quality = _clip((adx_value - 12.0) / 25.0)
        efficiency_quality = _clip(efficiency / 0.45)
        volume_quality = _clip(volume / 1.5)
        momentum_quality = _clip((momentum + 0.01) / 0.08)
        return _clip(
            0.25 * distance_quality
            + 0.20 * rsi_quality
            + 0.20 * adx_quality
            + 0.15 * efficiency_quality
            + 0.10 * volume_quality
            + 0.10 * momentum_quality
        )

    def _mean_reversion_score(
        self,
        frame: pd.DataFrame,
        ts: pd.Timestamp,
        snapshot: RegimeSnapshot,
    ) -> float:
        if snapshot.regime not in {MarketRegime.RANGE, MarketRegime.BULL_NORMAL}:
            return 0.0
        row = frame.loc[ts]
        close = _finite(row.get("close"))
        fast = _finite(row.get("fast_ma"))
        slow = _finite(row.get("slow_ma"))
        atr_value = _finite(row.get("atr"))
        rsi_value = _finite(row.get("rsi"))
        atr_pct = _finite(row.get("atr_percentile"), 100.0)
        volume = _finite(row.get("volume_ratio"))
        benchmark_flag = row.get("benchmark_bull")
        benchmark_bull = bool(benchmark_flag) if not pd.isna(benchmark_flag) else False
        if (
            min(close, fast, slow, atr_value) <= 0
            or not benchmark_bull
            or close < slow * 0.985
            or close >= fast
            or rsi_value > self.ensemble_config.mean_reversion_max_rsi
            or atr_pct >= self.ensemble_config.stress_atr_percentile
        ):
            return 0.0

        deviation_atr = (fast - close) / atr_value
        if deviation_atr <= 0 or deviation_atr > self.ensemble_config.mean_reversion_max_distance_atr:
            return 0.0
        deviation_quality = _clip(
            deviation_atr / self.ensemble_config.mean_reversion_max_distance_atr
        )
        oversold_quality = _clip(
            (self.ensemble_config.mean_reversion_max_rsi - rsi_value) / 14.0
        )
        vol_quality = 1.0 - _clip(atr_pct / self.ensemble_config.stress_atr_percentile)
        volume_quality = _clip(volume / 1.5)
        rebound_quality = 0.0
        location = int(frame.index.get_indexer([ts])[0])
        if location > 0:
            previous_close = _finite(frame.iloc[location - 1].get("close"))
            if previous_close > 0 and close > previous_close:
                rebound_quality = _clip((close / previous_close - 1.0) / 0.02)
        return _clip(
            0.30 * deviation_quality
            + 0.25 * oversold_quality
            + 0.20 * rebound_quality
            + 0.15 * vol_quality
            + 0.10 * volume_quality
        )

    @staticmethod
    def _weights(regime: MarketRegime) -> dict[str, float]:
        if regime is MarketRegime.BULL_EXPANSION:
            return {"breakout": 0.70, "pullback": 0.30, "mean_reversion": 0.00}
        if regime is MarketRegime.BULL_NORMAL:
            return {"breakout": 0.45, "pullback": 0.45, "mean_reversion": 0.10}
        if regime is MarketRegime.RANGE:
            return {"breakout": 0.10, "pullback": 0.25, "mean_reversion": 0.65}
        return {"breakout": 0.0, "pullback": 0.0, "mean_reversion": 0.0}

    @staticmethod
    def _entry_reason(scores: Mapping[str, float]) -> str:
        dominant = max(
            ("breakout", "pullback", "mean_reversion"),
            key=lambda name: (scores.get(name, 0.0), name),
        )
        return f"regime_ensemble_{dominant}_entry"

    @staticmethod
    def _annotate(
        decision: CryptoSignalDecision,
        snapshot: RegimeSnapshot,
        sleeve_scores: Mapping[str, float],
    ) -> CryptoSignalDecision:
        out = copy.deepcopy(decision)
        details = dict(out.details or {})
        details["ensemble"] = {
            "regime": snapshot.to_dict(),
            "sleeves": {key: float(value) for key, value in sleeve_scores.items()},
        }
        out.details = details
        return out
