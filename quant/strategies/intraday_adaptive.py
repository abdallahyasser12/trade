"""Regime-adaptive intraday crypto alpha ensemble.

This is the second-generation high-velocity hypothesis.  It deliberately avoids a
single always-on signal.  Every completed bar is decomposed into four causal sleeves:

* cross-sectional momentum continuation;
* prior-range breakout/volatility expansion;
* trend pullback continuation;
* range mean reversion with reversal confirmation.

A probabilistic market router built only from information available on the completed
bar controls how much each sleeve can contribute.  Stress/risk-off conditions can
block new exposure completely.  Quotes/order-book information is intentionally *not*
used here: live microstructure belongs in the execution/scanner layer so research and
production share the same bar-driven alpha hypothesis without look-ahead.

The strategy is long-only, deterministic, causal and implements the shared
``UniverseStrategy`` contract.  It is a hypothesis, not a profitability claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import atr
from models import BotAction


_EPS = 1e-12


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class AdaptiveIntradayConfig:
    """Structural parameters only; keep the research grid intentionally bounded."""

    fast_lookback: int = 3
    medium_lookback: int = 12
    slow_lookback: int = 36
    vol_lookback: int = 48
    volume_lookback: int = 24
    breakout_lookback: int = 20
    reversion_lookback: int = 24
    trend_fast_span: int = 18
    trend_slow_span: int = 54
    atr_baseline_lookback: int = 72
    top_k: int = 4
    exit_rank_buffer: int = 2
    min_entry_score: float = 0.24
    exit_score: float = 0.08
    min_volume_ratio: float = 0.65
    max_atr_ratio: float = 2.75
    stop_atr_multiplier: float = 1.65
    trailing_atr_multiplier: float = 2.05
    stress_block_probability: float = 0.70
    min_trend_probability: float = 0.42

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.fast_lookback < 1:
            errors.append("fast_lookback must be positive.")
        if not self.fast_lookback < self.medium_lookback < self.slow_lookback:
            errors.append("lookbacks must satisfy fast < medium < slow.")
        if self.vol_lookback < self.slow_lookback:
            errors.append("vol_lookback must cover slow_lookback.")
        if self.volume_lookback < 5 or self.breakout_lookback < 5:
            errors.append("volume/breakout lookbacks must be at least 5 bars.")
        if self.reversion_lookback < 10:
            errors.append("reversion_lookback must be at least 10 bars.")
        if not 2 <= self.trend_fast_span < self.trend_slow_span:
            errors.append("trend spans must satisfy 2 <= fast < slow.")
        if self.atr_baseline_lookback < 20:
            errors.append("atr_baseline_lookback must be at least 20 bars.")
        if self.top_k < 1 or self.exit_rank_buffer < 0:
            errors.append("top_k must be positive and exit_rank_buffer non-negative.")
        for label, value in {
            "min_entry_score": self.min_entry_score,
            "exit_score": self.exit_score,
            "min_volume_ratio": self.min_volume_ratio,
            "max_atr_ratio": self.max_atr_ratio,
            "stop_atr_multiplier": self.stop_atr_multiplier,
            "trailing_atr_multiplier": self.trailing_atr_multiplier,
            "stress_block_probability": self.stress_block_probability,
            "min_trend_probability": self.min_trend_probability,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if not 0.0 <= self.exit_score < self.min_entry_score <= 1.0:
            errors.append("scores must satisfy 0 <= exit_score < min_entry_score <= 1.")
        if self.min_volume_ratio <= 0 or self.max_atr_ratio <= 1:
            errors.append("volume floor must be positive and max_atr_ratio > 1.")
        if self.stop_atr_multiplier <= 0 or self.trailing_atr_multiplier <= 0:
            errors.append("ATR stop multipliers must be positive.")
        if not 0.0 <= self.stress_block_probability <= 1.0:
            errors.append("stress_block_probability must be in [0,1].")
        if not 0.0 <= self.min_trend_probability <= 1.0:
            errors.append("min_trend_probability must be in [0,1].")
        return errors


class AdaptiveIntradayStrategy:
    """Causal regime router + multi-sleeve intraday alpha hypothesis."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        adaptive: Optional[AdaptiveIntradayConfig] = None,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        self.adaptive = adaptive or AdaptiveIntradayConfig()
        errors = self.adaptive.validate()
        if errors:
            raise ValueError("Invalid adaptive intraday configuration: " + "; ".join(errors))
        if self.adaptive.top_k > len(config.symbols):
            raise ValueError("top_k cannot exceed configured universe size.")

    def required_bars(self) -> int:
        c = self.adaptive
        return max(
            c.vol_lookback + c.slow_lookback + 3,
            c.atr_baseline_lookback + self.config.atr_period + 3,
            c.trend_slow_span * 2,
            c.breakout_lookback + 3,
            c.reversion_lookback + 3,
            c.volume_lookback + 3,
        )

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Adaptive intraday bars require OHLCV columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Adaptive intraday requires a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Adaptive intraday bars must be chronological and unique.")

        c = self.adaptive
        out = bars.copy()
        close = out["close"].astype(float)
        out["atr"] = atr(out["high"], out["low"], close, self.config.atr_period)

        log_return = np.log(close / close.shift(1))
        sigma = log_return.rolling(c.vol_lookback, min_periods=c.vol_lookback).std(ddof=1)
        out["realized_vol"] = sigma

        for name, lookback in (
            ("fast", c.fast_lookback),
            ("medium", c.medium_lookback),
            ("slow", c.slow_lookback),
        ):
            raw = close / close.shift(lookback) - 1.0
            noise = sigma * math.sqrt(lookback)
            out[f"{name}_momentum"] = raw
            out[f"{name}_z"] = raw.where(noise > 0) / noise

        out["alpha_core"] = (
            0.50 * out["fast_z"]
            + 0.32 * out["medium_z"]
            + 0.18 * out["slow_z"]
        ).replace([np.inf, -np.inf], np.nan)

        out["trend_fast"] = close.ewm(span=c.trend_fast_span, adjust=False).mean()
        out["trend_slow"] = close.ewm(span=c.trend_slow_span, adjust=False).mean()
        out["trend_atr"] = (
            (out["trend_fast"] - out["trend_slow"])
            / out["atr"].where(out["atr"] > 0)
        ).replace([np.inf, -np.inf], np.nan)

        prior_high = out["high"].shift(1).rolling(
            c.breakout_lookback, min_periods=c.breakout_lookback
        ).max()
        out["prior_range_high"] = prior_high
        out["breakout_atr"] = (
            (close - prior_high) / out["atr"].where(out["atr"] > 0)
        ).replace([np.inf, -np.inf], np.nan)

        mean = close.rolling(c.reversion_lookback, min_periods=c.reversion_lookback).mean()
        std = close.rolling(c.reversion_lookback, min_periods=c.reversion_lookback).std(ddof=1)
        out["reversion_z"] = ((close - mean) / std.where(std > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["reversal_confirmation"] = (close > close.shift(1)).astype(float)

        avg_volume = out["volume"].rolling(
            c.volume_lookback, min_periods=c.volume_lookback
        ).median()
        out["volume_ratio"] = out["volume"] / avg_volume.where(avg_volume > 0)

        atr_baseline = out["atr"].rolling(
            c.atr_baseline_lookback, min_periods=c.atr_baseline_lookback
        ).median()
        out["atr_ratio"] = out["atr"] / atr_baseline.where(atr_baseline > 0)
        out["compression"] = (
            1.0 / out["atr_ratio"].where(out["atr_ratio"] > 0)
        ).clip(lower=0.0, upper=2.0)
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing adaptive intraday data for {sorted(missing)}")
        prepared = {
            symbol: self.prepare_symbol(bars_by_symbol[symbol])
            for symbol in self.config.symbols
        }

        alpha = pd.DataFrame(
            {symbol: frame["alpha_core"] for symbol, frame in prepared.items()}
        )
        ranks = alpha.rank(axis=1, ascending=False, method="first", na_option="keep")
        denominator = max(1, len(self.config.symbols) - 1)

        bullish_members = pd.DataFrame(
            {
                symbol: (
                    (frame["close"] > frame["trend_slow"])
                    & (frame["trend_fast"] > frame["trend_slow"])
                ).astype(float)
                for symbol, frame in prepared.items()
            }
        )
        breadth = bullish_members.mean(axis=1)
        dispersion = alpha.std(axis=1, ddof=0)

        benchmark = prepared[self.config.benchmark_symbol]
        bench_trend = benchmark["trend_atr"].fillna(0.0)
        bench_momentum = benchmark["medium_z"].fillna(0.0)
        bench_atr_ratio = benchmark["atr_ratio"].fillna(1.0)
        router_raw = 0.90 * bench_trend + 0.55 * bench_momentum + 1.60 * (breadth - 0.50)
        bull_probability = 1.0 / (1.0 + np.exp(-router_raw.clip(-20, 20)))
        stress_probability = ((bench_atr_ratio - 1.35) / 1.35).clip(lower=0.0, upper=1.0)
        range_probability = (
            (1.0 - stress_probability)
            * (1.0 - (2.0 * (bull_probability - 0.50).abs()).clip(upper=1.0))
        ).clip(lower=0.0, upper=1.0)
        risk_off_probability = (
            (1.0 - stress_probability - range_probability).clip(lower=0.0)
            * (1.0 - bull_probability)
        ).clip(lower=0.0, upper=1.0)

        for symbol, frame in prepared.items():
            frame["rank"] = ranks[symbol].reindex(frame.index)
            frame["leader_quality"] = (
                1.0 - (frame["rank"] - 1.0) / denominator
            ).clip(lower=0.0, upper=1.0)
            frame["market_breadth"] = breadth.reindex(frame.index)
            frame["cross_sectional_dispersion"] = dispersion.reindex(frame.index)
            frame["bull_probability"] = bull_probability.reindex(frame.index)
            frame["range_probability"] = range_probability.reindex(frame.index)
            frame["stress_probability"] = stress_probability.reindex(frame.index)
            frame["risk_off_probability"] = risk_off_probability.reindex(frame.index)
        return prepared

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
            return [self._hold(symbol, None, "market_context_unavailable") for symbol in self.config.symbols]
        return self.evaluate_prepared(prepared, common[-1], held_symbols)

    def evaluate_prepared(
        self,
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp | datetime,
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        ts = pd.Timestamp(timestamp)
        decisions = [
            self._evaluate_symbol(symbol, prepared[symbol], ts, symbol in held_symbols)
            for symbol in self.config.symbols
        ]
        buys = sorted(
            (d for d in decisions if d.action is BotAction.BUY),
            key=lambda d: (-(d.score if d.score is not None else -math.inf), d.symbol),
        )
        return buys + [d for d in decisions if d.action is not BotAction.BUY]

    def _evaluate_symbol(self, symbol: str, frame: pd.DataFrame, ts: pd.Timestamp, held: bool):
        if ts not in frame.index:
            return self._hold(symbol, ts, "bar_unavailable")
        row = frame.loc[ts]
        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, ts, "indicators_not_initialized")

        values = {
            name: _finite(row.get(name))
            for name in (
                "fast_z",
                "medium_z",
                "slow_z",
                "trend_atr",
                "breakout_atr",
                "reversion_z",
                "volume_ratio",
                "atr_ratio",
                "rank",
                "leader_quality",
                "market_breadth",
                "cross_sectional_dispersion",
                "bull_probability",
                "range_probability",
                "stress_probability",
                "risk_off_probability",
                "reversal_confirmation",
                "compression",
            )
        }
        if any(
            values[key] is None
            for key in (
                "fast_z",
                "medium_z",
                "trend_atr",
                "reversion_z",
                "volume_ratio",
                "atr_ratio",
                "rank",
                "leader_quality",
                "bull_probability",
                "range_probability",
                "stress_probability",
                "reversal_confirmation",
            )
        ):
            return self._hold(symbol, ts, "insufficient_history_for_adaptive_router")

        c = self.adaptive
        bull = _clip(values["bull_probability"] or 0.0)
        range_p = _clip(values["range_probability"] or 0.0)
        stress = _clip(values["stress_probability"] or 0.0)
        risk_off = _clip(values["risk_off_probability"] or 0.0)
        leader = _clip(values["leader_quality"] or 0.0)
        volume_ratio = values["volume_ratio"] or 0.0
        atr_ratio = values["atr_ratio"] or 99.0
        fast_z = values["fast_z"] or 0.0
        medium_z = values["medium_z"] or 0.0
        slow_z = values["slow_z"] or 0.0
        trend_atr = values["trend_atr"] or 0.0
        breakout_atr = values["breakout_atr"] or 0.0
        reversion_z = values["reversion_z"] or 0.0
        reversal = values["reversal_confirmation"] or 0.0
        compression = values["compression"] or 0.0

        volume_quality = _clip((volume_ratio - 0.50) / 1.25)
        volatility_quality = _clip((c.max_atr_ratio - atr_ratio) / max(_EPS, c.max_atr_ratio - 1.0))
        trend_strength = _sigmoid(0.75 * trend_atr + 0.45 * medium_z)
        momentum_strength = _sigmoid(0.70 * fast_z + 0.55 * medium_z + 0.25 * slow_z)

        momentum = leader * momentum_strength * bull * volume_quality
        breakout = (
            leader
            * bull
            * volume_quality
            * _clip(max(0.0, breakout_atr) / 1.25)
            * _clip(0.50 + 0.35 * compression)
        )
        pullback_shape = _clip((-fast_z + 0.25) / 1.75) * _clip((medium_z + 0.25) / 1.75)
        pullback = leader * bull * trend_strength * pullback_shape * volume_quality
        oversold = _clip((-reversion_z - 0.35) / 2.15)
        reversion = range_p * oversold * _clip(0.25 + 0.75 * reversal) * volume_quality

        trend_route = 0.48 * momentum + 0.30 * breakout + 0.22 * pullback
        range_route = 0.08 * momentum + 0.07 * breakout + 0.15 * pullback + 0.70 * reversion
        route_mass = max(_EPS, bull + range_p)
        combined = (
            (bull * trend_route + range_p * range_route) / route_mass
        ) * volatility_quality * (1.0 - stress)
        combined = _clip(combined)

        regime = (
            CryptoRegime.STRESS
            if stress >= c.stress_block_probability
            else CryptoRegime.BULL_TREND
            if bull >= max(range_p, c.min_trend_probability)
            else CryptoRegime.RANGE
        )
        details = {
            "adaptive_version": "2.0.0",
            "momentum_sleeve": momentum,
            "breakout_sleeve": breakout,
            "pullback_sleeve": pullback,
            "reversion_sleeve": reversion,
            "trend_route": trend_route,
            "range_route": range_route,
            "bull_probability": bull,
            "range_probability": range_p,
            "stress_probability": stress,
            "risk_off_probability": risk_off,
            "leader_quality": leader,
            "rank": values["rank"],
            "market_breadth": values["market_breadth"],
            "cross_sectional_dispersion": values["cross_sectional_dispersion"],
            "atr_ratio": atr_ratio,
            "reversion_z": reversion_z,
        }

        if held:
            if stress >= c.stress_block_probability:
                return self._decision(symbol, BotAction.EXIT, ts, "adaptive_stress_exit", regime, close, atr_value, combined, details)
            if fast_z < -1.10 and medium_z < -0.20:
                return self._decision(symbol, BotAction.EXIT, ts, "adaptive_momentum_failure", regime, close, atr_value, combined, details)
            if range_p > bull and reversion_z >= -0.05:
                return self._decision(symbol, BotAction.EXIT, ts, "adaptive_mean_recovered", regime, close, atr_value, combined, details)
            if (values["rank"] or math.inf) > c.top_k + c.exit_rank_buffer and bull >= range_p:
                return self._decision(symbol, BotAction.EXIT, ts, "adaptive_relative_strength_decay", regime, close, atr_value, combined, details)
            if combined < c.exit_score:
                return self._decision(symbol, BotAction.EXIT, ts, "adaptive_edge_decay", regime, close, atr_value, combined, details)
            return self._decision(symbol, BotAction.HOLD, ts, "adaptive_edge_intact", regime, close, atr_value, combined, details)

        filters = {
            "stress": stress < c.stress_block_probability,
            "risk_off": risk_off < 0.70,
            "volume": volume_ratio >= c.min_volume_ratio,
            "volatility": atr_ratio <= c.max_atr_ratio,
            "score": combined >= c.min_entry_score,
            "opportunity": (
                (values["rank"] or math.inf) <= c.top_k
                or reversion >= c.min_entry_score
            ),
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol,
                BotAction.HOLD,
                ts,
                "adaptive_blocked_" + "_".join(failed),
                regime,
                close,
                atr_value,
                combined,
                details,
                filters=filters,
            )

        stop = close - atr_value * c.stop_atr_multiplier
        if stop <= 0 or stop >= close:
            return self._hold(symbol, ts, "invalid_adaptive_atr_stop")
        dominant = max(
            {
                "momentum": momentum,
                "breakout": breakout,
                "pullback": pullback,
                "reversion": reversion,
            },
            key=lambda name: {
                "momentum": momentum,
                "breakout": breakout,
                "pullback": pullback,
                "reversion": reversion,
            }[name],
        )
        details["dominant_sleeve"] = dominant
        return self._decision(
            symbol,
            BotAction.BUY,
            ts,
            f"adaptive_{dominant}_entry",
            regime,
            close,
            atr_value,
            combined,
            details,
            filters=filters,
            stop_price=stop,
        )

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        if not all(
            math.isfinite(value) and value > 0
            for value in (current_stop, highest_completed_close, completed_close, current_atr)
        ):
            raise ValueError("Adaptive trailing-stop inputs must be finite and positive.")
        highest = max(highest_completed_close, completed_close)
        candidate = highest - current_atr * self.adaptive.trailing_atr_multiplier
        return max(current_stop, candidate) if candidate > 0 else current_stop, highest

    def _hold(self, symbol, timestamp, reason):
        return CryptoSignalDecision(
            symbol=symbol,
            action=BotAction.HOLD,
            timestamp=pd.Timestamp(timestamp).to_pydatetime() if timestamp is not None else None,
            reason=reason,
        )

    def _decision(
        self,
        symbol,
        action,
        timestamp,
        reason,
        regime,
        close,
        atr_value,
        score,
        details,
        *,
        filters=None,
        stop_price=None,
    ):
        return CryptoSignalDecision(
            symbol=symbol,
            action=action,
            timestamp=pd.Timestamp(timestamp).to_pydatetime(),
            reason=reason,
            regime=regime,
            close=close,
            atr=atr_value,
            score=score,
            stop_price=stop_price,
            volume_ratio=details.get("volume_ratio"),
            filters=filters or {},
            details=dict(details),
        )
