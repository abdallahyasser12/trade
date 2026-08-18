"""High-velocity cross-sectional crypto momentum hypothesis.

This strategy exists to create *more independent decision opportunities* on
5-minute/15-minute bars without weakening execution/risk gates. It is not a
profit claim. It ships RESEARCH-only through the registry and must pass the same
walk-forward, stressed-cost, shadow and paper gates as every other strategy.

Signal
------
For each symbol, short and medium horizon returns are normalized by trailing
realized volatility. The two z-like moves are combined with a bounded volume
surprise term, then ranked cross-sectionally on the exact same completed bar.
Only leaders above an absolute quality floor can enter. Held names use rank
hysteresis so the strategy does not churn on one-rank noise.
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
from indicators import atr, sma
from models import BotAction


@dataclass(frozen=True)
class IntradayMomentumConfig:
    fast_lookback: int = 3
    slow_lookback: int = 12
    vol_lookback: int = 36
    volume_lookback: int = 24
    top_k: int = 3
    exit_rank_buffer: int = 1
    min_score: float = 0.30
    min_volume_ratio: float = 0.80
    stop_atr_multiplier: float = 1.8
    trailing_atr_multiplier: float = 2.2
    benchmark_fast_ma: int = 24
    benchmark_slow_ma: int = 72
    require_benchmark_bull: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.fast_lookback < 1:
            errors.append("fast_lookback must be positive.")
        if self.slow_lookback <= self.fast_lookback:
            errors.append("slow_lookback must exceed fast_lookback.")
        if self.vol_lookback < max(10, self.slow_lookback):
            errors.append("vol_lookback must cover the slow horizon and at least 10 bars.")
        if self.volume_lookback < 5:
            errors.append("volume_lookback must be at least 5 bars.")
        if self.top_k < 1:
            errors.append("top_k must be positive.")
        if self.exit_rank_buffer < 0:
            errors.append("exit_rank_buffer cannot be negative.")
        for label, value in {
            "min_score": self.min_score,
            "min_volume_ratio": self.min_volume_ratio,
            "stop_atr_multiplier": self.stop_atr_multiplier,
            "trailing_atr_multiplier": self.trailing_atr_multiplier,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.min_score < 0:
            errors.append("min_score cannot be negative.")
        if self.min_volume_ratio <= 0:
            errors.append("min_volume_ratio must be positive.")
        if self.stop_atr_multiplier <= 0 or self.trailing_atr_multiplier <= 0:
            errors.append("ATR multipliers must be positive.")
        if self.benchmark_fast_ma < 2 or self.benchmark_slow_ma <= self.benchmark_fast_ma:
            errors.append("benchmark MA periods must satisfy 2 <= fast < slow.")
        return errors


class IntradayMomentumStrategy:
    """Causal, rank-based intraday momentum strategy."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        intraday: Optional[IntradayMomentumConfig] = None,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        self.intraday = intraday or IntradayMomentumConfig()
        errors = self.intraday.validate()
        if errors:
            raise ValueError("Invalid intraday momentum configuration: " + "; ".join(errors))
        if self.intraday.top_k > len(config.symbols):
            raise ValueError("top_k cannot exceed the configured universe size.")

    def required_bars(self) -> int:
        c = self.intraday
        return max(
            c.vol_lookback + c.slow_lookback + 2,
            c.volume_lookback + 2,
            c.benchmark_slow_ma + 2,
            self.config.atr_period + 2,
        )

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Intraday momentum bars require OHLCV columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Intraday momentum requires a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Intraday momentum bars must be chronological and unique.")

        c = self.intraday
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)
        log_return = np.log(out["close"] / out["close"].shift(1))
        sigma = log_return.rolling(c.vol_lookback, min_periods=c.vol_lookback).std(ddof=1)
        out["realized_vol"] = sigma
        fast_ret = out["close"] / out["close"].shift(c.fast_lookback) - 1.0
        slow_ret = out["close"] / out["close"].shift(c.slow_lookback) - 1.0
        fast_noise = sigma * math.sqrt(c.fast_lookback)
        slow_noise = sigma * math.sqrt(c.slow_lookback)
        fast_z = fast_ret.where(fast_noise > 0) / fast_noise
        slow_z = slow_ret.where(slow_noise > 0) / slow_noise
        avg_volume = out["volume"].rolling(
            c.volume_lookback, min_periods=c.volume_lookback
        ).mean()
        out["volume_ratio"] = out["volume"] / avg_volume.where(avg_volume > 0)
        # tanh prevents a single abnormal volume print from dominating returns.
        volume_term = np.tanh((out["volume_ratio"] - 1.0).fillna(0.0))
        out["score"] = 0.60 * fast_z + 0.40 * slow_z + 0.10 * volume_term
        out["score"] = out["score"].replace([np.inf, -np.inf], np.nan)
        out["fast_momentum"] = fast_ret
        out["slow_momentum"] = slow_ret
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing intraday data for {sorted(missing)}")
        prepared = {
            symbol: self.prepare_symbol(bars_by_symbol[symbol])
            for symbol in self.config.symbols
        }
        scores = pd.DataFrame({symbol: frame["score"] for symbol, frame in prepared.items()})
        ranks = scores.rank(axis=1, ascending=False, method="first", na_option="keep")
        benchmark = prepared[self.config.benchmark_symbol]
        fast = sma(benchmark["close"], self.intraday.benchmark_fast_ma)
        slow = sma(benchmark["close"], self.intraday.benchmark_slow_ma)
        benchmark_bull = (fast > slow) & (benchmark["close"] > slow)
        for symbol, frame in prepared.items():
            frame["rank"] = ranks[symbol].reindex(frame.index)
            frame["benchmark_bull"] = benchmark_bull.reindex(frame.index).astype("boolean")
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
        timestamp = pd.Timestamp(timestamp)
        decisions = [
            self._evaluate_symbol(symbol, prepared[symbol], timestamp, symbol in held_symbols)
            for symbol in self.config.symbols
        ]
        buys = sorted(
            (decision for decision in decisions if decision.action is BotAction.BUY),
            key=lambda decision: (-(decision.score or -math.inf), decision.symbol),
        )
        return buys + [decision for decision in decisions if decision.action is not BotAction.BUY]

    def _evaluate_symbol(self, symbol, frame, timestamp, held):
        if timestamp not in frame.index:
            return self._hold(symbol, timestamp, "bar_unavailable")
        row = frame.loc[timestamp]
        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        score = _finite(row.get("score"))
        rank = _finite(row.get("rank"))
        volume_ratio = _finite(row.get("volume_ratio"))
        bull_raw = row.get("benchmark_bull")
        benchmark_bull = (
            bool(bull_raw)
            if bull_raw is not None and bull_raw is not pd.NA
            else None
        )
        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        if score is None or rank is None or volume_ratio is None:
            return self._hold(symbol, timestamp, "insufficient_history_to_rank")

        c = self.intraday
        regime = CryptoRegime.BULL_TREND if benchmark_bull else CryptoRegime.RANGE
        values = {
            "close": close,
            "atr": atr_value,
            "volume_ratio": volume_ratio,
            "fast_momentum": _finite(row.get("fast_momentum")),
            "slow_momentum": _finite(row.get("slow_momentum")),
        }

        if held:
            if score <= 0:
                return self._decision(symbol, BotAction.EXIT, timestamp, "intraday_momentum_reversal", regime, values, score)
            if rank > c.top_k + c.exit_rank_buffer:
                return self._decision(symbol, BotAction.EXIT, timestamp, "intraday_rank_decay", regime, values, score)
            if c.require_benchmark_bull and benchmark_bull is False:
                return self._decision(symbol, BotAction.EXIT, timestamp, "intraday_benchmark_break", regime, values, score)
            return self._decision(symbol, BotAction.HOLD, timestamp, "intraday_rank_intact", regime, values, score)

        filters = {
            "rank": rank <= c.top_k,
            "score": score >= c.min_score,
            "volume": volume_ratio >= c.min_volume_ratio,
            "regime": (not c.require_benchmark_bull) or bool(benchmark_bull),
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol,
                BotAction.HOLD,
                timestamp,
                "blocked_" + "_".join(failed),
                regime,
                values,
                score,
                filters=filters,
            )
        stop = close - atr_value * c.stop_atr_multiplier
        if stop <= 0 or stop >= close:
            return self._hold(symbol, timestamp, "invalid_atr_stop")
        return self._decision(
            symbol,
            BotAction.BUY,
            timestamp,
            "intraday_cross_sectional_leader",
            regime,
            values,
            score,
            filters=filters,
            stop_price=stop,
        )

    def ratchet_stop(self, current_stop, highest_completed_close, completed_close, current_atr):
        if not all(
            math.isfinite(value) and value > 0
            for value in (current_stop, highest_completed_close, completed_close, current_atr)
        ):
            raise ValueError("Trailing-stop inputs must be finite and positive.")
        highest = max(highest_completed_close, completed_close)
        candidate = highest - current_atr * self.intraday.trailing_atr_multiplier
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
        values,
        score,
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
            close=values.get("close"),
            atr=values.get("atr"),
            volume_ratio=values.get("volume_ratio"),
            score=score,
            stop_price=stop_price,
            filters=filters or {},
            details={
                "fast_momentum": values.get("fast_momentum"),
                "slow_momentum": values.get("slow_momentum"),
            },
        )


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None
