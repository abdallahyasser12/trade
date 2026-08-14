"""Shared causal regime, breakout, and cross-sectional crypto strategy."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping, Optional

import pandas as pd

from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import adx, atr, atr_percentile, efficiency_ratio, rsi, sma, volume_ratio
from models import BotAction


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


class CryptoRegimeStrategy:
    """Deterministic long-only strategy used verbatim by live and backtest."""

    def __init__(self, config: CryptoStrategyConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config

    def required_bars(self) -> int:
        return self.config.warmup_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add causal indicators; rolling breakout deliberately excludes bar N."""

        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Crypto bars require open/high/low/close/volume columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Crypto bars need a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Crypto bars must be chronological and de-duplicated.")
        cfg = self.config
        out = bars.copy()
        out["fast_ma"] = sma(out["close"], cfg.fast_ma_period)
        out["slow_ma"] = sma(out["close"], cfg.slow_ma_period)
        out["regime_ma"] = sma(out["close"], cfg.regime_ma_period)
        out["regime_slope"] = (
            out["regime_ma"] / out["regime_ma"].shift(cfg.regime_slope_period) - 1.0
        )
        out["rsi"] = rsi(out["close"], cfg.rsi_period)
        out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
        out["atr_percentile"] = atr_percentile(
            out["atr"], cfg.atr_percentile_lookback
        )
        out["adx"] = adx(out["high"], out["low"], out["close"], cfg.adx_period)
        out["efficiency_ratio"] = efficiency_ratio(
            out["close"], cfg.efficiency_period
        )
        out["volume_ratio"] = volume_ratio(out["volume"], cfg.volume_period)
        # shift(1) is the central no-look-ahead invariant: today's close is
        # compared with the high of only the preceding breakout window.
        out["prior_breakout_high"] = (
            out["close"].shift(1).rolling(cfg.breakout_period, min_periods=cfg.breakout_period).max()
        )
        out["momentum"] = out["close"] / out["close"].shift(cfg.momentum_period) - 1.0
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing crypto data for {sorted(missing)}")
        prepared = {
            symbol: self.prepare_symbol(bars_by_symbol[symbol])
            for symbol in self.config.symbols
        }
        benchmark = prepared[self.config.benchmark_symbol]
        benchmark_return = (
            benchmark["close"]
            / benchmark["close"].shift(self.config.relative_strength_period)
            - 1.0
        )
        benchmark_bull = (
            (benchmark["close"] > benchmark["regime_ma"])
            & (benchmark["regime_slope"] > 0)
        )
        for symbol, frame in prepared.items():
            own_return = (
                frame["close"]
                / frame["close"].shift(self.config.relative_strength_period)
                - 1.0
            )
            # Exact timestamp alignment: a missing benchmark bar stays NaN/False
            # and blocks entry rather than borrowing information from another bar.
            frame["benchmark_return"] = benchmark_return.reindex(frame.index)
            frame["benchmark_bull"] = benchmark_bull.reindex(frame.index).astype("boolean")
            frame["relative_strength"] = own_return - frame["benchmark_return"]
        return prepared

    def evaluate(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        prepared = self.prepare_universe(bars_by_symbol)
        common = None
        for frame in prepared.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            return [
                self._hold(symbol, None, "market_context_unavailable")
                for symbol in self.config.symbols
            ]
        return self.evaluate_prepared(prepared, common[-1], held_symbols)

    def evaluate_prepared(
        self,
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp | datetime,
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        timestamp = pd.Timestamp(timestamp)
        decisions = [
            self._evaluate_symbol(
                symbol,
                prepared[symbol],
                timestamp,
                symbol in held_symbols,
            )
            for symbol in self.config.symbols
        ]
        # Stable deterministic candidate priority; HOLD/EXIT retain universe order.
        buys = sorted(
            (decision for decision in decisions if decision.action is BotAction.BUY),
            key=lambda decision: (-(decision.score or -math.inf), decision.symbol),
        )
        non_buys = [decision for decision in decisions if decision.action is not BotAction.BUY]
        return buys + non_buys

    def _evaluate_symbol(
        self,
        symbol: str,
        frame: pd.DataFrame,
        timestamp: pd.Timestamp,
        held: bool,
    ) -> CryptoSignalDecision:
        cfg = self.config
        if timestamp not in frame.index:
            return self._hold(symbol, timestamp, "missing_aligned_bar")
        location = int(frame.index.get_indexer([timestamp])[0])
        if location < self.required_bars() - 1:
            return self._hold(
                symbol,
                timestamp,
                "insufficient_data",
                {"bars_required": self.required_bars(), "bar_position": int(location)},
            )
        row = frame.iloc[location]
        previous = frame.iloc[location - 1]
        values = {
            key: _finite(row.get(key))
            for key in (
                "close",
                "fast_ma",
                "slow_ma",
                "rsi",
                "atr",
                "atr_percentile",
                "adx",
                "efficiency_ratio",
                "volume_ratio",
                "momentum",
                "relative_strength",
                "prior_breakout_high",
                "regime_slope",
            )
        }
        benchmark_value = row.get("benchmark_bull")
        benchmark_bull = bool(benchmark_value) if not pd.isna(benchmark_value) else False
        if any(value is None for value in values.values()):
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        close = values["close"]
        atr_value = values["atr"]
        if close <= 0 or atr_value <= 0:
            return self._hold(symbol, timestamp, "invalid_indicator_values")

        stress = values["atr_percentile"] > cfg.max_atr_percentile
        own_trend = (
            close > values["slow_ma"]
            and values["fast_ma"] > values["slow_ma"]
            and values["regime_slope"] > 0
        )
        if stress:
            regime = CryptoRegime.STRESS
        elif benchmark_bull and own_trend:
            regime = CryptoRegime.BULL_TREND
        else:
            regime = CryptoRegime.RANGE

        if held:
            trend_exit = close < values["fast_ma"]
            benchmark_exit = cfg.exit_on_benchmark_regime_break and not benchmark_bull
            if stress or trend_exit or benchmark_exit:
                reason = (
                    "volatility_stress_exit"
                    if stress
                    else "fast_trend_exit"
                    if trend_exit
                    else "benchmark_regime_exit"
                )
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, reason, regime, values
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "position_healthy", regime, values
            )

        previous_close = _finite(previous.get("close"))
        previous_breakout = _finite(previous.get("prior_breakout_high"))
        new_breakout = (
            previous_close is not None
            and previous_breakout is not None
            and previous_close <= previous_breakout
            and close > values["prior_breakout_high"]
        )
        filters = {
            "new_breakout": new_breakout,
            "trend": own_trend,
            "rsi": cfg.min_rsi <= values["rsi"] <= cfg.max_rsi,
            "adx": values["adx"] >= cfg.min_adx,
            "efficiency": values["efficiency_ratio"] >= cfg.min_efficiency_ratio,
            "volume": (
                _finite(row.get("volume")) is not None
                and float(row["volume"]) > 0
                and values["volume_ratio"] >= cfg.min_volume_ratio
            ),
            "volatility": not stress,
            "momentum": values["momentum"] > 0,
            "benchmark_regime": benchmark_bull,
            "relative_strength": (
                symbol == cfg.benchmark_symbol
                or values["relative_strength"] >= cfg.min_relative_strength
            ),
        }
        score = (
            values["momentum"] * 4.0
            + values["relative_strength"] * 2.0
            + values["adx"] / 100.0
            + values["efficiency_ratio"]
            + min(values["volume_ratio"], 3.0) / 3.0
            - values["atr_percentile"] / 200.0
        )
        if all(filters.values()):
            stop = close - atr_value * cfg.atr_stop_multiplier
            if stop > 0 and stop < close:
                return self._decision(
                    symbol,
                    BotAction.BUY,
                    timestamp,
                    "confirmed_regime_breakout",
                    regime,
                    values,
                    filters=filters,
                    score=score,
                    stop_price=stop,
                )
        failed = [name for name, passed in filters.items() if not passed]
        return self._decision(
            symbol,
            BotAction.HOLD,
            timestamp,
            "entry_filters_failed",
            regime,
            values,
            filters=filters,
            score=score,
            details={"failed_filters": failed},
        )

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        """Return a stop that can only tighten using a completed close."""

        if not all(
            math.isfinite(value) and value > 0
            for value in (current_stop, highest_completed_close, completed_close, current_atr)
        ):
            raise ValueError("Trailing-stop inputs must be finite and positive.")
        highest = max(highest_completed_close, completed_close)
        if not self.config.enable_trailing_stop:
            return current_stop, highest
        candidate = highest - current_atr * self.config.trailing_atr_multiplier
        if candidate <= 0:
            return current_stop, highest
        return max(current_stop, candidate), highest

    def _decision(
        self,
        symbol: str,
        action: BotAction,
        timestamp,
        reason: str,
        regime: CryptoRegime,
        values: dict[str, float],
        *,
        filters: Optional[dict[str, bool]] = None,
        score: Optional[float] = None,
        stop_price: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> CryptoSignalDecision:
        ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        return CryptoSignalDecision(
            symbol=symbol,
            action=action,
            timestamp=ts,
            reason=reason,
            regime=regime,
            score=score,
            close=values.get("close"),
            atr=values.get("atr"),
            stop_price=stop_price,
            rsi=values.get("rsi"),
            adx=values.get("adx"),
            efficiency_ratio=values.get("efficiency_ratio"),
            volume_ratio=values.get("volume_ratio"),
            atr_percentile=values.get("atr_percentile"),
            momentum=values.get("momentum"),
            relative_strength=values.get("relative_strength"),
            filters=filters or {},
            details=details or {},
        )

    @staticmethod
    def _hold(symbol: str, timestamp, reason: str, details: Optional[dict] = None):
        ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        return CryptoSignalDecision(
            symbol=symbol,
            action=BotAction.HOLD,
            timestamp=ts,
            reason=reason,
            details=details or {},
        )
