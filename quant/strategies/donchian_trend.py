"""Volatility-adjusted Donchian channel trend strategy.

Hypothesis: a NEW N-bar closing high, taken only when the tape is trending
(price above its trend MA, positive momentum, adequate path efficiency)
and volatility is not stressed, captures the fat right tail of trends;
ATR-scaled stops keep the risk per trade level across volatility regimes.

The module mirrors the production ``CryptoRegimeStrategy`` protocol shape
exactly so it plugs into the already-audited execution and research paths.
Two causal invariants carry over verbatim:

* every breakout reference excludes the current bar via ``shift(1)``, so
  bar N never trades against a channel that includes itself;
* entries fire only on the FIRST close through the prior channel top (the
  two-bar event rule), so a persistently extended market cannot re-trigger
  the same breakout bar after bar.

Exits for held positions are channel-based rather than MA-based: a close
that surrenders the configured fraction of the entry channel means the
breakout thesis has failed, independent of where any moving average sits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

import pandas as pd

from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import atr, atr_percentile, efficiency_ratio, sma, volume_ratio
from models import BotAction
from quant.strategies.config import IntradayStrategyConfig

# Score weights (fixed, versioned with the strategy: changing them changes
# ranking behavior and therefore requires a metadata version bump).
_SCORE_MOMENTUM_WEIGHT = 3.0
_SCORE_VOLUME_CAP = 3.0
_SCORE_ATR_PERCENTILE_DIVISOR = 200.0


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _validate_bars(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
        raise ValueError("Donchian bars require open/high/low/close/volume columns.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Donchian bars need a timezone-aware DatetimeIndex.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Donchian bars must be chronological and de-duplicated.")


@dataclass(frozen=True)
class DonchianTrendConfig(IntradayStrategyConfig):
    """Parameters specific to the Donchian trend family.

    ``momentum_period`` doubles as the efficiency-ratio window so momentum
    magnitude and path quality describe the same move; a separate knob
    would only add an over-fittable degree of freedom.
    """

    breakout_period: int = 20
    trend_ma_period: int = 50
    momentum_period: int = 20
    min_efficiency_ratio: float = 0.20
    exit_channel_fraction: float = 0.5

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.breakout_period < 2:
            errors.append("Breakout period must be at least 2 bars.")
        if self.trend_ma_period < 2:
            errors.append("Trend MA period must be at least 2 bars.")
        if self.momentum_period < 2:
            errors.append(
                "Momentum period must be at least 2 bars (it is also the "
                "efficiency-ratio window)."
            )
        if not (
            math.isfinite(self.min_efficiency_ratio)
            and 0.0 <= self.min_efficiency_ratio <= 1.0
        ):
            errors.append("Minimum efficiency ratio must be within [0, 1].")
        if not (
            math.isfinite(self.exit_channel_fraction)
            and 0.0 <= self.exit_channel_fraction <= 1.0
        ):
            errors.append("Exit channel fraction must be within [0, 1].")
        return errors

    def warmup_bars(self) -> int:
        # +3 on the breakout window: shift(1) consumes one bar and the
        # two-bar event rule needs an initialized previous row as well.
        return max(
            super().warmup_bars(),
            self.breakout_period + 3,
            self.trend_ma_period + 2,
            self.momentum_period + 2,
        )


class DonchianTrendStrategy:
    """Deterministic long-only Donchian breakout over a configured universe."""

    def __init__(self, config: DonchianTrendConfig):
        errors = config.validate()
        if errors:
            raise ValueError(
                "Invalid Donchian trend configuration: " + "; ".join(errors)
            )
        self.config = config

    def required_bars(self) -> int:
        return self.config.warmup_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add causal indicators; channel references exclude the current bar."""
        _validate_bars(bars)
        cfg = self.config
        out = bars.copy()
        # shift(1) is the central no-look-ahead invariant: the current close
        # is compared with the channel of only the PRECEDING window.
        prior_close = out["close"].shift(1)
        out["prior_breakout_high"] = prior_close.rolling(
            cfg.breakout_period, min_periods=cfg.breakout_period
        ).max()
        out["prior_breakout_low"] = prior_close.rolling(
            cfg.breakout_period, min_periods=cfg.breakout_period
        ).min()
        out["trend_ma"] = sma(out["close"], cfg.trend_ma_period)
        out["momentum"] = out["close"] / out["close"].shift(cfg.momentum_period) - 1.0
        out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
        out["atr_percentile"] = atr_percentile(out["atr"], cfg.atr_percentile_lookback)
        out["efficiency_ratio"] = efficiency_ratio(out["close"], cfg.momentum_period)
        out["volume_ratio"] = volume_ratio(out["volume"], cfg.volume_period)
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Each symbol is evaluated on its own channel; no cross-sectional
        context is required, so a missing symbol fails loudly instead of
        silently narrowing the universe."""
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing bar data for {sorted(missing)}")
        return {
            symbol: self.prepare_symbol(bars_by_symbol[symbol])
            for symbol in self.config.symbols
        }

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
        non_buys = [
            decision for decision in decisions if decision.action is not BotAction.BUY
        ]
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
                "prior_breakout_high",
                "prior_breakout_low",
                "trend_ma",
                "momentum",
                "atr",
                "atr_percentile",
                "efficiency_ratio",
                "volume_ratio",
            )
        }
        if any(value is None for value in values.values()):
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        close = values["close"]
        atr_value = values["atr"]
        if close <= 0 or atr_value <= 0:
            return self._hold(symbol, timestamp, "invalid_indicator_values")

        stress = values["atr_percentile"] > cfg.max_atr_percentile
        trend = close > values["trend_ma"]
        if stress:
            regime = CryptoRegime.STRESS
        elif trend:
            regime = CryptoRegime.BULL_TREND
        else:
            regime = CryptoRegime.RANGE

        if held:
            channel_midpoint = values["prior_breakout_low"] + cfg.exit_channel_fraction * (
                values["prior_breakout_high"] - values["prior_breakout_low"]
            )
            if close < channel_midpoint:
                return self._decision(
                    symbol,
                    BotAction.EXIT,
                    timestamp,
                    "channel_exit",
                    regime,
                    values,
                    details={"channel_midpoint": channel_midpoint},
                )
            if stress:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "volatility_stress_exit", regime, values
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
            "trend": trend,
            "momentum": values["momentum"] > 0,
            "efficiency": values["efficiency_ratio"] >= cfg.min_efficiency_ratio,
            "volume": values["volume_ratio"] >= cfg.min_volume_ratio,
            "volatility": not stress,
        }
        score = (
            values["momentum"] * _SCORE_MOMENTUM_WEIGHT
            + values["efficiency_ratio"]
            + min(values["volume_ratio"], _SCORE_VOLUME_CAP) / _SCORE_VOLUME_CAP
            - values["atr_percentile"] / _SCORE_ATR_PERCENTILE_DIVISOR
        )
        if all(filters.values()):
            stop = close - atr_value * cfg.atr_stop_multiplier
            if stop > 0 and stop < close:
                return self._decision(
                    symbol,
                    BotAction.BUY,
                    timestamp,
                    "donchian_breakout",
                    regime,
                    values,
                    filters=filters,
                    score=score,
                    stop_price=stop,
                )
            # A stop at or below zero means the ATR distance exceeds the
            # price itself; entering without a placeable protective stop is
            # never allowed, so this can only ADD a rejection.
            return self._decision(
                symbol,
                BotAction.HOLD,
                timestamp,
                "invalid_stop_distance",
                regime,
                values,
                filters=filters,
                score=score,
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
            efficiency_ratio=values.get("efficiency_ratio"),
            volume_ratio=values.get("volume_ratio"),
            atr_percentile=values.get("atr_percentile"),
            momentum=values.get("momentum"),
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
