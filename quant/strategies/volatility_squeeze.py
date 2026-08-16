"""Volatility-squeeze expansion breakout strategy.

Hypothesis: breakouts out of COMPRESSED ranges carry more follow-through
than breakouts from already-expanded ranges, because compression means the
market has agreed on price and the breakout resolves that agreement with
fresh participation (which is why an expansion-volume confirmation is
mandatory, not optional).

Compression is measured as the trailing percentile of the breakout-window
range divided by ATR — the same "range_compression" construction the quant
feature engine uses.  The percentile is shifted by ONE bar before the
entry test: the breakout bar's own range expansion must never be allowed
to un-squeeze the setup it is breaking out of, otherwise the filter would
be conditioning on the very event it gates.  The breakout itself uses the
production two-bar event rule with ``shift(1)`` channel references, so no
bar trades against a range that includes itself and a persistent extension
cannot re-trigger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

import pandas as pd

from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import atr, atr_percentile, sma, volume_ratio
from models import BotAction
from quant.strategies.config import IntradayStrategyConfig

# Score weight (fixed, versioned with the strategy: changing it changes
# ranking behavior and therefore requires a metadata version bump).
_SCORE_VOLUME_CAP = 3.0


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _validate_bars(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
        raise ValueError("Squeeze bars require open/high/low/close/volume columns.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Squeeze bars need a timezone-aware DatetimeIndex.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Squeeze bars must be chronological and de-duplicated.")


@dataclass(frozen=True)
class VolatilitySqueezeConfig(IntradayStrategyConfig):
    """Parameters specific to the squeeze-expansion family.

    ``breakout_period`` is also the range-measurement window: the
    compression being scored is exactly the channel the entry must break,
    so squeeze and breakout always describe the same structure.
    """

    squeeze_lookback: int = 100
    squeeze_percentile_max: float = 25.0
    breakout_period: int = 20
    trend_ma_period: int = 50
    min_expansion_volume_ratio: float = 1.5

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.squeeze_lookback < 20:
            errors.append("Squeeze lookback must be at least 20 bars.")
        if not (
            math.isfinite(self.squeeze_percentile_max)
            and 0.0 < self.squeeze_percentile_max <= 100.0
        ):
            errors.append("Squeeze percentile ceiling must be in (0, 100].")
        if self.breakout_period < 2:
            errors.append("Breakout period must be at least 2 bars.")
        if self.trend_ma_period < 2:
            errors.append("Trend MA period must be at least 2 bars.")
        if not (
            math.isfinite(self.min_expansion_volume_ratio)
            and self.min_expansion_volume_ratio > 0
        ):
            errors.append(
                "Minimum expansion volume ratio must be finite and greater than zero."
            )
        return errors

    def warmup_bars(self) -> int:
        # The squeeze percentile chains three windows (ATR, range, trailing
        # percentile) and is then shifted one bar; summing them is the
        # conservative bound that guarantees full initialization (+ pad).
        return max(
            super().warmup_bars(),
            self.atr_period + self.breakout_period + self.squeeze_lookback + 3,
            self.trend_ma_period + 2,
            self.breakout_period + 3,
        )


class VolatilitySqueezeStrategy:
    """Deterministic long-only squeeze-expansion breakout over a universe."""

    def __init__(self, config: VolatilitySqueezeConfig):
        errors = config.validate()
        if errors:
            raise ValueError(
                "Invalid volatility squeeze configuration: " + "; ".join(errors)
            )
        self.config = config

    def required_bars(self) -> int:
        return self.config.warmup_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add causal indicators; the squeeze percentile is shifted one bar
        so the breakout bar's own expansion cannot un-squeeze its setup."""
        _validate_bars(bars)
        cfg = self.config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
        bar_range = (
            out["high"].rolling(cfg.breakout_period, min_periods=cfg.breakout_period).max()
            - out["low"].rolling(cfg.breakout_period, min_periods=cfg.breakout_period).min()
        )
        range_vs_atr = (bar_range / out["atr"]).where(out["atr"] > 0)
        out["squeeze_percentile"] = atr_percentile(
            range_vs_atr, cfg.squeeze_lookback
        ).shift(1)
        # shift(1) keeps the current bar OUT of its own breakout reference —
        # the same no-look-ahead invariant the production crypto strategy uses.
        out["prior_breakout_high"] = (
            out["close"].shift(1).rolling(cfg.breakout_period, min_periods=cfg.breakout_period).max()
        )
        out["trend_ma"] = sma(out["close"], cfg.trend_ma_period)
        out["volume_ratio"] = volume_ratio(out["volume"], cfg.volume_period)
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Each symbol is scored against its own compression history; no
        cross-sectional context is required, so a missing symbol fails
        loudly instead of silently narrowing the universe."""
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
                "atr",
                "squeeze_percentile",
                "prior_breakout_high",
                "trend_ma",
                "volume_ratio",
            )
        }
        if any(value is None for value in values.values()):
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        close = values["close"]
        atr_value = values["atr"]
        if close <= 0 or atr_value <= 0:
            return self._hold(symbol, timestamp, "invalid_indicator_values")

        trend = close > values["trend_ma"]
        regime = CryptoRegime.BULL_TREND if trend else CryptoRegime.RANGE

        if held:
            if close < values["trend_ma"]:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "trend_break", regime, values
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
        squeeze = values["squeeze_percentile"] <= cfg.squeeze_percentile_max
        filters = {
            "squeeze": squeeze,
            "new_breakout": new_breakout,
            "volume": values["volume_ratio"] >= cfg.min_expansion_volume_ratio,
            "trend": trend,
        }
        score = (
            (cfg.squeeze_percentile_max - values["squeeze_percentile"])
            / cfg.squeeze_percentile_max
            + min(values["volume_ratio"], _SCORE_VOLUME_CAP) / _SCORE_VOLUME_CAP
        )
        if all(filters.values()):
            stop = close - atr_value * cfg.atr_stop_multiplier
            if stop > 0 and stop < close:
                return self._decision(
                    symbol,
                    BotAction.BUY,
                    timestamp,
                    "squeeze_expansion_breakout",
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
        merged_details = {"squeeze_percentile": values.get("squeeze_percentile")}
        merged_details.update(details or {})
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
            volume_ratio=values.get("volume_ratio"),
            filters=filters or {},
            details=merged_details,
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
