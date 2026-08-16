"""VWAP overextension reversion: buy deep intraday stretches in chop.

Hypothesis: an intraday price stretched far below the session VWAP in a
NON-trending tape reverts toward VWAP often enough to pay its costs.  The
efficiency-ratio ceiling is the heart of the hypothesis — the identical
stretch inside a clean downtrend is not a dislocation, it is the trend,
and buying it is knife-catching.  The RSI band excludes both mild noise
(upper bound) and free-fall conditions (lower bound).

Causal mechanics: VWAP and the z-distance are session-anchored cumulative
values through the current completed bar only (``quant.features``); the
session bar index gates out the first bars of a session where a VWAP of
one or two prints is too immature to define "stretched".

Implements the ``UniverseStrategy`` protocol exactly like the production
crypto strategies: long-only, one decision per configured symbol, BUYs
first by descending score, fail-closed HOLD on any degraded input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

import pandas as pd

from crypto_models import CryptoSignalDecision
from indicators import atr, atr_percentile, efficiency_ratio, rsi, volume_ratio
from models import BotAction
from quant.features import session_ids, session_vwap
from quant.strategies.config import IntradayStrategyConfig


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class VwapReversionConfig(IntradayStrategyConfig):
    """Reversion parameters on top of the shared intraday fields.

    ``entry_z`` is in ATR units below VWAP (negative).  ``exit_z`` is the
    reversion target; 0 means "touched VWAP".  ``min_session_bar`` skips
    the immature-VWAP bars at each session open.
    """

    entry_z: float = -2.0
    exit_z: float = 0.0
    max_efficiency_ratio: float = 0.35
    min_rsi: float = 15.0
    max_rsi: float = 45.0
    rsi_period: int = 14
    efficiency_period: int = 20
    min_session_bar: int = 2

    def validate(self) -> list[str]:
        errors = super().validate()
        if not math.isfinite(self.entry_z) or self.entry_z >= 0:
            errors.append("Entry z-score must be finite and negative.")
        if not math.isfinite(self.exit_z) or self.exit_z <= self.entry_z:
            errors.append("Exit z-score must sit above the entry z-score.")
        if not (0.0 < self.max_efficiency_ratio <= 1.0):
            errors.append("Efficiency ceiling must be in (0, 1].")
        if not (0.0 <= self.min_rsi < self.max_rsi <= 100.0):
            errors.append("RSI band must satisfy 0 <= minimum < maximum <= 100.")
        if self.rsi_period < 2:
            errors.append("RSI period must be at least 2.")
        if self.efficiency_period < 2:
            errors.append("Efficiency period must be at least 2.")
        if self.min_session_bar < 1:
            errors.append("Minimum session bar must be at least 1.")
        return errors

    def warmup_bars(self) -> int:
        return max(
            super().warmup_bars(),
            self.rsi_period + 2,
            self.efficiency_period + 2,
        )


class VwapReversionStrategy:
    """Deterministic long-only VWAP-stretch reversion strategy."""

    def __init__(self, config: VwapReversionConfig):
        if not isinstance(config, VwapReversionConfig):
            raise ValueError("VwapReversionStrategy requires a VwapReversionConfig.")
        errors = config.validate()
        if errors:
            raise ValueError("Invalid VWAP-reversion configuration: " + "; ".join(errors))
        self.config = config

    def required_bars(self) -> int:
        return self.config.warmup_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Intraday bars require open/high/low/close/volume columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Intraday bars need a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Intraday bars must be chronological and de-duplicated.")
        cfg = self.config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
        out["rsi"] = rsi(out["close"], cfg.rsi_period)
        out["efficiency_ratio"] = efficiency_ratio(out["close"], cfg.efficiency_period)
        out["volume_ratio"] = volume_ratio(out["volume"], cfg.volume_period)
        out["atr_percentile"] = atr_percentile(out["atr"], cfg.atr_percentile_lookback)
        anchor = cfg.session_anchor
        out["vwap"] = session_vwap(bars, anchor)
        out["vwap_z"] = ((out["close"] - out["vwap"]) / out["atr"]).where(out["atr"] > 0)
        session = session_ids(out.index, anchor)
        out["session_bar_index"] = session.groupby(session.values).cumcount()
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing intraday data for {sorted(missing)}")
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
            self._evaluate_symbol(symbol, prepared[symbol], timestamp, symbol in held_symbols)
            for symbol in self.config.symbols
        ]
        buys = sorted(
            (d for d in decisions if d.action is BotAction.BUY),
            key=lambda d: (-(d.score or -math.inf), d.symbol),
        )
        non_buys = [d for d in decisions if d.action is not BotAction.BUY]
        return buys + non_buys

    def _evaluate_symbol(
        self, symbol: str, frame: pd.DataFrame, timestamp: pd.Timestamp, held: bool
    ) -> CryptoSignalDecision:
        cfg = self.config
        if timestamp not in frame.index:
            return self._hold(symbol, timestamp, "missing_aligned_bar")
        location = int(frame.index.get_indexer([timestamp])[0])
        if location < self.required_bars() - 1:
            return self._hold(symbol, timestamp, "insufficient_data")
        row = frame.iloc[location]
        values = {
            key: _finite(row.get(key))
            for key in (
                "close",
                "atr",
                "rsi",
                "efficiency_ratio",
                "volume_ratio",
                "atr_percentile",
                "vwap",
                "vwap_z",
            )
        }
        session_bar = _finite(row.get("session_bar_index"))
        if any(value is None for value in values.values()) or session_bar is None:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        close = values["close"]
        atr_value = values["atr"]
        if close <= 0 or atr_value <= 0:
            return self._hold(symbol, timestamp, "invalid_indicator_values")

        if held:
            if values["vwap_z"] >= cfg.exit_z:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "reversion_complete", values
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "position_healthy", values
            )

        filters = {
            "stretch": values["vwap_z"] <= cfg.entry_z,
            "rsi": cfg.min_rsi <= values["rsi"] <= cfg.max_rsi,
            "chop_only": values["efficiency_ratio"] <= cfg.max_efficiency_ratio,
            "volatility": values["atr_percentile"] <= cfg.max_atr_percentile,
            "vwap_mature": session_bar >= cfg.min_session_bar,
        }
        score = -values["vwap_z"]  # deeper stretch scores higher
        if all(filters.values()):
            stop = close - atr_value * cfg.atr_stop_multiplier
            if 0 < stop < close:
                return self._decision(
                    symbol,
                    BotAction.BUY,
                    timestamp,
                    "vwap_overextension",
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
        values: dict[str, Optional[float]],
        *,
        filters: Optional[dict[str, bool]] = None,
        score: Optional[float] = None,
        stop_price: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> CryptoSignalDecision:
        ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        extra = dict(details or {})
        extra.setdefault("vwap", values.get("vwap"))
        extra.setdefault("vwap_z", values.get("vwap_z"))
        return CryptoSignalDecision(
            symbol=symbol,
            action=action,
            timestamp=ts,
            reason=reason,
            score=score,
            close=values.get("close"),
            atr=values.get("atr"),
            stop_price=stop_price,
            rsi=values.get("rsi"),
            efficiency_ratio=values.get("efficiency_ratio"),
            volume_ratio=values.get("volume_ratio"),
            atr_percentile=values.get("atr_percentile"),
            filters=filters or {},
            details=extra,
        )

    @staticmethod
    def _hold(symbol: str, timestamp, reason: str) -> CryptoSignalDecision:
        ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        return CryptoSignalDecision(
            symbol=symbol, action=BotAction.HOLD, timestamp=ts, reason=reason
        )
