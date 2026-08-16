"""Opening-range breakout: trade only the FIRST break of a session's range.

Hypothesis: the first bars of a session are a price-discovery auction; the
first close above that auction's high on clearly elevated relative volume
marks initiative buying whose intraday follow-through is worth more than
its costs.  Every later close above the range restates already-known
information, so only the first breakout of each session is tradable — one
shot per session by construction, not by bookkeeping.

Causal mechanics: ``quant.features.opening_range`` publishes ``or_high`` /
``or_low`` only on bars strictly after the range window, so no bar can
trade against a range that includes itself.  First-ness is decided from
the PREVIOUS completed bar's close against the same session's range —
bars <= N only, the same no-look-ahead invariant as the shift(1) breakout
reference in ``crypto_strategy.py``.

Implements the ``UniverseStrategy`` protocol exactly like the production
crypto strategies: long-only, one decision per configured symbol, BUYs
first by descending score, and fail-closed HOLD whenever an input is
missing or degraded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

import pandas as pd

from crypto_models import CryptoSignalDecision
from indicators import atr, atr_percentile, volume_ratio
from models import BotAction
from quant.features import opening_range, session_vwap
from quant.strategies.config import IntradayStrategyConfig


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class OpeningRangeConfig(IntradayStrategyConfig):
    """Opening-range parameters on top of the shared intraday fields.

    ``max_entry_session_bar`` bounds how late in the session a first
    breakout may still be taken: a range break near the close has a very
    different (worse) follow-through profile than a morning break, and the
    cutoff keeps fresh entries from being initiated into the final bars.
    """

    opening_range_bars: int = 4
    max_entry_session_bar: int = 48
    min_or_volume_ratio: float = 1.5

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.opening_range_bars < 1:
            errors.append("Opening range must span at least one bar.")
        if self.max_entry_session_bar <= self.opening_range_bars:
            errors.append("Entry cutoff must lie beyond the opening range window.")
        if not math.isfinite(self.min_or_volume_ratio) or self.min_or_volume_ratio <= 0:
            errors.append(
                "Minimum breakout volume ratio must be finite and greater than zero."
            )
        return errors

    def warmup_bars(self) -> int:
        # The session-scoped range needs its own window completed; the
        # inherited warmup covers the ATR, volume, and percentile lookbacks.
        return max(super().warmup_bars(), self.opening_range_bars + 2)


class OpeningRangeBreakoutStrategy:
    """Deterministic long-only first-breakout-of-session strategy."""

    def __init__(self, config: OpeningRangeConfig):
        if not isinstance(config, OpeningRangeConfig):
            raise ValueError(
                "OpeningRangeBreakoutStrategy requires an OpeningRangeConfig."
            )
        errors = config.validate()
        if errors:
            raise ValueError(
                "Invalid opening-range configuration: " + "; ".join(errors)
            )
        self.config = config

    def required_bars(self) -> int:
        return self.config.warmup_bars()

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add causal indicator and session-anchored range/VWAP columns."""

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
        out["volume_ratio"] = volume_ratio(out["volume"], cfg.volume_period)
        out["atr_percentile"] = atr_percentile(out["atr"], cfg.atr_percentile_lookback)
        anchor = cfg.session_anchor
        or_frame = opening_range(bars, anchor, cfg.opening_range_bars)
        for column in or_frame.columns:
            out[column] = or_frame[column]
        out["vwap"] = session_vwap(bars, anchor)
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

        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        volume_ratio_value = _finite(row.get("volume_ratio"))
        atr_percentile_value = _finite(row.get("atr_percentile"))
        or_high = _finite(row.get("or_high"))
        or_low = _finite(row.get("or_low"))
        vwap_value = _finite(row.get("vwap"))
        session_bar = _finite(row.get("session_bar_index"))
        complete_value = row.get("or_complete")
        or_complete = bool(complete_value) if not pd.isna(complete_value) else False

        if close is None or close <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")

        values = {
            "close": close,
            "atr": atr_value,
            "volume_ratio": volume_ratio_value,
            "atr_percentile": atr_percentile_value,
        }
        context = {
            "or_high": or_high,
            "or_low": or_low,
            "vwap": vwap_value,
            "session_bar_index": session_bar,
        }

        if held:
            # A missing session VWAP (zero-volume session) fails closed to
            # HOLD; the ratcheted protective stop remains the backstop.
            if vwap_value is None:
                return self._hold(symbol, timestamp, "indicators_not_initialized")
            if close < vwap_value:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "vwap_loss", values,
                    details=context,
                )
            # or_low is unknowable while a NEW session's range is still
            # forming; the range-failure exit applies only once it exists.
            if or_low is not None and close < or_low:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "range_failure", values,
                    details=context,
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "position_healthy", values,
                details=context,
            )

        needed = (atr_value, volume_ratio_value, atr_percentile_value, session_bar)
        if any(value is None for value in needed):
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        if atr_value <= 0:
            return self._hold(symbol, timestamp, "invalid_indicator_values")

        previous_close = _finite(previous.get("close"))
        previous_session_bar = _finite(previous.get("session_bar_index"))
        range_ready = or_complete and or_high is not None and or_low is not None
        # First-ness: the previous bar must belong to the SAME session (its
        # session position precedes this bar's) and must have closed at or
        # below the range high.  Bars inside the range window close at or
        # below or_high by construction, so bar ``opening_range_bars`` is
        # the earliest eligible entry.
        same_session = (
            previous_session_bar is not None
            and previous_session_bar < session_bar
        )
        first_breakout = (
            range_ready
            and same_session
            and previous_close is not None
            and previous_close <= or_high
            and close > or_high
        )
        filters = {
            "or_complete": range_ready,
            "first_breakout": first_breakout,
            "entry_window": session_bar <= cfg.max_entry_session_bar,
            "volume": (
                _finite(row.get("volume")) is not None
                and float(row["volume"]) > 0
                and volume_ratio_value >= cfg.min_or_volume_ratio
            ),
            "volatility": atr_percentile_value <= cfg.max_atr_percentile,
        }
        score = None
        if range_ready:
            score = (
                (close - or_high) / atr_value
                + min(volume_ratio_value, 3.0) / 3.0
            )
        if all(filters.values()):
            stop = max(or_low, close - atr_value * cfg.atr_stop_multiplier)
            if 0 < stop < close:
                return self._decision(
                    symbol,
                    BotAction.BUY,
                    timestamp,
                    "opening_range_breakout",
                    values,
                    filters=filters,
                    score=score,
                    stop_price=stop,
                    details=context,
                )
            details = dict(context)
            details["stop_candidate"] = stop
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "invalid_stop", values,
                filters=filters, score=score, details=details,
            )
        failed = [name for name, passed in filters.items() if not passed]
        details = dict(context)
        details["failed_filters"] = failed
        return self._decision(
            symbol, BotAction.HOLD, timestamp, "entry_filters_failed", values,
            filters=filters, score=score, details=details,
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
            for value in (
                current_stop,
                highest_completed_close,
                completed_close,
                current_atr,
            )
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
        return CryptoSignalDecision(
            symbol=symbol,
            action=action,
            timestamp=ts,
            reason=reason,
            score=score,
            close=values.get("close"),
            atr=values.get("atr"),
            stop_price=stop_price,
            volume_ratio=values.get("volume_ratio"),
            atr_percentile=values.get("atr_percentile"),
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
