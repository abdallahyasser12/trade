"""Cross-sectional risk-adjusted momentum — a different hypothesis, not a tuning.

The audited breakout strategy asks an *absolute* question of each symbol in
isolation: "is this one breaking out, trending, liquid, un-stressed, and
un-overbought all at once?" Seven walk-forward experiments established two
failures in that design:

1. **The conjunction rarely fires.** Ten simultaneous conditions produced 2
   out-of-sample trades across five years of daily data — at two symbols and
   again at eight. Trading frequency was accidental rather than structural.
2. **Its score is dimensionally incoherent.** Raw fractional momentum was
   summed with a normalised ADX, so a 25% move and a maximal trend reading
   contributed equally. Scores were not comparable across symbols, which is
   precisely what a ranking needs them to be.

This module asks a *relative* question instead: "which symbols in this
universe have the strongest move relative to their own noise, right now?"

    score = (close / close[-L] - 1) / (sigma * sqrt(L))

where ``sigma`` is the per-bar standard deviation of log returns. The
denominator makes the numerator comparable across assets of wildly different
volatility — a score of 1.0 means the move is one standard deviation larger
than noise alone would produce over that horizon, whether the symbol is BTC
or DOGE. That is the property the old score lacked.

Ranking is always defined, so *something* is always the best candidate: trade
frequency becomes a property of ``top_k`` rather than an accident of filter
alignment. Exit uses a rank buffer, because the single most expensive lesson
from the cost work is that turnover, not signal, was eating the edge.

Six parameters. The breakout config carries roughly fifteen. Every parameter
is overfitting surface, and this is a deliberate reduction.

Causality: every column is computed from completed bars only. The decision at
bar N uses closes through N and executes at the N+1 open, identical to the
audited strategy. Nothing here reindexes forward or fills backward.

This is a hypothesis to be tested, not a promise. It runs through the same
unmodified cost gate, benchmark comparison and promotion criteria as
everything else, and is expected to be judged by them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import atr, sma
from models import BotAction


@dataclass(frozen=True)
class CrossSectionalConfig:
    """Deliberately small parameter surface."""

    lookback: int = 30
    vol_lookback: int = 30
    top_k: int = 3
    # Minimum risk-adjusted move to enter at all, so the strategy can decline
    # to hold "the best of a uniformly bad universe".
    min_score: float = 0.5
    # Hysteresis: exit only once rank falls this far past top_k. Turnover was
    # the measured killer, so leaving is deliberately stickier than entering.
    exit_rank_buffer: int = 2
    # One regime gate, not ten filters.
    require_benchmark_bull: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.lookback < 2:
            errors.append("Cross-sectional lookback must be at least 2 bars.")
        if self.vol_lookback < 5:
            errors.append("Volatility lookback must be at least 5 bars.")
        if self.top_k < 1:
            errors.append("top_k must be at least 1.")
        if self.exit_rank_buffer < 0:
            errors.append("Exit rank buffer cannot be negative.")
        if not math.isfinite(self.min_score):
            errors.append("Minimum score must be finite.")
        if self.min_score < 0:
            errors.append("Minimum score cannot be negative; that would buy falling assets.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossSectionalMomentumStrategy:
    """Rank-based long-only selection, interface-compatible with the base engine."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        cross_config: Optional[CrossSectionalConfig] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        self.cross_config = cross_config or CrossSectionalConfig()
        cross_errors = self.cross_config.validate()
        if cross_errors:
            raise ValueError(
                "Invalid cross-sectional configuration: " + "; ".join(cross_errors)
            )

    def required_bars(self) -> int:
        """Enough history for the slowest column plus the regime filter."""

        cross = self.cross_config
        return max(
            self.config.regime_ma_period + 2,
            cross.lookback + cross.vol_lookback + 2,
            self.config.atr_period + 2,
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Crypto bars require open/high/low/close/volume columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Crypto bars need a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Crypto bars must be chronological and de-duplicated.")

        cross = self.cross_config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)
        out["regime_ma"] = sma(out["close"], self.config.regime_ma_period)

        # Per-bar volatility of log returns. Log returns so that a symbol which
        # doubled and one which halved contribute symmetrically.
        log_return = np.log(out["close"] / out["close"].shift(1))
        sigma = log_return.rolling(
            cross.vol_lookback, min_periods=cross.vol_lookback
        ).std(ddof=1)
        out["realized_vol"] = sigma

        momentum = out["close"] / out["close"].shift(cross.lookback) - 1.0
        out["momentum"] = momentum
        # Scale the move by the noise you would expect over the same horizon.
        # A non-positive or non-finite sigma yields NaN, which never ranks and
        # therefore never trades — degenerate inputs fail closed.
        horizon_noise = sigma * math.sqrt(cross.lookback)
        out["score"] = momentum.where(horizon_noise > 0) / horizon_noise
        out["score"] = out["score"].replace([np.inf, -np.inf], np.nan)
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
        benchmark_bull = benchmark["close"] > benchmark["regime_ma"]

        # Cross-sectional rank per timestamp. Built from a wide frame so every
        # symbol is ranked against exactly the same bar, never a neighbouring
        # one. NaN scores stay unranked and cannot be selected.
        scores = pd.DataFrame(
            {symbol: frame["score"] for symbol, frame in prepared.items()}
        )
        # rank(ascending=False) gives 1 to the strongest score.
        ranks = scores.rank(axis=1, ascending=False, method="first", na_option="keep")
        eligible = scores.notna().sum(axis=1)

        for symbol, frame in prepared.items():
            frame["rank"] = ranks[symbol].reindex(frame.index)
            frame["eligible_count"] = eligible.reindex(frame.index)
            frame["benchmark_bull"] = (
                benchmark_bull.reindex(frame.index).astype("boolean")
            )
        return prepared

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
                symbol, prepared[symbol], timestamp, symbol in held_symbols
            )
            for symbol in self.config.symbols
        ]
        # Best rank first, so scarce capital reaches the strongest candidate.
        buys = sorted(
            (d for d in decisions if d.action is BotAction.BUY),
            key=lambda d: (-(d.score or -math.inf), d.symbol),
        )
        non_buys = [d for d in decisions if d.action is not BotAction.BUY]
        return buys + non_buys

    def _evaluate_symbol(
        self,
        symbol: str,
        frame: pd.DataFrame,
        timestamp: pd.Timestamp,
        held: bool,
    ) -> CryptoSignalDecision:
        if timestamp not in frame.index:
            return self._hold(symbol, timestamp, "bar_unavailable")
        row = frame.loc[timestamp]

        score = _finite(row.get("score"))
        rank = _finite(row.get("rank"))
        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        bull_raw = row.get("benchmark_bull")
        benchmark_bull = bool(bull_raw) if bull_raw is not None and bull_raw is not pd.NA else None

        values = {
            "close": close,
            "atr": atr_value,
            "momentum": _finite(row.get("momentum")),
        }

        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        if score is None or rank is None:
            return self._hold(symbol, timestamp, "insufficient_history_to_rank")

        cross = self.cross_config
        regime = (
            CryptoRegime.BULL_TREND if benchmark_bull else CryptoRegime.RANGE
        )

        if held:
            # Leaving is stickier than entering: turnover was the measured
            # killer, so a held position keeps its slot until rank decays
            # past the buffer or the move turns outright negative.
            if score <= 0:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "risk_adjusted_momentum_negative",
                    regime, values, score=score,
                )
            if rank > cross.top_k + cross.exit_rank_buffer:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "rank_decayed_out_of_book",
                    regime, values, score=score,
                    details={"rank": rank, "exit_rank": cross.top_k + cross.exit_rank_buffer},
                )
            if cross.require_benchmark_bull and benchmark_bull is False:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "benchmark_regime_break",
                    regime, values, score=score,
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "rank_intact", regime, values,
                score=score, details={"rank": rank},
            )

        filters = {
            "rank": rank <= cross.top_k,
            "score": score >= cross.min_score,
            "regime": (not cross.require_benchmark_bull) or bool(benchmark_bull),
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol, BotAction.HOLD, timestamp, f"blocked_{'_'.join(failed)}",
                regime, values, filters=filters, score=score,
                details={"rank": rank},
            )
        return self._decision(
            symbol, BotAction.BUY, timestamp, "cross_sectional_leader",
            regime, values, filters=filters, score=score,
            stop_price=close - atr_value * self.config.atr_stop_multiplier,
            details={"rank": rank, "eligible": _finite(row.get("eligible_count"))},
        )

    # ------------------------------------------------------------------
    # Shared machinery — identical semantics to the audited strategy
    # ------------------------------------------------------------------

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        """Return a stop that can only tighten, using a completed close."""

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
        values: dict,
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


def _finite(value) -> Optional[float]:
    """None for anything that is not a usable finite number."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
