"""Volatility-state expansion — a strategy that trades the second moment.

Every other member of the council trades the *first* moment. The audited
breakout strategy asks "is this symbol moving up?", the cross-sectional
module asks "which symbol is moving up most per unit of noise?", and the
ensemble asks both. They disagree about ranking and about filters, but they
all draw their information from the same place: the direction of price. Two
estimators of the same quantity are correlated no matter how differently
they are parameterised, which is why a council of directional strategies
diversifies far less than its member count suggests.

This module reads a different number. Its state variable is realized
volatility — the standard deviation of log returns — and its thesis is a
property of that variable rather than of price:

    volatility clusters, and it mean-reverts. A sustained CONTRACTION is
    followed by an EXPANSION. The contraction is observable; its timing is
    not, and the direction of the resolution is not.

So the entry is two-sided in its information and one-sided only in its
execution. The coil is detected in the second moment — realized vol sitting
in a low percentile of its own trailing distribution — and the *direction*
is not predicted at all: the strategy waits for a completed bar to resolve
the coil upward through the ceiling it was coiling under, and only then
buys. Nothing about "is price trending" enters the decision. There is
deliberately no benchmark-direction gate, because that single filter is what
would re-import first-moment information and rebuild the correlation this
module exists to avoid.

The consequence is a genuinely different failure mode. A directional
strategy fails in a choppy, non-trending tape. This one *feeds* on the
transition out of exactly that tape, and instead fails when volatility is
persistently high — when there is no compression to find. Those two failure
modes do not coincide, which is the entire justification for the module.

Percentile, not level. Raw volatility is not comparable across symbols
(DOGE and BTC live in different ranges) nor across epochs (2021 and 2023 do
not share a distribution). Ranking vol[N] inside its own trailing window
makes "compressed" mean the same thing everywhere, which is what lets one
threshold serve the whole universe.

Causality: every column is computed from completed bars only. The decision
at bar N uses closes through N and executes at the N+1 open, identical to
the audited strategy. Nothing here reindexes forward or fills backward. The
percentile rank in particular is delegated to ``indicators.atr_percentile``,
whose window ends at and includes bar N — a hand-rolled ranker that ranks
within the whole series is the single most common leak in this class of
strategy, and it is invisible in every test except truncation.

Six parameters. This is a hypothesis to be tested, not a promise. It runs
through the same unmodified cost gate, benchmark comparison and promotion
criteria as everything else, and is expected to be judged by them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from indicators import atr, atr_percentile
from models import BotAction


@dataclass(frozen=True)
class VolatilityRegimeConfig:
    """Deliberately small parameter surface: six numbers, no filter stack."""

    # Window for realized volatility (std of log returns).
    vol_lookback: int = 20
    # Trailing window the realized vol is ranked *within*. This is what makes
    # "compressed" comparable across symbols and across epochs.
    percentile_lookback: int = 120
    # Window whose ceiling the coil must resolve through, and over which the
    # coil itself is looked for. One window serves both so the two halves of
    # the signal describe the same stretch of tape.
    breakout_period: int = 10
    # The coiled state: vol in the bottom this-many percent of its own past.
    compression_percentile: float = 20.0
    # The exhausted state: vol in the top (100 - this) percent. Entry is
    # refused above it and a held position is closed at it — one traversal of
    # the state variable is the whole trade.
    expansion_percentile: float = 85.0
    # Minimum resolution size, measured in ATR so it is scale-free. Zero
    # would admit a breakout that clears the ceiling by a rounding error.
    min_thrust_atr: float = 0.25

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.vol_lookback < 5:
            errors.append("Volatility lookback must be at least 5 bars.")
        if self.percentile_lookback < 20:
            errors.append("Percentile lookback must span at least 20 bars.")
        if self.percentile_lookback <= self.vol_lookback:
            errors.append(
                "Percentile lookback must exceed the volatility lookback; "
                "ranking vol against fewer bars than it is measured over is noise."
            )
        if self.breakout_period < 2:
            errors.append("Breakout confirmation window must be at least 2 bars.")
        finite = {
            "compression percentile": self.compression_percentile,
            "expansion percentile": self.expansion_percentile,
            "minimum thrust": self.min_thrust_atr,
        }
        for label, value in finite.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if math.isfinite(self.compression_percentile) and math.isfinite(
            self.expansion_percentile
        ):
            if not (0 < self.compression_percentile < self.expansion_percentile <= 100):
                errors.append(
                    "Percentile band must satisfy 0 < compression < expansion <= 100."
                )
        if math.isfinite(self.min_thrust_atr) and self.min_thrust_atr < 0:
            errors.append(
                "Minimum thrust cannot be negative; that would buy a failed resolution."
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VolatilityRegimeStrategyConfig(CryptoStrategyConfig):
    """A base config that carries its own volatility-state parameters.

    The audited walk-forward searches by calling ``dataclasses.replace`` on
    whatever config it was handed, and de-duplicates candidates by their
    ``to_dict``. Frozen-dataclass inheritance means the replacement preserves
    both this subclass and this field, and ``asdict`` recurses into it — so
    two candidates that differ only in ``vol_lookback`` or
    ``compression_percentile`` are genuinely distinct rather than being
    collapsed into one.

    Without it the grid can only vary keys this strategy never reads, which
    would run N identical backtests per fold, decide the argmax arbitrarily,
    and report a candidate count — and therefore a multiple-testing penalty —
    for a search that never happened.
    """

    volatility: VolatilityRegimeConfig = VolatilityRegimeConfig()

    def validate(self) -> list[str]:
        return super().validate() + self.volatility.validate()


def with_volatility_config(
    config: CryptoStrategyConfig, volatility: VolatilityRegimeConfig
) -> VolatilityRegimeStrategyConfig:
    """Attach volatility-state parameters to any base config, idempotently."""

    if isinstance(config, VolatilityRegimeStrategyConfig):
        return replace(config, volatility=volatility)
    values = {field.name: getattr(config, field.name) for field in fields(CryptoStrategyConfig)}
    return VolatilityRegimeStrategyConfig(**values, volatility=volatility)


def resolve_volatility_config(
    config: CryptoStrategyConfig, fallback: Optional[VolatilityRegimeConfig] = None
) -> VolatilityRegimeConfig:
    """Prefer parameters the candidate carries; fall back to the run default."""

    carried = getattr(config, "volatility", None)
    if isinstance(carried, VolatilityRegimeConfig):
        return carried
    return fallback or VolatilityRegimeConfig()


class VolatilityRegimeStrategy:
    """Compression-to-expansion selection, interface-compatible with the base engine."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        vol_config: Optional[VolatilityRegimeConfig] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        # A candidate produced by the walk-forward search carries the
        # parameters it was selected with; an explicit argument still wins.
        self.vol_config = vol_config or resolve_volatility_config(config)
        vol_errors = self.vol_config.validate()
        if vol_errors:
            raise ValueError(
                "Invalid volatility-regime configuration: " + "; ".join(vol_errors)
            )

    def required_bars(self) -> int:
        """Enough history for the slowest column, chained windows included."""

        vol = self.vol_config
        return max(
            # ATR: one differencing bar plus the Wilder window.
            self.config.atr_period + 2,
            # Realized vol: one log-return bar plus the deviation window.
            vol.vol_lookback + 2,
            # Percentile rank of realized vol: two chained windows, so they add.
            vol.vol_lookback + vol.percentile_lookback + 2,
            # The coil is a shifted rolling maximum *of that percentile*: three
            # chained windows, and +3 rather than +2 because the window itself
            # is shifted by one bar.
            vol.vol_lookback + vol.percentile_lookback + vol.breakout_period + 3,
            # The breakout ceiling is a shifted rolling maximum of close.
            vol.breakout_period + 3,
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

        vol = self.vol_config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)

        # Log returns, so a symbol that doubled and one that halved contribute
        # symmetrically to the dispersion estimate. A non-positive close is
        # masked before the logarithm rather than after: log(0) is -inf and an
        # infinity here would survive into the percentile rank.
        close = out["close"].astype(float)
        positive_close = close.where(close > 0)
        log_return = np.log(positive_close / positive_close.shift(1))
        log_return = log_return.replace([np.inf, -np.inf], np.nan)
        sigma = log_return.rolling(
            vol.vol_lookback, min_periods=vol.vol_lookback
        ).std(ddof=1)
        out["realized_vol"] = sigma.replace([np.inf, -np.inf], np.nan)

        # The state variable. ``atr_percentile`` is a generic causal rolling
        # percentile rank despite its name: for bar N the window is the last
        # ``lookback`` values ending at and INCLUDING bar N, and the result is
        # the share of that window at or below the current value. It is
        # deliberately reused rather than re-implemented — a hand-rolled
        # ranker that ranks within the whole column is the leak this class of
        # strategy dies of, and it is invisible outside a truncation test.
        ranked = atr_percentile(out["realized_vol"], vol.percentile_lookback)
        # Honest warmup: the helper skips NaNs inside its window, which would
        # rank an early bar against a handful of values and call the answer a
        # percentile. Require the whole ranking window to be populated.
        window_full = (
            out["realized_vol"]
            .notna()
            .astype(float)
            .rolling(vol.percentile_lookback, min_periods=vol.percentile_lookback)
            .min()
        )
        out["vol_percentile"] = ranked.where(window_full > 0)

        # The coil: the LOOSEST the state variable got over the window the
        # breakout is measured across, using bars strictly before N. A rolling
        # maximum, not a minimum, because the thesis is a *sustained*
        # contraction — one quiet bar inside an otherwise noisy stretch is not
        # a coil, and a minimum would call it one. shift(1) is the central
        # no-look-ahead invariant here: bar N's own volatility is already
        # contaminated by the resolution we are trying to confirm, so it must
        # not be allowed to describe the compression that preceded it.
        out["coil_percentile"] = (
            out["vol_percentile"]
            .shift(1)
            .rolling(vol.breakout_period, min_periods=vol.breakout_period)
            .max()
        )

        # The ceiling the coil has to resolve through, from the preceding
        # window only: today's close is compared with the high of bars < N.
        out["prior_breakout_high"] = (
            out["close"]
            .shift(1)
            .rolling(vol.breakout_period, min_periods=vol.breakout_period)
            .max()
        )

        # How far past that ceiling the completed bar closed, measured in ATR
        # so it is comparable across symbols. A non-positive or missing ATR
        # yields NaN, which never scores and therefore never trades —
        # degenerate inputs fail closed instead of dividing by zero.
        atr_series = out["atr"]
        thrust = (out["close"] - out["prior_breakout_high"]).where(atr_series > 0) / atr_series
        out["thrust"] = thrust.replace([np.inf, -np.inf], np.nan)

        # Priority among simultaneous candidates: a deep coil resolving hard
        # outranks a shallow coil drifting through its ceiling. Both factors
        # are dimensionless, so the product is comparable across symbols.
        squeeze = (100.0 - out["coil_percentile"]) / 100.0
        out["score"] = (squeeze * out["thrust"]).replace([np.inf, -np.inf], np.nan)
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

        # The one thing a single symbol cannot see: whether the whole universe
        # is in a volatility shock. An idiosyncratic squeeze resolving during a
        # market-wide expansion is far more likely to be a gap than a coil
        # releasing. Note this stays inside the second moment — it is a
        # dispersion statistic, not a direction, so it does not smuggle the
        # correlation with the directional members back in through the gate.
        states = pd.DataFrame(
            {symbol: frame["vol_percentile"] for symbol, frame in prepared.items()}
        )
        # axis=1: one statistic per timestamp, computed only from that
        # timestamp's readings. Never a mean over the column, which would be
        # the classic full-sample look-ahead.
        eligible = states.notna().sum(axis=1)
        # ``median`` skips NaN, so a timestamp at which only one symbol is
        # measurable returns that symbol's own reading — and the "no market-wide
        # shock" gate silently collapses into the ``unexhausted`` filter the
        # entry already applies, while still reporting as passed. A statistic
        # with no cross-sectional content must read as absent, not as calm; the
        # evaluator treats a missing reading as a refusal. Derived from the
        # configured universe rather than a new parameter, and a one-symbol
        # universe keeps its degenerate-but-honest single reading.
        min_participants = 2 if len(self.config.symbols) > 1 else 1
        universe_state = states.median(axis=1).where(eligible >= min_participants)

        for symbol, frame in prepared.items():
            # Exact timestamp alignment: a missing cross-symbol reading stays
            # NaN and blocks entry rather than borrowing information from
            # another bar. Plain reindex, no method, no fill.
            frame["universe_vol_percentile"] = universe_state.reindex(frame.index)
            frame["eligible_count"] = eligible.reindex(frame.index)
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
        # Deepest coil resolving hardest first, so scarce capital reaches the
        # strongest candidate; HOLD/EXIT retain universe order.
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

        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        vol_percentile = _finite(row.get("vol_percentile"))
        coil = _finite(row.get("coil_percentile"))
        ceiling = _finite(row.get("prior_breakout_high"))
        thrust = _finite(row.get("thrust"))
        score = _finite(row.get("score"))
        universe_state = _finite(row.get("universe_vol_percentile"))

        values = {"close": close, "atr": atr_value}

        # A BUY without a finite positive ATR sizes to a zero-width stop in the
        # backtest and raises in live, so this gate comes before everything.
        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        # Gate each side on the columns it actually reads. The exit consults
        # ``vol_percentile`` and nothing else, but ``coil_percentile`` chains one
        # more rolling window on top of it, so there is a stretch of exactly
        # ``breakout_period`` bars at the head of every prepared history where the
        # state variable is readable and the entry-side columns are not. Demanding
        # the entry columns here pinned a held position in HOLD across that whole
        # stretch — precisely the window a crash-recovered engine restarts into —
        # even when volatility had already reached the exhausted end and the trade
        # was settled. The entry-side requirement is unchanged, only moved below
        # the exit so it gates the decision that actually depends on it.
        if vol_percentile is None:
            return self._hold(symbol, timestamp, "volatility_state_not_initialized")

        vol = self.vol_config
        # The regime label describes the state variable, not the trend: coiled
        # is RANGE, exhausted is STRESS, and the traversal in between is the
        # only window in which this strategy wants to be long.
        if vol_percentile >= vol.expansion_percentile:
            regime = CryptoRegime.STRESS
        elif vol_percentile <= vol.compression_percentile:
            regime = CryptoRegime.RANGE
        else:
            regime = CryptoRegime.BULL_TREND

        diagnostics = {
            "vol_percentile": vol_percentile,
            "coil_percentile": coil,
            "thrust_atr": thrust,
            "prior_breakout_high": ceiling,
            "universe_vol_percentile": universe_state,
        }

        if held:
            # One traversal of the state variable is the whole trade. The bet
            # was compression -> expansion; once volatility has reached the
            # exhausted end of its own distribution the bet has been settled,
            # whichever way price went. Everything else is left to the
            # protective stop the engine already carries, which is why there is
            # no second, trend-shaped exit here: inventing one would make the
            # exit directional and undo the decorrelation.
            if vol_percentile >= vol.expansion_percentile:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "volatility_expansion_complete",
                    regime, values, score=score, details=diagnostics,
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "expansion_underway", regime, values,
                score=score, details=diagnostics,
            )

        if None in (coil, ceiling, thrust, score):
            return self._hold(symbol, timestamp, "volatility_state_not_initialized")

        filters = {
            # Compression held across the whole window being resolved, not
            # merely on one bar of it.
            "compression": coil <= vol.compression_percentile,
            # ...and has not already spent itself.
            "unexhausted": vol_percentile < vol.expansion_percentile,
            # The completed bar resolved the coil upward. Direction is read off
            # the tape, never predicted.
            "breakout": close > ceiling,
            "thrust": thrust >= vol.min_thrust_atr,
            # No market-wide volatility shock; a missing reading fails closed.
            "universe_calm": (
                universe_state is not None
                and universe_state < vol.expansion_percentile
            ),
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol, BotAction.HOLD, timestamp, f"blocked_{'_'.join(failed)}",
                regime, values, filters=filters, score=score, details=diagnostics,
            )
        return self._decision(
            symbol, BotAction.BUY, timestamp, "compression_resolved_upward",
            regime, values, filters=filters, score=score,
            stop_price=close - atr_value * self.config.atr_stop_multiplier,
            details=diagnostics | {"eligible": _finite(row.get("eligible_count"))},
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
        # ``atr_percentile`` is deliberately left unset. The field is an input
        # to the risk layer's volatility de-risking, not a diagnostic, and this
        # strategy enters precisely when volatility is compressed — reporting a
        # low percentile there would enlarge every position it takes. The
        # sizing layer keeps its own conservative default; the volatility state
        # this strategy actually reads is reported in ``details`` instead.
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
