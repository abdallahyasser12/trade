"""Market breadth — a universe-level state variable, not another price filter.

Every other decision engine in this repository asks a question about one symbol
at a time. The audited breakout strategy asks "is *this* symbol breaking out,
trending, liquid and un-stressed?"; the cross-sectional engine asks "is *this*
symbol the strongest in the universe right now?" — a relative question, but one
still answered from a single instant of a single price path. Both are, in the
end, functions of price. Stacking more of them buys correlated opinions: when
the price path is misleading, every member is misled at the same moment and for
the same reason.

Breadth measures something a price series structurally cannot express: **how
many**.

    breadth(t) = (# symbols above their own trend at t) / (# measurable at t)

The numerator is a *count*. It is invariant to the size of any individual move,
which is precisely the quantity momentum is made of. A universe can print a new
high on the back of one symbol while the other seven roll over — index up,
breadth down. That divergence is not a lagged or rescaled version of momentum;
it is a different observable, and it is the reason this module earns a slot in
an ensemble rather than adding another correlated vote to it.

The trading rules follow directly from that one measurement:

* **Risk-on** requires breadth above a floor *and* not deteriorating over a
  fixed window. "High but falling" is the divergence case, and it is treated as
  risk-off — that asymmetry is the entire point of measuring the slope.
* **Selection** is not a separate model. Inside a risk-on universe the strategy
  buys the symbols that are themselves above their own trend, ordered by how far
  above it they sit, in their own volatility units, times how *persistently*
  they have been there. A symbol that crossed one bar ago scores near zero even
  with a wide gap; participation, not one lucky close, is what breadth is about.
* **Exit** has two levels, and they do different jobs. The universe-level one
  is the crash guard: when breadth falls under the exit floor the whole book
  leaves at once, on one shared signal, costing nothing while the universe is
  healthy. The symbol-level one is just the entry condition read backwards — a
  position below its own trend is by definition no longer participating, and
  claiming otherwise would make the score a fiction. Expect the symbol-level
  rule to fire far more often; the breadth floor is not a frequent exit and is
  not meant to be. Turnover was the measured killer in every prior experiment
  here, so the exit floor sits well below the entry floor, and that gap is the
  only hysteresis this design carries.

Two failure modes learned the expensive way, both encoded structurally rather
than by convention:

1. **Symbols without enough history are excluded from the denominator, never
   counted as "below trend".** Counting a warming-up symbol as a negative vote
   biases breadth down at the start of every series and at every universe
   change, which manufactures a fake risk-off regime exactly where the backtest
   has the most data.
2. **Breadth is built from a wide frame aligned on exact timestamps**, the same
   construction the cross-sectional rank uses. Every symbol is counted against
   the *same* bar; a missing bar is missing, never borrowed from a neighbour.

Six parameters. Every parameter is overfitting surface, and a state variable
that needs a dozen of them is not a state variable.

Causality: every column is computed from completed bars only. The decision at
bar N uses closes through N and executes at the N+1 open, identical to the
audited strategy. Nothing here reindexes forward or fills backward, and a
denominator that could be zero yields NaN rather than a trade.

This is a hypothesis to be tested, not a promise. It runs through the same
unmodified cost gate, benchmark comparison and promotion criteria as everything
else, and is expected to be judged by them.
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
from indicators import atr, sma
from models import BotAction


@dataclass(frozen=True)
class BreadthConfig:
    """Deliberately small parameter surface: one trend, one state, two floors."""

    # The trend each symbol is measured against. This is the only per-symbol
    # model in the strategy; everything else is a property of the universe.
    trend_ma_period: int = 50
    # Raw breadth over a small universe is a coarse staircase (0, 1/3, 2/3, 1).
    # Smoothing turns it into a state variable instead of a coin flip.
    breadth_smoothing: int = 5
    # Window for "is breadth deteriorating?" and, per symbol, for how
    # persistently it has held its own trend. One window, two uses — the second
    # is the same question asked of one symbol instead of all of them.
    breadth_slope_period: int = 10
    # Risk-on floor: the fraction of the universe that must be participating.
    entry_breadth: float = 0.60
    # Collapse floor: below this the whole book leaves. Well under the entry
    # floor, because the gap between them is the only hysteresis here and
    # turnover was the measured killer.
    exit_breadth: float = 0.35
    # Minimum symbols with enough history for breadth to mean anything. Below
    # this the fraction is NaN and nothing trades, rather than a 1-of-1 reading
    # masquerading as universe health.
    min_participation: int = 2

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.trend_ma_period < 5:
            errors.append("Breadth trend MA must span at least 5 bars.")
        if self.breadth_smoothing < 1:
            errors.append("Breadth smoothing must be at least 1 bar.")
        if self.breadth_slope_period < 1:
            errors.append("Breadth slope period must be at least 1 bar.")
        if self.min_participation < 2:
            errors.append(
                "Breadth needs at least 2 measurable symbols; a one-symbol "
                "fraction is not a universe statistic."
            )
        for label, value in {
            "entry breadth threshold": self.entry_breadth,
            "exit breadth threshold": self.exit_breadth,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label.capitalize()} must be finite.")
        if math.isfinite(self.entry_breadth) and not (0 < self.entry_breadth <= 1):
            errors.append("Entry breadth threshold must be in (0, 1].")
        if math.isfinite(self.exit_breadth) and not (0 <= self.exit_breadth < 1):
            errors.append("Exit breadth threshold must be in [0, 1).")
        if (
            math.isfinite(self.entry_breadth)
            and math.isfinite(self.exit_breadth)
            and self.exit_breadth >= self.entry_breadth
        ):
            errors.append(
                "Exit breadth threshold must sit below the entry threshold; "
                "without that gap the book churns on every crossing."
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BreadthStrategyConfig(CryptoStrategyConfig):
    """A base config that carries its own breadth parameters.

    The audited walk-forward searches by calling ``dataclasses.replace`` on
    whatever config it was handed and de-duplicates candidates by their
    ``to_dict``. Frozen-dataclass inheritance means the replacement preserves
    both this subclass and this field, and ``asdict`` recurses into it — so two
    candidates that differ only in ``trend_ma_period`` or ``entry_breadth`` are
    genuinely distinct rather than collapsing into one.

    Without it the grid could only vary keys this engine never reads, which
    would run N identical backtests per fold, pick the argmax arbitrarily, and
    report a candidate count — and therefore a multiple-testing penalty — for a
    search that never happened.
    """

    breadth: BreadthConfig = BreadthConfig()

    def validate(self) -> list[str]:
        return super().validate() + self.breadth.validate()


def with_breadth_config(
    config: CryptoStrategyConfig, breadth: BreadthConfig
) -> BreadthStrategyConfig:
    """Attach breadth parameters to any base config, idempotently."""

    if isinstance(config, BreadthStrategyConfig):
        return replace(config, breadth=breadth)
    values = {field.name: getattr(config, field.name) for field in fields(CryptoStrategyConfig)}
    return BreadthStrategyConfig(**values, breadth=breadth)


def resolve_breadth_config(
    config: CryptoStrategyConfig, fallback: Optional[BreadthConfig] = None
) -> BreadthConfig:
    """Prefer parameters the candidate carries; fall back to the run default."""

    carried = getattr(config, "breadth", None)
    if isinstance(carried, BreadthConfig):
        return carried
    return fallback or BreadthConfig()


class BreadthStrategy:
    """Universe-health gating with participation selection, engine-compatible."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        breadth_config: Optional[BreadthConfig] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        # A candidate produced by the walk-forward search carries the parameters
        # it was selected with; an explicit argument still wins.
        self.breadth_config = breadth_config or resolve_breadth_config(config)
        breadth_errors = self.breadth_config.validate()
        if breadth_errors:
            raise ValueError("Invalid breadth configuration: " + "; ".join(breadth_errors))
        if self.breadth_config.min_participation > len(config.symbols):
            # Otherwise breadth is NaN at every bar and the strategy silently
            # never trades, which is the one failure mode that looks like a
            # result instead of a bug.
            raise ValueError(
                "Invalid breadth configuration: min_participation "
                f"({self.breadth_config.min_participation}) exceeds the "
                f"{len(config.symbols)}-symbol universe."
            )

    def required_bars(self) -> int:
        """Enough history for the slowest column, including the breadth slope."""

        breadth = self.breadth_config
        return max(
            self.config.atr_period + 2,
            breadth.trend_ma_period + 2,
            # Persistence above a symbol's own trend: two chained windows.
            breadth.trend_ma_period + breadth.breadth_slope_period + 2,
            # Smoothed breadth, then differenced over the slope window: three.
            breadth.trend_ma_period
            + breadth.breadth_smoothing
            + breadth.breadth_slope_period
            + 2,
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

        breadth = self.breadth_config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)
        out["trend_ma"] = sma(out["close"], breadth.trend_ma_period)

        # The atom breadth is counted from. It is deliberately float 1.0/0.0/NaN
        # rather than a bare boolean comparison: `close > NaN` is False in
        # pandas, which would count every warming-up symbol as a vote against
        # the universe. NaN here means "not measurable", and NaN is skipped by
        # both the numerator and the denominator in prepare_universe.
        measurable = (
            out["trend_ma"].notna() & (out["trend_ma"] > 0) & out["close"].notna()
        )
        membership = (out["close"] > out["trend_ma"]).astype(float).where(measurable)
        out["above_trend"] = membership

        # How persistently this symbol has held its own trend, over the same
        # window the universe slope uses. rolling(min_periods=window) means a
        # single unmeasurable bar inside the window yields NaN rather than a
        # value fabricated from partial history.
        out["participation"] = membership.rolling(
            breadth.breadth_slope_period, min_periods=breadth.breadth_slope_period
        ).mean()

        # Distance above the symbol's own trend, as a fraction (diagnostic) and
        # in the symbol's own ATR units (comparable across BTC and DOGE).
        out["trend_gap"] = (
            out["close"] / out["trend_ma"].where(out["trend_ma"] > 0) - 1.0
        )
        gap_in_atr = (out["close"] - out["trend_ma"]) / out["atr"].where(out["atr"] > 0)

        # Selection score. A zero or non-finite ATR yields NaN, which never
        # sorts and therefore never trades — degenerate inputs fail closed.
        out["score"] = gap_in_atr * out["participation"]
        out["trend_gap"] = out["trend_gap"].replace([np.inf, -np.inf], np.nan)
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

        breadth_cfg = self.breadth_config

        # The one genuinely cross-sectional computation. Built from a wide frame
        # so every symbol is counted against exactly the same bar, never a
        # neighbouring one — identical construction to the cross-sectional rank.
        membership = pd.DataFrame(
            {symbol: frame["above_trend"] for symbol, frame in prepared.items()}
        )
        # Denominator: symbols that *can* be measured at this bar. A symbol
        # still warming up, or missing a bar entirely, is excluded rather than
        # counted as below trend; counting it would bias breadth down at the
        # start of every series and at every universe change.
        measurable = membership.notna().sum(axis=1)
        participating = membership.sum(axis=1, min_count=1)
        raw_breadth = participating.divide(measurable.where(measurable > 0))
        raw_breadth = raw_breadth.replace([np.inf, -np.inf], np.nan).where(
            measurable >= breadth_cfg.min_participation
        )

        smoothed = raw_breadth.rolling(
            breadth_cfg.breadth_smoothing, min_periods=breadth_cfg.breadth_smoothing
        ).mean()
        # Divergence term: the level alone cannot distinguish "healthy" from
        # "still high but emptying out". A plain difference of the smoothed
        # series against its own past — no future bar is consulted.
        change = smoothed - smoothed.shift(breadth_cfg.breadth_slope_period)

        # Nullable booleans: an undefined breadth is pd.NA, not False, so the
        # evaluator can tell "the universe is unhealthy" apart from "we do not
        # know yet" and refuse to enter on either.
        risk_on = ((smoothed >= breadth_cfg.entry_breadth) & (change >= 0)).astype(
            "boolean"
        ).where(smoothed.notna() & change.notna())
        collapse = (smoothed < breadth_cfg.exit_breadth).astype("boolean").where(
            smoothed.notna()
        )

        for symbol, frame in prepared.items():
            # Exact timestamp alignment: a missing universe bar stays NaN/pd.NA
            # and blocks entry rather than borrowing information from another
            # bar. Plain reindex, never a fill method.
            frame["breadth"] = smoothed.reindex(frame.index)
            frame["breadth_change"] = change.reindex(frame.index)
            frame["breadth_participants"] = measurable.reindex(frame.index)
            frame["risk_on"] = risk_on.reindex(frame.index).astype("boolean")
            frame["breadth_collapse"] = collapse.reindex(frame.index).astype("boolean")
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
        # Strongest participation first, so scarce capital reaches the symbols
        # most convincingly inside the healthy part of the universe.
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
        score = _finite(row.get("score"))
        breadth = _finite(row.get("breadth"))
        breadth_change = _finite(row.get("breadth_change"))
        participation = _finite(row.get("participation"))
        participants = _finite(row.get("breadth_participants"))
        membership = _finite(row.get("above_trend"))
        above_trend = None if membership is None else membership > 0.5
        risk_on = _optional_bool(row.get("risk_on"))
        collapsed = _optional_bool(row.get("breadth_collapse"))

        values = {
            "close": close,
            "atr": atr_value,
            "momentum": _finite(row.get("trend_gap")),
        }
        state = {
            "breadth": breadth,
            "breadth_change": breadth_change,
            "participants": participants,
            "participation": participation,
        }

        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")

        if collapsed is True:
            regime = CryptoRegime.STRESS
        elif risk_on is True:
            regime = CryptoRegime.BULL_TREND
        elif breadth is not None:
            regime = CryptoRegime.RANGE
        else:
            regime = CryptoRegime.UNKNOWN

        if held:
            # One universe-level gate empties the whole book at once: cheap,
            # shared, and low-turnover compared with per-symbol crash rules.
            if collapsed is True:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "breadth_collapse",
                    regime, values, score=score, details=state,
                )
            if above_trend is False:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "left_own_trend",
                    regime, values, score=score, details=state,
                )
            if above_trend is None or collapsed is None:
                # Universe health is unknown, not bad. Do nothing and let the
                # engine's protective stop carry the position; churning the
                # book on a warmup gap is the expensive mistake here.
                return self._hold(symbol, timestamp, "breadth_state_unavailable", state)
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "participating_in_healthy_breadth",
                regime, values, score=score, details=state,
            )

        if score is None or above_trend is None or risk_on is None:
            return self._hold(symbol, timestamp, "breadth_not_initialized", state)

        stop_price = close - atr_value * self.config.atr_stop_multiplier
        filters = {
            # Universe first: the gate is what makes this a breadth strategy
            # rather than a slow moving-average crossover.
            "breadth_risk_on": risk_on,
            "own_trend": above_trend,
            # A "stop" that is not strictly between zero and the entry price is
            # not a stop, and a BUY carrying one is a signal whose own risk is
            # undefined. Reachable whenever ATR grows past close /
            # atr_stop_multiplier — a violent expansion in a symbol that is
            # still above its trend. The audited strategy refuses the entry in
            # exactly this case rather than emitting it; so does this one.
            "protective_stop": math.isfinite(stop_price) and 0.0 < stop_price < close,
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol, BotAction.HOLD, timestamp, f"blocked_{'_'.join(failed)}",
                regime, values, filters=filters, score=score, details=state,
            )
        return self._decision(
            symbol, BotAction.BUY, timestamp, "breadth_confirmed_participation",
            regime, values, filters=filters, score=score,
            stop_price=stop_price,
            details=state,
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


def _optional_bool(value) -> Optional[bool]:
    """None for a missing nullable boolean, so "unknown" never reads as False."""

    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return bool(value)
