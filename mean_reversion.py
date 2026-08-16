"""Short-horizon mean reversion — the opposite sign, deliberately.

Every decision engine in this repository so far asks the same question with
different vocabulary. The audited breakout asks "is this one breaking out?".
The ensemble asks it twice and votes. The cross-sectional engine asks "which
one is moving most relative to its own noise?". All three are long-only
trend-followers reading the same closes, and the field assessment shows what
that costs: eight experiments, and the seven trend configurations bled a
near-constant amount in nine folds out of nine, independent of whether the
benchmark did +83% or -38%. That flatness is the tell. Members which share a
hypothesis do not diversify each other — they agree, and when the hypothesis
is wrong they agree about being wrong, all on the same bar.

This module takes the other side on purpose:

    z = (close - rolling_mean(N)) / rolling_std(N)

and buys when ``z`` is sufficiently *negative* — that is, precisely when a
momentum reading of the same bars is at its weakest. The sign of the entry
condition is inverted with respect to every existing member, so the
correlation between this member's entry timing and theirs is negative by
construction rather than by fortunate parameterisation. That is the entire
argument for its existence. A council member that adds information must be
able to be right on the days the incumbents are wrong; a member that merely
re-derives the incumbents' signal with new constants adds variance and
friction and nothing else.

What it is not: a bet against trend. The long-horizon filter (``close >
trend_ma``) is retained and is deliberately much slower than the z-score
window. The hypothesis is "an intact uptrend that has been pushed
statistically far below its own recent mean tends to snap back", not "falling
things rise". Buying weakness without a trend filter is buying a falling
knife, and nothing in the evidence base supports that.

Turnover is the known killer, and mean reversion is the family most exposed
to it. Round-trip friction here is 75 bps, and the diagnosed gradient across
experiments 1, 6 and 7 was that profit factor rose monotonically as cost per
unit of risk fell — the signal only ever had a chance once friction stopped
dominating. A raw z-score crossing has no turnover ceiling at all: a series
oscillating around its mean triggers on nearly every bar. So entries are
rate-limited by ``cooldown_bars``, which honours at most one entry signal per
``cooldown_bars + 1`` bars per symbol and simultaneously imposes a minimum
holding period of the same length before the reversion exit may fire. The
resulting worst case is a single auditable number that should be computed
before any walk-forward is run:

    max round trips / year / symbol = bars_per_year / (cooldown_bars + 1)
    worst-case friction              = that count x 75 bps

At ``1Day`` bars with the default cooldown of 10 that ceiling is ~33 round
trips a year per symbol; on ``1Hour`` bars the same setting permits ~800, which
is indefensible. The parameter is not a formality — it must be sized against
the timeframe, and the ceiling above is how you check that it was.

The cooldown alone is not sufficient, because it can only damp the exits it is
allowed to damp, and it is never allowed to damp a risk exit. The second
turnover defect is structural and specific to this hypothesis: a dip deep
enough to enter sits near the long trend line by construction, so an exit that
tests the same line is adjacent to the entry and whipsaws out within a few
bars. The trend-break exit therefore carries a dead band one dispersion unit
wide — the trend is broken when price is below the long MA by more than the
short-horizon noise the strategy is buying, not when it merely touches it.
That reuses the z-score's own denominator and so costs no parameter.

Six parameters. Every parameter is overfitting surface, and the reduction is
deliberate: no oscillator band, no volume gate, no volatility percentile, no
per-symbol tuning. The one cross-sectional column is the same benchmark regime
gate the audited strategy already uses.

Causality: every column is computed from completed bars only. The decision at
bar N uses closes through N and executes at the N+1 open, identical to the
audited strategy. Nothing here reindexes forward or fills backward. The
cooldown state is a forward recursion over completed triggers, so the value at
bar N is a function of bars <= N and truncating history cannot change it.

Degenerate inputs fail closed. A flat window has zero dispersion, which would
make ``z`` infinite; the division is masked to NaN instead, and NaN never
compares true against any threshold, so it can never trade.

This is a hypothesis to be tested, not a promise. It runs through the same
unmodified cost gate, benchmark comparison and promotion criteria as
everything else, and is expected to be judged by them.
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
class MeanReversionConfig:
    """Deliberately small parameter surface: six knobs, no per-symbol tuning."""

    # Short horizon: the window the deviation is measured against.
    z_period: int = 20
    # Long horizon: the trend that must still be intact for a dip to be a dip
    # rather than a breakdown. Validated to be strictly slower than z_period,
    # because if the two horizons collapse this stops being "buy the dip in an
    # uptrend" and becomes an unconditional bet against the last N bars.
    trend_ma_period: int = 100
    # Enter at z <= -entry_z. Expressed as a positive number so the sign of the
    # inversion is visible at the call site.
    entry_z: float = 1.5
    # Exit once z has climbed back to exit_z. Zero means "exit at the mean",
    # which is the conservative reading of reversion; a positive value holds
    # for overshoot and a negative value takes the money early.
    exit_z: float = 0.0
    # The turnover bound. Honours at most one entry signal per
    # cooldown_bars + 1 bars per symbol, and holds an open position for at
    # least that long before the reversion exit is allowed to fire. Friction is
    # 75 bps round trip and mean reversion is naturally the highest-turnover
    # family in this repository; without this the strategy pays the spread on
    # every oscillation and the cost gate is the only thing standing between it
    # and the failure mode already diagnosed in experiments 1 and 7.
    cooldown_bars: int = 10
    # One regime gate, not ten filters. Buying weakness while the benchmark
    # itself is below its regime MA is the falling-knife case.
    require_benchmark_bull: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.z_period < 5:
            errors.append(
                "Z-score window must be at least 5 bars; a shorter one measures "
                "quantisation noise rather than deviation."
            )
        if self.trend_ma_period < 3:
            errors.append("Trend MA period must be at least 3 bars.")
        if self.trend_ma_period <= self.z_period:
            errors.append(
                "Trend MA period must be strictly longer than the z-score window; "
                "otherwise the two horizons collapse and this is no longer a dip "
                "inside an uptrend."
            )
        if self.cooldown_bars < 1:
            errors.append(
                "Cooldown must be at least 1 bar; unbounded turnover is the "
                "measured failure mode of this strategy family."
            )
        if not math.isfinite(self.entry_z):
            errors.append("Entry z-score must be finite.")
        elif self.entry_z <= 0:
            errors.append(
                "Entry z-score must be positive; a non-positive threshold buys "
                "strength and turns this into another momentum member."
            )
        if not math.isfinite(self.exit_z):
            errors.append("Exit z-score must be finite.")
        if (
            math.isfinite(self.entry_z)
            and math.isfinite(self.exit_z)
            and self.exit_z <= -self.entry_z
        ):
            errors.append(
                "Exit z-score must sit above the entry z-score; otherwise the "
                "entry and exit conditions overlap and a position is closed on "
                "the bar it opens."
            )
        # Both thresholds are compared against a z-score measured over the very
        # window the point itself belongs to, and that statistic is bounded:
        # |z| <= (n - 1) / sqrt(n) exactly, for a window of n observations.
        # A threshold at or beyond that ceiling is not merely strict, it is
        # unreachable — the comparison can never be true, on any data, and
        # nothing downstream reports an error because NaN is not involved. The
        # ceiling bites at exactly the sizes a search explores: at the minimum
        # legal z_period of 5 it is 1.789, barely above the default entry_z of
        # 1.5, so a grid over (z_period, entry_z) walks into it immediately and
        # candidates that can never open a trade are reported as honest
        # zero-trade results rather than as invalid configurations. An
        # unreachable exit_z is worse, because it fails silently in the other
        # direction: entries fire normally and the reversion exit never does,
        # so the member quietly stops being mean reversion and becomes a
        # hold-until-stopped trend follower whose backtest looks plausible.
        if self.z_period >= 5:
            ceiling = (self.z_period - 1) / math.sqrt(self.z_period)
            if math.isfinite(self.entry_z) and self.entry_z >= ceiling:
                errors.append(
                    f"Entry z-score {self.entry_z} is unreachable for a "
                    f"{self.z_period}-bar window: a close scored against its "
                    f"own window satisfies |z| <= {ceiling:.3f}, so this "
                    "threshold can never trigger and the strategy would never "
                    "trade. Lengthen z_period or lower entry_z."
                )
            if math.isfinite(self.exit_z) and self.exit_z >= ceiling:
                errors.append(
                    f"Exit z-score {self.exit_z} is unreachable for a "
                    f"{self.z_period}-bar window (|z| <= {ceiling:.3f}), so the "
                    "reversion exit could never fire and positions would leave "
                    "only through a risk exit. Lengthen z_period or lower "
                    "exit_z."
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeanReversionStrategyConfig(CryptoStrategyConfig):
    """A base config that carries its own mean-reversion parameters.

    Identical mechanics to ``CrossSectionalStrategyConfig`` and for the same
    reason: the audited walk-forward searches by calling ``dataclasses.replace``
    on whatever config it was handed and de-duplicates candidates by their
    ``to_dict``. Frozen-dataclass inheritance preserves both this subclass and
    this field, and ``asdict`` recurses into it — so two candidates differing
    only in ``z_period`` or ``cooldown_bars`` are genuinely distinct rather than
    collapsing into one.

    Without it the grid could only vary keys this engine never reads, which
    would run N identical backtests per fold, pick the argmax arbitrarily, and
    report a multiple-testing penalty for a search that never happened.
    """

    mean_reversion: MeanReversionConfig = MeanReversionConfig()

    def validate(self) -> list[str]:
        return super().validate() + self.mean_reversion.validate()


def with_mean_reversion_config(
    config: CryptoStrategyConfig, mean_reversion: MeanReversionConfig
) -> MeanReversionStrategyConfig:
    """Attach mean-reversion parameters to any base config, idempotently."""

    if isinstance(config, MeanReversionStrategyConfig):
        return replace(config, mean_reversion=mean_reversion)
    values = {field.name: getattr(config, field.name) for field in fields(CryptoStrategyConfig)}
    return MeanReversionStrategyConfig(**values, mean_reversion=mean_reversion)


def resolve_mean_reversion_config(
    config: CryptoStrategyConfig, fallback: Optional[MeanReversionConfig] = None
) -> MeanReversionConfig:
    """Prefer parameters the candidate carries; fall back to the run default."""

    carried = getattr(config, "mean_reversion", None)
    if isinstance(carried, MeanReversionConfig):
        return carried
    return fallback or MeanReversionConfig()


class MeanReversionStrategy:
    """Long-only oversold-dip selection, interface-compatible with the base engine."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        mean_reversion_config: Optional[MeanReversionConfig] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        # A candidate produced by the walk-forward search carries the
        # parameters it was selected with; an explicit argument still wins.
        self.mr_config = mean_reversion_config or resolve_mean_reversion_config(config)
        mr_errors = self.mr_config.validate()
        if mr_errors:
            raise ValueError(
                "Invalid mean-reversion configuration: " + "; ".join(mr_errors)
            )

    def required_bars(self) -> int:
        """Enough history for the slowest column plus the turnover state."""

        mr = self.mr_config
        return max(
            # Benchmark regime filter, read off the benchmark frame.
            self.config.regime_ma_period + 2,
            self.config.atr_period + 2,
            mr.z_period + 2,
            mr.trend_ma_period + 2,
            # Chained: the cooldown recursion consumes triggers, and a trigger
            # needs both the z-score and the trend filter to be initialized.
            max(mr.z_period, mr.trend_ma_period) + mr.cooldown_bars + 2,
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

        mr = self.mr_config
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)
        # Two moving averages with two different jobs, deliberately not merged.
        # ``regime_ma`` is the audited benchmark filter and is only ever read
        # off the benchmark frame in prepare_universe; ``trend_ma`` is this
        # strategy's own long-horizon gate and is read per symbol.
        out["regime_ma"] = sma(out["close"], self.config.regime_ma_period)
        out["trend_ma"] = sma(out["close"], mr.trend_ma_period)

        # Deviation of the completed close from its own recent mean, measured
        # in units of its own recent dispersion. Both windows end at and
        # include bar N, and min_periods equals the window so warmup is honest
        # rather than fabricated from partial history.
        mean = sma(out["close"], mr.z_period)
        dispersion = out["close"].rolling(mr.z_period, min_periods=mr.z_period).std(ddof=1)
        out["dispersion"] = dispersion

        # Masking the numerator on ``dispersion > 0`` is what makes a flat
        # series fail closed: a zero-dispersion window yields NaN, never
        # +/-inf, and NaN compares false against every threshold below, so a
        # degenerate symbol cannot trade. The explicit inf-scrub covers the
        # residual float cases.
        zscore = (out["close"] - mean).where(dispersion > 0) / dispersion
        zscore = zscore.replace([np.inf, -np.inf], np.nan)
        out["zscore"] = zscore
        # The sign flip is the whole thesis. ``score`` is "how oversold", so it
        # is large exactly when a momentum score over the same window is most
        # negative, and the shared BUY ordering below then hands scarce capital
        # to the deepest dip rather than the strongest trend.
        out["score"] = -zscore

        # Diagnostic only, never gated on: the plain short-horizon return over
        # the same window. Kept because it is the number the momentum members
        # trade off, so a side-by-side of the two columns is the cheapest
        # available check that this member really is taking the other side.
        out["momentum"] = out["close"] / out["close"].shift(mr.z_period) - 1.0
        out["momentum"] = out["momentum"].replace([np.inf, -np.inf], np.nan)

        # Raw per-symbol entry condition. Deliberately excludes the benchmark
        # gate so that prepare_symbol stays usable standalone (crash recovery
        # calls it for one symbol), and so the rate limit below is a property
        # of the symbol's own oscillation rather than of the market backdrop.
        # A NaN on either side compares false, so warmup never triggers.
        trigger = (out["score"] >= mr.entry_z) & (out["close"] > out["trend_ma"])
        ready, bars_since = _cooldown_state(
            trigger.to_numpy(dtype=bool), mr.cooldown_bars
        )
        out["entry_ready"] = pd.Series(ready, index=out.index)
        out["bars_since_entry_signal"] = pd.Series(bars_since, index=out.index)
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

        # The only genuinely cross-symbol column: whether the benchmark itself
        # is above its regime MA. Buying an oversold dip while the whole market
        # is below its own trend is the falling-knife case this gate exists to
        # refuse.
        benchmark = prepared[self.config.benchmark_symbol]
        benchmark_bull = benchmark["close"] > benchmark["regime_ma"]

        for symbol, frame in prepared.items():
            # Exact timestamp alignment: a missing benchmark bar stays pd.NA and
            # blocks entry rather than borrowing information from another bar.
            # ``.reindex`` with no ``method=`` is a pure label-align, not a fill;
            # the nullable "boolean" dtype is what keeps the gap distinguishable
            # from an honest False.
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
        # Deepest dip first, so scarce capital reaches the most stretched
        # candidate. Both consumers walk the BUYs in list order and stop when
        # slots run out, so this ordering is load-bearing, not cosmetic.
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
        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        trend_ma = _finite(row.get("trend_ma"))
        dispersion = _finite(row.get("dispersion"))
        entry_ready = _finite(row.get("entry_ready"))
        bars_since = _finite(row.get("bars_since_entry_signal"))
        bull_raw = row.get("benchmark_bull")
        benchmark_bull = bool(bull_raw) if bull_raw is not None and bull_raw is not pd.NA else None

        values = {
            "close": close,
            "atr": atr_value,
            "momentum": _finite(row.get("momentum")),
        }

        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        if score is None or trend_ma is None:
            return self._hold(symbol, timestamp, "insufficient_history_to_score")

        mr = self.mr_config
        in_uptrend = close > trend_ma
        # Hysteresis, and the reason it is not optional. Entry requires
        # ``close > trend_ma``, and a dip deep enough to be worth buying sits
        # close to that line by construction — measured on the reference
        # universe, a quarter of entries land within one dispersion unit of it.
        # If the exit tested the same line, the two gates would be adjacent and
        # the strategy would whipsaw out one to four bars after entering, paying
        # the full 75 bps round trip for noise. The dead band is one dispersion
        # unit wide: the trend counts as broken only once price is below the
        # long MA by more than the short-horizon noise this strategy exists to
        # buy. It costs no parameter — ``dispersion`` is the same denominator
        # the z-score already uses — and it is dimensionally coherent, both
        # sides being in price units. If dispersion is unavailable the band
        # collapses to the bare line, which exits sooner: unknown fails closed.
        trend_break = close < trend_ma - dispersion if dispersion is not None else close < trend_ma
        regime = CryptoRegime.BULL_TREND if benchmark_bull else CryptoRegime.RANGE
        details = {
            "zscore": -score,
            "trend_ma": trend_ma,
            "bars_since_entry_signal": bars_since,
        }

        if held:
            # Risk exits are evaluated first and are never damped. The holding
            # period bounds turnover; it is not allowed to bound risk, so a
            # broken trend or a benchmark regime break leaves on the next open
            # no matter how recent the entry was. The protective stop itself
            # lives in the risk layer and is untouched here.
            if trend_break:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "trend_filter_broken",
                    regime, values, score=score, details=details,
                )
            if mr.require_benchmark_bull and benchmark_bull is False:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "benchmark_regime_break",
                    regime, values, score=score, details=details,
                )
            # Reverted: z has climbed back to exit_z, so the oversold score has
            # fallen to -exit_z. This is the discretionary exit — the one that
            # would otherwise fire on a single violent snapback bar and pay the
            # full 75 bps for a few hours of exposure — so it is the one the
            # minimum holding period damps.
            if score <= -mr.exit_z:
                if bars_since is not None and bars_since <= mr.cooldown_bars:
                    return self._decision(
                        symbol, BotAction.HOLD, timestamp, "minimum_hold_active",
                        regime, values, score=score, details=details,
                    )
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "reversion_complete",
                    regime, values, score=score, details=details,
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "reversion_pending",
                regime, values, score=score, details=details,
            )

        filters = {
            # score >= entry_z is exactly z <= -entry_z: statistically oversold.
            "oversold": score >= mr.entry_z,
            "trend": in_uptrend,
            "regime": (not mr.require_benchmark_bull) or bool(benchmark_bull),
            # An unknown cooldown state (warmup) fails closed rather than open:
            # not knowing how recently this symbol traded is not a licence to
            # trade it again.
            "cooldown": entry_ready is not None and entry_ready > 0,
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol, BotAction.HOLD, timestamp, f"blocked_{'_'.join(failed)}",
                regime, values, filters=filters, score=score, details=details,
            )
        return self._decision(
            symbol, BotAction.BUY, timestamp, "oversold_dip_in_uptrend",
            regime, values, filters=filters, score=score,
            stop_price=close - atr_value * self.config.atr_stop_multiplier,
            details=details,
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
        """Return a stop that can only tighten, using a completed close.

        Byte-for-byte the audited semantics. Both callers clamp the result
        below the close themselves, so nothing is clamped here.
        """

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


def _cooldown_state(
    trigger: np.ndarray, cooldown_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rate-limit raw entry triggers to one per ``cooldown_bars + 1`` bars.

    Returns ``(entry_ready, bars_since_entry_signal)``:

    * ``entry_ready`` is 1.0 on a bar whose raw trigger is honoured and 0.0
      everywhere else. A trigger is honoured only if the previously honoured
      one was more than ``cooldown_bars`` bars ago, so a persistently oversold
      run produces one entry signal, not one per bar. That distinction is the
      difference between a bounded and an unbounded friction bill.
    * ``bars_since_entry_signal`` counts bars since the most recent honoured
      trigger, and is NaN before the first one. The evaluation layer reads it
      as the minimum holding period: the reversion exit is suppressed while it
      is ``<= cooldown_bars``. A trigger honoured at bar N is acted on at the
      N+1 open, and the first bar on which the reversion exit may fire is
      N + cooldown_bars + 1, so a position is held for at least
      ``cooldown_bars + 1`` bars before it can be closed on signal.

    This counter is a signal-side proxy for the position, not the position
    itself — ``evaluate_prepared`` is handed a set of held symbols and no entry
    timestamps, so the decision layer cannot see fills and must not pretend to.
    On the entry side the proxy is conservative in both directions: a trigger
    the portfolio layer declined to act on still consumes the cooldown (fewer
    entries), and an unknown state blocks entry (fewer entries again).

    On the exit side it is not, and the asymmetry is worth stating plainly. The
    counter is reset by *any* honoured trigger, including one that fires while
    a position is already open — which cannot be an entry, because the symbol
    is already held. The stated invariant still holds, since every position is
    held at least ``cooldown_bars + 1`` bars; but a re-trigger mid-hold re-arms
    the suppression, so the reversion exit on an already-mature position can be
    deferred by up to a further ``cooldown_bars``. The effect is bounded, it
    lowers turnover and it cannot touch a risk exit, but it does lengthen
    exposure rather than shorten it. Removing it needs the position's entry bar,
    which this interface deliberately does not carry.

    Strictly causal: the state at bar N is a function of triggers at bars <= N
    only, so truncating the history leaves every earlier value unchanged.
    """

    count = len(trigger)
    entry_ready = np.zeros(count, dtype=float)
    bars_since = np.full(count, np.nan, dtype=float)
    last: Optional[int] = None
    for position in range(count):
        if trigger[position] and (last is None or position - last > cooldown_bars):
            entry_ready[position] = 1.0
            last = position
        if last is not None:
            bars_since[position] = float(position - last)
    return entry_ready, bars_since


def _finite(value) -> Optional[float]:
    """None for anything that is not a usable finite number."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
