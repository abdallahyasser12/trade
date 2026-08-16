"""A council of decorrelated strategies that votes on each decision.

Why this is built the way it is
-------------------------------

The obvious design — every member votes BUY/HOLD, enter on a majority or on
unanimity — was measured and discarded before this file was written. Two facts
killed it, and both are worth stating because they are not obvious:

**Combining correlated strategies makes things worse, not better.** The regime
ensemble already blends breakout, pullback and mean-reversion sleeves over the
same bars. Out of sample it scored Sharpe −1.80 against the plain breakout
strategy's −0.40, taking 164 trades where the base took 82. Blending members
that share an information source concentrates their shared error and doubles
the friction paid for it. A council only earns its keep if its members
disagree for *different reasons*.

**Decorrelated members cannot reach unanimity.** Measured BUY rates on the same
universe: cross-sectional 15.6%, breadth 22.9%, volatility-regime 0.76%,
mean-reversion 0.40%, with pairwise correlations between −0.01 and 0.03. That
decorrelation is exactly what makes them worth combining — and it means the
probability of all four agreeing is the product of those rates, around one in a
million. A unanimity rule would take zero trades across any realistic history.
Sparse firing and independence are mathematically incompatible with unanimity.

So the council combines *scores*, not votes.

Each member already emits a score per symbol per bar, but on incompatible
scales — cross-sectional spans roughly [−2, +4] while breadth spans [−1.4,
+6.2]. Summing them raw would weight breadth about twice as heavily as
cross-sectional for no reason but its units. That is precisely the
dimensional-incoherence defect that sank the original strategy, and repeating
it here would be inexcusable. Every member's score is therefore converted to
its cross-sectional rank within the universe at that timestamp, mapped to
[−1, +1]. Rank is unitless, always defined, and comparable across members by
construction.

Rank alone is not enough, because it discards level: a member convinced that
every symbol is terrible still ranks its least-terrible symbol at +1. So the
council uses both kinds of information:

* the **combined rank score** decides *which* symbols are relatively strongest,
* the members' **actions** decide whether anything is worth owning at all —
  ``min_proposers`` members must independently issue BUY, and ``min_vetoers``
  members must simultaneously issue EXIT to force a held position out.

Both are counts, not flags, and both are searched rather than assumed.
``min_proposers`` spans the whole spectrum from "any single member may propose"
to full unanimity; the data decides how much agreement to demand.

Turnover
--------

Friction is 75 bps a round trip and turnover has been the measured killer in
every experiment run against this repository. A council that enters on a
majority would trade *more* than any of its members, which is the failure mode
of the regime ensemble. Entering therefore requires ``min_proposers`` agreeing
and a combined score above ``entry_threshold``, while holding continues until
the combined score falls below a strictly lower ``exit_threshold``. The gap
between the two is a hysteresis band that exists solely to stop the book
churning around a decision boundary.

The single-member veto that an earlier draft of this file defaulted to has been
removed for cause, and the reasoning is recorded on ``min_vetoers`` because it
is the most counter-intuitive result here: an exit rule that fires when *any*
member dissents looks conservative and is in fact the most aggressive setting
available, because decorrelated members dissent about something 94.4% of the
time. It turned the best-performing configuration into the worst.

This is a hypothesis, not a promise. It runs through the same unmodified
simulator, cost gate, benchmark comparison and promotion criteria as every
other engine, and the only question that matters about it is whether it beats
its own best single member. If it does not, it is machinery that costs money.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from breadth import BreadthConfig, BreadthStrategy
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from crypto_config import CryptoStrategyConfig
from crypto_models import CryptoRegime, CryptoSignalDecision
from crypto_strategy import CryptoRegimeStrategy
from indicators import atr, sma
from mean_reversion import MeanReversionConfig, MeanReversionStrategy
from models import BotAction
from volatility_regime import VolatilityRegimeConfig, VolatilityRegimeStrategy

# Name -> (strategy class, its config class). Membership is by name so that a
# council configuration stays a plain, serialisable, hashable thing that can
# travel through the walk-forward search and the cache fingerprint.
MEMBER_REGISTRY: dict[str, tuple[type, Optional[type]]] = {
    "cross": (CrossSectionalMomentumStrategy, CrossSectionalConfig),
    "meanrev": (MeanReversionStrategy, MeanReversionConfig),
    "volregime": (VolatilityRegimeStrategy, VolatilityRegimeConfig),
    "breadth": (BreadthStrategy, BreadthConfig),
    "base": (CryptoRegimeStrategy, None),
}

VOTE_MODES = ("weighted", "majority", "unanimous")


@dataclass(frozen=True)
class CouncilConfig:
    """Deliberately small surface: eight parameters for four strategies."""

    members: tuple[str, ...] = ("cross", "meanrev", "volregime", "breadth")
    weights: tuple[float, ...] = ()
    # How much independent agreement an entry needs. 1 means any member may
    # propose; len(members) is unanimity. Measured BUY rates make unanimity
    # unreachable for decorrelated members, which is why this is searched
    # rather than assumed.
    min_proposers: int = 2
    # Combined rank score in [-1, +1] required to open.
    entry_threshold: float = 0.25
    # Strictly lower than entry_threshold: the gap is a hysteresis band that
    # exists to stop the book churning across a boundary at 75 bps a round trip.
    exit_threshold: float = -0.15
    top_k: int = 3
    # Fail closed rather than trade on one opinion when members abstain.
    min_voters: int = 2
    # How many members must simultaneously issue EXIT to force a held position
    # out. 0 disables the veto entirely.
    #
    # This is a count rather than a flag because the flag version was measured
    # and was catastrophic. With four decorrelated members, the probability
    # that at least one issues EXIT on any given bar is 94.4%. A single-member
    # veto therefore made holding almost impossible: entry required two members
    # to agree (uncommon), exit required one to dissent (near-certain), and the
    # book churned at 198 trades against its best member's 69, with profit
    # factor collapsing from 3.49 to 0.51. The asymmetry intended to *reduce*
    # turnover maximised it. Requiring a quorum keeps the crash guard while
    # ignoring the routine disagreement that decorrelation guarantees.
    min_vetoers: int = 0
    vote_mode: str = "weighted"

    def resolved_weights(self) -> tuple[float, ...]:
        if not self.weights:
            return tuple(1.0 for _ in self.members)
        return tuple(float(w) for w in self.weights)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.members:
            errors.append("A council needs at least one member.")
        if len(set(self.members)) != len(self.members):
            errors.append("Council members must be distinct.")
        unknown = [name for name in self.members if name not in MEMBER_REGISTRY]
        if unknown:
            errors.append(f"Unknown council members: {sorted(unknown)}.")
        if self.weights and len(self.weights) != len(self.members):
            errors.append("Council weights must match the member count.")
        if any(not math.isfinite(w) or w <= 0 for w in self.resolved_weights()):
            errors.append("Council weights must be finite and positive.")
        if not 1 <= self.min_proposers <= max(1, len(self.members)):
            errors.append("min_proposers must be between 1 and the member count.")
        if self.min_voters < 1:
            errors.append("min_voters must be at least 1.")
        if self.min_voters > len(self.members):
            errors.append("min_voters cannot exceed the member count.")
        if self.top_k < 1:
            errors.append("top_k must be at least 1.")
        for label, value in (
            ("entry threshold", self.entry_threshold),
            ("exit threshold", self.exit_threshold),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                errors.append(f"Council {label} must be finite and within [-1, 1].")
        if math.isfinite(self.entry_threshold) and math.isfinite(self.exit_threshold):
            if self.exit_threshold >= self.entry_threshold:
                errors.append(
                    "Exit threshold must be strictly below entry threshold; "
                    "without the gap the book churns across the boundary."
                )
        if self.min_vetoers < 0:
            errors.append("min_vetoers cannot be negative.")
        if self.min_vetoers > len(self.members):
            errors.append(
                "min_vetoers cannot exceed the member count; the veto could never fire."
            )
        if self.vote_mode not in VOTE_MODES:
            errors.append(f"Vote mode must be one of {VOTE_MODES}.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["members"] = list(self.members)
        data["weights"] = list(self.resolved_weights())
        return data


class CouncilStrategy:
    """Rank-normalised weighted council, interface-compatible with the base engine."""

    def __init__(
        self,
        config: CryptoStrategyConfig,
        council_config: Optional[CouncilConfig] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto strategy configuration: " + "; ".join(errors))
        self.config = config
        self.council_config = council_config or resolve_council_config(config)
        council_errors = self.council_config.validate()
        if council_errors:
            raise ValueError("Invalid council configuration: " + "; ".join(council_errors))

        self.members: dict[str, Any] = {}
        for name in self.council_config.members:
            strategy_cls, config_cls = MEMBER_REGISTRY[name]
            self.members[name] = (
                strategy_cls(config) if config_cls is None
                else strategy_cls(config, config_cls())
            )
        self._weights = dict(
            zip(self.council_config.members, self.council_config.resolved_weights())
        )

    def required_bars(self) -> int:
        """The slowest member gates the council; anything less starves a voter."""

        return max(member.required_bars() for member in self.members.values())

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Single-symbol path, used by live-engine recovery for atr/close only.

        The council score is cross-sectional and therefore undefined for one
        symbol in isolation, so this deliberately returns only the columns the
        recovery path actually reads rather than inventing a score.
        """

        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(bars, pd.DataFrame) or required.difference(bars.columns):
            raise ValueError("Crypto bars require open/high/low/close/volume columns.")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Crypto bars need a timezone-aware DatetimeIndex.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Crypto bars must be chronological and de-duplicated.")
        out = bars.copy()
        out["atr"] = atr(out["high"], out["low"], out["close"], self.config.atr_period)
        out["regime_ma"] = sma(out["close"], self.config.regime_ma_period)
        return out

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = set(self.config.symbols).difference(bars_by_symbol)
        if missing:
            raise ValueError(f"Missing crypto data for {sorted(missing)}")

        prepared_by_member = {
            name: member.prepare_universe(bars_by_symbol)
            for name, member in self.members.items()
        }

        # Start from the union index so a symbol missing from one member's
        # frame still lines up; reindex without any fill method, so a missing
        # bar stays NaN and simply does not vote.
        base = {
            symbol: self.prepare_symbol(bars_by_symbol[symbol])
            for symbol in self.config.symbols
        }

        # Per member: a wide score frame, ranked across symbols at each single
        # timestamp. Ranking is what makes members with incompatible score
        # scales comparable, and it is done on a wide frame so every symbol is
        # measured against exactly the same bar rather than a neighbouring one.
        normalised: dict[str, pd.DataFrame] = {}
        for name, prepared in prepared_by_member.items():
            wide = pd.DataFrame(
                {
                    symbol: _score_column(prepared.get(symbol))
                    for symbol in self.config.symbols
                }
            )
            normalised[name] = _rank_to_signed_unit(wide)

        weight_total = pd.DataFrame(
            0.0,
            index=normalised[self.council_config.members[0]].index,
            columns=list(self.config.symbols),
        )
        weighted_sum = weight_total.copy()
        voter_count = weight_total.copy()
        for name, frame in normalised.items():
            weight = self._weights[name]
            present = frame.notna()
            weighted_sum = weighted_sum.add(frame.fillna(0.0) * weight, fill_value=0.0)
            weight_total = weight_total.add(present * weight, fill_value=0.0)
            voter_count = voter_count.add(present.astype(float), fill_value=0.0)

        combined = weighted_sum.where(weight_total > 0) / weight_total
        # Fail closed: below the quorum the council has no opinion at all,
        # rather than an opinion held by one member.
        combined = combined.where(voter_count >= self.council_config.min_voters)
        combined = combined.replace([np.inf, -np.inf], np.nan)
        ranks = combined.rank(axis=1, ascending=False, method="first", na_option="keep")

        for symbol in self.config.symbols:
            frame = base[symbol]
            frame["council_score"] = combined[symbol].reindex(frame.index)
            frame["council_rank"] = ranks[symbol].reindex(frame.index)
            frame["council_voters"] = voter_count[symbol].reindex(frame.index)
            for name in self.council_config.members:
                frame[f"vote_{name}"] = normalised[name][symbol].reindex(frame.index)
        self._prepared_by_member = prepared_by_member
        return base

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
        member_prepared = getattr(self, "_prepared_by_member", None)

        # One call per member per bar, with the real held set: a held symbol
        # yields that member's exit opinion (the veto) and an unheld one yields
        # its entry opinion (the proposal). Both are needed and this is the
        # cheapest way to get them.
        member_actions: dict[str, dict[str, BotAction]] = {}
        if member_prepared:
            for name, member in self.members.items():
                try:
                    decisions = member.evaluate_prepared(
                        member_prepared[name], timestamp, held_symbols
                    )
                except Exception:  # noqa: BLE001 - a broken member abstains
                    continue
                member_actions[name] = {d.symbol: d.action for d in decisions}

        decisions = [
            self._evaluate_symbol(
                symbol,
                prepared[symbol],
                timestamp,
                symbol in held_symbols,
                member_actions,
            )
            for symbol in self.config.symbols
        ]
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
        member_actions: Mapping[str, Mapping[str, BotAction]],
    ) -> CryptoSignalDecision:
        if timestamp not in frame.index:
            return self._hold(symbol, timestamp, "bar_unavailable")
        row = frame.loc[timestamp]

        score = _finite(row.get("council_score"))
        rank = _finite(row.get("council_rank"))
        close = _finite(row.get("close"))
        atr_value = _finite(row.get("atr"))
        cfg = self.council_config

        values = {"close": close, "atr": atr_value, "momentum": score}
        if close is None or close <= 0 or atr_value is None or atr_value <= 0:
            return self._hold(symbol, timestamp, "indicators_not_initialized")
        if score is None or rank is None:
            return self._hold(symbol, timestamp, "council_quorum_not_met")

        proposers = [
            name for name, actions in member_actions.items()
            if actions.get(symbol) is BotAction.BUY
        ]
        vetoes = [
            name for name, actions in member_actions.items()
            if actions.get(symbol) is BotAction.EXIT
        ]
        details = {
            "rank": rank,
            "proposers": sorted(proposers),
            "vetoes": sorted(vetoes),
            "voters": _finite(row.get("council_voters")),
        }
        regime = CryptoRegime.BULL_TREND if score > 0 else CryptoRegime.RANGE

        if held:
            if cfg.min_vetoers and len(vetoes) >= cfg.min_vetoers:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "member_veto",
                    regime, values, score=score, details=details,
                )
            # Hysteresis: leaving needs a strictly worse score than entering
            # required, because turnover at 75 bps is the measured killer.
            if score <= cfg.exit_threshold:
                return self._decision(
                    symbol, BotAction.EXIT, timestamp, "council_score_decayed",
                    regime, values, score=score, details=details,
                )
            return self._decision(
                symbol, BotAction.HOLD, timestamp, "council_conviction_intact",
                regime, values, score=score, details=details,
            )

        required = self._required_proposers()
        filters = {
            "score": score >= cfg.entry_threshold,
            "rank": rank <= cfg.top_k,
            "quorum": len(proposers) >= required,
            "veto": not (cfg.min_vetoers and len(vetoes) >= cfg.min_vetoers),
        }
        if not all(filters.values()):
            failed = [name for name, passed in filters.items() if not passed]
            return self._decision(
                symbol, BotAction.HOLD, timestamp, f"blocked_{'_'.join(failed)}",
                regime, values, filters=filters, score=score, details=details,
            )
        return self._decision(
            symbol, BotAction.BUY, timestamp, "council_agreed",
            regime, values, filters=filters, score=score,
            stop_price=close - atr_value * self.config.atr_stop_multiplier,
            details=details,
        )

    def _required_proposers(self) -> int:
        """``vote_mode`` is a coarse preset over the same underlying count."""

        cfg = self.council_config
        if cfg.vote_mode == "unanimous":
            return len(cfg.members)
        if cfg.vote_mode == "majority":
            return len(cfg.members) // 2 + 1
        return cfg.min_proposers

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


@dataclass(frozen=True)
class CouncilStrategyConfig(CryptoStrategyConfig):
    """A base config carrying its council parameters, for the walk-forward search.

    Same rationale as ``CrossSectionalStrategyConfig``: the audited candidate
    builder searches by calling ``dataclasses.replace`` and de-duplicates by
    ``to_dict``, so parameters living on a separate object are invisible to it
    and every candidate would collapse into one.
    """

    council: CouncilConfig = field(default_factory=CouncilConfig)

    def validate(self) -> list[str]:
        return super().validate() + self.council.validate()


def resolve_council_config(
    config: CryptoStrategyConfig, fallback: Optional[CouncilConfig] = None
) -> CouncilConfig:
    carried = getattr(config, "council", None)
    if isinstance(carried, CouncilConfig):
        return carried
    return fallback or CouncilConfig()


def _score_column(frame: Optional[pd.DataFrame]) -> pd.Series:
    """A member's per-symbol score, or an all-NaN column when it cannot vote."""

    if frame is None or "score" not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame["score"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _rank_to_signed_unit(wide: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank per timestamp, mapped to [-1, +1].

    Rank rather than the raw score, because members score on incompatible
    scales and summing them raw would weight a member by its units rather than
    by its information. NaN scores stay NaN: a member that cannot score a
    symbol abstains instead of voting zero, which would otherwise be an
    opinion.
    """

    if wide.empty:
        return wide
    ranks = wide.rank(axis=1, ascending=True, method="average", na_option="keep")
    counts = wide.notna().sum(axis=1)
    # With a single eligible symbol the ranking carries no information, so it
    # sits at neutral rather than being handed a spurious +1.
    denominator = (counts - 1).replace(0, np.nan)
    scaled = (ranks.sub(1).div(denominator, axis=0)) * 2.0 - 1.0
    scaled = scaled.where(counts.gt(1), 0.0).where(wide.notna())
    return scaled.replace([np.inf, -np.inf], np.nan)


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
