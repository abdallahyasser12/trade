"""Exit intelligence: trade management independent of entries.

One deterministic policy object owns everything that happens after a fill:
protective-stop ratcheting, optional partial profit-taking, optional
break-even, optional time stops, and the extreme-winner state machine

    NORMAL_POSITION -> STRONG_WINNER -> RUNNER

The policy is pure: it reads a ``ManagedPosition`` snapshot plus one
COMPLETED bar's values and returns instructions.  Instructions produced
from bar N are executable no earlier than bar N+1 — the policy never
retro-tests a stop against the bar that created it (the same contract as
``RiskManager.calculate_ratchet_stop``).

Nothing here is assumed to improve expectancy.  Every behavior is a flag
with a conservative default of OFF (except plain ATR trailing, which the
production engines already use), so the research simulator can measure
each exit feature's incremental value before anything enables it in a
deployed configuration.  In particular break-even is NOT enabled merely
because it feels safe — that is an empirical question (section 34).

The runner mechanics implement section 35: a position that reaches an
exceptional R-multiple with intact trend quality first LOCKS a profit
floor (STRONG_WINNER), then optionally reduces to a residual runner with
a wide trail so a genuine +100% move is never cut short by the standard
trail, while the locked floor guarantees the trade can no longer round
trip to a loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


class PositionPhase(str, Enum):
    NORMAL = "NORMAL_POSITION"
    STRONG_WINNER = "STRONG_WINNER"
    RUNNER = "RUNNER"


class ExitInstructionKind(str, Enum):
    UPDATE_STOP = "UPDATE_STOP"        # raise the protective stop
    PARTIAL_EXIT = "PARTIAL_EXIT"      # sell a fraction at next opportunity
    FULL_EXIT = "FULL_EXIT"            # close the position at next opportunity


@dataclass(frozen=True)
class ExitInstruction:
    kind: ExitInstructionKind
    reason: str
    new_stop: Optional[float] = None
    exit_fraction: Optional[float] = None  # fraction of ORIGINAL quantity


@dataclass(frozen=True)
class ExitPolicyConfig:
    """Every exit behavior, individually researchable.

    R-based triggers are measured from the completed close against the
    initial risk (entry - initial stop).
    """

    trailing_atr_multiplier: float = 3.0
    enable_trailing_stop: bool = True
    enable_partial_take: bool = False
    partial_take_r: float = 2.0
    partial_take_fraction: float = 0.5
    enable_break_even: bool = False
    break_even_trigger_r: float = 1.5
    enable_time_stop: bool = False
    time_stop_bars: int = 20
    time_stop_max_r: float = 0.5
    enable_runner: bool = False
    strong_winner_trigger_r: float = 2.5
    strong_winner_lock_r: float = 1.0
    runner_trigger_r: float = 4.0
    runner_fraction: float = 0.25
    runner_trail_atr_multiplier: float = 5.0
    # Trend-quality floor for promoting a winner: skip promotion when the
    # bar's efficiency ratio is known and below this (a chop-driven spike
    # is not a trend).  Unknown efficiency does not block promotion.
    runner_min_efficiency: float = 0.30

    def validate(self) -> list[str]:
        errors: list[str] = []
        positive = {
            "trailing ATR multiplier": self.trailing_atr_multiplier,
            "partial-take trigger": self.partial_take_r,
            "break-even trigger": self.break_even_trigger_r,
            "strong-winner trigger": self.strong_winner_trigger_r,
            "runner trigger": self.runner_trigger_r,
            "runner trail multiplier": self.runner_trail_atr_multiplier,
        }
        for label, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and greater than zero.")
        if not (0.0 < self.partial_take_fraction < 1.0):
            errors.append("Partial-take fraction must be in (0, 1).")
        if not (0.0 < self.runner_fraction < 1.0):
            errors.append("Runner fraction must be in (0, 1).")
        if self.enable_time_stop and self.time_stop_bars < 1:
            errors.append("Time stop must allow at least one bar.")
        if not math.isfinite(self.time_stop_max_r):
            errors.append("Time-stop R ceiling must be finite.")
        if not math.isfinite(self.strong_winner_lock_r) or self.strong_winner_lock_r < 0:
            errors.append("Strong-winner lock must be finite and non-negative.")
        if self.strong_winner_lock_r >= self.strong_winner_trigger_r:
            errors.append("Strong-winner lock must sit below its trigger.")
        if self.enable_runner and self.runner_trigger_r <= self.strong_winner_trigger_r:
            errors.append("Runner trigger must exceed the strong-winner trigger.")
        if self.runner_trail_atr_multiplier < self.trailing_atr_multiplier:
            errors.append("Runner trail must be at least as wide as the normal trail.")
        if not (0.0 <= self.runner_min_efficiency <= 1.0):
            errors.append("Runner efficiency floor must be within [0, 1].")
        return errors


@dataclass(frozen=True)
class ManagedPosition:
    """Causal snapshot of one open long position under management.

    ``original_qty`` is the fill quantity at entry; ``remaining_fraction``
    is the fraction of it still open (1.0 before any partial exits).
    ``highest_close`` covers completed bars only.
    """

    entry_price: float
    initial_stop_price: float
    current_stop_price: float
    highest_close: float
    bars_held: int
    remaining_fraction: float = 1.0
    phase: PositionPhase = PositionPhase.NORMAL
    partial_taken: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        values = {
            "entry price": self.entry_price,
            "initial stop": self.initial_stop_price,
            "current stop": self.current_stop_price,
            "highest close": self.highest_close,
        }
        for label, value in values.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and positive.")
        if self.initial_stop_price >= self.entry_price:
            errors.append("Initial stop must sit below the entry price.")
        if not (0.0 < self.remaining_fraction <= 1.0):
            errors.append("Remaining fraction must be in (0, 1].")
        if self.bars_held < 0:
            errors.append("Bars held cannot be negative.")
        return errors

    @property
    def initial_risk(self) -> float:
        return self.entry_price - self.initial_stop_price

    def r_multiple(self, price: float) -> float:
        return (price - self.entry_price) / self.initial_risk


@dataclass
class ManagementUpdate:
    """The policy's verdict for one completed bar."""

    position: ManagedPosition
    instructions: list[ExitInstruction] = field(default_factory=list)


class TradeManagementPolicy:
    """Deterministic post-entry trade management."""

    def __init__(self, config: Optional[ExitPolicyConfig] = None):
        self.config = config or ExitPolicyConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid exit policy configuration: " + "; ".join(errors))

    def on_completed_bar(
        self,
        position: ManagedPosition,
        *,
        close: float,
        atr_value: float,
        efficiency_ratio: Optional[float] = None,
    ) -> ManagementUpdate:
        """Evaluate management rules on one completed bar.

        Returns the next position snapshot plus instructions.  A FULL_EXIT
        instruction ends evaluation; stop updates returned here must be
        applied to future execution only.
        """
        errors = position.validate()
        if errors:
            raise ValueError("Invalid managed position: " + "; ".join(errors))
        if not math.isfinite(close) or close <= 0:
            raise ValueError("Completed close must be finite and positive.")
        if not math.isfinite(atr_value) or atr_value <= 0:
            raise ValueError("ATR must be finite and positive.")

        cfg = self.config
        instructions: list[ExitInstruction] = []
        highest = max(position.highest_close, close)
        r_now = position.r_multiple(close)
        phase = position.phase
        partial_taken = position.partial_taken
        remaining = position.remaining_fraction
        stop = position.current_stop_price

        # --- time stop: stagnant trades only -----------------------------
        if (
            cfg.enable_time_stop
            and position.bars_held + 1 >= cfg.time_stop_bars
            and r_now < cfg.time_stop_max_r
        ):
            updated = replace(
                position,
                highest_close=highest,
                bars_held=position.bars_held + 1,
            )
            return ManagementUpdate(
                position=updated,
                instructions=[
                    ExitInstruction(
                        kind=ExitInstructionKind.FULL_EXIT,
                        reason="time_stop_stagnant",
                    )
                ],
            )

        # --- phase promotion ---------------------------------------------
        quality_ok = (
            efficiency_ratio is None
            or not math.isfinite(efficiency_ratio)
            or efficiency_ratio >= cfg.runner_min_efficiency
        )
        if (
            cfg.enable_runner
            and phase is PositionPhase.NORMAL
            and r_now >= cfg.strong_winner_trigger_r
            and quality_ok
        ):
            phase = PositionPhase.STRONG_WINNER
        if (
            cfg.enable_runner
            and phase is PositionPhase.STRONG_WINNER
            and r_now >= cfg.runner_trigger_r
            and quality_ok
        ):
            phase = PositionPhase.RUNNER
            if remaining > cfg.runner_fraction:
                take = remaining - cfg.runner_fraction
                instructions.append(
                    ExitInstruction(
                        kind=ExitInstructionKind.PARTIAL_EXIT,
                        reason="runner_reduction",
                        exit_fraction=take,
                    )
                )
                remaining = cfg.runner_fraction

        # --- standard partial take ---------------------------------------
        if (
            cfg.enable_partial_take
            and not partial_taken
            and phase is not PositionPhase.RUNNER
            and r_now >= cfg.partial_take_r
        ):
            take = min(cfg.partial_take_fraction, remaining)
            if take > 0 and take < remaining:
                instructions.append(
                    ExitInstruction(
                        kind=ExitInstructionKind.PARTIAL_EXIT,
                        reason="partial_profit_take",
                        exit_fraction=take,
                    )
                )
                remaining -= take
                partial_taken = True

        # --- stop management ----------------------------------------------
        candidates = [stop]
        if cfg.enable_trailing_stop:
            trail_multiplier = (
                cfg.runner_trail_atr_multiplier
                if phase is PositionPhase.RUNNER
                else cfg.trailing_atr_multiplier
            )
            candidates.append(highest - atr_value * trail_multiplier)
        if cfg.enable_break_even and r_now >= cfg.break_even_trigger_r:
            candidates.append(position.entry_price)
        if phase in (PositionPhase.STRONG_WINNER, PositionPhase.RUNNER):
            # The profit floor a promoted winner may never surrender.
            candidates.append(
                position.entry_price + cfg.strong_winner_lock_r * position.initial_risk
            )
        next_stop = max(candidates)
        if next_stop > stop:
            instructions.append(
                ExitInstruction(
                    kind=ExitInstructionKind.UPDATE_STOP,
                    reason="stop_ratchet",
                    new_stop=next_stop,
                )
            )
            stop = next_stop

        updated = replace(
            position,
            current_stop_price=stop,
            highest_close=highest,
            bars_held=position.bars_held + 1,
            remaining_fraction=remaining,
            phase=phase,
            partial_taken=partial_taken,
        )
        return ManagementUpdate(position=updated, instructions=instructions)
