"""Monte Carlo trade-sequence stress for promoted candidates (spec §38).

Resampling the OBSERVED out-of-sample R-multiples answers a question the
walk-forward aggregate cannot: how bad can the same edge look when the
same trades arrive in an unlucky order?  Drawdown depth, losing-streak
length, and a risk-of-ruin proxy are all order statistics — one lucky
ordering in the backtest hides them.

Two blocks are reported: BASELINE resamples the trades as measured, and
STRESSED first degrades them deterministically (some winners demoted to
losses, remaining winners shrunk, a flat cost subtracted from every
trade) so the gate reads survival under a worse world, not the world we
happened to record.  Determinism everywhere: one seeded generator, and
the winner demotion picks the SMALLEST winners — the conservative choice,
because demoting large winners would understate how much of the
expectancy the tail carries.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

# Below this many trades a resampled distribution is noise wearing a
# distribution costume; refusing is the honest behavior.
MIN_TRADES_FOR_STRESS = 5


@dataclass(frozen=True)
class StressConfig:
    num_paths: int = 2000
    seed: int = 7
    win_rate_haircut: float = 0.10
    payoff_haircut: float = 0.15
    extra_cost_r: float = 0.05
    ruin_drawdown_r: float = 10.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not (100 <= self.num_paths <= 100_000):
            errors.append("Monte Carlo paths must be between 100 and 100000.")
        if not (0.0 <= self.win_rate_haircut < 1.0):
            errors.append("Win-rate haircut must be in [0, 1).")
        if not (0.0 <= self.payoff_haircut < 1.0):
            errors.append("Payoff haircut must be in [0, 1).")
        if not math.isfinite(self.extra_cost_r) or self.extra_cost_r < 0:
            errors.append("Extra cost must be finite and non-negative.")
        if not math.isfinite(self.ruin_drawdown_r) or self.ruin_drawdown_r <= 0:
            errors.append("Ruin drawdown threshold must be finite and positive.")
        return errors


@dataclass
class StressBlock:
    """Distribution summary over resampled paths for one trade set."""

    trades: int
    expectancy_r: float
    final_r_p5: float
    final_r_p50: float
    final_r_p95: float
    max_drawdown_r_p5: float
    max_drawdown_r_p50: float
    max_drawdown_r_p95: float
    longest_loss_streak_p95: float
    risk_of_ruin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StressReport:
    config: StressConfig
    baseline: StressBlock
    stressed: StressBlock

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "baseline": self.baseline.to_dict(),
            "stressed": self.stressed.to_dict(),
        }


def _apply_haircuts(r_multiples: np.ndarray, config: StressConfig) -> np.ndarray:
    stressed = r_multiples.copy()
    winner_positions = np.flatnonzero(stressed > 0)
    if len(winner_positions) and config.win_rate_haircut > 0:
        demote_count = math.ceil(len(winner_positions) * config.win_rate_haircut)
        # Demote the SMALLEST winners: conservative, deterministic.
        by_size = winner_positions[np.argsort(stressed[winner_positions])]
        stressed[by_size[:demote_count]] = -1.0
    winner_positions = np.flatnonzero(stressed > 0)
    stressed[winner_positions] *= 1.0 - config.payoff_haircut
    stressed -= config.extra_cost_r
    return stressed


def _resample_block(
    r_multiples: np.ndarray, config: StressConfig, rng: np.random.Generator
) -> StressBlock:
    n = len(r_multiples)
    finals = np.empty(config.num_paths)
    drawdowns = np.empty(config.num_paths)
    streaks = np.empty(config.num_paths)
    ruined = 0
    for path in range(config.num_paths):
        sample = rng.choice(r_multiples, size=n, replace=True)
        curve = np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:]
        drawdown = float(np.max(peak - curve))
        finals[path] = curve[-1]
        drawdowns[path] = drawdown
        losses = sample <= 0
        longest = current = 0
        for is_loss in losses:
            current = current + 1 if is_loss else 0
            longest = max(longest, current)
        streaks[path] = longest
        if drawdown >= config.ruin_drawdown_r:
            ruined += 1
    return StressBlock(
        trades=n,
        expectancy_r=float(np.mean(r_multiples)),
        final_r_p5=float(np.percentile(finals, 5)),
        final_r_p50=float(np.percentile(finals, 50)),
        final_r_p95=float(np.percentile(finals, 95)),
        max_drawdown_r_p5=float(np.percentile(drawdowns, 5)),
        max_drawdown_r_p50=float(np.percentile(drawdowns, 50)),
        max_drawdown_r_p95=float(np.percentile(drawdowns, 95)),
        longest_loss_streak_p95=float(np.percentile(streaks, 95)),
        risk_of_ruin=ruined / config.num_paths,
    )


def run_trade_stress(
    r_multiples: Sequence[float], config: StressConfig | None = None
) -> StressReport:
    """Resample observed R-multiples; report baseline and stressed blocks."""
    config = config or StressConfig()
    errors = config.validate()
    if errors:
        raise ValueError("Invalid stress configuration: " + "; ".join(errors))
    values = np.asarray(list(r_multiples), dtype=float)
    if len(values) < MIN_TRADES_FOR_STRESS:
        raise ValueError(
            f"Trade-sequence stress needs at least {MIN_TRADES_FOR_STRESS} trades; "
            f"got {len(values)} — a distribution from nothing is fabrication."
        )
    if not np.isfinite(values).all():
        raise ValueError("R-multiples must all be finite.")
    rng = np.random.Generator(np.random.PCG64(config.seed))
    baseline = _resample_block(values, config, rng)
    rng = np.random.Generator(np.random.PCG64(config.seed))
    stressed = _resample_block(_apply_haircuts(values, config), config, rng)
    return StressReport(config=config, baseline=baseline, stressed=stressed)
