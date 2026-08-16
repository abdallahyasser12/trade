"""Does the signal survive deployment, or was it only good while it was tiny?

Ten walk-forward experiments produced one configuration with positive
out-of-sample expectancy: cross-sectional momentum, profit factor 1.48, Sharpe
0.52, positive under stress. It still failed promotion, and by a wide margin —
+2.01% against a benchmark that returned +112.48%.

The gap is not the entries. Sizing is risk-based, so position notional is
``risk_per_trade_pct`` divided by the distance to the stop. At the default
0.35% risk and the 4x ATR stop the search converged on, a daily crypto stop
sits roughly 6-20% from entry, which puts each position at about 1.8-5.8% of
equity. Three positions means the book holds roughly 82-95% cash at all times.
A portfolio that is ~90% in cash cannot track a benchmark that rose 112%,
however good its entries are.

So the question this module answers is narrow and specific: **if the same
signal is given more capital, does its quality survive?**

That is a diagnostic, not an optimisation, and the distinction is the whole
point. Sizing does not change the signal — it scales it. If expectancy is
positive then the highest-risk setting will almost always show the highest
return, so "pick the best return" would be selecting leverage and calling it
edge. The reading that means something is different:

* **Sharpe and profit factor are scale-invariant under pure scaling.** If the
  edge is real and merely under-deployed, they should stay roughly flat as
  risk rises while return scales up with it. That is the good outcome, and it
  says the constraint really was deployment.
* **If they decay as risk rises, scaling is not the answer.** Something
  non-linear is being paid for: positions colliding with the exposure caps,
  the heat ceiling forcing fewer concurrent holdings and so less
  diversification, or larger positions simply being unable to get the same
  fills. Whatever the mechanism, more capital makes it worse rather than
  better, and the strategy is not merely under-deployed.
* **Where it breaks matters as much as whether it breaks.** Each point reports
  which constraint bound it, so a flat return across two risk levels reads as
  "the position cap stopped it growing" rather than as "the signal saturated".

On comparing against buy-and-hold
---------------------------------

The promotion gate compares raw returns, which is correct for a long-only book
that is choosing between running this and holding the benchmark. But raw
return conflates two very different failures: having no edge, and having an
edge deployed at a tenth of the benchmark's volatility. Those need different
responses, so this module also reports the scale-free comparison — the
strategy's Sharpe against the benchmark's own Sharpe over the identical
window.

A higher Sharpe means that *at matched volatility* the strategy would out-earn
the benchmark. It does not mean it will, and the difference is not academic:
matching a 60-70% annualised volatility with a book sized off 4x ATR stops
requires leverage the risk limits do not permit and that this repository has
never tested. The scale-free number is diagnostic information about the
signal. It is not a promotion argument, and nothing here loosens a gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult

# Bars per year by timeframe, for annualising a Sharpe from an equity curve.
BARS_PER_YEAR = {
    "15Min": 365 * 24 * 4,
    "1Hour": 365 * 24,
    "4Hour": 365 * 6,
    "1Day": 365,
}

# Research-only ceilings. The sweep must be able to raise the exposure caps or
# it measures the caps rather than the sizing, but it must not raise them
# without bound either. These are the limits of the study, not of the engine —
# the live caps (max_live_total_exposure_pct, max_live_order_notional) are
# never touched by anything in this module.
STUDY_MAX_POSITION_PCT = 60.0
STUDY_MAX_TOTAL_EXPOSURE_PCT = 100.0
STUDY_MAX_HEAT_PCT = 10.0


@dataclass(frozen=True)
class SizingGrid:
    """The deployment levels to test, from current to the config's ceiling."""

    risk_per_trade_pcts: tuple[float, ...] = (0.35, 0.70, 1.05, 1.40, 2.00)
    position_counts: tuple[int, ...] = (3, 5, 8)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.risk_per_trade_pcts:
            errors.append("The sizing grid needs at least one risk level.")
        if not self.position_counts:
            errors.append("The sizing grid needs at least one position count.")
        for risk in self.risk_per_trade_pcts:
            if not math.isfinite(risk) or not 0 < risk <= 2:
                errors.append(f"Risk per trade {risk} is outside the config's (0, 2]% range.")
        for count in self.position_counts:
            if not 1 <= count <= 12:
                errors.append(f"Position count {count} is outside the config's [1, 12] range.")
        return errors

    def points(self) -> list[tuple[float, int]]:
        return [
            (risk, count)
            for risk in sorted(self.risk_per_trade_pcts)
            for count in sorted(self.position_counts)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_per_trade_pcts": list(self.risk_per_trade_pcts),
            "position_counts": list(self.position_counts),
        }


@dataclass(frozen=True)
class SizingPoint:
    """One deployment level, and what it cost to get there."""

    risk_per_trade_pct: float
    max_positions: int
    heat_pct: float
    num_trades: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]
    annualised_volatility_pct: float
    exposure_time_pct: float
    implied_position_pct: Optional[float]
    binding_constraint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SizingStudy:
    points: list[SizingPoint]
    benchmark_return_pct: float
    benchmark_sharpe: Optional[float]
    benchmark_volatility_pct: float
    grid: SizingGrid

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "benchmark": {
                "return_pct": self.benchmark_return_pct,
                "sharpe": self.benchmark_sharpe,
                "volatility_pct": self.benchmark_volatility_pct,
            },
            "points": [point.to_dict() for point in self.points],
            "verdict": self.verdict(),
        }

    def quality_holds(self, tolerance: float = 0.8) -> Optional[bool]:
        """Did risk-adjusted quality survive the largest deployment tested?

        Compares the best Sharpe at the smallest risk level against the best at
        the largest. None when either level produced nothing comparable, which
        is itself worth surfacing rather than papering over with a default.
        """

        low, high = self._extremes()
        if low is None or high is None or low <= 0:
            return None
        return high >= low * tolerance

    def _extremes(self) -> tuple[Optional[float], Optional[float]]:
        if not self.points:
            return None, None
        risks = sorted({point.risk_per_trade_pct for point in self.points})
        if len(risks) < 2:
            return None, None

        def best(risk: float) -> Optional[float]:
            values = [
                point.sharpe_ratio
                for point in self.points
                if point.risk_per_trade_pct == risk
                and point.sharpe_ratio is not None
                and math.isfinite(point.sharpe_ratio)
            ]
            return max(values) if values else None

        return best(risks[0]), best(risks[-1])

    def verdict(self) -> str:
        low, high = self._extremes()
        if low is None or high is None:
            return (
                "Inconclusive: at least one risk level produced no comparable "
                "Sharpe, so scaling behaviour could not be measured."
            )
        held = self.quality_holds()
        detail = f"best Sharpe {low:.2f} at the lowest risk level vs {high:.2f} at the highest"
        if held:
            return (
                f"Quality held under deployment ({detail}). Risk-adjusted "
                "performance is roughly scale-invariant, which is what an edge "
                "that was merely under-deployed looks like."
            )
        return (
            f"Quality decayed under deployment ({detail}). Scaling is not the "
            "answer: something non-linear is being paid for as positions grow, "
            "so more capital makes this worse rather than better."
        )


def scaled_risk_config(
    base: CryptoRiskConfig, risk_per_trade_pct: float, max_positions: int
) -> CryptoRiskConfig:
    """Scale the whole risk envelope coherently, not just one field.

    Raising ``risk_per_trade_pct`` alone measures almost nothing: the heat cap
    defaults to the product of the old risk and the old position count, so it
    binds immediately and the extra risk budget is never spent. The exposure
    caps bind next, for the same reason. Every ceiling therefore moves together
    with the thing it is meant to bound, and each is clamped at a
    research-only limit so the sweep cannot wander into configurations the
    engine would refuse.

    The live caps are deliberately absent. They are a separate safety ceiling
    and no research sweep has any business moving them.
    """

    heat = min(STUDY_MAX_HEAT_PCT, risk_per_trade_pct * max_positions)
    position_pct = min(
        STUDY_MAX_POSITION_PCT,
        max(base.max_position_pct, risk_per_trade_pct * 12.0),
    )
    total_exposure = min(
        STUDY_MAX_TOTAL_EXPOSURE_PCT,
        max(base.max_total_exposure_pct, position_pct * max_positions),
    )
    return replace(
        base,
        risk_per_trade_pct=risk_per_trade_pct,
        max_positions=max_positions,
        max_portfolio_heat_pct=heat,
        max_position_pct=position_pct,
        max_total_exposure_pct=total_exposure,
        # The live ceiling must never exceed the paper one, and raising the
        # paper cap would otherwise leave this untouched and inconsistent.
        max_live_total_exposure_pct=min(
            base.max_live_total_exposure_pct, total_exposure
        ),
    )


def annualised_stats(
    curve: pd.Series, timeframe: str
) -> tuple[float, float, Optional[float]]:
    """Annualised return, annualised volatility, and Sharpe from an equity curve.

    Sharpe is computed as the mean of log returns over their standard
    deviation, annualised by the square root of the period count. That
    definition is *exactly* scale-invariant: scaling every log return by k
    scales both the mean and the deviation by k, so the ratio is untouched.
    Scale invariance is the entire diagnostic - it is what lets a flat Sharpe
    across deployment levels mean "the edge survived" rather than "the
    arithmetic happened to work out".

    The naive alternative - annualised *geometric* return divided by
    volatility - is not scale-invariant, because compounding is non-linear and
    leverage introduces volatility drag. Measured on the same series, doubling
    the log returns moved that ratio from -0.95 to -0.79, which would have read
    as quality improving under deployment when nothing about the signal had
    changed. Volatility drag is real and belongs in the return column, where it
    shows up as compounded return growing sub-linearly with risk; it does not
    belong inside the quality measure.

    The benchmark is put through this identical function for the same reason:
    a comparison between two differently-computed Sharpes means nothing.
    """

    values = pd.to_numeric(pd.Series(curve), errors="coerce").dropna()
    if len(values) < 3 or (values <= 0).any():
        return 0.0, 0.0, None
    periods = BARS_PER_YEAR.get(timeframe)
    if not periods:
        raise ValueError(f"Unsupported timeframe for annualisation: {timeframe!r}")

    returns = np.log(values / values.shift(1)).dropna()
    if returns.empty:
        return 0.0, 0.0, None
    deviation = float(returns.std(ddof=1))
    volatility = deviation * math.sqrt(periods)
    total = float(values.iloc[-1] / values.iloc[0])
    years = len(returns) / periods
    # Geometric: what actually compounded, drag included.
    annual_return = (total ** (1.0 / years) - 1.0) if years > 0 else 0.0
    # Scale-invariant: signal quality, leverage divided out.
    sharpe = (
        float(returns.mean()) / deviation * math.sqrt(periods)
        if deviation > 0 and math.isfinite(deviation)
        else None
    )
    return annual_return * 100.0, volatility * 100.0, sharpe


def implied_position_pct(result: CryptoBacktestResult, risk_per_trade_pct: float) -> Optional[float]:
    """Notional per position as a percent of equity, from measured cost-to-risk.

    ``cost_to_risk`` is round-trip friction divided by the stop distance, so
    the stop distance falls straight out of it, and notional is the risk budget
    divided by that distance. This is what makes "the book is 90% cash"
    measurable rather than asserted.
    """

    if not result.median_cost_to_risk or not result.round_trip_cost_bps:
        return None
    if not math.isfinite(result.median_cost_to_risk) or result.median_cost_to_risk <= 0:
        return None
    stop_distance_bps = result.round_trip_cost_bps / result.median_cost_to_risk
    if stop_distance_bps <= 0:
        return None
    return risk_per_trade_pct * 10_000.0 / stop_distance_bps


def binding_constraint(
    risk: CryptoRiskConfig, implied_pct: Optional[float], max_positions: int
) -> str:
    """Which ceiling actually stopped this configuration deploying more.

    Without this a flat return across two risk levels is ambiguous - it could
    mean the signal saturated or it could mean a cap bound - and those call for
    opposite responses.
    """

    if implied_pct is None:
        return "unknown (no trades to measure)"
    if implied_pct >= risk.max_position_pct:
        return f"per-position cap ({risk.max_position_pct:g}%)"
    if implied_pct * max_positions >= risk.max_total_exposure_pct:
        return f"total exposure cap ({risk.max_total_exposure_pct:g}%)"
    # Strictly greater: the shipped default sets the heat cap to exactly
    # risk x positions, so an inclusive test would report heat as binding in
    # every configuration including the one it was designed to permit.
    if risk.risk_per_trade_pct * max_positions > risk.max_portfolio_heat_pct:
        return f"portfolio heat cap ({risk.max_portfolio_heat_pct:g}%)"
    return "none (sizing was free to scale)"


def run_sizing_study(
    backtest_fn,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    base_risk: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    grid: Optional[SizingGrid] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> SizingStudy:
    """Run one backtest per deployment level and report how quality scaled.

    ``backtest_fn`` takes the audited signature, so this works against any
    decision engine without knowing which one it is.
    """

    resolved = grid or SizingGrid()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid sizing grid: " + "; ".join(errors))
    config_errors = strategy_config.validate()
    if config_errors:
        raise ValueError("Invalid strategy configuration: " + "; ".join(config_errors))

    # Fail here, with the mismatch named, rather than fifteen backtests deep.
    # A caller that drops a symbol from the universe without also dropping it
    # from the strategy config produces a "Missing bars" error from inside
    # prepare_universe that points at the strategy rather than at the caller.
    missing = set(strategy_config.symbols).difference(bars_by_symbol)
    extra = set(bars_by_symbol).difference(strategy_config.symbols)
    if missing or extra:
        raise ValueError(
            "The strategy universe and the supplied bars disagree: "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}. "
            "A caller that drops symbols must drop them from both."
        )

    points: list[SizingPoint] = []
    benchmark_return = 0.0
    benchmark_sharpe: Optional[float] = None
    benchmark_volatility = 0.0

    for risk_pct, positions in resolved.points():
        risk_config = scaled_risk_config(base_risk, risk_pct, positions)
        invalid = risk_config.validate()
        if invalid:
            raise ValueError(
                f"Sizing point (risk={risk_pct}, positions={positions}) produced an "
                "invalid risk configuration: " + "; ".join(invalid)
            )
        result = backtest_fn(
            bars_by_symbol,
            strategy_config,
            risk_config,
            backtest_config,
            active_start=active_start,
        )
        _, volatility, sharpe = annualised_stats(
            result.equity_curve, strategy_config.timeframe
        )
        if benchmark_sharpe is None:
            benchmark_return = result.benchmark_return_pct
            _, benchmark_volatility, benchmark_sharpe = annualised_stats(
                result.benchmark_curve, strategy_config.timeframe
            )
        implied = implied_position_pct(result, risk_pct)
        points.append(
            SizingPoint(
                risk_per_trade_pct=risk_pct,
                max_positions=positions,
                heat_pct=risk_config.max_portfolio_heat_pct,
                num_trades=result.num_trades,
                total_return_pct=result.total_return_pct,
                max_drawdown_pct=result.max_drawdown_pct,
                sharpe_ratio=sharpe,
                profit_factor=result.profit_factor,
                annualised_volatility_pct=volatility,
                exposure_time_pct=result.exposure_time_pct,
                implied_position_pct=implied,
                binding_constraint=binding_constraint(risk_config, implied, positions),
            )
        )

    return SizingStudy(
        points=points,
        benchmark_return_pct=benchmark_return,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_volatility_pct=benchmark_volatility,
        grid=resolved,
    )
