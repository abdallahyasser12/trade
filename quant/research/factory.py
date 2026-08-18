"""Walk-forward research factory for ``UniverseStrategy`` families.

Chronological rolling train -> embargo -> test research with training-only
candidate selection, stressed execution costs, Monte Carlo trade-sequence stress,
and deflated-Sharpe multiple-testing correction. A research failure can never
silently become an out-of-sample candidate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from quant.exits import ExitPolicyConfig
from quant.research.multiplicity import deflated_sharpe_probability
from quant.research.simulator import SimulationConfig, SimulationResult, run_simulation
from quant.research.stress import StressConfig, run_trade_stress
from quant.store import QuantStore
from quant.strategies.base import UniverseStrategy

StrategyFactory = Callable[[dict[str, Any]], UniverseStrategy]


@dataclass(frozen=True)
class QuantWalkForwardConfig:
    train_bars: int = 500
    test_bars: int = 150
    step_bars: int = 150
    embargo_bars: int = 1
    min_train_trades: int = 3
    max_candidates: int = 32
    stress_slippage_bps: float = 25.0
    stress_spread_bps: float = 15.0
    stress_fee_bps: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.train_bars < 50:
            errors.append("Training window must contain at least 50 bars.")
        if self.test_bars < 10:
            errors.append("Test window must contain at least 10 bars.")
        if self.step_bars < self.test_bars:
            errors.append("Step size must be at least the test size (overlap is forbidden).")
        if self.embargo_bars < 1:
            errors.append("Embargo must contain at least one bar.")
        if self.min_train_trades < 1:
            errors.append("Minimum training trades must be at least one.")
        if not (1 <= self.max_candidates <= 256):
            errors.append("Candidate count must be between 1 and 256.")
        for label, value in {
            "stress slippage": self.stress_slippage_bps,
            "stress spread": self.stress_spread_bps,
            "stress fee": self.stress_fee_bps,
        }.items():
            if not math.isfinite(value) or value < 0:
                errors.append(f"{label} must be finite and non-negative.")
        return errors


@dataclass(frozen=True)
class QuantPromotionCriteria:
    min_folds: int = 3
    min_oos_trades: int = 15
    min_positive_fold_ratio: float = 0.60
    min_compounded_return_pct: float = 0.0
    min_benchmark_excess_return_pct: float = 0.0
    min_aggregate_sharpe: float = 0.50
    min_profit_factor: float = 1.10
    max_fold_drawdown_pct: float = 20.0
    min_stressed_return_pct: float = 0.0
    min_deflated_sharpe_probability: float = 0.50
    max_ruin_probability: float = 0.05

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.min_folds < 2:
            errors.append("Promotion requires at least two out-of-sample folds.")
        if self.min_oos_trades < 1:
            errors.append("Promotion trade count must be positive.")
        if not (0.0 <= self.min_positive_fold_ratio <= 1.0):
            errors.append("Positive-fold ratio must be between 0 and 1.")
        if not (0.0 <= self.min_deflated_sharpe_probability <= 1.0):
            errors.append("Deflated-Sharpe probability threshold must be in [0, 1].")
        if not (0.0 <= self.max_ruin_probability <= 1.0):
            errors.append("Ruin-probability ceiling must be in [0, 1].")
        for label, value in {
            "minimum compounded return": self.min_compounded_return_pct,
            "minimum benchmark excess": self.min_benchmark_excess_return_pct,
            "minimum aggregate Sharpe": self.min_aggregate_sharpe,
            "minimum profit factor": self.min_profit_factor,
            "maximum fold drawdown": self.max_fold_drawdown_pct,
            "minimum stressed return": self.min_stressed_return_pct,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.min_profit_factor <= 0 or self.max_fold_drawdown_pct <= 0:
            errors.append("Profit-factor and drawdown gates must be positive.")
        return errors


@dataclass
class GateResult:
    name: str
    value: Optional[float]
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantFoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_params: dict[str, Any]
    training_score: float
    test_result: SimulationResult
    stressed_result: SimulationResult


@dataclass
class QuantResearchReport:
    strategy_id: str
    folds: list[QuantFoldResult]
    candidate_count: int
    trial_count: int
    compounded_return_pct: float
    benchmark_compounded_return_pct: float
    stressed_compounded_return_pct: float
    positive_fold_ratio: float
    aggregate_sharpe: Optional[float]
    per_bar_sharpe: Optional[float]
    profit_factor: Optional[float]
    worst_fold_drawdown_pct: float
    total_oos_trades: int
    oos_r_multiples: list[float]
    stress_report: Optional[Any]
    dsr_report: Optional[Any]
    gates: list[GateResult]
    passed: bool
    latest_selected_params: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)

    def gates_dict(self) -> dict[str, dict[str, Any]]:
        return {gate.name: gate.to_dict() for gate in self.gates}


def _robust_training_score(
    result: SimulationResult, *, min_trades: int, candidate_count: int
) -> float:
    if result.num_trades < min_trades:
        return -math.inf
    sharpe = result.sharpe_ratio if result.sharpe_ratio is not None else -2.0
    calmar = result.total_return_pct / max(1.0, result.max_drawdown_pct)
    pf = result.profit_factor
    pf_term = 0.15 * math.log1p(min(4.0, pf)) if pf is not None and math.isfinite(pf) else 0.0
    observations = max(2, len(result.equity_curve))
    penalty = math.sqrt(2.0 * math.log(max(2, candidate_count)) / observations)
    return sharpe + 0.30 * calmar + pf_term - 0.02 * result.max_drawdown_pct - penalty


def _compounded(returns_pct: Sequence[float]) -> float:
    wealth = 1.0
    for value in returns_pct:
        wealth *= max(0.0, 1.0 + value / 100.0)
    return (wealth - 1.0) * 100.0


def run_quant_walk_forward(
    *,
    strategy_id: str,
    strategy_factory: StrategyFactory,
    candidate_params: Sequence[dict[str, Any]],
    bars_by_symbol: dict[str, pd.DataFrame],
    sim_config: SimulationConfig,
    exit_policy: Optional[ExitPolicyConfig] = None,
    walk_config: Optional[QuantWalkForwardConfig] = None,
    criteria: Optional[QuantPromotionCriteria] = None,
    stress_config: Optional[StressConfig] = None,
    prior_trials: int = 0,
) -> QuantResearchReport:
    walk_config = walk_config or QuantWalkForwardConfig()
    criteria = criteria or QuantPromotionCriteria()
    for config, label in ((walk_config, "walk-forward"), (criteria, "promotion")):
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid {label} configuration: " + "; ".join(errors))
    if not candidate_params:
        raise ValueError("At least one candidate parameter set is required.")
    if len(candidate_params) > walk_config.max_candidates:
        raise ValueError(
            f"{len(candidate_params)} candidates exceed the configured cap of "
            f"{walk_config.max_candidates}; a larger grid must raise the cap explicitly."
        )
    if prior_trials < 0:
        raise ValueError("Prior trial count cannot be negative.")

    strategies = [strategy_factory(dict(params)) for params in candidate_params]
    warmups = [strategy.required_bars() for strategy in strategies]

    common: Optional[pd.DatetimeIndex] = None
    for frame in bars_by_symbol.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        raise ValueError("No aligned history across the universe.")
    common = common.sort_values()

    stressed_sim = replace(
        sim_config,
        slippage_bps=max(sim_config.slippage_bps, walk_config.stress_slippage_bps),
        spread_bps=max(sim_config.spread_bps, walk_config.stress_spread_bps),
        fee_bps=max(sim_config.fee_bps, walk_config.stress_fee_bps),
    )

    def slice_bars(start_pos: int, end_pos: int) -> dict[str, pd.DataFrame]:
        window = common[start_pos:end_pos]
        return {
            symbol: frame.loc[frame.index.isin(window)]
            for symbol, frame in bars_by_symbol.items()
        }

    folds: list[QuantFoldResult] = []
    train_start = 0
    while True:
        train_end = train_start + walk_config.train_bars
        test_start = train_end + walk_config.embargo_bars
        test_end = test_start + walk_config.test_bars
        if test_end > len(common):
            break

        best_score = -math.inf
        best_ordinal: Optional[int] = None
        for ordinal, (_params, strategy, warmup) in enumerate(
            zip(candidate_params, strategies, warmups)
        ):
            if walk_config.train_bars < warmup + 2:
                continue
            try:
                train_result = run_simulation(
                    strategy,
                    slice_bars(train_start, train_end),
                    config=sim_config,
                    exit_policy=exit_policy,
                )
            except ValueError:
                continue
            score = _robust_training_score(
                train_result,
                min_trades=walk_config.min_train_trades,
                candidate_count=len(candidate_params),
            )
            if math.isfinite(score) and (best_ordinal is None or score > best_score):
                best_score = score
                best_ordinal = ordinal

        if best_ordinal is None:
            raise ValueError(
                "No candidate satisfied minimum training evidence in walk-forward fold "
                f"{common[train_start].isoformat()}..{common[train_end - 1].isoformat()}; "
                "refusing to select candidate 0 by default or skip the fold."
            )

        selected_params = dict(candidate_params[best_ordinal])
        selected_strategy = strategies[best_ordinal]
        selected_warmup = warmups[best_ordinal]
        oos_start = test_start - (selected_warmup - 1)
        if oos_start < 0:
            raise ValueError("Test window cannot supply candidate warmup; extend train_bars.")
        test_result = run_simulation(
            selected_strategy,
            slice_bars(oos_start, test_end),
            config=sim_config,
            exit_policy=exit_policy,
        )
        stressed_result = run_simulation(
            selected_strategy,
            slice_bars(oos_start, test_end),
            config=stressed_sim,
            exit_policy=exit_policy,
        )
        folds.append(
            QuantFoldResult(
                train_start=common[train_start],
                train_end=common[train_end - 1],
                test_start=common[test_start],
                test_end=common[test_end - 1],
                selected_params=selected_params,
                training_score=best_score,
                test_result=test_result,
                stressed_result=stressed_result,
            )
        )
        train_start += walk_config.step_bars

    if not folds:
        raise ValueError(
            "Not enough aligned history for a single walk-forward fold: "
            f"{len(common)} bars, need at least "
            f"{walk_config.train_bars + walk_config.embargo_bars + walk_config.test_bars}."
        )

    fold_returns = [fold.test_result.total_return_pct for fold in folds]
    benchmark_returns = [fold.test_result.benchmark_return_pct for fold in folds]
    stressed_returns = [fold.stressed_result.total_return_pct for fold in folds]
    compounded = _compounded(fold_returns)
    benchmark_compounded = _compounded(benchmark_returns)
    stressed_compounded = _compounded(stressed_returns)
    positive_ratio = sum(1 for value in fold_returns if value > 0) / len(folds)
    worst_drawdown = max(fold.test_result.max_drawdown_pct for fold in folds)
    total_trades = sum(fold.test_result.num_trades for fold in folds)

    pooled_returns: list[float] = []
    for fold in folds:
        pooled_returns.extend(fold.test_result.equity_curve.pct_change().dropna().tolist())
    per_bar_sharpe: Optional[float] = None
    aggregate_sharpe: Optional[float] = None
    return_skew = 0.0
    return_kurtosis = 3.0
    if len(pooled_returns) >= 20:
        series = pd.Series(pooled_returns, dtype=float)
        std = float(series.std(ddof=1))
        if std > 0:
            per_bar_sharpe = float(series.mean() / std)
            aggregate_sharpe = per_bar_sharpe * math.sqrt(sim_config.annualization_factor)
        skew = float(series.skew())
        excess_kurtosis = float(series.kurt())
        if math.isfinite(skew):
            return_skew = skew
        if math.isfinite(excess_kurtosis):
            return_kurtosis = max(1.0, excess_kurtosis + 3.0)

    pooled_pnl = [
        trade.net_pnl
        for fold in folds
        for trade in fold.test_result.trades
        if trade.net_pnl is not None
    ]
    wins = sum(value for value in pooled_pnl if value > 0)
    losses = abs(sum(value for value in pooled_pnl if value <= 0))
    profit_factor: Optional[float] = None
    if losses > 0:
        profit_factor = wins / losses
    elif wins > 0:
        profit_factor = math.inf

    oos_r = [r for fold in folds for r in fold.test_result.r_multiples]
    stress_report = None
    ruin_probability: Optional[float] = None
    try:
        stress_report = run_trade_stress(oos_r, stress_config or StressConfig())
        report_dict = stress_report.to_dict()
        stressed_block = report_dict.get("stressed") or {}
        baseline_block = report_dict.get("baseline") or {}
        ruin_probability = stressed_block.get("risk_of_ruin", baseline_block.get("risk_of_ruin"))
    except ValueError:
        stress_report = None

    dsr_report = None
    dsr_probability: Optional[float] = None
    trial_count = len(candidate_params) + prior_trials
    if per_bar_sharpe is not None:
        dsr_report = deflated_sharpe_probability(
            per_bar_sharpe,
            n_trials=max(1, trial_count),
            n_observations=len(pooled_returns),
            skewness=return_skew,
            kurtosis=return_kurtosis,
            variance_sharpe=1.0 / max(2, len(pooled_returns) - 1),
        )
        dsr_probability = float(getattr(dsr_report, "dsr_probability"))

    def gate(name: str, value: Optional[float], threshold: float, ok: bool) -> GateResult:
        return GateResult(name=name, value=value, threshold=threshold, passed=ok)

    gates = [
        gate("min_folds", float(len(folds)), float(criteria.min_folds), len(folds) >= criteria.min_folds),
        gate("min_oos_trades", float(total_trades), float(criteria.min_oos_trades), total_trades >= criteria.min_oos_trades),
        gate("min_positive_fold_ratio", positive_ratio, criteria.min_positive_fold_ratio, positive_ratio >= criteria.min_positive_fold_ratio),
        gate("min_compounded_return_pct", compounded, criteria.min_compounded_return_pct, compounded >= criteria.min_compounded_return_pct),
        gate("min_benchmark_excess_return_pct", compounded - benchmark_compounded, criteria.min_benchmark_excess_return_pct, compounded - benchmark_compounded >= criteria.min_benchmark_excess_return_pct),
        gate("min_aggregate_sharpe", aggregate_sharpe, criteria.min_aggregate_sharpe, aggregate_sharpe is not None and aggregate_sharpe >= criteria.min_aggregate_sharpe),
        gate("min_profit_factor", None if profit_factor is None or math.isinf(profit_factor) else profit_factor, criteria.min_profit_factor, profit_factor is not None and profit_factor >= criteria.min_profit_factor),
        gate("max_fold_drawdown_pct", worst_drawdown, criteria.max_fold_drawdown_pct, worst_drawdown <= criteria.max_fold_drawdown_pct),
        gate("min_stressed_return_pct", stressed_compounded, criteria.min_stressed_return_pct, stressed_compounded >= criteria.min_stressed_return_pct),
        gate("min_deflated_sharpe_probability", dsr_probability, criteria.min_deflated_sharpe_probability, dsr_probability is not None and dsr_probability >= criteria.min_deflated_sharpe_probability),
        gate("max_ruin_probability", ruin_probability, criteria.max_ruin_probability, ruin_probability is not None and ruin_probability <= criteria.max_ruin_probability),
    ]
    passed = all(g.passed for g in gates)

    return QuantResearchReport(
        strategy_id=strategy_id,
        folds=folds,
        candidate_count=len(candidate_params),
        trial_count=trial_count,
        compounded_return_pct=compounded,
        benchmark_compounded_return_pct=benchmark_compounded,
        stressed_compounded_return_pct=stressed_compounded,
        positive_fold_ratio=positive_ratio,
        aggregate_sharpe=aggregate_sharpe,
        per_bar_sharpe=per_bar_sharpe,
        profit_factor=profit_factor,
        worst_fold_drawdown_pct=worst_drawdown,
        total_oos_trades=total_trades,
        oos_r_multiples=oos_r,
        stress_report=stress_report,
        dsr_report=dsr_report,
        gates=gates,
        passed=passed,
        latest_selected_params=folds[-1].selected_params,
        protocol={
            "walk_config": asdict(walk_config),
            "criteria": asdict(criteria),
            "sim_config": sim_config.to_dict(),
            "stressed_sim_config": stressed_sim.to_dict(),
            "exit_policy": asdict(exit_policy) if exit_policy else None,
            "universe": sorted(bars_by_symbol),
            "data_window": f"{common[0].isoformat()}..{common[-1].isoformat()}",
            "oos_return_skewness": return_skew,
            "oos_return_kurtosis": return_kurtosis,
        },
    )


def record_research_report(
    store: QuantStore,
    report: QuantResearchReport,
    *,
    git_commit: Optional[str] = None,
    seed: Optional[int] = None,
    hypothesis_id: Optional[str] = None,
) -> int:
    return store.record_research_run(
        strategy_id=report.strategy_id,
        params=report.latest_selected_params,
        universe=list(report.protocol.get("universe", [])),
        data_window=str(report.protocol.get("data_window", "")),
        candidate_count=report.candidate_count,
        # Persist only trials performed by this run; QuantStore accumulates lineage.
        trial_count=report.candidate_count,
        passed=report.passed,
        gates=report.gates_dict(),
        cost_assumptions={
            "sim": report.protocol.get("sim_config"),
            "stressed": report.protocol.get("stressed_sim_config"),
        },
        hypothesis_id=hypothesis_id,
        git_commit=git_commit,
        seed=seed,
        result={
            "compounded_return_pct": report.compounded_return_pct,
            "benchmark_compounded_return_pct": report.benchmark_compounded_return_pct,
            "stressed_compounded_return_pct": report.stressed_compounded_return_pct,
            "aggregate_sharpe": report.aggregate_sharpe,
            "profit_factor": None if report.profit_factor is None or math.isinf(report.profit_factor) else report.profit_factor,
            "positive_fold_ratio": report.positive_fold_ratio,
            "worst_fold_drawdown_pct": report.worst_fold_drawdown_pct,
            "total_oos_trades": report.total_oos_trades,
            "folds": len(report.folds),
        },
    )
