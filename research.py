"""Chronological strategy research and promotion controls.

This module deliberately does not contain trading rules.  It evaluates
``TrendStrategy``/``RiskManager`` exclusively through ``run_backtest`` so a
candidate cannot receive a special research-only implementation.  Candidate
selection happens on a rolling training window; the following embargoed window
is untouched until selection is complete.  Cost stress and a fixed baseline are
evaluated on that same out-of-sample window.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import BacktestConfig, RiskConfig, StrategyConfig, bars_per_year
from models import BacktestResult


TUNABLE_PARAMETERS = frozenset(
    {
        "short_ma_period",
        "long_ma_period",
        "rsi_entry_floor",
        "rsi_overbought",
        "atr_multiplier",
        "min_volume_ratio",
        "max_atr_percentile",
        "min_adx",
        "min_efficiency_ratio",
        "regime_ma_period",
        "regime_slope_period",
        "relative_strength_period",
        "min_relative_strength",
        "trailing_atr_multiplier",
        "break_even_trigger_r",
    }
)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Bounded rolling train/embargo/test protocol."""

    train_bars: int = 756
    test_bars: int = 126
    step_bars: int = 126
    embargo_bars: int = 1
    min_train_trades: int = 5
    max_candidates: int = 64
    stress_slippage_bps: float = 10.0
    stress_commission_per_trade: float = 1.0

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
        if not math.isfinite(self.stress_slippage_bps) or self.stress_slippage_bps < 0:
            errors.append("Stress slippage must be finite and non-negative.")
        if (
            not math.isfinite(self.stress_commission_per_trade)
            or self.stress_commission_per_trade < 0
        ):
            errors.append("Stress commission must be finite and non-negative.")
        return errors


@dataclass(frozen=True)
class PromotionCriteria:
    """Evidence gates required before a candidate may enter paper testing."""

    min_folds: int = 3
    min_oos_trades: int = 15
    min_positive_fold_ratio: float = 0.60
    min_compounded_return_pct: float = 0.0
    min_excess_return_pct: float = 0.0
    min_aggregate_sharpe: float = 0.50
    min_profit_factor: float = 1.10
    max_fold_drawdown_pct: float = 20.0
    min_stressed_return_pct: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.min_folds < 2:
            errors.append("Promotion requires at least two out-of-sample folds.")
        if self.min_oos_trades < 1:
            errors.append("Promotion trade count must be positive.")
        if not (0.0 <= self.min_positive_fold_ratio <= 1.0):
            errors.append("Positive-fold ratio must be between 0 and 1.")
        for label, value in {
            "minimum compounded return": self.min_compounded_return_pct,
            "minimum excess return": self.min_excess_return_pct,
            "minimum aggregate Sharpe": self.min_aggregate_sharpe,
            "minimum profit factor": self.min_profit_factor,
            "maximum fold drawdown": self.max_fold_drawdown_pct,
            "minimum stressed return": self.min_stressed_return_pct,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.min_profit_factor <= 0:
            errors.append("Minimum profit factor must be positive.")
        if self.max_fold_drawdown_pct <= 0:
            errors.append("Maximum fold drawdown must be positive.")
        return errors


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_config: StrategyConfig
    training_score: float
    test_result: BacktestResult
    baseline_result: BacktestResult
    stressed_result: BacktestResult


@dataclass
class PromotionDecision:
    promoted: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    candidate_count: int
    compounded_return_pct: float
    baseline_compounded_return_pct: float
    stressed_compounded_return_pct: float
    median_fold_return_pct: float
    positive_fold_ratio: float
    aggregate_sharpe: Optional[float]
    profit_factor: Optional[float]
    worst_fold_drawdown_pct: float
    total_oos_trades: int
    latest_selected_config: StrategyConfig
    promotion: PromotionDecision
    protocol: dict[str, Any] = field(default_factory=dict)


def build_candidate_configs(
    base: StrategyConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    *,
    max_candidates: int = 64,
) -> list[StrategyConfig]:
    """Create a deterministic, validated, tightly bounded candidate set."""
    unknown = set(parameter_grid) - TUNABLE_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported tunable parameters: {sorted(unknown)}")
    if not parameter_grid:
        candidates = [replace(base)]
    else:
        keys = sorted(parameter_grid)
        value_lists = [list(parameter_grid[key]) for key in keys]
        if any(not values for values in value_lists):
            raise ValueError("Every parameter-grid dimension needs at least one value.")
        raw_count = math.prod(len(values) for values in value_lists)
        if raw_count > max_candidates:
            raise ValueError(
                f"Parameter grid creates {raw_count} candidates; maximum is {max_candidates}."
            )
        candidates = [
            replace(base, **dict(zip(keys, values, strict=True)))
            for values in itertools.product(*value_lists)
        ]

    unique: dict[tuple, StrategyConfig] = {}
    invalid_messages: list[str] = []
    for candidate in candidates:
        errors = candidate.validate()
        if errors:
            invalid_messages.extend(errors)
            continue
        identity = tuple(sorted(candidate.to_dict().items()))
        unique.setdefault(identity, candidate)
    if not unique:
        detail = "; ".join(sorted(set(invalid_messages)))
        raise ValueError(
            "Parameter grid produced no valid candidates."
            + (f" {detail}" if detail else "")
        )
    return list(unique.values())


def _robust_training_score(
    result: BacktestResult,
    *,
    min_trades: int,
    candidate_count: int,
) -> float:
    """Risk-adjusted objective with an explicit multiple-testing penalty."""
    if result.num_trades < min_trades:
        return -math.inf
    sharpe = result.sharpe_ratio if result.sharpe_ratio is not None else -2.0
    calmar = result.total_return_pct / max(1.0, result.max_drawdown_pct)
    profit_factor = min(4.0, result.profit_factor or 0.0)
    observations = max(2, len(result.equity_curve) - 1)
    selection_penalty = math.sqrt(2.0 * math.log(max(1, candidate_count)) / observations)
    return (
        sharpe
        + 0.30 * calmar
        + 0.15 * math.log1p(profit_factor)
        - 0.02 * result.max_drawdown_pct
        - selection_penalty
    )


def _compounded_return(results: Iterable[BacktestResult]) -> float:
    wealth = 1.0
    for result in results:
        wealth *= max(0.0, 1.0 + result.total_return_pct / 100.0)
    return (wealth - 1.0) * 100.0


def _aggregate_sharpe(
    results: Iterable[BacktestResult],
    timeframe: str,
    risk_free_rate: float,
) -> Optional[float]:
    """Annualized OOS Sharpe using the configured annual risk-free rate."""
    returns: list[float] = []
    for result in results:
        returns.extend(result.equity_curve.pct_change().dropna().astype(float).tolist())
    if len(returns) < 20:
        return None
    values = np.asarray(returns, dtype=float)
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    annualization = bars_per_year(timeframe)
    per_bar_risk_free = risk_free_rate / annualization
    excess = values - per_bar_risk_free
    return float(excess.mean() / std * math.sqrt(annualization))


def _aggregate_profit_factor(results: Iterable[BacktestResult]) -> Optional[float]:
    wins = 0.0
    losses = 0.0
    for result in results:
        for trade in result.trades:
            if not trade.is_closed or trade.pnl is None:
                continue
            if trade.pnl > 0:
                wins += trade.pnl
            else:
                losses += -trade.pnl
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else None


def evaluate_promotion(
    *,
    folds: list[WalkForwardFold],
    compounded_return_pct: float,
    baseline_compounded_return_pct: float,
    stressed_compounded_return_pct: float,
    positive_fold_ratio: float,
    aggregate_sharpe: Optional[float],
    profit_factor: Optional[float],
    worst_fold_drawdown_pct: float,
    total_oos_trades: int,
    criteria: PromotionCriteria,
) -> PromotionDecision:
    """Apply every gate; no weighted score can hide a failed safety metric."""
    errors = criteria.validate()
    if errors:
        raise ValueError("Invalid promotion criteria: " + "; ".join(errors))
    failures: list[str] = []
    if len(folds) < criteria.min_folds:
        failures.append(f"folds {len(folds)} < {criteria.min_folds}")
    if total_oos_trades < criteria.min_oos_trades:
        failures.append(f"OOS trades {total_oos_trades} < {criteria.min_oos_trades}")
    if positive_fold_ratio < criteria.min_positive_fold_ratio:
        failures.append(
            f"positive-fold ratio {positive_fold_ratio:.2f} < "
            f"{criteria.min_positive_fold_ratio:.2f}"
        )
    if compounded_return_pct < criteria.min_compounded_return_pct:
        failures.append(
            f"OOS return {compounded_return_pct:.2f}% < "
            f"{criteria.min_compounded_return_pct:.2f}%"
        )
    excess = compounded_return_pct - baseline_compounded_return_pct
    if excess < criteria.min_excess_return_pct:
        failures.append(
            f"excess return {excess:.2f}% < {criteria.min_excess_return_pct:.2f}%"
        )
    if aggregate_sharpe is None or aggregate_sharpe < criteria.min_aggregate_sharpe:
        rendered = "n/a" if aggregate_sharpe is None else f"{aggregate_sharpe:.2f}"
        failures.append(
            f"aggregate Sharpe {rendered} < {criteria.min_aggregate_sharpe:.2f}"
        )
    if profit_factor is None or profit_factor < criteria.min_profit_factor:
        rendered = "n/a" if profit_factor is None else f"{profit_factor:.2f}"
        failures.append(
            f"profit factor {rendered} < {criteria.min_profit_factor:.2f}"
        )
    if worst_fold_drawdown_pct > criteria.max_fold_drawdown_pct:
        failures.append(
            f"worst fold drawdown {worst_fold_drawdown_pct:.2f}% > "
            f"{criteria.max_fold_drawdown_pct:.2f}%"
        )
    if stressed_compounded_return_pct < criteria.min_stressed_return_pct:
        failures.append(
            f"stressed return {stressed_compounded_return_pct:.2f}% < "
            f"{criteria.min_stressed_return_pct:.2f}%"
        )
    return PromotionDecision(promoted=not failures, failures=failures)


def run_walk_forward(
    price_data: pd.DataFrame,
    base_strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    backtest_config: BacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: WalkForwardConfig,
    promotion_criteria: PromotionCriteria,
    market_data: Optional[pd.DataFrame] = None,
) -> WalkForwardResult:
    """Run bounded rolling optimization and untouched out-of-sample tests."""
    walk_errors = walk_config.validate()
    if walk_errors:
        raise ValueError("Invalid walk-forward configuration: " + "; ".join(walk_errors))
    criteria_errors = promotion_criteria.validate()
    if criteria_errors:
        raise ValueError("Invalid promotion criteria: " + "; ".join(criteria_errors))
    if not isinstance(price_data, pd.DataFrame) or not isinstance(
        price_data.index, pd.DatetimeIndex
    ):
        raise ValueError("price_data must be a datetime-indexed DataFrame")
    candidates = build_candidate_configs(
        base_strategy_config,
        parameter_grid,
        max_candidates=walk_config.max_candidates,
    )
    required_total = (
        walk_config.train_bars + walk_config.embargo_bars + walk_config.test_bars
    )
    if len(price_data) < required_total:
        raise ValueError(
            f"Walk-forward protocol needs at least {required_total} bars; "
            f"received {len(price_data)}."
        )

    forced_close = replace(backtest_config, liquidate_at_end=True)
    stressed = replace(
        forced_close,
        slippage_bps=max(
            forced_close.slippage_bps,
            walk_config.stress_slippage_bps,
        ),
        commission_per_trade=max(
            forced_close.commission_per_trade,
            walk_config.stress_commission_per_trade,
        ),
    )
    folds: list[WalkForwardFold] = []
    max_candidate_warmup = max(candidate.warmup_bars() for candidate in candidates)
    train_start = 0
    while True:
        train_end = train_start + walk_config.train_bars
        test_start = train_end + walk_config.embargo_bars
        test_end = test_start + walk_config.test_bars
        if test_end > len(price_data):
            break

        training_context_start = max(0, train_start - max_candidate_warmup - 2)
        training_slice = price_data.iloc[training_context_start:train_end]
        training_activation = price_data.index[train_start]
        scored: list[tuple[float, int, StrategyConfig]] = []
        for ordinal, candidate in enumerate(candidates):
            try:
                result = run_backtest(
                    training_slice,
                    candidate,
                    risk_config,
                    forced_close,
                    market_data=market_data,
                    active_start=(
                        training_activation if training_context_start < train_start else None
                    ),
                )
                score = _robust_training_score(
                    result,
                    min_trades=walk_config.min_train_trades,
                    candidate_count=len(candidates),
                )
            except ValueError:
                score = -math.inf
            scored.append((score, -ordinal, candidate))
        best_score, _, selected = max(scored, key=lambda item: (item[0], item[1]))
        if not math.isfinite(best_score):
            raise ValueError(
                "No candidate met the minimum training-trade requirement in fold "
                f"starting {price_data.index[train_start].isoformat()}."
            )

        context_bars = max(
            selected.warmup_bars(), base_strategy_config.warmup_bars()
        ) + 2
        context_start = max(0, test_start - context_bars)
        test_slice = price_data.iloc[context_start:test_end]
        activation = price_data.index[test_start]
        selected_result = run_backtest(
            test_slice,
            selected,
            risk_config,
            forced_close,
            market_data=market_data,
            active_start=activation,
        )
        baseline_result = run_backtest(
            test_slice,
            base_strategy_config,
            risk_config,
            forced_close,
            market_data=market_data,
            active_start=activation,
        )
        stressed_result = run_backtest(
            test_slice,
            selected,
            risk_config,
            stressed,
            market_data=market_data,
            active_start=activation,
        )
        folds.append(
            WalkForwardFold(
                train_start=price_data.index[train_start],
                train_end=price_data.index[train_end - 1],
                test_start=activation,
                test_end=price_data.index[test_end - 1],
                selected_config=selected,
                training_score=best_score,
                test_result=selected_result,
                baseline_result=baseline_result,
                stressed_result=stressed_result,
            )
        )
        train_start += walk_config.step_bars

    if not folds:
        raise ValueError("Walk-forward protocol produced no complete folds.")

    candidate_results = [fold.test_result for fold in folds]
    baseline_results = [fold.baseline_result for fold in folds]
    stressed_results = [fold.stressed_result for fold in folds]
    compounded = _compounded_return(candidate_results)
    baseline_compounded = _compounded_return(baseline_results)
    stressed_compounded = _compounded_return(stressed_results)
    fold_returns = [result.total_return_pct for result in candidate_results]
    positive_ratio = sum(value > 0 for value in fold_returns) / len(fold_returns)
    aggregate_sharpe = _aggregate_sharpe(
        candidate_results,
        base_strategy_config.timeframe,
        backtest_config.risk_free_rate,
    )
    profit_factor = _aggregate_profit_factor(candidate_results)
    worst_drawdown = max(result.max_drawdown_pct for result in candidate_results)
    total_trades = sum(result.num_trades for result in candidate_results)
    promotion = evaluate_promotion(
        folds=folds,
        compounded_return_pct=compounded,
        baseline_compounded_return_pct=baseline_compounded,
        stressed_compounded_return_pct=stressed_compounded,
        positive_fold_ratio=positive_ratio,
        aggregate_sharpe=aggregate_sharpe,
        profit_factor=profit_factor,
        worst_fold_drawdown_pct=worst_drawdown,
        total_oos_trades=total_trades,
        criteria=promotion_criteria,
    )
    return WalkForwardResult(
        folds=folds,
        candidate_count=len(candidates),
        compounded_return_pct=compounded,
        baseline_compounded_return_pct=baseline_compounded,
        stressed_compounded_return_pct=stressed_compounded,
        median_fold_return_pct=float(median(fold_returns)),
        positive_fold_ratio=positive_ratio,
        aggregate_sharpe=aggregate_sharpe,
        profit_factor=profit_factor,
        worst_fold_drawdown_pct=worst_drawdown,
        total_oos_trades=total_trades,
        latest_selected_config=folds[-1].selected_config,
        promotion=promotion,
        protocol={
            "train_bars": walk_config.train_bars,
            "test_bars": walk_config.test_bars,
            "step_bars": walk_config.step_bars,
            "embargo_bars": walk_config.embargo_bars,
            "candidate_count": len(candidates),
            "selection": "training-only risk-adjusted score with multiple-testing penalty",
            "execution": "shared run_backtest; next-bar fills; forced fold-end liquidation",
            "stress_slippage_bps": stressed.slippage_bps,
            "stress_commission_per_trade": stressed.commission_per_trade,
        },
    )
