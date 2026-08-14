"""Bounded chronological walk-forward promotion for the shared crypto system."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from crypto_backtest import run_crypto_backtest
from crypto_config import (
    CryptoBacktestConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
    crypto_bars_per_year,
)
from crypto_models import CryptoBacktestResult


TUNABLE_CRYPTO_PARAMETERS = frozenset(
    {
        "fast_ma_period",
        "slow_ma_period",
        "breakout_period",
        "momentum_period",
        "min_rsi",
        "max_rsi",
        "atr_stop_multiplier",
        "trailing_atr_multiplier",
        "max_atr_percentile",
        "min_adx",
        "min_efficiency_ratio",
        "min_volume_ratio",
        "min_relative_strength",
    }
)


@dataclass(frozen=True)
class CryptoWalkForwardConfig:
    train_bars: int = 1_500
    test_bars: int = 360
    step_bars: int = 360
    embargo_bars: int = 1
    min_train_trades: int = 4
    max_candidates: int = 32
    stress_slippage_bps: float = 35.0
    stress_taker_fee_bps: float = 50.0
    stress_spread_bps: float = 20.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.train_bars < 100:
            errors.append("Training window needs at least 100 aligned bars.")
        if self.test_bars < 20:
            errors.append("Test window needs at least 20 aligned bars.")
        if self.step_bars < self.test_bars:
            errors.append("Step size must prevent overlapping test windows.")
        if self.embargo_bars < 1:
            errors.append("Embargo needs at least one bar.")
        if self.min_train_trades < 1:
            errors.append("Minimum training trades must be positive.")
        if not 1 <= self.max_candidates <= 128:
            errors.append("Candidate count must be between 1 and 128.")
        for label, value in {
            "stress slippage": self.stress_slippage_bps,
            "stress taker fee": self.stress_taker_fee_bps,
            "stress spread": self.stress_spread_bps,
        }.items():
            if not math.isfinite(value) or value < 0:
                errors.append(f"{label} must be finite and non-negative.")
        return errors


@dataclass(frozen=True)
class CryptoPromotionCriteria:
    min_folds: int = 3
    min_oos_trades: int = 12
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
            errors.append("Promotion needs at least two test folds.")
        if self.min_oos_trades < 1:
            errors.append("Promotion trade count must be positive.")
        if not 0 <= self.min_positive_fold_ratio <= 1:
            errors.append("Positive-fold ratio must be between 0 and 1.")
        for label, value in {
            "minimum return": self.min_compounded_return_pct,
            "minimum excess return": self.min_excess_return_pct,
            "minimum Sharpe": self.min_aggregate_sharpe,
            "minimum profit factor": self.min_profit_factor,
            "maximum drawdown": self.max_fold_drawdown_pct,
            "minimum stressed return": self.min_stressed_return_pct,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.min_profit_factor <= 0 or self.max_fold_drawdown_pct <= 0:
            errors.append("Profit-factor and drawdown gates must be positive.")
        return errors


@dataclass
class CryptoPromotionDecision:
    promoted: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class CryptoWalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_config: CryptoStrategyConfig
    training_score: float
    test_result: CryptoBacktestResult
    baseline_result: CryptoBacktestResult
    stressed_result: CryptoBacktestResult


@dataclass
class CryptoWalkForwardResult:
    folds: list[CryptoWalkForwardFold]
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
    latest_selected_config: CryptoStrategyConfig
    promotion: CryptoPromotionDecision
    protocol: dict[str, Any]


def build_crypto_candidates(
    base: CryptoStrategyConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    *,
    max_candidates: int,
) -> list[CryptoStrategyConfig]:
    unknown = set(parameter_grid).difference(TUNABLE_CRYPTO_PARAMETERS)
    if unknown:
        raise ValueError(f"Unsupported crypto tunables: {sorted(unknown)}")
    keys = sorted(parameter_grid)
    lists = [list(parameter_grid[key]) for key in keys]
    if any(not values for values in lists):
        raise ValueError("Each crypto parameter-grid dimension needs a value.")
    count = math.prod(len(values) for values in lists) if lists else 1
    if count > max_candidates:
        raise ValueError(f"Crypto grid creates {count} candidates; maximum is {max_candidates}.")
    raw = (
        [replace(base)]
        if not keys
        else [
            replace(base, **dict(zip(keys, values, strict=True)))
            for values in itertools.product(*lists)
        ]
    )
    unique: dict[str, CryptoStrategyConfig] = {}
    for candidate in raw:
        if candidate.validate():
            continue
        identity = repr(sorted(candidate.to_dict().items()))
        unique.setdefault(identity, candidate)
    if not unique:
        raise ValueError("Crypto grid produced no valid configurations.")
    return list(unique.values())


def _score(result: CryptoBacktestResult, min_trades: int, candidate_count: int) -> float:
    if result.num_trades < min_trades:
        return -math.inf
    sharpe = result.sharpe_ratio if result.sharpe_ratio is not None else -2.0
    calmar = result.total_return_pct / max(1.0, result.max_drawdown_pct)
    pf = min(4.0, result.profit_factor or 0.0)
    observations = max(2, len(result.equity_curve) - 1)
    selection_penalty = math.sqrt(2 * math.log(max(1, candidate_count)) / observations)
    return sharpe + 0.30 * calmar + 0.15 * math.log1p(pf) - 0.02 * result.max_drawdown_pct - selection_penalty


def _compound(results: list[CryptoBacktestResult]) -> float:
    wealth = 1.0
    for result in results:
        wealth *= max(0.0, 1.0 + result.total_return_pct / 100.0)
    return (wealth - 1.0) * 100.0


def _aggregate_sharpe(
    results: list[CryptoBacktestResult], timeframe: str, risk_free_rate: float
) -> Optional[float]:
    returns: list[float] = []
    for result in results:
        returns.extend(result.equity_curve.pct_change().dropna().tolist())
    if len(returns) < 20:
        return None
    values = np.asarray(returns, dtype=float)
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    factor = crypto_bars_per_year(timeframe)
    return float((values - risk_free_rate / factor).mean() / std * math.sqrt(factor))


def _profit_factor(results: list[CryptoBacktestResult]) -> Optional[float]:
    wins = sum(
        float(trade.pnl)
        for result in results
        for trade in result.trades
        if trade.pnl is not None and trade.pnl > 0
    )
    losses = sum(
        -float(trade.pnl)
        for result in results
        for trade in result.trades
        if trade.pnl is not None and trade.pnl < 0
    )
    return wins / losses if losses else math.inf if wins else None


def evaluate_crypto_promotion(
    folds: list[CryptoWalkForwardFold],
    criteria: CryptoPromotionCriteria,
) -> tuple[CryptoPromotionDecision, dict[str, Any]]:
    criteria_errors = criteria.validate()
    if criteria_errors:
        raise ValueError("Invalid crypto promotion criteria: " + "; ".join(criteria_errors))
    if not folds:
        raise ValueError("Crypto promotion evaluation needs at least one completed fold.")
    tests = [fold.test_result for fold in folds]
    baselines = [fold.baseline_result for fold in folds]
    stressed = [fold.stressed_result for fold in folds]
    compounded = _compound(tests)
    baseline = _compound(baselines)
    stressed_return = _compound(stressed)
    positive_ratio = sum(result.total_return_pct > 0 for result in tests) / len(tests)
    total_trades = sum(result.num_trades for result in tests)
    sharpe = _aggregate_sharpe(
        tests,
        folds[0].selected_config.timeframe,
        folds[0].test_result.config_snapshot["backtest"]["risk_free_rate"],
    )
    pf = _profit_factor(tests)
    worst_drawdown = max(result.max_drawdown_pct for result in tests)
    failures: list[str] = []
    checks = [
        (len(folds) >= criteria.min_folds, f"folds {len(folds)} < {criteria.min_folds}"),
        (total_trades >= criteria.min_oos_trades, f"OOS trades {total_trades} < {criteria.min_oos_trades}"),
        (positive_ratio >= criteria.min_positive_fold_ratio, f"positive-fold ratio {positive_ratio:.2f} < {criteria.min_positive_fold_ratio:.2f}"),
        (compounded >= criteria.min_compounded_return_pct, f"OOS return {compounded:.2f}% < {criteria.min_compounded_return_pct:.2f}%"),
        (compounded - baseline >= criteria.min_excess_return_pct, f"excess return {compounded - baseline:.2f}% < {criteria.min_excess_return_pct:.2f}%"),
        (sharpe is not None and sharpe >= criteria.min_aggregate_sharpe, f"aggregate Sharpe {'n/a' if sharpe is None else f'{sharpe:.2f}'} < {criteria.min_aggregate_sharpe:.2f}"),
        (pf is not None and pf >= criteria.min_profit_factor, f"profit factor {'n/a' if pf is None else f'{pf:.2f}'} < {criteria.min_profit_factor:.2f}"),
        (worst_drawdown <= criteria.max_fold_drawdown_pct, f"worst drawdown {worst_drawdown:.2f}% > {criteria.max_fold_drawdown_pct:.2f}%"),
        (stressed_return >= criteria.min_stressed_return_pct, f"stressed return {stressed_return:.2f}% < {criteria.min_stressed_return_pct:.2f}%"),
    ]
    failures.extend(message for passed, message in checks if not passed)
    metrics = {
        "compounded": compounded,
        "baseline": baseline,
        "stressed": stressed_return,
        "positive_ratio": positive_ratio,
        "total_trades": total_trades,
        "sharpe": sharpe,
        "profit_factor": pf,
        "worst_drawdown": worst_drawdown,
    }
    return CryptoPromotionDecision(not failures, failures), metrics


def run_crypto_walk_forward(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    base_strategy: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: CryptoWalkForwardConfig,
    criteria: CryptoPromotionCriteria,
) -> CryptoWalkForwardResult:
    errors = walk_config.validate() + criteria.validate()
    if errors:
        raise ValueError("Invalid crypto walk-forward configuration: " + "; ".join(errors))
    candidates = build_crypto_candidates(
        base_strategy, parameter_grid, max_candidates=walk_config.max_candidates
    )
    common: Optional[pd.DatetimeIndex] = None
    reference_symbol = base_strategy.symbols[0]
    for symbol in base_strategy.symbols:
        if symbol not in bars_by_symbol:
            raise ValueError(f"Missing bars for {symbol}.")
        index = bars_by_symbol[symbol].index
        if common is None:
            common = index
        elif not index.equals(common):
            raise ValueError(
                "Crypto walk-forward requires exact timestamp alignment; "
                f"{symbol} differs from {reference_symbol}."
            )
    if common is None:
        raise ValueError("Crypto walk-forward received no aligned history.")
    common = common.sort_values()
    required = walk_config.train_bars + walk_config.embargo_bars + walk_config.test_bars
    if len(common) < required:
        raise ValueError(f"Crypto walk-forward needs {required} aligned bars; received {len(common)}.")

    forced = replace(backtest_config, liquidate_at_end=True)
    stressed_config = replace(
        forced,
        slippage_bps=max(forced.slippage_bps, walk_config.stress_slippage_bps),
        taker_fee_bps=max(forced.taker_fee_bps, walk_config.stress_taker_fee_bps),
        spread_bps=max(forced.spread_bps, walk_config.stress_spread_bps),
    )
    maximum_warmup = max(candidate.warmup_bars() for candidate in candidates)
    folds: list[CryptoWalkForwardFold] = []
    train_start = 0
    while True:
        train_end = train_start + walk_config.train_bars
        test_start = train_end + walk_config.embargo_bars
        test_end = test_start + walk_config.test_bars
        if test_end > len(common):
            break
        train_context = max(0, train_start - maximum_warmup - 2)
        train_index = common[train_context:train_end]
        train_data = {symbol: bars_by_symbol[symbol].loc[train_index] for symbol in base_strategy.symbols}
        activation = common[train_start]
        scored: list[tuple[float, int, CryptoStrategyConfig]] = []
        for ordinal, candidate in enumerate(candidates):
            try:
                result = run_crypto_backtest(
                    train_data,
                    candidate,
                    risk_config,
                    forced,
                    active_start=activation if train_context < train_start else None,
                )
                score = _score(result, walk_config.min_train_trades, len(candidates))
            except ValueError:
                score = -math.inf
            scored.append((score, -ordinal, candidate))
        best_score, _, selected = max(scored, key=lambda item: (item[0], item[1]))
        if not math.isfinite(best_score):
            raise ValueError(f"No crypto candidate met training evidence in fold {len(folds) + 1}.")

        context = max(selected.warmup_bars(), base_strategy.warmup_bars()) + 2
        context_start = max(0, test_start - context)
        test_index = common[context_start:test_end]
        test_data = {symbol: bars_by_symbol[symbol].loc[test_index] for symbol in base_strategy.symbols}
        test_activation = common[test_start]
        selected_result = run_crypto_backtest(
            test_data, selected, risk_config, forced, active_start=test_activation
        )
        baseline_result = run_crypto_backtest(
            test_data, base_strategy, risk_config, forced, active_start=test_activation
        )
        stressed_result = run_crypto_backtest(
            test_data, selected, risk_config, stressed_config, active_start=test_activation
        )
        folds.append(
            CryptoWalkForwardFold(
                train_start=common[train_start],
                train_end=common[train_end - 1],
                test_start=test_activation,
                test_end=common[test_end - 1],
                selected_config=selected,
                training_score=best_score,
                test_result=selected_result,
                baseline_result=baseline_result,
                stressed_result=stressed_result,
            )
        )
        train_start += walk_config.step_bars
    if not folds:
        raise ValueError("Crypto walk-forward produced no complete folds.")
    promotion, metrics = evaluate_crypto_promotion(folds, criteria)
    return CryptoWalkForwardResult(
        folds=folds,
        candidate_count=len(candidates),
        compounded_return_pct=metrics["compounded"],
        baseline_compounded_return_pct=metrics["baseline"],
        stressed_compounded_return_pct=metrics["stressed"],
        median_fold_return_pct=float(median(fold.test_result.total_return_pct for fold in folds)),
        positive_fold_ratio=metrics["positive_ratio"],
        aggregate_sharpe=metrics["sharpe"],
        profit_factor=metrics["profit_factor"],
        worst_fold_drawdown_pct=metrics["worst_drawdown"],
        total_oos_trades=metrics["total_trades"],
        latest_selected_config=folds[-1].selected_config,
        promotion=promotion,
        protocol={
            "train_bars": walk_config.train_bars,
            "test_bars": walk_config.test_bars,
            "step_bars": walk_config.step_bars,
            "embargo_bars": walk_config.embargo_bars,
            "candidate_count": len(candidates),
            "selection": "training-only risk-adjusted score with multiple-testing penalty",
            "execution": "shared next-bar crypto backtest with forced fold liquidation",
            "stress": stressed_config.to_dict(),
        },
    )
