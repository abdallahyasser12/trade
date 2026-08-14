"""Walk-forward research adapter for the regime-aware crypto ensemble."""

from __future__ import annotations

import types
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

import crypto_research
from advanced_crypto_backtest import run_regime_ensemble_backtest
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    CryptoWalkForwardResult,
)
from regime_ensemble import EnsembleConfig


def run_regime_ensemble_walk_forward(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    base_strategy: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: CryptoWalkForwardConfig,
    criteria: CryptoPromotionCriteria,
    *,
    ensemble_config: Optional[EnsembleConfig] = None,
) -> CryptoWalkForwardResult:
    """Run the audited chronological research protocol with ensemble decisions.

    ``crypto_research.run_crypto_walk_forward`` is executed from its original code
    object with an isolated globals dictionary in which only
    ``run_crypto_backtest`` is rebound. Candidate generation, embargoes,
    non-overlapping test folds, stress tests, multiple-testing penalty, and
    promotion gates therefore remain unchanged.
    """

    resolved = ensemble_config or EnsembleConfig()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid ensemble configuration: " + "; ".join(errors))

    def ensemble_backtest(*args, **kwargs):
        return run_regime_ensemble_backtest(
            *args,
            ensemble_config=resolved,
            **kwargs,
        )

    original = crypto_research.run_crypto_walk_forward
    namespace = dict(original.__globals__)
    namespace["run_crypto_backtest"] = ensemble_backtest
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_regime_ensemble_walk_forward_impl",
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    injected.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    result = injected(
        bars_by_symbol,
        base_strategy,
        risk_config,
        backtest_config,
        parameter_grid,
        walk_config,
        criteria,
    )
    protocol = dict(result.protocol)
    protocol["decision_engine"] = "regime_ensemble_v2"
    protocol["ensemble"] = {
        "min_buy_score": resolved.min_buy_score,
        "min_regime_confidence": resolved.min_regime_confidence,
        "stress_atr_percentile": resolved.stress_atr_percentile,
        "min_breadth_for_trend": resolved.min_breadth_for_trend,
        "pullback_max_distance_atr": resolved.pullback_max_distance_atr,
        "mean_reversion_max_rsi": resolved.mean_reversion_max_rsi,
        "mean_reversion_max_distance_atr": resolved.mean_reversion_max_distance_atr,
        "require_positive_benchmark_momentum": resolved.require_positive_benchmark_momentum,
    }
    result.protocol = protocol
    return result
