"""Backtest adapter for the regime-aware ensemble.

This module reuses the audited ``crypto_backtest.run_crypto_backtest`` code object
without mutating module globals. Only the strategy constructor is replaced in an
isolated globals dictionary, so accounting, next-bar execution, costs, stops,
fractional sizing, and portfolio-risk mechanics remain byte-for-byte identical to
the production crypto backtester.
"""

from __future__ import annotations

import types
from typing import Mapping, Optional

import pandas as pd

import crypto_backtest
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult
from models import AssetInfo
from regime_ensemble import EnsembleConfig, RegimeAwareEnsembleStrategy


def _strategy_factory(ensemble_config: EnsembleConfig):
    def build(strategy_config: CryptoStrategyConfig) -> RegimeAwareEnsembleStrategy:
        return RegimeAwareEnsembleStrategy(strategy_config, ensemble_config)

    return build


def run_regime_ensemble_backtest(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    ensemble_config: Optional[EnsembleConfig] = None,
    assets: Optional[Mapping[str, AssetInfo]] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> CryptoBacktestResult:
    """Run the existing causal simulator with the ensemble strategy injected.

    A fresh function object receives a private copy of the original backtest
    globals. The repository-level ``crypto_backtest.CryptoRegimeStrategy`` name
    is never changed, so classic and ensemble research can safely run in parallel.
    """

    resolved = ensemble_config or EnsembleConfig()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid ensemble configuration: " + "; ".join(errors))

    original = crypto_backtest.run_crypto_backtest
    namespace = dict(original.__globals__)
    namespace["CryptoRegimeStrategy"] = _strategy_factory(resolved)
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_regime_ensemble_backtest_impl",
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    injected.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return injected(
        bars_by_symbol,
        strategy_config,
        risk_config,
        backtest_config,
        assets=assets,
        active_start=active_start,
    )
