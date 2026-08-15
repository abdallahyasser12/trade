"""Backtest and walk-forward adapters for the cross-sectional strategy.

Uses the same technique as ``advanced_crypto_backtest``: the audited backtest
and walk-forward code objects are executed with an isolated globals dictionary
in which only the strategy constructor (or the backtest function) is rebound.
Accounting, next-bar execution, cost gating, fractional sizing, portfolio heat,
stops, candidate generation, embargo, stress runs and every promotion gate
therefore remain byte-for-byte the audited implementations.

Nothing here weakens a gate. A different decision layer is the only difference.
"""

from __future__ import annotations

import types
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

import crypto_backtest
import crypto_research
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    CryptoWalkForwardResult,
)
from models import AssetInfo


def _strategy_factory(cross_config: CrossSectionalConfig):
    def build(strategy_config: CryptoStrategyConfig) -> CrossSectionalMomentumStrategy:
        return CrossSectionalMomentumStrategy(strategy_config, cross_config)

    return build


def run_cross_sectional_backtest(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    cross_config: Optional[CrossSectionalConfig] = None,
    assets: Optional[Mapping[str, AssetInfo]] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> CryptoBacktestResult:
    """Run the audited simulator with the cross-sectional decision layer."""

    resolved = cross_config or CrossSectionalConfig()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid cross-sectional configuration: " + "; ".join(errors))

    original = crypto_backtest.run_crypto_backtest
    namespace = dict(original.__globals__)
    namespace["CryptoRegimeStrategy"] = _strategy_factory(resolved)
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_cross_sectional_backtest_impl",
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


def run_cross_sectional_walk_forward(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    base_strategy: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: CryptoWalkForwardConfig,
    criteria: CryptoPromotionCriteria,
    *,
    cross_config: Optional[CrossSectionalConfig] = None,
) -> CryptoWalkForwardResult:
    """Run the audited walk-forward protocol against cross-sectional decisions."""

    resolved = cross_config or CrossSectionalConfig()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid cross-sectional configuration: " + "; ".join(errors))

    def cross_backtest(*args, **kwargs):
        return run_cross_sectional_backtest(*args, cross_config=resolved, **kwargs)

    original = crypto_research.run_crypto_walk_forward
    namespace = dict(original.__globals__)
    namespace["run_crypto_backtest"] = cross_backtest
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_cross_sectional_walk_forward_impl",
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
    protocol["decision_engine"] = "cross_sectional_momentum"
    protocol["cross_sectional"] = resolved.to_dict()
    result.protocol = protocol
    return result
