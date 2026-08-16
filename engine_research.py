"""Generic research adapters for any registered decision engine.

``cross_sectional_research`` and ``council_research`` each carry a bespoke copy
of the code-object injection. That was fine for two engines and is not fine for
six, so this module holds the one generic version: give it a factory that turns
a ``CryptoStrategyConfig`` into a strategy object, and it will run the audited
backtest or walk-forward against that decision layer.

The injection itself is unchanged. The audited functions are executed with an
isolated globals dictionary in which only the strategy constructor (or the
backtest function) is rebound, so accounting, next-bar execution, cost gating,
fractional sizing, portfolio heat, stops, embargo, stress runs and every
promotion gate remain byte-for-byte the audited implementations.

This exists to answer one question the repository could not previously ask:
does a council of strategies beat its own best single member, judged by the
same gates? Answering it requires running each member alone through the
identical protocol, which requires exactly this.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Iterable, Mapping, Optional

import pandas as pd

import crypto_backtest
import crypto_research
from breadth import BreadthConfig, BreadthStrategy
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    CryptoWalkForwardResult,
)
from crypto_strategy import CryptoRegimeStrategy
from mean_reversion import MeanReversionConfig, MeanReversionStrategy
from models import AssetInfo
from volatility_regime import VolatilityRegimeConfig, VolatilityRegimeStrategy

# Engines that take a single fixed configuration and expose no searchable axis
# of their own. Running them with an empty grid gives a one-candidate protocol:
# no in-sample selection at all, and therefore no multiple-testing burden to
# discount. That is the cleanest possible comparison baseline for the council.
SOLO_ENGINES: dict[str, Callable[[CryptoStrategyConfig], Any]] = {
    "meanrev": lambda cfg: MeanReversionStrategy(cfg, MeanReversionConfig()),
    "volregime": lambda cfg: VolatilityRegimeStrategy(cfg, VolatilityRegimeConfig()),
    "breadth": lambda cfg: BreadthStrategy(cfg, BreadthConfig()),
    "base": CryptoRegimeStrategy,
}


def run_injected_backtest(
    strategy_factory: Callable[[CryptoStrategyConfig], Any],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    assets: Optional[Mapping[str, AssetInfo]] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> CryptoBacktestResult:
    """Run the audited simulator against an arbitrary decision layer."""

    original = crypto_backtest.run_crypto_backtest
    namespace = dict(original.__globals__)
    namespace["CryptoRegimeStrategy"] = strategy_factory
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_injected_backtest_impl",
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


def run_injected_walk_forward(
    strategy_factory: Callable[[CryptoStrategyConfig], Any],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    base_strategy: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: CryptoWalkForwardConfig,
    criteria: CryptoPromotionCriteria,
    *,
    engine_name: str = "injected",
) -> CryptoWalkForwardResult:
    """Run the audited walk-forward protocol against an arbitrary decision layer."""

    def injected_backtest(*args, **kwargs):
        return run_injected_backtest(strategy_factory, *args, **kwargs)

    original = crypto_research.run_crypto_walk_forward
    namespace = dict(original.__globals__)
    namespace["run_crypto_backtest"] = injected_backtest
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_injected_walk_forward_impl",
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
    protocol["decision_engine"] = engine_name
    result.protocol = protocol
    return result
