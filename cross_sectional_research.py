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
from dataclasses import replace
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

import crypto_backtest
import crypto_research
from cross_sectional import (
    CrossSectionalConfig,
    CrossSectionalMomentumStrategy,
    resolve_cross_config,
    with_cross_config,
)
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    CryptoWalkForwardResult,
)
from models import AssetInfo

# The grid key under which cross-sectional candidates travel. It is not a
# CryptoStrategyConfig field, so the audited builder would reject it; the
# injected builder below pops it and hands the remaining dimensions on.
CROSS_GRID_KEY = "cross"


def _strategy_factory(cross_config: CrossSectionalConfig):
    def build(strategy_config: CryptoStrategyConfig) -> CrossSectionalMomentumStrategy:
        # A searched candidate carries the parameters it is being judged on.
        # Falling back to the run-level default here would silently evaluate
        # every candidate identically.
        return CrossSectionalMomentumStrategy(
            strategy_config, resolve_cross_config(strategy_config, cross_config)
        )

    return build


def _candidate_builder(default_cross: CrossSectionalConfig):
    """A builder that can search cross-sectional parameters.

    Delegates every base dimension to the audited builder, so the tunable
    allowlist, per-candidate validation and the candidate cap all still apply
    exactly as written. The only addition is a cross-sectional axis, and the
    combined count is re-checked against the same cap.
    """

    def build_crypto_candidates(
        base: CryptoStrategyConfig,
        parameter_grid: Mapping[str, Iterable[Any]],
        *,
        max_candidates: int,
    ) -> list[CryptoStrategyConfig]:
        grid = dict(parameter_grid)
        raw_cross = list(grid.pop(CROSS_GRID_KEY, [default_cross]))
        if not raw_cross:
            raise ValueError("The cross-sectional grid dimension needs a value.")

        unique_cross: dict[str, CrossSectionalConfig] = {}
        for value in raw_cross:
            if not isinstance(value, CrossSectionalConfig):
                raise ValueError(
                    "Cross-sectional grid values must be CrossSectionalConfig instances."
                )
            errors = value.validate()
            if errors:
                raise ValueError(
                    "Invalid cross-sectional candidate: " + "; ".join(errors)
                )
            unique_cross.setdefault(repr(sorted(value.to_dict().items())), value)
        crosses = list(unique_cross.values())

        base_candidates = crypto_research.build_crypto_candidates(
            with_cross_config(base, crosses[0]), grid, max_candidates=max_candidates
        )
        combined = [
            replace(candidate, cross=cross)
            for cross in crosses
            for candidate in base_candidates
        ]
        if len(combined) > max_candidates:
            raise ValueError(
                f"Cross-sectional grid creates {len(combined)} candidates; "
                f"maximum is {max_candidates}."
            )
        return combined

    return build_crypto_candidates


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

    resolved = resolve_cross_config(strategy_config, cross_config)
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
        return run_cross_sectional_backtest(*args, cross_config=None, **kwargs)

    original = crypto_research.run_crypto_walk_forward
    namespace = dict(original.__globals__)
    namespace["run_crypto_backtest"] = cross_backtest
    namespace["build_crypto_candidates"] = _candidate_builder(resolved)
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
        with_cross_config(base_strategy, resolved),
        risk_config,
        backtest_config,
        parameter_grid,
        walk_config,
        criteria,
    )
    protocol = dict(result.protocol)
    protocol["decision_engine"] = "cross_sectional_momentum"
    protocol["cross_sectional_default"] = resolved.to_dict()
    # Report what was actually searched, so the candidate count in the verdict
    # can be read against the axis it came from rather than assumed.
    searched = [
        value.to_dict() for value in parameter_grid.get(CROSS_GRID_KEY, []) or []
    ]
    protocol["cross_sectional_grid"] = searched or [resolved.to_dict()]
    protocol["cross_sectional_selected"] = resolve_cross_config(
        result.latest_selected_config, resolved
    ).to_dict()
    result.protocol = protocol
    return result
