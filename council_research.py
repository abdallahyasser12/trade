"""Backtest and walk-forward adapters for the council strategy.

Same technique as ``cross_sectional_research``: the audited backtest and
walk-forward code objects are executed with an isolated globals dictionary in
which only the strategy constructor (or the backtest function, or the candidate
builder) is rebound. Accounting, next-bar execution, cost gating, fractional
sizing, portfolio heat, stops, embargo, stress runs and every promotion gate
therefore remain byte-for-byte the audited implementations.

Nothing here weakens a gate. A different decision layer is the only difference.
"""

from __future__ import annotations

import types
from dataclasses import fields, replace
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

import crypto_backtest
import crypto_research
from council import (
    CouncilConfig,
    CouncilStrategy,
    CouncilStrategyConfig,
    resolve_council_config,
)
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_models import CryptoBacktestResult
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    CryptoWalkForwardResult,
)
from models import AssetInfo

COUNCIL_GRID_KEY = "council"


def with_council_config(
    config: CryptoStrategyConfig, council: CouncilConfig
) -> CouncilStrategyConfig:
    """Attach council parameters to any base config, idempotently."""

    if isinstance(config, CouncilStrategyConfig):
        return replace(config, council=council)
    values = {field.name: getattr(config, field.name) for field in fields(CryptoStrategyConfig)}
    return CouncilStrategyConfig(**values, council=council)


def _strategy_factory(default_council: CouncilConfig):
    def build(strategy_config: CryptoStrategyConfig) -> CouncilStrategy:
        return CouncilStrategy(
            strategy_config, resolve_council_config(strategy_config, default_council)
        )

    return build


def _candidate_builder(default_council: CouncilConfig):
    """A builder that can search council parameters.

    Delegates every base dimension to the audited builder, so the tunable
    allowlist, per-candidate validation and the candidate cap all still apply.
    The combined count is re-checked against the same cap so the council axis
    cannot smuggle candidates past it.
    """

    def build_crypto_candidates(
        base: CryptoStrategyConfig,
        parameter_grid: Mapping[str, Iterable[Any]],
        *,
        max_candidates: int,
    ) -> list[CryptoStrategyConfig]:
        grid = dict(parameter_grid)
        raw = list(grid.pop(COUNCIL_GRID_KEY, [default_council]))
        if not raw:
            raise ValueError("The council grid dimension needs a value.")

        unique: dict[str, CouncilConfig] = {}
        for value in raw:
            if not isinstance(value, CouncilConfig):
                raise ValueError(
                    "Council grid values must be CouncilConfig instances."
                )
            errors = value.validate()
            if errors:
                raise ValueError("Invalid council candidate: " + "; ".join(errors))
            unique.setdefault(repr(sorted(value.to_dict().items())), value)
        councils = list(unique.values())

        base_candidates = crypto_research.build_crypto_candidates(
            with_council_config(base, councils[0]), grid, max_candidates=max_candidates
        )
        combined = [
            replace(candidate, council=council)
            for council in councils
            for candidate in base_candidates
        ]
        if len(combined) > max_candidates:
            raise ValueError(
                f"Council grid creates {len(combined)} candidates; "
                f"maximum is {max_candidates}."
            )
        return combined

    return build_crypto_candidates


def run_council_backtest(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    council_config: Optional[CouncilConfig] = None,
    assets: Optional[Mapping[str, AssetInfo]] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> CryptoBacktestResult:
    """Run the audited simulator with the council decision layer."""

    resolved = resolve_council_config(strategy_config, council_config)
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid council configuration: " + "; ".join(errors))

    original = crypto_backtest.run_crypto_backtest
    namespace = dict(original.__globals__)
    namespace["CryptoRegimeStrategy"] = _strategy_factory(resolved)
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_council_backtest_impl",
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


def run_council_walk_forward(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    base_strategy: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    parameter_grid: Mapping[str, Iterable[Any]],
    walk_config: CryptoWalkForwardConfig,
    criteria: CryptoPromotionCriteria,
    *,
    council_config: Optional[CouncilConfig] = None,
) -> CryptoWalkForwardResult:
    """Run the audited walk-forward protocol against council decisions."""

    resolved = council_config or CouncilConfig()
    errors = resolved.validate()
    if errors:
        raise ValueError("Invalid council configuration: " + "; ".join(errors))

    def council_backtest(*args, **kwargs):
        return run_council_backtest(*args, council_config=None, **kwargs)

    original = crypto_research.run_crypto_walk_forward
    namespace = dict(original.__globals__)
    namespace["run_crypto_backtest"] = council_backtest
    namespace["build_crypto_candidates"] = _candidate_builder(resolved)
    injected = types.FunctionType(
        original.__code__,
        namespace,
        name="run_council_walk_forward_impl",
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    injected.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    result = injected(
        bars_by_symbol,
        with_council_config(base_strategy, resolved),
        risk_config,
        backtest_config,
        parameter_grid,
        walk_config,
        criteria,
    )
    protocol = dict(result.protocol)
    protocol["decision_engine"] = "council"
    protocol["council_default"] = resolved.to_dict()
    searched = [
        value.to_dict() for value in parameter_grid.get(COUNCIL_GRID_KEY, []) or []
    ]
    protocol["council_grid"] = searched or [resolved.to_dict()]
    protocol["council_selected"] = resolve_council_config(
        result.latest_selected_config, resolved
    ).to_dict()
    result.protocol = protocol
    return result
