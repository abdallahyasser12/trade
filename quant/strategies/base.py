"""The one strategy interface every quant-desk strategy implements.

The protocol below is the SAME shape the existing production crypto
strategies already implement (``CryptoRegimeStrategy``,
``CrossSectionalMomentumStrategy``, ``MeanReversionStrategy``,
``VolatilityRegimeStrategy``, ``RegimeAwareEnsembleStrategy``): the
existing engines, backtester, and walk-forward research harness all
consume it.  Formalizing it here — instead of inventing a second
interface — means every new intraday strategy plugs into the already
audited execution and research paths unchanged.

A strategy is a hypothesis, not an assumed edge.  ``StrategyMetadata``
carries the hypothesis text, natural horizon, and market scope so the
research factory and lifecycle manager can treat strategies as data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import pandas as pd

from crypto_models import CryptoSignalDecision
from quant.contracts import Market


@runtime_checkable
class UniverseStrategy(Protocol):
    """Structural interface of a multi-asset, long-only, bar-driven strategy.

    Contract, enforced by the shared research/causality tests:

    * ``prepare_symbol`` adds indicator columns using bar N data for bar N
      values only (strictly causal — appending future bars must never
      change an already-computed value).
    * ``evaluate_prepared`` reads exactly one timestamp's completed row
      (plus history) and returns one decision per configured symbol, BUYs
      first in deterministic priority order.
    * ``ratchet_stop`` may only tighten a protective stop, never loosen it.
    """

    def required_bars(self) -> int:
        """Completed bars needed before signals can be generated safely."""
        ...

    def prepare_symbol(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``bars`` with causal indicator columns added."""
        ...

    def prepare_universe(
        self, bars_by_symbol: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Prepare every configured symbol plus cross-sectional context."""
        ...

    def evaluate(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        """Prepare and evaluate the latest common completed bar."""
        ...

    def evaluate_prepared(
        self,
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp | datetime,
        held_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> list[CryptoSignalDecision]:
        """Evaluate one aligned completed bar across the prepared universe."""
        ...

    def ratchet_stop(
        self,
        current_stop: float,
        highest_completed_close: float,
        completed_close: float,
        current_atr: float,
    ) -> tuple[float, float]:
        """Return (next_stop, next_highest_close); the stop may only rise."""
        ...


@dataclass(frozen=True)
class StrategyMetadata:
    """Registry-facing description of one strategy family.

    ``strategy_id`` is the stable key used by the lifecycle store, the
    decision journal, and attribution records; changing behavior without
    bumping ``version`` corrupts every downstream measurement, so any
    change to signal logic MUST bump it.
    """

    strategy_id: str
    name: str
    version: str
    hypothesis: str
    markets: tuple[Market, ...]
    natural_timeframes: tuple[str, ...]
    horizon_bars: int
    required_sources: tuple[str, ...] = ("price",)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.strategy_id or not self.strategy_id.replace("_", "").isalnum():
            errors.append("strategy_id must be a non-empty snake_case identifier.")
        if not self.version:
            errors.append("version is required.")
        if not self.hypothesis:
            errors.append("A strategy must state its hypothesis explicitly.")
        if not self.markets:
            errors.append("At least one market is required.")
        if not self.natural_timeframes:
            errors.append("At least one natural timeframe is required.")
        if self.horizon_bars < 0:
            errors.append("horizon_bars cannot be negative.")
        if "price" not in self.required_sources:
            errors.append("Every bar-driven strategy requires the price source.")
        return errors
