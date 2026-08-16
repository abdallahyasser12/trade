"""Point-in-time-safe dynamic universe construction (spec §5).

This module never fetches: it filters CALLER-PROVIDED per-symbol stats,
so the same code serves live scanning (stats from today's feed) and
historical research (stats as they were on the historical date).  The
survivorship-bias guard is structural: a stat snapshot dated after the
requested as-of moment refuses the whole build, because a universe built
from future knowledge poisons every backtest run on it.

Every exclusion carries a named reason — the platform's "no silent
rejections" rule applies to symbols exactly as it applies to trades.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from quant.contracts import Market


@dataclass(frozen=True)
class UniverseFilterConfig:
    # Penny exclusion defaults ON (spec §5).
    min_price: float = 5.0
    max_price: Optional[float] = None
    min_avg_dollar_volume: float = 5_000_000.0
    min_history_bars: int = 250
    max_spread_bps: float = 30.0
    max_symbols: int = 200
    allow_missing_spread: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.min_price) or self.min_price < 0:
            errors.append("Minimum price must be finite and non-negative.")
        if self.max_price is not None and (
            not math.isfinite(self.max_price) or self.max_price <= self.min_price
        ):
            errors.append("Maximum price must exceed the minimum price.")
        if not math.isfinite(self.min_avg_dollar_volume) or self.min_avg_dollar_volume <= 0:
            errors.append("Minimum dollar volume must be finite and positive.")
        if self.min_history_bars < 1:
            errors.append("Minimum history must be at least one bar.")
        if not math.isfinite(self.max_spread_bps) or self.max_spread_bps <= 0:
            errors.append("Maximum spread must be finite and positive.")
        if not (1 <= self.max_symbols <= 5000):
            errors.append("Maximum symbols must be between 1 and 5000.")
        return errors


@dataclass(frozen=True)
class SymbolStats:
    """Per-symbol facts as of one moment; the caller owns their honesty."""

    symbol: str
    market: Market
    asof: datetime
    price: float
    avg_dollar_volume: float
    history_bars: int
    avg_spread_bps: Optional[float] = None
    tradable: bool = True
    halted: bool = False
    fractionable: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.symbol:
            errors.append("Symbol is required.")
        if self.asof.tzinfo is None:
            errors.append("Stats timestamp must be timezone-aware.")
        for label, value in {
            "price": self.price,
            "average dollar volume": self.avg_dollar_volume,
        }.items():
            if not math.isfinite(value) or value < 0:
                errors.append(f"{label} must be finite and non-negative.")
        if self.history_bars < 0:
            errors.append("History bar count cannot be negative.")
        return errors


@dataclass
class UniverseSelection:
    asof: datetime
    selected: list[str]
    rejected: dict[str, str]
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof.isoformat(),
            "selected": list(self.selected),
            "rejected": dict(self.rejected),
            "config_snapshot": dict(self.config_snapshot),
        }


def build_universe(
    stats: list[SymbolStats],
    config: UniverseFilterConfig,
    asof: datetime,
) -> UniverseSelection:
    """Filter and rank symbols using only information dated <= ``asof``."""
    errors = config.validate()
    if errors:
        raise ValueError("Invalid universe configuration: " + "; ".join(errors))
    if asof.tzinfo is None:
        raise ValueError("Universe as-of moment must be timezone-aware.")

    future_dated = [s.symbol for s in stats if s.asof > asof]
    if future_dated:
        raise ValueError(
            "Point-in-time violation: stats for "
            f"{sorted(future_dated)} are dated after the requested as-of "
            f"moment {asof.isoformat()}; a universe built from future "
            "knowledge poisons every backtest run on it."
        )

    survivors: list[SymbolStats] = []
    rejected: dict[str, str] = {}
    for stat in stats:
        stat_errors = stat.validate()
        if stat_errors:
            rejected[stat.symbol or "<missing>"] = "invalid_stats"
            continue
        if not stat.tradable:
            rejected[stat.symbol] = "not_tradable"
        elif stat.halted:
            rejected[stat.symbol] = "halted"
        elif stat.price < config.min_price:
            rejected[stat.symbol] = "below_minimum_price"
        elif config.max_price is not None and stat.price > config.max_price:
            rejected[stat.symbol] = "above_maximum_price"
        elif stat.avg_dollar_volume < config.min_avg_dollar_volume:
            rejected[stat.symbol] = "insufficient_dollar_volume"
        elif stat.history_bars < config.min_history_bars:
            rejected[stat.symbol] = "insufficient_history"
        elif stat.avg_spread_bps is None and not config.allow_missing_spread:
            rejected[stat.symbol] = "spread_unknown"
        elif stat.avg_spread_bps is not None and stat.avg_spread_bps > config.max_spread_bps:
            rejected[stat.symbol] = "spread_too_wide"
        else:
            survivors.append(stat)

    # Liquidity-ranked, deterministic tie-break by symbol; truncation to
    # the cap is itself a named rejection so nothing disappears silently.
    survivors.sort(key=lambda s: (-s.avg_dollar_volume, s.symbol))
    selected = [s.symbol for s in survivors[: config.max_symbols]]
    for stat in survivors[config.max_symbols :]:
        rejected[stat.symbol] = "beyond_symbol_cap"

    return UniverseSelection(
        asof=asof,
        selected=selected,
        rejected=rejected,
        config_snapshot=asdict(config),
    )
