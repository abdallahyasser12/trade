"""Central strategy registry: one place that knows every strategy family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from crypto_config import CryptoStrategyConfig
from quant.contracts import Market
from quant.strategies.base import StrategyMetadata, UniverseStrategy

StrategyFactory = Callable[[CryptoStrategyConfig], UniverseStrategy]


@dataclass(frozen=True)
class StrategySpec:
    metadata: StrategyMetadata
    factory: StrategyFactory


class StrategyRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, StrategySpec] = {}

    def register(self, metadata: StrategyMetadata, factory: StrategyFactory) -> None:
        errors = metadata.validate()
        if errors:
            raise ValueError(
                f"Invalid metadata for {metadata.strategy_id!r}: " + "; ".join(errors)
            )
        if metadata.strategy_id in self._specs:
            raise ValueError(f"Strategy {metadata.strategy_id!r} is already registered.")
        self._specs[metadata.strategy_id] = StrategySpec(metadata, factory)

    def get(self, strategy_id: str) -> StrategySpec:
        if strategy_id not in self._specs:
            raise KeyError(
                f"Unknown strategy {strategy_id!r}. Registered: {sorted(self._specs)}"
            )
        return self._specs[strategy_id]

    def create(self, strategy_id: str, config: CryptoStrategyConfig) -> UniverseStrategy:
        return self.get(strategy_id).factory(config)

    def list(self, market: Optional[Market] = None) -> list[StrategyMetadata]:
        entries = [spec.metadata for spec in self._specs.values()]
        if market is not None:
            entries = [meta for meta in entries if market in meta.markets]
        return sorted(entries, key=lambda meta: meta.strategy_id)

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)


def _make_regime_breakout(config):
    from crypto_strategy import CryptoRegimeStrategy
    return CryptoRegimeStrategy(config)


def _make_cross_sectional(config):
    from cross_sectional import CrossSectionalMomentumStrategy, resolve_cross_config
    return CrossSectionalMomentumStrategy(config, resolve_cross_config(config))


def _make_mean_reversion(config):
    from mean_reversion import MeanReversionStrategy, resolve_mean_reversion_config
    return MeanReversionStrategy(config, resolve_mean_reversion_config(config))


def _make_volatility_regime(config):
    from volatility_regime import VolatilityRegimeStrategy, resolve_volatility_config
    return VolatilityRegimeStrategy(config, resolve_volatility_config(config))


def _make_donchian_trend(config):
    from quant.strategies.donchian_trend import DonchianTrendStrategy
    return DonchianTrendStrategy(config)


def _make_opening_range_breakout(config):
    from quant.strategies.opening_range import OpeningRangeBreakoutStrategy
    return OpeningRangeBreakoutStrategy(config)


def _make_vwap_reversion(config):
    from quant.strategies.vwap_reversion import VwapReversionStrategy
    return VwapReversionStrategy(config)


def _make_volatility_squeeze(config):
    from quant.strategies.volatility_squeeze import VolatilitySqueezeStrategy
    return VolatilitySqueezeStrategy(config)


def _make_intraday_momentum(config):
    from quant.strategies.intraday_momentum import IntradayMomentumStrategy
    return IntradayMomentumStrategy(config)


def build_default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        StrategyMetadata(
            strategy_id="crypto_regime_breakout",
            name="Regime breakout (production crypto)",
            version="1.0.0",
            hypothesis=(
                "Confirmed breakouts above the prior range, taken only in a "
                "benchmark-confirmed bull regime with volume and trend-quality "
                "confirmation, earn more than they cost after fees."
            ),
            markets=(Market.CRYPTO,),
            natural_timeframes=("1Hour", "4Hour"),
            horizon_bars=24,
            tags=("breakout", "trend"),
        ),
        _make_regime_breakout,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="crypto_cross_sectional",
            name="Cross-sectional momentum",
            version="1.0.0",
            hypothesis=(
                "Ranking the universe by risk-adjusted momentum and holding the "
                "strongest names earns positive net expectancy. Existing real-data "
                "evidence is positive but insufficient for promotion."
            ),
            markets=(Market.CRYPTO,),
            natural_timeframes=("4Hour", "1Day"),
            horizon_bars=42,
            tags=("momentum", "cross_sectional"),
        ),
        _make_cross_sectional,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="crypto_intraday_momentum",
            name="Intraday cross-sectional momentum",
            version="1.0.0",
            hypothesis=(
                "On liquid crypto pairs, short-horizon risk-adjusted momentum leaders "
                "with non-degenerate volume persist long enough on 5-15 minute bars "
                "to cover spread, slippage and fees; rank hysteresis limits churn. "
                "This is a new unproven hypothesis and ships RESEARCH-only."
            ),
            markets=(Market.CRYPTO,),
            natural_timeframes=("5Min", "15Min"),
            horizon_bars=12,
            tags=("intraday", "momentum", "cross_sectional", "high_velocity"),
        ),
        _make_intraday_momentum,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="crypto_mean_reversion",
            name="Oversold mean reversion",
            version="1.0.0",
            hypothesis=(
                "Sharp short-term dislocations below a rising long-term trend revert "
                "often enough to pay their costs."
            ),
            markets=(Market.CRYPTO,),
            natural_timeframes=("1Hour", "4Hour"),
            horizon_bars=12,
            tags=("mean_reversion",),
        ),
        _make_mean_reversion,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="crypto_volatility_regime",
            name="Volatility-regime trend",
            version="1.0.0",
            hypothesis=(
                "Trend entries filtered by a volatility-regime classifier avoid the "
                "chop that destroys unfiltered trend systems."
            ),
            markets=(Market.CRYPTO,),
            natural_timeframes=("4Hour", "1Day"),
            horizon_bars=30,
            tags=("trend", "volatility"),
        ),
        _make_volatility_regime,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="donchian_trend",
            name="Volatility-adjusted Donchian trend",
            version="1.0.0",
            hypothesis=(
                "New N-bar highs with ATR-scaled stops capture the fat right tail of "
                "trends; volatility scaling keeps risk per trade level."
            ),
            markets=(Market.CRYPTO, Market.US_EQUITY),
            natural_timeframes=("4Hour", "1Day"),
            horizon_bars=40,
            tags=("trend", "breakout", "runner"),
        ),
        _make_donchian_trend,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="opening_range_breakout",
            name="Opening-range breakout",
            version="1.0.0",
            hypothesis=(
                "A break of the first-hour range on elevated relative volume continues "
                "intraday more often than it reverses, net of spread."
            ),
            markets=(Market.US_EQUITY,),
            natural_timeframes=("5Min", "15Min"),
            horizon_bars=24,
            tags=("intraday", "breakout", "momentum"),
        ),
        _make_opening_range_breakout,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="vwap_reversion",
            name="VWAP overextension reversion",
            version="1.0.0",
            hypothesis=(
                "Intraday stretches far below session VWAP in a non-trending tape "
                "revert toward VWAP with enough frequency to pay costs."
            ),
            markets=(Market.US_EQUITY, Market.CRYPTO),
            natural_timeframes=("5Min", "15Min"),
            horizon_bars=12,
            tags=("intraday", "mean_reversion", "vwap"),
        ),
        _make_vwap_reversion,
    )
    registry.register(
        StrategyMetadata(
            strategy_id="volatility_squeeze",
            name="Volatility-squeeze expansion",
            version="1.0.0",
            hypothesis=(
                "Breakouts out of multi-bar volatility compression carry more "
                "follow-through than breakouts from already-expanded ranges."
            ),
            markets=(Market.US_EQUITY, Market.CRYPTO),
            natural_timeframes=("15Min", "1Hour", "4Hour"),
            horizon_bars=20,
            tags=("breakout", "volatility"),
        ),
        _make_volatility_squeeze,
    )
    return registry
