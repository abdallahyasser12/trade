"""Broker-backed adapters for high-velocity intraday crypto hypotheses.

Execution semantics stay inside ``QuantControlledCryptoTradeEngine``.  Strategy
versions are isolated because lifecycle/attribution evidence from the original
single-sleeve momentum hypothesis must never be reused by the adaptive ensemble.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from quant.intraday_lifecycle import intraday_degradation_monitor
from quant.lifecycle import LifecycleManager
from quant.production_runtime import QuantControlledCryptoTradeEngine
from quant.scanner import QuantScanner
from quant.store import QuantStore
from quant.strategies.intraday_adaptive import (
    AdaptiveIntradayConfig,
    AdaptiveIntradayStrategy,
)
from quant.strategies.intraday_momentum import (
    IntradayMomentumConfig,
    IntradayMomentumStrategy,
)
from state_store import StateStore

LEGACY_STRATEGY_ID = "crypto_intraday_momentum"
LEGACY_STRATEGY_VERSION = "1.0.0"
ADAPTIVE_STRATEGY_ID = "crypto_intraday_adaptive"
ADAPTIVE_STRATEGY_VERSION = "2.0.0"

# Public defaults used by the production runtime.
STRATEGY_ID = ADAPTIVE_STRATEGY_ID
STRATEGY_VERSION = ADAPTIVE_STRATEGY_VERSION


class _IntradayControlledEngine(QuantControlledCryptoTradeEngine):
    def _base_kwargs(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        quant_store: Optional[QuantStore],
        lifecycle: Optional[LifecycleManager],
        scanner: Optional[QuantScanner],
        now_fn: Optional[Callable[[], datetime]],
    ) -> dict:
        kwargs = {
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "quant_store": quant_store,
            "lifecycle": lifecycle,
            "scanner": scanner,
            "degradation_monitor": intraday_degradation_monitor(),
        }
        if now_fn is not None:
            kwargs["now_fn"] = now_fn
        return kwargs


class IntradayAdaptiveCryptoTradeEngine(_IntradayControlledEngine):
    """Production adapter for the regime-adaptive v2 hypothesis."""

    def __init__(
        self,
        broker: AlpacaBroker,
        data_feed: AlpacaCryptoDataFeed,
        store: StateStore,
        strategy_config: CryptoStrategyConfig,
        risk_config: CryptoRiskConfig,
        engine_config: CryptoEngineConfig,
        *,
        adaptive_config: Optional[AdaptiveIntradayConfig] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[QuantScanner] = None,
    ) -> None:
        super().__init__(
            broker,
            data_feed,
            store,
            strategy_config,
            risk_config,
            engine_config,
            **self._base_kwargs(
                strategy_id=ADAPTIVE_STRATEGY_ID,
                strategy_version=ADAPTIVE_STRATEGY_VERSION,
                quant_store=quant_store,
                lifecycle=lifecycle,
                scanner=scanner,
                now_fn=now_fn,
            ),
        )
        self.adaptive_config = adaptive_config or AdaptiveIntradayConfig()
        self.strategy = AdaptiveIntradayStrategy(strategy_config, self.adaptive_config)


class IntradayMomentumCryptoTradeEngine(_IntradayControlledEngine):
    """Backward-compatible v1 adapter; no v2 evidence is shared with it."""

    def __init__(
        self,
        broker: AlpacaBroker,
        data_feed: AlpacaCryptoDataFeed,
        store: StateStore,
        strategy_config: CryptoStrategyConfig,
        risk_config: CryptoRiskConfig,
        engine_config: CryptoEngineConfig,
        *,
        intraday_config: Optional[IntradayMomentumConfig] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[QuantScanner] = None,
    ) -> None:
        super().__init__(
            broker,
            data_feed,
            store,
            strategy_config,
            risk_config,
            engine_config,
            **self._base_kwargs(
                strategy_id=LEGACY_STRATEGY_ID,
                strategy_version=LEGACY_STRATEGY_VERSION,
                quant_store=quant_store,
                lifecycle=lifecycle,
                scanner=scanner,
                now_fn=now_fn,
            ),
        )
        self.intraday_config = intraday_config or IntradayMomentumConfig()
        self.strategy = IntradayMomentumStrategy(strategy_config, self.intraday_config)


def make_intraday_adaptive_crypto_engine(
    broker: AlpacaBroker,
    data_feed: AlpacaCryptoDataFeed,
    store: StateStore,
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    engine_config: CryptoEngineConfig,
    *,
    adaptive_config: Optional[AdaptiveIntradayConfig] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    quant_store: Optional[QuantStore] = None,
    lifecycle: Optional[LifecycleManager] = None,
    scanner: Optional[QuantScanner] = None,
) -> IntradayAdaptiveCryptoTradeEngine:
    return IntradayAdaptiveCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        adaptive_config=adaptive_config,
        now_fn=now_fn,
        quant_store=quant_store,
        lifecycle=lifecycle,
        scanner=scanner,
    )


def make_intraday_momentum_crypto_engine(
    broker: AlpacaBroker,
    data_feed: AlpacaCryptoDataFeed,
    store: StateStore,
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    engine_config: CryptoEngineConfig,
    *,
    intraday_config: Optional[IntradayMomentumConfig] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    quant_store: Optional[QuantStore] = None,
    lifecycle: Optional[LifecycleManager] = None,
    scanner: Optional[QuantScanner] = None,
) -> IntradayMomentumCryptoTradeEngine:
    return IntradayMomentumCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        intraday_config=intraday_config,
        now_fn=now_fn,
        quant_store=quant_store,
        lifecycle=lifecycle,
        scanner=scanner,
    )
