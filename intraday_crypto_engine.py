"""Broker-backed adapter for the high-velocity intraday momentum hypothesis.

Strategy logic changes; execution does not. Broker-backed entries pass through the
Quant-controlled lifecycle runtime and use intraday-specific degradation thresholds.
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
from quant.strategies.intraday_momentum import (
    IntradayMomentumConfig,
    IntradayMomentumStrategy,
)
from state_store import StateStore

STRATEGY_ID = "crypto_intraday_momentum"
STRATEGY_VERSION = "1.0.0"


class IntradayMomentumCryptoTradeEngine(QuantControlledCryptoTradeEngine):
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
        kwargs = {
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "quant_store": quant_store,
            "lifecycle": lifecycle,
            "scanner": scanner,
            "degradation_monitor": intraday_degradation_monitor(),
        }
        if now_fn is not None:
            kwargs["now_fn"] = now_fn
        super().__init__(
            broker,
            data_feed,
            store,
            strategy_config,
            risk_config,
            engine_config,
            **kwargs,
        )
        self.intraday_config = intraday_config or IntradayMomentumConfig()
        self.strategy = IntradayMomentumStrategy(
            strategy_config,
            self.intraday_config,
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
