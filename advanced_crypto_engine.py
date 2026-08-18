"""Live/paper engine adapter for the regime-aware ensemble.

Broker execution/protection still comes from the audited crypto engine. Quant Desk
is now authoritative for lifecycle, measured-alpha and pre-trade portfolio gates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from quant.lifecycle import LifecycleManager
from quant.production_runtime import QuantControlledCryptoTradeEngine
from quant.scanner import QuantScanner
from quant.store import QuantStore
from regime_ensemble import EnsembleConfig, RegimeAwareEnsembleStrategy
from state_store import StateStore


STRATEGY_ID = "crypto_regime_ensemble"


class RegimeAwareCryptoTradeEngine(QuantControlledCryptoTradeEngine):
    """Regime ensemble using the production execution path plus Quant Desk gate."""

    def __init__(
        self,
        broker: AlpacaBroker,
        data_feed: AlpacaCryptoDataFeed,
        store: StateStore,
        strategy_config: CryptoStrategyConfig,
        risk_config: CryptoRiskConfig,
        engine_config: CryptoEngineConfig,
        *,
        ensemble_config: Optional[EnsembleConfig] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[QuantScanner] = None,
    ):
        kwargs = {
            "strategy_id": STRATEGY_ID,
            "quant_store": quant_store,
            "lifecycle": lifecycle,
            "scanner": scanner,
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
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.strategy = RegimeAwareEnsembleStrategy(strategy_config, self.ensemble_config)


def make_regime_aware_crypto_engine(
    broker: AlpacaBroker,
    data_feed: AlpacaCryptoDataFeed,
    store: StateStore,
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    engine_config: CryptoEngineConfig,
    *,
    ensemble_config: Optional[EnsembleConfig] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    quant_store: Optional[QuantStore] = None,
    lifecycle: Optional[LifecycleManager] = None,
    scanner: Optional[QuantScanner] = None,
) -> RegimeAwareCryptoTradeEngine:
    return RegimeAwareCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        ensemble_config=ensemble_config,
        now_fn=now_fn,
        quant_store=quant_store,
        lifecycle=lifecycle,
        scanner=scanner,
    )
