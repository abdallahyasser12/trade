"""Live/paper engine adapter for the regime-aware ensemble."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from crypto_engine import CryptoTradeEngine
from regime_ensemble import EnsembleConfig, RegimeAwareEnsembleStrategy
from state_store import StateStore


class RegimeAwareCryptoTradeEngine(CryptoTradeEngine):
    """Use the production execution engine with the ensemble decision layer.

    The parent class still owns broker verification, account-mode isolation,
    session-loss latching, quote/spread validation, correlation gates,
    fractional sizing, deterministic client IDs, partial-fill reconciliation,
    protective stop-limit management, watchdog exits, and crash recovery.
    """

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
    ):
        kwargs = {}
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
        self.strategy = RegimeAwareEnsembleStrategy(
            strategy_config,
            self.ensemble_config,
        )


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
) -> RegimeAwareCryptoTradeEngine:
    """Explicit factory used by deployments that opt into ensemble-v2."""

    return RegimeAwareCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        ensemble_config=ensemble_config,
        now_fn=now_fn,
    )
