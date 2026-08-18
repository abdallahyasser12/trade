"""Live/paper adapter for cross-sectional momentum.

The strategy still uses the audited ``CryptoTradeEngine`` order/protection path,
but broker-backed entries now pass through ``QuantControlledCryptoTradeEngine``:
research/shadow states cannot reach a broker, PAPER requires measured shadow
expectancy, and LIVE requires lifecycle evidence/canary state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from quant.lifecycle import LifecycleManager
from quant.production_runtime import QuantControlledCryptoTradeEngine
from quant.scanner import QuantScanner
from quant.store import QuantStore
from state_store import StateStore


STRATEGY_ID = "crypto_cross_sectional"


class CrossSectionalCryptoTradeEngine(QuantControlledCryptoTradeEngine):
    """Production execution engine with cross-sectional decisions + Quant Desk gate."""

    def __init__(
        self,
        broker: AlpacaBroker,
        data_feed: AlpacaCryptoDataFeed,
        store: StateStore,
        strategy_config: CryptoStrategyConfig,
        risk_config: CryptoRiskConfig,
        engine_config: CryptoEngineConfig,
        *,
        cross_config: Optional[CrossSectionalConfig] = None,
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
        self.cross_config = cross_config or CrossSectionalConfig()
        errors = self.cross_config.validate()
        if errors:
            raise ValueError("Invalid cross-sectional configuration: " + "; ".join(errors))
        self.strategy = CrossSectionalMomentumStrategy(strategy_config, self.cross_config)


def make_cross_sectional_crypto_engine(
    broker: AlpacaBroker,
    data_feed: AlpacaCryptoDataFeed,
    store: StateStore,
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    engine_config: CryptoEngineConfig,
    *,
    cross_config: Optional[CrossSectionalConfig] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    quant_store: Optional[QuantStore] = None,
    lifecycle: Optional[LifecycleManager] = None,
    scanner: Optional[QuantScanner] = None,
) -> CrossSectionalCryptoTradeEngine:
    return CrossSectionalCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        cross_config=cross_config,
        now_fn=now_fn,
        quant_store=quant_store,
        lifecycle=lifecycle,
        scanner=scanner,
    )
