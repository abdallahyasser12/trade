"""Live/paper engine adapter for the cross-sectional momentum strategy.

Deliberately identical in shape to ``advanced_crypto_engine`` — subclass the
production engine, replace ``self.strategy``, change nothing else. The parent
retains broker verification, account-mode isolation, session-loss latching,
quote/spread validation, correlation gates, fractional sizing, deterministic
client order IDs, partial-fill reconciliation, protective stop-limit
management, the stop watchdog, and crash recovery.

A note on why this exists and what it does not mean.

Being selectable is not being cleared. This engine has run through the
unmodified promotion gates once (experiment 8) and FAILED them: it beat every
prior configuration on expectancy — profit factor 1.48, Sharpe 0.52, positive
under stress — and still lost decisively to buy-and-hold because risk-based
sizing keeps the book roughly 82-95% in cash. Wiring it makes the strategy
reachable so that paper testing can happen; it does not assert the strategy is
profitable, and no gate was changed to permit it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from crypto_engine import CryptoTradeEngine
from state_store import StateStore


class CrossSectionalCryptoTradeEngine(CryptoTradeEngine):
    """Production execution engine with the cross-sectional decision layer."""

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
        quant_journal=None,
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
            quant_journal=quant_journal,
            **kwargs,
        )
        self.cross_config = cross_config or CrossSectionalConfig()
        # Validated here rather than deep inside a trading loop: a bad config
        # must fail at construction, before the engine can place anything.
        errors = self.cross_config.validate()
        if errors:
            raise ValueError(
                "Invalid cross-sectional configuration: " + "; ".join(errors)
            )
        self.strategy = CrossSectionalMomentumStrategy(
            strategy_config,
            self.cross_config,
        )


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
    quant_journal=None,
) -> CrossSectionalCryptoTradeEngine:
    """Explicit factory used by deployments that opt into cross-sectional."""

    return CrossSectionalCryptoTradeEngine(
        broker,
        data_feed,
        store,
        strategy_config,
        risk_config,
        engine_config,
        cross_config=cross_config,
        now_fn=now_fn,
        quant_journal=quant_journal,
    )
