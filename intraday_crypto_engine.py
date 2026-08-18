"""Broker-backed adapters for high-velocity intraday crypto hypotheses.

Execution semantics stay inside ``QuantControlledCryptoTradeEngine``. Strategy
versions are isolated and every intraday construction path enforces the strict
high-velocity lifecycle policy. A caller may supply a stricter lifecycle manager, but
cannot weaken sample, expectancy, PF, drawdown or execution-evidence requirements.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from broker import AlpacaBroker
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_data_feed import AlpacaCryptoDataFeed
from quant.intraday_lifecycle import (
    intraday_degradation_monitor,
    intraday_promotion_requirements,
)
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

STRATEGY_ID = ADAPTIVE_STRATEGY_ID
STRATEGY_VERSION = ADAPTIVE_STRATEGY_VERSION


def _assert_not_weaker_lifecycle(lifecycle: LifecycleManager) -> None:
    minimum = intraday_promotion_requirements()
    supplied = lifecycle.requirements
    failures: list[str] = []

    for field in (
        "min_research_trades",
        "min_shadow_trades",
        "min_paper_trades",
        "min_measured_execution_samples",
        "min_canary_trades",
    ):
        if getattr(supplied, field) < getattr(minimum, field):
            failures.append(field)

    for field in (
        "min_shadow_expectancy_r",
        "min_shadow_profit_factor",
        "min_paper_expectancy_r",
        "min_paper_profit_factor",
        "min_canary_expectancy_r",
        "min_canary_profit_factor",
    ):
        if getattr(supplied, field) < getattr(minimum, field):
            failures.append(field)

    for field in (
        "max_shadow_drawdown_r",
        "max_paper_drawdown_r",
        "max_canary_drawdown_r",
        "max_open_operational_defects",
    ):
        if getattr(supplied, field) > getattr(minimum, field):
            failures.append(field)

    if failures:
        raise ValueError(
            "Intraday engine lifecycle is weaker than the production policy: "
            + ", ".join(sorted(set(failures)))
        )


def _strict_intraday_context(
    quant_store: Optional[QuantStore],
    lifecycle: Optional[LifecycleManager],
) -> tuple[QuantStore, LifecycleManager]:
    if lifecycle is not None:
        desk_store = quant_store or lifecycle.store
        if lifecycle.store is not desk_store:
            raise ValueError(
                "Intraday lifecycle and QuantStore must reference the same store instance."
            )
        _assert_not_weaker_lifecycle(lifecycle)
        return desk_store, lifecycle

    desk_store = quant_store or QuantStore()
    strict = LifecycleManager(
        desk_store,
        requirements=intraday_promotion_requirements(),
    )
    return desk_store, strict


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
        desk_store, strict_lifecycle = _strict_intraday_context(
            quant_store,
            lifecycle,
        )
        kwargs = {
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "quant_store": desk_store,
            "lifecycle": strict_lifecycle,
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
