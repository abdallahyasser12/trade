"""Cross-sectional adapter safety and Quant Desk lifecycle integration tests."""

from datetime import timedelta

import pytest

from config import AccountMode
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from cross_sectional_engine import (
    STRATEGY_ID,
    CrossSectionalCryptoTradeEngine,
    make_cross_sectional_crypto_engine,
)
from crypto_config import CryptoEngineConfig, CryptoRiskConfig
from crypto_engine import CRYPTO_KILL_SWITCH_KEY, CryptoTradeEngine
from models import AccountInfo
from quant.contracts import StrategyState
from quant.production_runtime import QuantControlledCryptoTradeEngine
from quant.store import QuantStore
from state_store import StateStore
from tests.crypto_helpers import breakout_bars, compact_crypto_config
from tests.test_crypto_engine import FakeBroker, FakeFeed


def _make(tmp_path, cross_config=None, broker=None, lifecycle_state=StrategyState.PAPER):
    frame = breakout_bars(periods=31)
    now = frame.index[-1].to_pydatetime() + timedelta(hours=1, seconds=10)
    broker = broker or FakeBroker()
    store = StateStore(str(tmp_path / "crypto.db"))
    quant_store = QuantStore(str(tmp_path / "quant.db"))
    if lifecycle_state is not StrategyState.RESEARCH:
        quant_store.set_strategy_state(
            STRATEGY_ID, StrategyState.RESEARCH, lifecycle_state, reason="test"
        )
    feed = FakeFeed({"BTC/USD": frame}, now)
    engine = make_cross_sectional_crypto_engine(
        broker,
        feed,
        store,
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        CryptoEngineConfig(poll_interval_seconds=5),
        cross_config=cross_config,
        now_fn=lambda: now,
        quant_store=quant_store,
    )
    return engine, broker, store, quant_store, now


def test_it_uses_audited_execution_engine_plus_quant_gate(tmp_path):
    engine, _, _, _, _ = _make(tmp_path)
    assert isinstance(engine, CryptoTradeEngine)
    assert isinstance(engine, QuantControlledCryptoTradeEngine)
    assert isinstance(engine.strategy, CrossSectionalMomentumStrategy)


def test_adapter_itself_does_not_fork_execution_methods(tmp_path):
    overridden = {
        name
        for name, value in vars(CrossSectionalCryptoTradeEngine).items()
        if callable(value) and not name.startswith("__")
    }
    assert overridden == set(), f"adapter forked execution methods: {overridden}"


def test_an_invalid_configuration_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="cross-sectional"):
        _make(tmp_path, cross_config=CrossSectionalConfig(top_k=0))
    with pytest.raises(ValueError, match="falling assets"):
        _make(tmp_path, cross_config=CrossSectionalConfig(min_score=-1.0))


def test_it_carries_the_configuration_it_was_given(tmp_path):
    engine, _, _, _, _ = _make(
        tmp_path, cross_config=CrossSectionalConfig(top_k=1, lookback=12)
    )
    assert engine.strategy.cross_config.top_k == 1
    assert engine.strategy.cross_config.lookback == 12


def test_research_state_cannot_start_broker_backed_adapter(tmp_path):
    engine, _, _, quant_store, _ = _make(
        tmp_path, lifecycle_state=StrategyState.RESEARCH
    )
    try:
        with pytest.raises(Exception, match="requires lifecycle state PAPER"):
            engine._startup()
    finally:
        quant_store.close()


def test_the_session_loss_latch_still_arms_the_kill_switch(tmp_path):
    engine, broker, store, _, _ = _make(tmp_path)
    engine._startup()
    engine._starting_equity = 100_000
    broker.account = AccountInfo(97_000, 97_000, 97_000, "ACTIVE", "USD", False, 0, False)
    engine.run_iteration()
    assert engine.risk.session_halted
    assert store.get_kv(CRYPTO_KILL_SWITCH_KEY) is not None
    assert broker.market_calls == []


def test_it_is_hardwired_to_the_same_paper_isolation(tmp_path):
    engine, broker, _, _, _ = _make(tmp_path)
    assert broker.account_mode is AccountMode.PAPER
    assert engine.engine_config.account_mode is AccountMode.PAPER


def test_stop_watchdog_remains_the_audited_parent_implementation(tmp_path):
    engine, _, _, _, _ = _make(tmp_path)
    assert (
        CrossSectionalCryptoTradeEngine._run_stop_watchdogs
        is CryptoTradeEngine._run_stop_watchdogs
    )
