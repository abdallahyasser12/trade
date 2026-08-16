"""The cross-sectional live adapter must inherit every audited safety property.

Swapping a decision layer is the only difference this engine is permitted to
make. If any of these fail, the adapter has quietly forked the execution path
that the rest of the test suite exists to protect.
"""

from datetime import timedelta

import pytest

from config import AccountMode
from cross_sectional import CrossSectionalConfig, CrossSectionalMomentumStrategy
from cross_sectional_engine import (
    CrossSectionalCryptoTradeEngine,
    make_cross_sectional_crypto_engine,
)
from crypto_config import CryptoEngineConfig, CryptoRiskConfig
from crypto_engine import CRYPTO_KILL_SWITCH_KEY, CryptoTradeEngine
from models import AccountInfo
from state_store import StateStore
from tests.crypto_helpers import breakout_bars, compact_crypto_config
from tests.test_crypto_engine import FakeBroker, FakeFeed, asset


def _make(tmp_path, cross_config=None, broker=None):
    frame = breakout_bars(periods=31)
    now = frame.index[-1].to_pydatetime() + timedelta(hours=1, seconds=10)
    broker = broker or FakeBroker()
    store = StateStore(str(tmp_path / "crypto.db"))
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
    )
    return engine, broker, store, now


def test_it_is_the_production_engine_with_only_the_strategy_replaced(tmp_path):
    engine, _, _, _ = _make(tmp_path)
    assert isinstance(engine, CryptoTradeEngine)
    assert isinstance(engine.strategy, CrossSectionalMomentumStrategy)


def test_every_execution_method_is_inherited_not_overridden(tmp_path):
    """The adapter must not fork the audited execution path.

    Anything it redefines is code the rest of the suite is no longer testing,
    so the set of overrides is pinned to exactly the constructor.
    """

    overridden = {
        name
        for name, value in vars(CrossSectionalCryptoTradeEngine).items()
        if callable(value) and not name.startswith("__")
    }
    assert overridden == set(), f"adapter forked execution methods: {overridden}"


def test_an_invalid_configuration_is_refused_at_construction(tmp_path):
    """A bad config must fail before an engine exists, not mid-session."""

    with pytest.raises(ValueError, match="cross-sectional"):
        _make(tmp_path, cross_config=CrossSectionalConfig(top_k=0))
    with pytest.raises(ValueError, match="falling assets"):
        _make(tmp_path, cross_config=CrossSectionalConfig(min_score=-1.0))


def test_it_carries_the_configuration_it_was_given(tmp_path):
    engine, _, _, _ = _make(tmp_path, cross_config=CrossSectionalConfig(top_k=1, lookback=12))
    assert engine.strategy.cross_config.top_k == 1
    assert engine.strategy.cross_config.lookback == 12


def test_the_session_loss_latch_still_arms_the_kill_switch(tmp_path):
    """Inherited, but worth proving: this is the last line before a bad day."""

    engine, broker, store, _ = _make(tmp_path)
    engine._startup()
    engine._starting_equity = 100_000
    broker.account = AccountInfo(97_000, 97_000, 97_000, "ACTIVE", "USD", False, 0, False)
    engine.run_iteration()
    assert engine.risk.session_halted
    assert store.get_kv(CRYPTO_KILL_SWITCH_KEY) is not None
    assert broker.market_calls == []


def test_it_is_hardwired_to_the_same_paper_isolation(tmp_path):
    engine, broker, _, _ = _make(tmp_path)
    assert broker.account_mode is AccountMode.PAPER
    assert engine.engine_config.account_mode is AccountMode.PAPER


def test_the_stop_watchdog_is_the_audited_one(tmp_path):
    """Reached through the adapter, the watchdog must behave identically."""

    engine, _, _, _ = _make(tmp_path)
    assert (
        CrossSectionalCryptoTradeEngine._run_stop_watchdogs
        is CryptoTradeEngine._run_stop_watchdogs
    )
