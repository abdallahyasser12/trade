"""Broker-backed Quant Desk runtime must make lifecycle state authoritative."""

from datetime import timedelta

import pytest

from broker import BrokerError
from config import AccountMode
from crypto_config import CryptoEngineConfig, CryptoRiskConfig
from quant.contracts import StrategyState
from quant.production_runtime import QuantControlledCryptoTradeEngine
from quant.store import QuantStore
from state_store import StateStore
from tests.crypto_helpers import breakout_bars, compact_crypto_config
from tests.test_crypto_engine import FakeBroker, FakeFeed


class LiveFakeBroker(FakeBroker):
    account_mode = AccountMode.LIVE


def make_runtime(tmp_path, *, mode=AccountMode.PAPER):
    frame = breakout_bars(periods=31)
    now = frame.index[-1].to_pydatetime() + timedelta(hours=1, seconds=10)
    broker = FakeBroker() if mode is AccountMode.PAPER else LiveFakeBroker()
    state_store = StateStore(str(tmp_path / f"state-{mode.value}.db"))
    quant_store = QuantStore(str(tmp_path / f"quant-{mode.value}.db"))
    feed = FakeFeed({"BTC/USD": frame}, now)
    engine = QuantControlledCryptoTradeEngine(
        broker,
        feed,
        state_store,
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        CryptoEngineConfig(account_mode=mode, poll_interval_seconds=5),
        strategy_id="runtime_test",
        quant_store=quant_store,
        now_fn=lambda: now,
    )
    return engine, quant_store


def test_research_cannot_reach_paper_broker(tmp_path):
    engine, quant_store = make_runtime(tmp_path)
    try:
        with pytest.raises(BrokerError, match="requires lifecycle state PAPER"):
            engine._startup()
    finally:
        quant_store.close()


def test_paper_state_can_start_paper_runtime(tmp_path):
    engine, quant_store = make_runtime(tmp_path)
    try:
        quant_store.set_strategy_state(
            "runtime_test", StrategyState.RESEARCH, StrategyState.PAPER, reason="test"
        )
        engine._startup()
        assert engine._starting_equity == 100_000
    finally:
        quant_store.close()


def test_paper_state_cannot_reach_live_broker(tmp_path):
    engine, quant_store = make_runtime(tmp_path, mode=AccountMode.LIVE)
    try:
        quant_store.set_strategy_state(
            "runtime_test", StrategyState.RESEARCH, StrategyState.PAPER, reason="test"
        )
        with pytest.raises(BrokerError, match="LIVE_CANARY/LIVE"):
            engine._startup()
    finally:
        quant_store.close()


def test_live_canary_can_start_live_runtime(tmp_path):
    engine, quant_store = make_runtime(tmp_path, mode=AccountMode.LIVE)
    try:
        quant_store.set_strategy_state(
            "runtime_test",
            StrategyState.RESEARCH,
            StrategyState.LIVE_CANARY,
            reason="test",
            operator_ack=True,
        )
        engine._startup()
        assert engine.engine_config.account_mode is AccountMode.LIVE
    finally:
        quant_store.close()
