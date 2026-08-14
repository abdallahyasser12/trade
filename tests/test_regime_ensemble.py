from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import crypto_backtest
from advanced_crypto_backtest import run_regime_ensemble_backtest
from advanced_crypto_engine import RegimeAwareCryptoTradeEngine
from crypto_config import (
    CryptoBacktestConfig,
    CryptoEngineConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)
from crypto_models import CryptoSignalDecision
from models import BotAction
from regime_ensemble import (
    EnsembleConfig,
    MarketRegime,
    RegimeAwareEnsembleStrategy,
)


def _strategy_config() -> CryptoStrategyConfig:
    return CryptoStrategyConfig(
        symbols=("BTC/USD", "ETH/USD"),
        benchmark_symbol="BTC/USD",
    )


def _prepared(*, atr_percentile: float = 40.0) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    frames = {}
    values = {
        "BTC/USD": (109.0, 110.0, 109.7, 100.0, 90.0, 2.0),
        "ETH/USD": (49.5, 50.0, 49.8, 46.0, 42.0, 1.0),
    }
    for symbol, (previous, close, fast, slow, regime_ma, atr_value) in values.items():
        frames[symbol] = pd.DataFrame(
            {
                "close": [previous, close],
                "fast_ma": [fast - 0.1, fast],
                "slow_ma": [slow, slow],
                "regime_ma": [regime_ma, regime_ma],
                "regime_slope": [0.01, 0.01],
                "rsi": [52.0, 54.0],
                "atr": [atr_value, atr_value],
                "atr_percentile": [atr_percentile, atr_percentile],
                "adx": [30.0, 30.0],
                "efficiency_ratio": [0.40, 0.40],
                "volume_ratio": [1.2, 1.3],
                "momentum": [0.03, 0.04],
                "relative_strength": [0.0, 0.01],
                "prior_breakout_high": [close + 5.0, close + 5.0],
                "benchmark_bull": [True, True],
            },
            index=index,
        )
    return frames


def _decision(symbol: str, action: BotAction) -> CryptoSignalDecision:
    return CryptoSignalDecision(
        symbol=symbol,
        action=action,
        timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        reason="base",
        score=0.8,
        close=100.0,
        atr=2.0,
        filters={"new_breakout": action is BotAction.BUY, "trend": True},
    )


def test_stress_regime_blocks_all_new_entries():
    strategy = RegimeAwareEnsembleStrategy(
        _strategy_config(),
        EnsembleConfig(min_buy_score=0.1, min_regime_confidence=0.0),
    )
    strategy.base.evaluate_prepared = lambda *_args, **_kwargs: [
        _decision("BTC/USD", BotAction.BUY),
        _decision("ETH/USD", BotAction.BUY),
    ]
    prepared = _prepared(atr_percentile=99.0)
    decisions = strategy.evaluate_prepared(prepared, prepared["BTC/USD"].index[-1])
    assert all(decision.action is BotAction.HOLD for decision in decisions)
    assert all(decision.reason == "ensemble_regime_blocks_entry" for decision in decisions)
    assert {
        decision.details["ensemble"]["regime"]["regime"] for decision in decisions
    } == {MarketRegime.STRESS.value}


def test_audited_exit_is_never_suppressed_by_ensemble():
    strategy = RegimeAwareEnsembleStrategy(_strategy_config())
    strategy.base.evaluate_prepared = lambda *_args, **_kwargs: [
        _decision("BTC/USD", BotAction.EXIT),
        _decision("ETH/USD", BotAction.HOLD),
    ]
    prepared = _prepared()
    decisions = strategy.evaluate_prepared(
        prepared,
        prepared["BTC/USD"].index[-1],
        {"BTC/USD"},
    )
    btc = next(decision for decision in decisions if decision.symbol == "BTC/USD")
    assert btc.action is BotAction.EXIT
    assert btc.reason == "base"


def test_pullback_sleeve_can_create_entry_in_healthy_bull_regime():
    strategy = RegimeAwareEnsembleStrategy(
        _strategy_config(),
        EnsembleConfig(min_buy_score=0.18, min_regime_confidence=0.20),
    )
    strategy.base.evaluate_prepared = lambda *_args, **_kwargs: [
        _decision("BTC/USD", BotAction.HOLD),
        _decision("ETH/USD", BotAction.HOLD),
    ]
    prepared = _prepared()
    decisions = strategy.evaluate_prepared(prepared, prepared["BTC/USD"].index[-1])
    buys = [decision for decision in decisions if decision.action is BotAction.BUY]
    assert buys
    assert all("pullback" in decision.reason for decision in buys)
    assert all(decision.stop_price < decision.close for decision in buys)


def test_equal_score_candidates_have_stable_symbol_tiebreak():
    strategy = RegimeAwareEnsembleStrategy(
        _strategy_config(),
        EnsembleConfig(min_buy_score=0.18, min_regime_confidence=0.20),
    )
    strategy.base.evaluate_prepared = lambda *_args, **_kwargs: [
        _decision("ETH/USD", BotAction.HOLD),
        _decision("BTC/USD", BotAction.HOLD),
    ]
    prepared = _prepared()
    prepared["ETH/USD"] = prepared["BTC/USD"].copy()
    decisions = strategy.evaluate_prepared(prepared, prepared["BTC/USD"].index[-1])
    buys = [decision.symbol for decision in decisions if decision.action is BotAction.BUY]
    assert buys == sorted(buys)


def test_backtest_adapter_uses_private_strategy_binding(monkeypatch):
    classic = crypto_backtest.CryptoRegimeStrategy
    fake_globals = {"CryptoRegimeStrategy": classic}
    exec(
        "def fake_backtest(bars_by_symbol, strategy_config, risk_config, "
        "backtest_config, *, assets=None, active_start=None):\n"
        "    return CryptoRegimeStrategy(strategy_config).__class__.__name__\n",
        fake_globals,
    )
    monkeypatch.setattr(
        crypto_backtest,
        "run_crypto_backtest",
        fake_globals["fake_backtest"],
    )
    result = run_regime_ensemble_backtest(
        {},
        _strategy_config(),
        CryptoRiskConfig(),
        CryptoBacktestConfig(),
    )
    assert result == "RegimeAwareEnsembleStrategy"
    assert crypto_backtest.CryptoRegimeStrategy is classic


class _DummyFeed:
    def set_interrupt_event(self, _event):
        self.event_set = True


def test_engine_adapter_reuses_production_execution_engine():
    engine = RegimeAwareCryptoTradeEngine(
        broker=object(),
        data_feed=_DummyFeed(),
        store=object(),
        strategy_config=_strategy_config(),
        risk_config=CryptoRiskConfig(),
        engine_config=CryptoEngineConfig(),
    )
    assert isinstance(engine.strategy, RegimeAwareEnsembleStrategy)
    assert engine.strategy_config.symbols == ("BTC/USD", "ETH/USD")
