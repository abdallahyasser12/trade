"""Adversarial causal and deterministic crypto decision scenarios."""

from crypto_models import CryptoRegime
from crypto_strategy import CryptoRegimeStrategy
from models import BotAction
from tests.crypto_helpers import breakout_bars, compact_crypto_config


def test_clean_regime_breakout_generates_buy():
    bars = breakout_bars()
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    decision = strategy.evaluate({"BTC/USD": bars})[0]
    # Fixture's latest bars move away from the breakout; inspect the exact
    # breakout close to prove the causal event rule.
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.action is BotAction.BUY
    assert decision.regime is CryptoRegime.BULL_TREND
    assert all(decision.filters.values())
    assert decision.stop_price < decision.close


def test_breakout_with_zero_volume_is_hold():
    bars = breakout_bars(zero_volume_at=30)
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.action is BotAction.HOLD
    assert decision.filters["volume"] is False


def test_breakout_with_overbought_rsi_is_hold():
    bars = breakout_bars()
    strategy = CryptoRegimeStrategy(compact_crypto_config(max_rsi=70.0))
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[30], set())[0]
    assert decision.rsi > 70
    assert decision.action is BotAction.HOLD
    assert decision.filters["rsi"] is False


def test_continuously_bullish_breakout_does_not_repeat_buy_for_ten_bars():
    bars = breakout_bars(periods=42)
    for offset, position in enumerate(range(31, 42), start=1):
        bars.iloc[position, bars.columns.get_loc("close")] = 112.5 + offset * 0.5
        bars.iloc[position, bars.columns.get_loc("open")] = 112.5 + (offset - 0.5) * 0.5
        bars.iloc[position, bars.columns.get_loc("high")] = bars.iloc[position]["close"] + 1
        bars.iloc[position, bars.columns.get_loc("low")] = bars.iloc[position]["open"] - 1
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    actions = [
        strategy.evaluate_prepared(prepared, bars.index[position], set())[0].action
        for position in range(30, 41)
    ]
    assert actions[0] is BotAction.BUY
    assert actions[1:] == [BotAction.HOLD] * 10


def test_bearish_fast_trend_break_exits_held_position():
    bars = breakout_bars(gap_at=32)
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[32], {"BTC/USD"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason in {"fast_trend_exit", "benchmark_regime_exit", "volatility_stress_exit"}


def test_future_prices_cannot_change_historical_decision():
    bars = breakout_bars(periods=40)
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    timestamp = bars.index[30]
    prefix = bars.loc[:timestamp]
    prefix_decision = strategy.evaluate_prepared(
        strategy.prepare_universe({"BTC/USD": prefix}), timestamp, set()
    )[0]
    altered = bars.copy()
    altered.loc[altered.index > timestamp, ["open", "high", "low", "close"]] *= 10
    full_decision = strategy.evaluate_prepared(
        strategy.prepare_universe({"BTC/USD": altered}), timestamp, set()
    )[0]
    assert full_decision.to_dict() == prefix_decision.to_dict()


def test_trailing_stop_can_only_tighten_from_completed_close():
    strategy = CryptoRegimeStrategy(compact_crypto_config())
    tighter, high = strategy.ratchet_stop(95.0, 100.0, 110.0, 2.0)
    assert high == 110.0
    assert tighter >= 95.0
    unchanged, _ = strategy.ratchet_stop(tighter, high, 105.0, 10.0)
    assert unchanged == tighter
