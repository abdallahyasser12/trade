"""Research-integrity regressions for the generic Quant Desk factory."""

import numpy as np
import pandas as pd
import pytest

from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.research.factory import QuantWalkForwardConfig, run_quant_walk_forward
from quant.research.simulator import SimulationConfig


class NeverTradeStrategy:
    def __init__(self, **_params):
        self.config = type("Cfg", (), {"benchmark_symbol": "TEST", "symbols": ("TEST",)})()

    def required_bars(self):
        return 10

    def prepare_symbol(self, bars):
        out = bars.copy()
        out["atr"] = 1.0
        return out

    def prepare_universe(self, bars_by_symbol):
        return {symbol: self.prepare_symbol(frame) for symbol, frame in bars_by_symbol.items()}

    def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
        row = prepared["TEST"].loc[pd.Timestamp(timestamp)]
        return [
            CryptoSignalDecision(
                symbol="TEST",
                action=BotAction.HOLD,
                timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                reason="never_trade",
                close=float(row["close"]),
                atr=1.0,
            )
        ]

    def ratchet_stop(self, current_stop, highest, close, atr_value):
        return current_stop, max(highest, close)


def bars(periods=520):
    index = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    close = 100.0 * np.cumprod(np.full(periods, 1.0002))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )


def test_all_invalid_training_candidates_fail_closed_instead_of_selecting_zero():
    with pytest.raises(ValueError, match="No candidate satisfied minimum training evidence"):
        run_quant_walk_forward(
            strategy_id="never_trade",
            strategy_factory=lambda params: NeverTradeStrategy(**params),
            candidate_params=[{"x": 1}, {"x": 2}],
            bars_by_symbol={"TEST": bars()},
            sim_config=SimulationConfig(
                slippage_bps=0,
                spread_bps=0,
                fee_bps=0,
                annualization_factor=8760,
            ),
            walk_config=QuantWalkForwardConfig(
                train_bars=150,
                test_bars=80,
                step_bars=80,
                embargo_bars=2,
                min_train_trades=3,
            ),
        )
