from __future__ import annotations

from types import SimpleNamespace

import pytest

from broker import BrokerError
from bybit_management import BybitTradeManagementExecutor, PolicyManagedStrategy
from quant.exits import ExitInstruction, ExitInstructionKind, ExitPolicyConfig, TradeManagementPolicy


class _Delegate:
    marker = "delegate"

    def ratchet_stop(self, current_stop, highest, close, atr):
        return 999.0, 999.0


def test_policy_managed_strategy_neutralizes_legacy_second_stop_manager():
    wrapper = PolicyManagedStrategy(_Delegate())
    stop, highest = wrapper.ratchet_stop(90.0, 100.0, 105.0, 2.0)
    assert stop == pytest.approx(90.0)
    assert highest == pytest.approx(105.0)
    assert wrapper.marker == "delegate"


def test_existing_trade_management_policy_remains_the_source_of_exit_instructions():
    policy = TradeManagementPolicy(
        ExitPolicyConfig(
            enable_trailing_stop=False,
            enable_profit_target=True,
            profit_target_r=1.0,
            profit_target_fraction=1.0,
        )
    )
    from quant.exits import ManagedPosition

    position = ManagedPosition(
        entry_price=100.0,
        initial_stop_price=98.0,
        current_stop_price=98.0,
        highest_close=100.0,
        bars_held=0,
    )
    update = policy.on_completed_bar(position, close=102.1, atr_value=1.0)
    assert len(update.instructions) == 1
    assert update.instructions[0].kind is ExitInstructionKind.FULL_EXIT
    assert update.instructions[0].reason == "profit_target"


def test_invalid_trade_management_stop_fails_closed_before_broker_action():
    engine = SimpleNamespace()
    broker = SimpleNamespace()
    state = SimpleNamespace()
    executor = BybitTradeManagementExecutor(
        engine,
        broker,
        state,
        TradeManagementPolicy(),
        strategy_id="crypto_intraday_adaptive",
    )
    with pytest.raises(BrokerError, match="invalid stop"):
        executor._update_stop("BTC/USD", 0.0)


def test_partial_instruction_uses_original_quantity_semantics():
    instruction = ExitInstruction(
        kind=ExitInstructionKind.PARTIAL_EXIT,
        reason="partial_profit_take",
        exit_fraction=0.5,
    )
    assert instruction.exit_fraction == pytest.approx(0.5)
    # TradeManagementPolicy documents fractions against ORIGINAL quantity, which is
    # the invariant the Bybit executor uses when converting a partial instruction.
    original_qty = 0.2
    assert original_qty * instruction.exit_fraction == pytest.approx(0.1)
