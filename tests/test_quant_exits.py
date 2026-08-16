"""Exit-policy state machine and stop-ratchet contract tests."""

import pytest

from quant.exits import (
    ExitInstructionKind,
    ExitPolicyConfig,
    ManagedPosition,
    PositionPhase,
    TradeManagementPolicy,
)


def make_position(**overrides):
    base = dict(
        entry_price=100.0,
        initial_stop_price=95.0,
        current_stop_price=95.0,
        highest_close=100.0,
        bars_held=0,
    )
    base.update(overrides)
    return ManagedPosition(**base)


def runner_config(**overrides):
    base = dict(
        enable_runner=True,
        strong_winner_trigger_r=2.0,
        strong_winner_lock_r=1.0,
        runner_trigger_r=4.0,
        runner_fraction=0.25,
        runner_trail_atr_multiplier=5.0,
        trailing_atr_multiplier=3.0,
    )
    base.update(overrides)
    return ExitPolicyConfig(**base)


def test_config_validation_rejects_inverted_triggers():
    bad = ExitPolicyConfig(enable_runner=True, runner_trigger_r=1.0, strong_winner_trigger_r=2.0)
    assert any("Runner trigger" in error for error in bad.validate())


def test_trailing_stop_ratchets_up_and_never_down():
    policy = TradeManagementPolicy(ExitPolicyConfig(trailing_atr_multiplier=3.0))
    position = make_position()
    update = policy.on_completed_bar(position, close=110.0, atr_value=2.0)
    # New stop = 110 - 6 = 104, above the old 95.
    assert update.position.current_stop_price == pytest.approx(104.0)
    kinds = [i.kind for i in update.instructions]
    assert ExitInstructionKind.UPDATE_STOP in kinds
    # A weaker bar afterwards must not lower the stop.
    update2 = policy.on_completed_bar(update.position, close=100.0, atr_value=5.0)
    assert update2.position.current_stop_price == pytest.approx(104.0)
    assert all(i.kind is not ExitInstructionKind.UPDATE_STOP for i in update2.instructions)


def test_break_even_only_after_trigger():
    config = ExitPolicyConfig(
        enable_trailing_stop=False, enable_break_even=True, break_even_trigger_r=1.5
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    # +1R close: no break-even yet.
    update = policy.on_completed_bar(position, close=105.0, atr_value=2.0)
    assert update.position.current_stop_price == pytest.approx(95.0)
    # +1.5R close: stop moves to entry, not beyond.
    update = policy.on_completed_bar(update.position, close=107.5, atr_value=2.0)
    assert update.position.current_stop_price == pytest.approx(100.0)


def test_partial_take_happens_exactly_once():
    config = ExitPolicyConfig(
        enable_partial_take=True, partial_take_r=2.0, partial_take_fraction=0.5,
        enable_trailing_stop=False,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    update = policy.on_completed_bar(position, close=110.0, atr_value=2.0)
    partials = [i for i in update.instructions if i.kind is ExitInstructionKind.PARTIAL_EXIT]
    assert len(partials) == 1
    assert partials[0].exit_fraction == pytest.approx(0.5)
    assert update.position.remaining_fraction == pytest.approx(0.5)
    # A later, even better bar must not take another partial.
    update2 = policy.on_completed_bar(update.position, close=120.0, atr_value=2.0)
    assert not [i for i in update2.instructions if i.kind is ExitInstructionKind.PARTIAL_EXIT]


def test_time_stop_fires_only_when_stagnant():
    config = ExitPolicyConfig(enable_time_stop=True, time_stop_bars=3, time_stop_max_r=0.5)
    policy = TradeManagementPolicy(config)
    position = make_position(bars_held=2)
    # Stagnant at +0.2R after 3 bars: exit.
    update = policy.on_completed_bar(position, close=101.0, atr_value=2.0)
    assert [i.kind for i in update.instructions] == [ExitInstructionKind.FULL_EXIT]
    # Same age but up +2R: keep it.
    winner = make_position(bars_held=2)
    update = policy.on_completed_bar(winner, close=110.0, atr_value=2.0)
    assert all(i.kind is not ExitInstructionKind.FULL_EXIT for i in update.instructions)


def test_runner_progression_and_profit_floor():
    policy = TradeManagementPolicy(runner_config())
    position = make_position()
    # +2R with clean trend: promote to STRONG_WINNER; profit floor at +1R locks.
    update = policy.on_completed_bar(
        position, close=110.0, atr_value=2.0, efficiency_ratio=0.8
    )
    assert update.position.phase is PositionPhase.STRONG_WINNER
    assert update.position.current_stop_price >= 105.0 - 1e-9
    # +4R: promote to RUNNER and reduce to the runner fraction.
    update = policy.on_completed_bar(
        update.position, close=120.0, atr_value=2.0, efficiency_ratio=0.8
    )
    assert update.position.phase is PositionPhase.RUNNER
    partials = [i for i in update.instructions if i.kind is ExitInstructionKind.PARTIAL_EXIT]
    assert len(partials) == 1
    assert partials[0].reason == "runner_reduction"
    assert partials[0].exit_fraction == pytest.approx(0.75)
    assert update.position.remaining_fraction == pytest.approx(0.25)
    # Runner trails with the WIDE multiplier: 120 - 2*5 = 110.
    assert update.position.current_stop_price == pytest.approx(110.0)


def test_poor_trend_quality_blocks_promotion():
    policy = TradeManagementPolicy(runner_config(runner_min_efficiency=0.5))
    position = make_position()
    update = policy.on_completed_bar(
        position, close=110.0, atr_value=2.0, efficiency_ratio=0.1
    )
    assert update.position.phase is PositionPhase.NORMAL


def test_unknown_efficiency_does_not_block_promotion():
    policy = TradeManagementPolicy(runner_config())
    update = TradeManagementPolicy(runner_config()).on_completed_bar(
        make_position(), close=110.0, atr_value=2.0, efficiency_ratio=None
    )
    assert update.position.phase is PositionPhase.STRONG_WINNER
    del policy


def test_invalid_inputs_raise():
    policy = TradeManagementPolicy(ExitPolicyConfig())
    with pytest.raises(ValueError, match="close"):
        policy.on_completed_bar(make_position(), close=float("nan"), atr_value=1.0)
    with pytest.raises(ValueError, match="ATR"):
        policy.on_completed_bar(make_position(), close=100.0, atr_value=0.0)
    with pytest.raises(ValueError, match="Invalid managed position"):
        policy.on_completed_bar(
            make_position(initial_stop_price=101.0), close=100.0, atr_value=1.0
        )
