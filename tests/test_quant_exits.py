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


def test_profit_target_full_exit_and_only_once():
    config = ExitPolicyConfig(
        enable_profit_target=True,
        profit_target_r=1.5,
        profit_target_fraction=1.0,
        trailing_atr_multiplier=3.0,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    # +1.0R: below target, nothing happens.
    update = policy.on_completed_bar(position, close=105.0, atr_value=2.0)
    assert all(i.kind is not ExitInstructionKind.FULL_EXIT for i in update.instructions)
    # +1.6R: full exit at target, evaluation ends.
    update2 = policy.on_completed_bar(update.position, close=108.0, atr_value=2.0)
    assert [i.kind for i in update2.instructions] == [ExitInstructionKind.FULL_EXIT]
    assert update2.instructions[0].reason == "profit_target"
    assert update2.position.target_taken


def test_profit_target_partial_banks_once_and_leaves_runner_path_open():
    config = runner_config(
        enable_profit_target=True,
        profit_target_r=1.2,
        profit_target_fraction=0.6,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    update = policy.on_completed_bar(position, close=107.0, atr_value=2.0)
    partials = [i for i in update.instructions if i.kind is ExitInstructionKind.PARTIAL_EXIT]
    assert len(partials) == 1
    assert partials[0].reason == "profit_target"
    assert partials[0].exit_fraction == pytest.approx(0.6)
    assert update.position.remaining_fraction == pytest.approx(0.4)
    assert update.position.target_taken
    # The target never fires twice.
    update2 = policy.on_completed_bar(update.position, close=108.0, atr_value=2.0)
    assert all(i.reason != "profit_target" for i in update2.instructions)


def test_profit_target_is_skipped_for_promoted_runner():
    config = runner_config(
        enable_profit_target=True,
        profit_target_r=1.0,
        profit_target_fraction=1.0,
    )
    policy = TradeManagementPolicy(config)
    position = make_position(
        phase=PositionPhase.RUNNER,
        remaining_fraction=0.25,
        highest_close=125.0,
        current_stop_price=105.0,
        bars_held=10,
    )
    update = policy.on_completed_bar(position, close=126.0, atr_value=2.0)
    assert all(i.reason != "profit_target" for i in update.instructions)


def test_smart_exit_requires_known_collapsed_hold_score():
    config = ExitPolicyConfig(
        enable_smart_exit=True,
        smart_exit_hold_score=35.0,
        smart_exit_min_r=0.0,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    # No hold score: the flag alone must be inert.
    update = policy.on_completed_bar(position, close=104.0, atr_value=2.0, hold_score=None)
    assert all(i.kind is not ExitInstructionKind.FULL_EXIT for i in update.instructions)
    # Healthy hold score: no exit.
    update2 = policy.on_completed_bar(position, close=104.0, atr_value=2.0, hold_score=70.0)
    assert all(i.kind is not ExitInstructionKind.FULL_EXIT for i in update2.instructions)
    # Collapsed hold score in profit: full exit.
    update3 = policy.on_completed_bar(position, close=104.0, atr_value=2.0, hold_score=20.0)
    assert [i.kind for i in update3.instructions] == [ExitInstructionKind.FULL_EXIT]
    assert update3.instructions[0].reason == "edge_decay_exit"


def test_smart_exit_respects_r_floor():
    config = ExitPolicyConfig(
        enable_smart_exit=True,
        smart_exit_hold_score=35.0,
        smart_exit_min_r=0.5,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    # +0.2R with a collapsed hold score: below the floor, stop stays in charge.
    update = policy.on_completed_bar(position, close=101.0, atr_value=2.0, hold_score=10.0)
    assert all(i.kind is not ExitInstructionKind.FULL_EXIT for i in update.instructions)


def test_adaptive_trail_tightens_only_on_weak_hold():
    config = ExitPolicyConfig(
        trailing_atr_multiplier=3.0,
        enable_adaptive_trail=True,
        tightened_trail_atr_multiplier=1.5,
        tighten_hold_score=55.0,
    )
    policy = TradeManagementPolicy(config)
    position = make_position()
    strong = policy.on_completed_bar(position, close=110.0, atr_value=2.0, hold_score=80.0)
    assert strong.position.current_stop_price == pytest.approx(110.0 - 3.0 * 2.0)
    weak = policy.on_completed_bar(position, close=110.0, atr_value=2.0, hold_score=40.0)
    assert weak.position.current_stop_price == pytest.approx(110.0 - 1.5 * 2.0)
    # Without a hold score the flag is inert.
    unknown = policy.on_completed_bar(position, close=110.0, atr_value=2.0)
    assert unknown.position.current_stop_price == pytest.approx(104.0)


def test_adaptive_trail_never_tightens_a_runner():
    config = runner_config(
        enable_adaptive_trail=True,
        tightened_trail_atr_multiplier=1.5,
        tighten_hold_score=55.0,
    )
    policy = TradeManagementPolicy(config)
    position = make_position(
        phase=PositionPhase.RUNNER,
        remaining_fraction=0.25,
        highest_close=130.0,
        current_stop_price=105.0,
        bars_held=10,
    )
    update = policy.on_completed_bar(position, close=140.0, atr_value=2.0, hold_score=10.0)
    # Runner trail = 140 - 5*2 = 130, not the tightened 140 - 3.
    assert update.position.current_stop_price == pytest.approx(130.0)


def test_new_flags_default_off_and_hold_score_alone_changes_nothing():
    policy = TradeManagementPolicy(ExitPolicyConfig())
    position = make_position()
    plain = policy.on_completed_bar(position, close=110.0, atr_value=2.0)
    with_hold = policy.on_completed_bar(position, close=110.0, atr_value=2.0, hold_score=1.0)
    assert plain.position == with_hold.position
    assert [ (i.kind, i.reason) for i in plain.instructions ] == [
        (i.kind, i.reason) for i in with_hold.instructions
    ]


def test_scalp_preset_validates_and_new_config_bounds_are_enforced():
    from quant.exits import scalp_exit_policy

    TradeManagementPolicy(scalp_exit_policy())  # must not raise
    assert any(
        "Profit-target fraction" in error
        for error in ExitPolicyConfig(profit_target_fraction=0.0).validate()
    )
    assert any(
        "Smart-exit hold score" in error
        for error in ExitPolicyConfig(smart_exit_hold_score=150.0).validate()
    )
    assert any(
        "Tightened trail" in error
        for error in ExitPolicyConfig(
            trailing_atr_multiplier=2.0, tightened_trail_atr_multiplier=3.0
        ).validate()
    )
