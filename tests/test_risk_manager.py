"""Unit tests for risk_manager.RiskManager.

Pure-math tests: no network, no broker, no market data.  Every case uses
hand-computable numbers so the expected share counts and percentages can
be verified in one's head.
"""

from __future__ import annotations

import pytest

from config import RiskConfig
from risk_manager import RiskManager


def make_manager(**overrides) -> RiskManager:
    """RiskManager with the default RiskConfig, selectively overridden."""
    return RiskManager(RiskConfig(**overrides))


# ---------------------------------------------------------------------------
# Constructor / config validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_accepts_default_config(self):
        cfg = RiskConfig()
        manager = RiskManager(cfg)
        assert manager.config is cfg

    @pytest.mark.parametrize(
        "overrides",
        [
            {"risk_per_trade_pct": 0.0},
            {"risk_per_trade_pct": -1.0},
            {"risk_per_trade_pct": 15.0},  # above the 10% conservative cap
            {"max_position_pct": 0.0},
            {"max_position_pct": 150.0},
            {"max_daily_loss_pct": 0.0},
            {"max_daily_loss_pct": -3.0},
            {"min_position_value": -1.0},
            {"min_shares": 0},
        ],
        ids=[
            "risk_pct_zero",
            "risk_pct_negative",
            "risk_pct_above_cap",
            "position_pct_zero",
            "position_pct_above_100",
            "daily_loss_zero",
            "daily_loss_negative",
            "min_value_negative",
            "min_shares_zero",
        ],
    )
    def test_rejects_invalid_config(self, overrides):
        """Any RiskConfig.validate() error must abort construction."""
        with pytest.raises(ValueError, match="Invalid risk configuration"):
            RiskManager(RiskConfig(**overrides))


# ---------------------------------------------------------------------------
# calculate_stop_price
# ---------------------------------------------------------------------------


class TestCalculateStopPrice:
    def test_basic_atr_stop(self):
        manager = make_manager()
        assert manager.calculate_stop_price(100.0, 2.0, 2.0) == pytest.approx(96.0)

    @pytest.mark.parametrize(
        "entry,atr,mult",
        [
            (float("nan"), 2.0, 2.0),
            (100.0, float("inf"), 2.0),
            (100.0, 2.0, float("nan")),
        ],
    )
    def test_rejects_non_finite_stop_inputs(self, entry, atr, mult):
        with pytest.raises(ValueError):
            make_manager().calculate_stop_price(entry, atr, mult)


    def test_position_size_rejects_non_finite_inputs_without_math_exceptions(self):
        manager = make_manager()
        for kwargs in (
            dict(equity=float("nan"), buying_power=1_000, entry_price=10, stop_price=9),
            dict(equity=1_000, buying_power=float("inf"), entry_price=10, stop_price=9),
            dict(equity=1_000, buying_power=1_000, entry_price=float("nan"), stop_price=9),
            dict(equity=1_000, buying_power=1_000, entry_price=10, stop_price=float("nan")),
        ):
            assert manager.calculate_position_size(**kwargs).approved is False

    @pytest.mark.parametrize(
        "entry,atr,mult",
        [
            (0.0, 2.0, 2.0),
            (-100.0, 2.0, 2.0),
            (100.0, 0.0, 2.0),
            (100.0, -2.0, 2.0),
            (100.0, 2.0, 0.0),
            (100.0, 2.0, -1.0),
        ],
        ids=[
            "entry_zero",
            "entry_negative",
            "atr_zero",
            "atr_negative",
            "multiplier_zero",
            "multiplier_negative",
        ],
    )
    def test_rejects_non_positive_inputs(self, entry, atr, mult):
        manager = make_manager()
        with pytest.raises(ValueError):
            manager.calculate_stop_price(entry, atr, mult)

    @pytest.mark.parametrize(
        "entry,atr,mult",
        [
            (10.0, 5.0, 2.0),  # stop exactly 0
            (10.0, 6.0, 2.0),  # stop negative
        ],
        ids=["stop_exactly_zero", "stop_negative"],
    )
    def test_never_returns_non_positive_stop(self, entry, atr, mult):
        """When the ATR distance swallows the entry price the trade must be
        refused via ValueError, never expressed as a stop <= 0."""
        manager = make_manager()
        with pytest.raises(ValueError):
            manager.calculate_stop_price(entry, atr, mult)


# ---------------------------------------------------------------------------
# calculate_position_size — sizing math
# ---------------------------------------------------------------------------


class TestPositionSizeHappyPath:
    def test_exposure_capped_sizing(self):
        """equity 100k / risk 1% / entry 100 / stop 96: risk budget 1000 over
        $4 risk-per-share allows 250 shares, but the 20% exposure cap
        (20k / $100 = 200 shares) binds."""
        manager = make_manager()  # defaults: risk 1%, exposure 20%
        result = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=200_000.0,
            entry_price=100.0,
            stop_price=96.0,
        )
        assert result.shares_by_risk == 250
        assert result.shares_by_exposure == 200
        assert result.shares_by_buying_power == 2000
        assert result.qty == 200
        assert result.position_value == pytest.approx(20_000.0)
        assert result.risk_amount == pytest.approx(800.0)  # 200 * $4
        assert result.rejection_reason is None
        assert result.approved is True

    def test_risk_cap_binds_when_exposure_is_loose(self):
        """With the exposure cap widened to 100% the risk budget (250 shares)
        becomes the binding constraint."""
        manager = make_manager(max_position_pct=100.0)
        result = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=200_000.0,
            entry_price=100.0,
            stop_price=96.0,
        )
        assert result.shares_by_risk == 250
        assert result.shares_by_exposure == 1000
        assert result.qty == 250
        assert result.risk_amount == pytest.approx(1000.0)
        assert result.approved is True

    def test_buying_power_cap_binds_when_small(self):
        """buying_power 5000 at $100/share caps the position at 50 shares even
        though risk and exposure would allow more."""
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=5_000.0,
            entry_price=100.0,
            stop_price=96.0,
        )
        assert result.shares_by_buying_power == 50
        assert result.qty == 50
        assert result.position_value == pytest.approx(5_000.0)
        assert result.approved is True

    def test_fractional_shares_floored_never_rounded_up(self):
        """Each cap is floored individually: risk 1000/$3 = 333.33 -> 333,
        exposure 20_000/$333 = 60.06 -> 60, buying power 5050/$100 -> 50."""
        manager = make_manager()

        by_risk = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=10_000_000.0,
            entry_price=100.0,
            stop_price=97.0,
        )
        assert by_risk.shares_by_risk == 333  # floor(1000 / 3), not 334

        by_exposure = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=10_000_000.0,
            entry_price=333.0,
            stop_price=1.0,
        )
        assert by_exposure.shares_by_exposure == 60  # floor(20_000 / 333)

        by_bp = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=5_050.0,
            entry_price=100.0,
            stop_price=96.0,
        )
        assert by_bp.shares_by_buying_power == 50  # floor(50.5)
        assert by_bp.qty == 50

    def test_qty_is_int(self):
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=100_000.0,
            buying_power=200_000.0,
            entry_price=100.0,
            stop_price=96.0,
        )
        assert isinstance(result.qty, int)


# ---------------------------------------------------------------------------
# calculate_position_size — rejections
# ---------------------------------------------------------------------------


def assert_rejected(result, reason: str) -> None:
    """A rejected sizing result must be inert: qty 0, not approved, reason set."""
    assert result.qty == 0
    assert result.approved is False
    assert result.rejection_reason == reason
    assert result.position_value == 0.0
    assert result.risk_amount == 0.0


class TestPositionSizeRejections:
    @pytest.mark.parametrize("equity", [0.0, -5_000.0], ids=["zero", "negative"])
    def test_non_positive_equity(self, equity):
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=equity, buying_power=200_000.0, entry_price=100.0, stop_price=96.0
        )
        assert_rejected(result, "non_positive_equity")

    @pytest.mark.parametrize(
        "stop", [100.0, 104.0, 0.0, -1.0], ids=["at_entry", "above_entry", "zero", "negative"]
    )
    def test_invalid_stop_price(self, stop):
        """Stops at/above entry (no long risk distance) and non-positive stops
        are all rejected."""
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=100_000.0, buying_power=200_000.0, entry_price=100.0, stop_price=stop
        )
        assert_rejected(result, "invalid_stop_price")

    @pytest.mark.parametrize("buying_power", [0.0, -1_000.0], ids=["zero", "negative"])
    def test_no_buying_power(self, buying_power):
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=100_000.0, buying_power=buying_power, entry_price=100.0, stop_price=96.0
        )
        assert_rejected(result, "no_buying_power")

    def test_zero_quantity_when_risk_budget_too_small(self):
        """equity 1000 at 1% risk gives a $10 budget; a $20 risk-per-share trade
        floors to 0 shares and is rejected as zero_quantity."""
        manager = make_manager()
        result = manager.calculate_position_size(
            equity=1_000.0, buying_power=200_000.0, entry_price=100.0, stop_price=80.0
        )
        assert result.shares_by_risk == 0
        assert_rejected(result, "zero_quantity")

    def test_below_min_position_value(self):
        """5 shares at $10 is a $50 position, under the default $100 minimum."""
        manager = make_manager()  # min_position_value=100.0
        result = manager.calculate_position_size(
            equity=100_000.0, buying_power=50.0, entry_price=10.0, stop_price=9.5
        )
        assert result.shares_by_buying_power == 5
        assert_rejected(result, "position_too_small")

    def test_below_min_shares(self):
        """With min_shares=10 (and no value floor), a 5-share fill is refused
        even though its dollar value passes."""
        manager = make_manager(min_shares=10, min_position_value=0.0)
        result = manager.calculate_position_size(
            equity=100_000.0, buying_power=50.0, entry_price=10.0, stop_price=9.5
        )
        assert result.shares_by_buying_power == 5
        assert_rejected(result, "position_too_small")


# ---------------------------------------------------------------------------
# Session loss and daily loss limit
# ---------------------------------------------------------------------------


class TestSessionLossPct:
    def test_loss_is_positive_pct(self):
        manager = make_manager()
        assert manager.session_loss_pct(100_000.0, 99_000.0) == pytest.approx(1.0)

    def test_gain_is_negative_pct(self):
        """Sign convention: a session gain must come back as a negative
        'loss' percentage, never be clamped to positive."""
        manager = make_manager()
        assert manager.session_loss_pct(100_000.0, 101_000.0) == pytest.approx(-1.0)

    def test_flat_session_is_zero(self):
        manager = make_manager()
        assert manager.session_loss_pct(100_000.0, 100_000.0) == 0.0

    @pytest.mark.parametrize("start", [0.0, -10.0], ids=["zero", "negative"])
    def test_non_positive_starting_equity_is_zero(self, start):
        """A degenerate baseline cannot produce a meaningful percentage; the
        method must return 0.0 rather than divide by zero."""
        manager = make_manager()
        assert manager.session_loss_pct(start, 50_000.0) == 0.0


class TestCheckDailyLossLimit:
    def test_below_limit_false(self):
        manager = make_manager()  # max_daily_loss_pct=3.0
        assert manager.check_daily_loss_limit(100_000.0, 97_500.0) is False  # 2.5%

    def test_exactly_at_limit_true(self):
        """>= semantics: hitting the limit exactly must halt the session."""
        manager = make_manager()  # max_daily_loss_pct=3.0
        assert manager.check_daily_loss_limit(100_000.0, 97_000.0) is True  # 3.0%

    def test_exactly_at_limit_true_binary_exact(self):
        """Same >= check with values exact in binary floating point (25% of
        100k), so the boundary result cannot hinge on rounding luck."""
        manager = make_manager(max_daily_loss_pct=25.0)
        assert manager.check_daily_loss_limit(100_000.0, 75_000.0) is True

    def test_above_limit_true(self):
        manager = make_manager()
        assert manager.check_daily_loss_limit(100_000.0, 96_000.0) is True  # 4.0%

    def test_gain_false(self):
        manager = make_manager()
        assert manager.check_daily_loss_limit(100_000.0, 105_000.0) is False
