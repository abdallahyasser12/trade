"""High-velocity intraday profile.

The requested ~2%/day objective is recorded here as an *aspirational measurement*,
never as a parameter-selection or promotion condition. Optimizing repeatedly until
a backtest prints 2%/day is textbook multiple-testing/overfit; this project instead
optimizes for robust net expectancy and separately reports whether the aspirational
return happened on untouched OOS data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from config import AccountMode
from crypto_config import (
    CryptoBacktestConfig,
    CryptoEngineConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)

# Verified as active/tradable Alpaca USD crypto pairs on 2026-08-18. The broker
# still verifies every symbol at startup because exchange listings can change.
HIGH_VELOCITY_CRYPTO_UNIVERSE = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
    "ADA/USD",
    "UNI/USD",
    "AAVE/USD",
)


@dataclass(frozen=True)
class IntradayObjective:
    aspirational_daily_return_pct: float = 2.0
    target_trades_per_day_min: float = 6.0
    target_trades_per_day_max: float = 40.0
    # A high-turnover strategy that needs more than this in one session is not
    # allowed to chase the return target by increasing risk.
    max_session_loss_pct: float = 2.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.aspirational_daily_return_pct):
            errors.append("Daily-return objective must be finite.")
        if self.target_trades_per_day_min <= 0:
            errors.append("Minimum daily trade target must be positive.")
        if self.target_trades_per_day_max < self.target_trades_per_day_min:
            errors.append("Maximum daily trade target must be >= minimum target.")
        if not (0 < self.max_session_loss_pct <= 5):
            errors.append("High-velocity session-loss ceiling must be in (0, 5]%.")
        return errors


DEFAULT_INTRADAY_OBJECTIVE = IntradayObjective()

# Scalp objective: more, shorter trades — but the same absolute session-loss
# ceiling. Frequency may never buy itself extra risk budget.
SCALP_INTRADAY_OBJECTIVE = IntradayObjective(
    aspirational_daily_return_pct=2.0,
    target_trades_per_day_min=15.0,
    target_trades_per_day_max=80.0,
    max_session_loss_pct=2.0,
)


def make_high_velocity_strategy_config(
    *,
    symbols: tuple[str, ...] = HIGH_VELOCITY_CRYPTO_UNIVERSE,
    timeframe: str = "5Min",
) -> CryptoStrategyConfig:
    """Short-bar config used by the new intraday strategy/research stack."""
    return CryptoStrategyConfig(
        symbols=symbols,
        timeframe=timeframe,
        benchmark_symbol="BTC/USD",
        fast_ma_period=12,
        slow_ma_period=48,
        regime_ma_period=96,
        regime_slope_period=12,
        breakout_period=12,
        momentum_period=12,
        relative_strength_period=36,
        rsi_period=14,
        atr_period=14,
        atr_stop_multiplier=1.8,
        trailing_atr_multiplier=2.2,
        atr_percentile_lookback=120,
        max_atr_percentile=95.0,
        adx_period=14,
        min_adx=15.0,
        efficiency_period=12,
        min_efficiency_ratio=0.15,
        volume_period=24,
        min_volume_ratio=0.7,
        min_relative_strength=-0.05,
    )


def make_high_velocity_risk_config() -> CryptoRiskConfig:
    """Frequency comes from the universe/timeframe, not oversized risk."""
    return CryptoRiskConfig(
        risk_per_trade_pct=0.15,
        max_position_pct=7.5,
        max_total_exposure_pct=30.0,
        max_session_loss_pct=DEFAULT_INTRADAY_OBJECTIVE.max_session_loss_pct,
        max_positions=6,
        min_order_notional=20.0,
        max_pairwise_correlation=0.80,
        correlation_lookback=120,
        min_risk_scale=0.25,
        drawdown_risk_reduction_start_pct=0.75,
        volatility_target_percentile=50.0,
        max_live_order_notional=1_000.0,
        max_live_total_exposure_pct=10.0,
        max_cost_to_risk=0.35,
        max_portfolio_heat_pct=0.90,
    )


def make_scalp_strategy_config(
    *,
    symbols: tuple[str, ...] = HIGH_VELOCITY_CRYPTO_UNIVERSE,
    timeframe: str = "1Min",
) -> CryptoStrategyConfig:
    """One-minute scalp scaffolding: the same gates, horizons rescaled.

    This is a RESEARCH hypothesis config for the scalp sleeve (many very short
    trades targeting small profits, exits from quant/exits.scalp_exit_policy).
    It deploys through exactly the same chronological research -> OOS ->
    lifecycle evidence path as every other configuration; nothing here reaches
    a broker by existing.
    """
    return CryptoStrategyConfig(
        symbols=symbols,
        timeframe=timeframe,
        benchmark_symbol="BTC/USD",
        fast_ma_period=9,
        slow_ma_period=30,
        regime_ma_period=90,
        regime_slope_period=10,
        breakout_period=15,
        momentum_period=10,
        relative_strength_period=30,
        rsi_period=14,
        atr_period=14,
        atr_stop_multiplier=1.5,
        trailing_atr_multiplier=1.8,
        atr_percentile_lookback=120,
        max_atr_percentile=95.0,
        adx_period=14,
        min_adx=12.0,
        efficiency_period=10,
        min_efficiency_ratio=0.12,
        volume_period=30,
        min_volume_ratio=0.8,
        min_relative_strength=-0.05,
    )


def make_scalp_risk_config() -> CryptoRiskConfig:
    """Scalp risk: smaller unit risk and a stricter cost gate, same ceilings.

    A scalp's edge per trade is small, so execution cost may consume at most
    30% of the risked amount (vs 35% for the 5-minute profile) and each trade
    risks less equity. The session-loss ceiling, exposure caps, correlation
    and heat limits are identical — higher frequency never relaxes them.
    """
    return CryptoRiskConfig(
        risk_per_trade_pct=0.10,
        max_position_pct=7.5,
        max_total_exposure_pct=30.0,
        max_session_loss_pct=SCALP_INTRADAY_OBJECTIVE.max_session_loss_pct,
        max_positions=6,
        min_order_notional=20.0,
        max_pairwise_correlation=0.80,
        correlation_lookback=120,
        min_risk_scale=0.25,
        drawdown_risk_reduction_start_pct=0.75,
        volatility_target_percentile=50.0,
        max_live_order_notional=1_000.0,
        max_live_total_exposure_pct=10.0,
        max_cost_to_risk=0.30,
        max_portfolio_heat_pct=0.90,
    )


def make_high_velocity_engine_config(
    *, account_mode: AccountMode = AccountMode.PAPER
) -> CryptoEngineConfig:
    return CryptoEngineConfig(
        account_mode=account_mode,
        poll_interval_seconds=5.0,
        bar_completion_tolerance_seconds=2.0,
        staleness_bars=2,
        quote_max_age_seconds=8.0,
        max_spread_bps=30.0,
        stop_limit_buffer_bps=35.0,
        stop_watchdog_grace_seconds=3.0,
        stop_watchdog_min_confirmations=2,
        position_open_poll_interval_seconds=1.0,
        taker_fee_bps=25.0,
        expected_slippage_bps=8.0,
        entry_order_timeout_seconds=45.0,
        fill_poll_attempts=5,
    )


def make_high_velocity_backtest_config(
    *,
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
) -> CryptoBacktestConfig:
    return CryptoBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        slippage_bps=8.0,
        taker_fee_bps=25.0,
        spread_bps=10.0,
    )


@dataclass(frozen=True)
class IntradayEvaluation:
    trades: int
    observed_days: float
    trades_per_day: float
    total_return_pct: float
    geometric_daily_return_pct: float
    activity_target_met: bool
    aspirational_return_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "observed_days": self.observed_days,
            "trades_per_day": self.trades_per_day,
            "total_return_pct": self.total_return_pct,
            "geometric_daily_return_pct": self.geometric_daily_return_pct,
            "activity_target_met": self.activity_target_met,
            "aspirational_return_met": self.aspirational_return_met,
        }


def evaluate_intraday_result(result, objective: IntradayObjective = DEFAULT_INTRADAY_OBJECTIVE):
    """Measure speed and return without converting either into an overfit gate."""
    errors = objective.validate()
    if errors:
        raise ValueError("Invalid intraday objective: " + "; ".join(errors))
    curve = result.equity_curve
    if len(curve) < 2:
        raise ValueError("Intraday evaluation requires at least two equity observations.")
    elapsed_seconds = (curve.index[-1] - curve.index[0]).total_seconds()
    observed_days = max(elapsed_seconds / 86_400.0, 1.0 / 288.0)
    trades_per_day = result.num_trades / observed_days
    growth = max(0.0, 1.0 + result.total_return_pct / 100.0)
    geometric_daily = (
        (growth ** (1.0 / observed_days) - 1.0) * 100.0 if growth > 0 else -100.0
    )
    return IntradayEvaluation(
        trades=result.num_trades,
        observed_days=observed_days,
        trades_per_day=trades_per_day,
        total_return_pct=result.total_return_pct,
        geometric_daily_return_pct=geometric_daily,
        activity_target_met=(
            objective.target_trades_per_day_min
            <= trades_per_day
            <= objective.target_trades_per_day_max
        ),
        aspirational_return_met=geometric_daily >= objective.aspirational_daily_return_pct,
    )
