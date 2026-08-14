"""Typed, immutable configuration for the long-only Alpaca crypto system."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Optional

from config import AccountMode, TIMEFRAMES, default_poll_interval


DEFAULT_CRYPTO_UNIVERSE = ("BTC/USD", "ETH/USD", "SOL/USD")
CRYPTO_SESSION_SYMBOL = "CRYPTO_PORTFOLIO"
_CRYPTO_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}/USD$")


def validate_crypto_symbol(symbol: str) -> Optional[str]:
    """Return a validation error for a non-USD Alpaca crypto pair."""

    if not isinstance(symbol, str) or not symbol:
        return "Crypto symbol is required."
    if not _CRYPTO_SYMBOL_RE.fullmatch(symbol):
        return (
            f"'{symbol}' is not a valid USD crypto pair; use uppercase BASE/USD "
            "format (for example BTC/USD)."
        )
    if symbol == "USD/USD":
        return "USD/USD is not a tradable crypto pair."
    return None


def crypto_bars_per_year(timeframe: str) -> float:
    """Annualization factor for a market that trades continuously."""

    factors = {
        "1Min": 365.25 * 24 * 60,
        "5Min": 365.25 * 24 * 12,
        "15Min": 365.25 * 24 * 4,
        "1Hour": 365.25 * 24,
        "1Day": 365.25,
    }
    try:
        return factors[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported crypto timeframe: {timeframe!r}") from exc


@dataclass(frozen=True)
class CryptoStrategyConfig:
    """Causal multi-factor regime and breakout decision model."""

    symbols: tuple[str, ...] = DEFAULT_CRYPTO_UNIVERSE
    timeframe: str = "1Hour"
    benchmark_symbol: str = "BTC/USD"
    fast_ma_period: int = 20
    slow_ma_period: int = 100
    regime_ma_period: int = 200
    regime_slope_period: int = 24
    breakout_period: int = 20
    momentum_period: int = 24
    relative_strength_period: int = 72
    rsi_period: int = 14
    min_rsi: float = 50.0
    max_rsi: float = 72.0
    atr_period: int = 14
    atr_stop_multiplier: float = 2.5
    trailing_atr_multiplier: float = 3.5
    atr_percentile_lookback: int = 120
    max_atr_percentile: float = 90.0
    adx_period: int = 14
    min_adx: float = 20.0
    efficiency_period: int = 20
    min_efficiency_ratio: float = 0.25
    volume_period: int = 24
    min_volume_ratio: float = 0.8
    min_relative_strength: float = -0.02
    exit_on_benchmark_regime_break: bool = True
    enable_trailing_stop: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.symbols:
            errors.append("Crypto universe needs at least one symbol.")
        if len(self.symbols) > 12:
            errors.append("Crypto universe is limited to 12 symbols per engine.")
        if len(set(self.symbols)) != len(self.symbols):
            errors.append("Crypto universe cannot contain duplicate symbols.")
        for symbol in self.symbols:
            error = validate_crypto_symbol(symbol)
            if error:
                errors.append(error)
        benchmark_error = validate_crypto_symbol(self.benchmark_symbol)
        if benchmark_error:
            errors.append("Benchmark: " + benchmark_error)
        if self.benchmark_symbol not in self.symbols:
            errors.append("Benchmark symbol must be included in the crypto universe.")
        if self.timeframe not in TIMEFRAMES:
            errors.append(f"Timeframe must be one of {TIMEFRAMES}.")
        integer_minima = {
            "fast MA": (self.fast_ma_period, 2),
            "slow MA": (self.slow_ma_period, 3),
            "regime MA": (self.regime_ma_period, 3),
            "regime slope": (self.regime_slope_period, 1),
            "breakout": (self.breakout_period, 2),
            "momentum": (self.momentum_period, 2),
            "relative strength": (self.relative_strength_period, 2),
            "RSI": (self.rsi_period, 2),
            "ATR": (self.atr_period, 2),
            "ATR percentile lookback": (self.atr_percentile_lookback, 20),
            "ADX": (self.adx_period, 2),
            "efficiency": (self.efficiency_period, 2),
            "volume": (self.volume_period, 1),
        }
        for label, (value, minimum) in integer_minima.items():
            if value < minimum:
                errors.append(f"{label} period must be at least {minimum}.")
        if self.fast_ma_period >= self.slow_ma_period:
            errors.append("Fast MA period must be smaller than slow MA period.")
        if self.slow_ma_period > self.regime_ma_period:
            errors.append("Slow MA period cannot exceed the benchmark regime MA period.")
        finite = {
            "minimum RSI": self.min_rsi,
            "maximum RSI": self.max_rsi,
            "ATR stop multiplier": self.atr_stop_multiplier,
            "trailing ATR multiplier": self.trailing_atr_multiplier,
            "maximum ATR percentile": self.max_atr_percentile,
            "minimum ADX": self.min_adx,
            "minimum efficiency ratio": self.min_efficiency_ratio,
            "minimum volume ratio": self.min_volume_ratio,
            "minimum relative strength": self.min_relative_strength,
        }
        for label, value in finite.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if not (0 <= self.min_rsi < self.max_rsi <= 100):
            errors.append("RSI entry band must satisfy 0 <= minimum < maximum <= 100.")
        if self.atr_stop_multiplier <= 0 or self.trailing_atr_multiplier <= 0:
            errors.append("ATR stop multipliers must be greater than zero.")
        if not (0 < self.max_atr_percentile <= 100):
            errors.append("Maximum ATR percentile must be in (0, 100].")
        if not (0 <= self.min_adx <= 100):
            errors.append("Minimum ADX must be between 0 and 100.")
        if not (0 <= self.min_efficiency_ratio <= 1):
            errors.append("Minimum efficiency ratio must be between 0 and 1.")
        if self.min_volume_ratio <= 0:
            errors.append("Minimum volume ratio must be greater than zero.")
        if not (-1 <= self.min_relative_strength <= 5):
            errors.append("Minimum relative strength must be between -1 and 5.")
        return errors

    def warmup_bars(self) -> int:
        return max(
            self.slow_ma_period + 2,
            self.regime_ma_period + self.regime_slope_period + 2,
            self.breakout_period + 3,
            self.momentum_period + 2,
            self.relative_strength_period + 2,
            self.rsi_period + 2,
            self.atr_period + self.atr_percentile_lookback + 2,
            self.adx_period * 2 + 2,
            self.efficiency_period + 2,
            self.volume_period + 2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        return data


@dataclass(frozen=True)
class CryptoRiskConfig:
    """Portfolio, loss, and fractional-sizing constraints."""

    risk_per_trade_pct: float = 0.35
    max_position_pct: float = 10.0
    max_total_exposure_pct: float = 30.0
    max_session_loss_pct: float = 2.0
    max_positions: int = 3
    min_order_notional: float = 20.0
    max_pairwise_correlation: float = 0.85
    correlation_lookback: int = 120
    min_risk_scale: float = 0.25
    drawdown_risk_reduction_start_pct: float = 0.75
    volatility_target_percentile: float = 50.0
    max_live_order_notional: float = 1_000.0
    max_live_total_exposure_pct: float = 10.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{name} must be finite.")
        if not (0 < self.risk_per_trade_pct <= 2):
            errors.append("Crypto risk per trade must be in (0, 2]%.")
        if not (0 < self.max_position_pct <= 100):
            errors.append("Maximum position exposure must be in (0, 100]%.")
        if not (0 < self.max_total_exposure_pct <= 100):
            errors.append("Maximum portfolio exposure must be in (0, 100]%.")
        if self.max_position_pct > self.max_total_exposure_pct:
            errors.append("Per-position exposure cannot exceed total exposure.")
        if not (0 < self.max_session_loss_pct <= 20):
            errors.append("Maximum session loss must be in (0, 20]%.")
        if not (1 <= self.max_positions <= 12):
            errors.append("Maximum positions must be between 1 and 12.")
        if self.min_order_notional <= 0:
            errors.append("Minimum order notional must be greater than zero.")
        if not (0 <= self.max_pairwise_correlation <= 1):
            errors.append("Maximum pairwise correlation must be between 0 and 1.")
        if self.correlation_lookback < 20:
            errors.append("Correlation lookback must contain at least 20 bars.")
        if not (0 < self.min_risk_scale <= 1):
            errors.append("Minimum risk scale must be in (0, 1].")
        if not (0 <= self.drawdown_risk_reduction_start_pct < self.max_session_loss_pct):
            errors.append("Risk reduction must start before the session-loss halt.")
        if not (0 < self.volatility_target_percentile <= 100):
            errors.append("Volatility target percentile must be in (0, 100].")
        if not (0 < self.max_live_order_notional <= 10_000):
            errors.append("Live order notional cap must be in (0, $10,000].")
        if not (0 < self.max_live_total_exposure_pct <= self.max_total_exposure_pct):
            errors.append("Live total exposure cap must be positive and no larger than paper cap.")
        if self.max_live_total_exposure_pct > 20:
            errors.append("Live total exposure cannot exceed the absolute 20% safety ceiling.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CryptoEngineConfig:
    """Immutable 24/7 engine and execution policy."""

    account_mode: AccountMode = AccountMode.PAPER
    poll_interval_seconds: Optional[float] = None
    max_runtime_seconds: Optional[float] = None
    bar_completion_tolerance_seconds: float = 5.0
    staleness_bars: int = 3
    quote_max_age_seconds: float = 120.0
    max_spread_bps: float = 40.0
    stop_limit_buffer_bps: float = 50.0
    stop_watchdog_grace_seconds: float = 15.0
    entry_order_timeout_seconds: float = 120.0
    fill_poll_attempts: int = 3

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            AccountMode(self.account_mode)
        except ValueError:
            errors.append("Account mode must be paper or live.")
        for label, value in {
            "polling interval": self.poll_interval_seconds,
            "maximum runtime": self.max_runtime_seconds,
        }.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                errors.append(f"{label} must be finite and greater than zero.")
        if self.poll_interval_seconds is not None and self.poll_interval_seconds < 5:
            errors.append("Polling interval must be at least 5 seconds.")
        for label, value in {
            "bar completion tolerance": self.bar_completion_tolerance_seconds,
            "quote maximum age": self.quote_max_age_seconds,
            "maximum spread": self.max_spread_bps,
            "stop-limit buffer": self.stop_limit_buffer_bps,
            "stop watchdog grace": self.stop_watchdog_grace_seconds,
            "entry order timeout": self.entry_order_timeout_seconds,
        }.items():
            if not math.isfinite(value) or value < 0:
                errors.append(f"{label} must be finite and non-negative.")
        if self.staleness_bars < 1:
            errors.append("Staleness window must contain at least one bar.")
        if self.quote_max_age_seconds <= 0:
            errors.append("Quote maximum age must be greater than zero.")
        if not (0 < self.max_spread_bps <= 500):
            errors.append("Maximum spread must be in (0, 500] bps.")
        if not (0 < self.stop_limit_buffer_bps <= 1_000):
            errors.append("Stop-limit buffer must be in (0, 1000] bps.")
        if self.entry_order_timeout_seconds < 30:
            errors.append("Entry order timeout must be at least 30 seconds.")
        if self.fill_poll_attempts < 1 or self.fill_poll_attempts > 20:
            errors.append("Fill poll attempts must be between 1 and 20.")
        return errors

    def resolved_poll_interval(self, timeframe: str) -> float:
        return self.poll_interval_seconds or default_poll_interval(timeframe)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["account_mode"] = AccountMode(self.account_mode).value
        return data


@dataclass(frozen=True)
class CryptoBacktestConfig:
    """Realistic percentage-cost portfolio simulation settings."""

    start_date: date = field(default_factory=lambda: date(2023, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))
    initial_capital: float = 100_000.0
    slippage_bps: float = 10.0
    taker_fee_bps: float = 25.0
    spread_bps: float = 5.0
    risk_free_rate: float = 0.0
    liquidate_at_end: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            return ["Start and end dates must be valid dates."]
        if self.end_date <= self.start_date:
            errors.append("End date must be after start date.")
        for label, value in {
            "initial capital": self.initial_capital,
            "slippage": self.slippage_bps,
            "taker fee": self.taker_fee_bps,
            "spread": self.spread_bps,
            "risk-free rate": self.risk_free_rate,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.initial_capital <= 0:
            errors.append("Initial capital must be greater than zero.")
        if min(self.slippage_bps, self.taker_fee_bps, self.spread_bps) < 0:
            errors.append("Slippage, fee, and spread assumptions cannot be negative.")
        if max(self.slippage_bps, self.taker_fee_bps, self.spread_bps) > 2_000:
            errors.append("Individual cost assumptions cannot exceed 2000 bps.")
        if not (-0.5 <= self.risk_free_rate <= 0.5):
            errors.append("Risk-free rate must be between -50% and 50% as a decimal.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data
