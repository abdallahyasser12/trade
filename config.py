"""Configuration: environment loading, typed config objects, validation,
timeframe mapping, and application constants.

This module never performs network requests.
"""

from __future__ import annotations

import os
import math
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

APP_NAME = "Trend & Crypto Bot"
CLIENT_ORDER_PREFIX = "trendbot"
DEFAULT_DB_PATH = str(Path("data") / "trading_bot.db")

# UI-facing timeframe strings.  This is the single mapping layer between the
# UI and Alpaca timeframe objects — do not convert timeframes anywhere else.
TIMEFRAMES: list[str] = ["1Min", "5Min", "15Min", "1Hour", "4Hour", "1Day"]
INTRADAY_TIMEFRAMES = frozenset({"1Min", "5Min", "15Min", "1Hour", "4Hour"})

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1Min": 60,
    "5Min": 5 * 60,
    "15Min": 15 * 60,
    "1Hour": 60 * 60,
    "4Hour": 4 * 60 * 60,
    "1Day": 24 * 60 * 60,
}

# Approximate completed regular-session bars per year, used only for Sharpe
# annualization.  252 trading days; 6.5 regular-session hours per day.
_BARS_PER_YEAR: dict[str, float] = {
    "1Min": 252 * 390.0,
    "5Min": 252 * 78.0,
    "15Min": 252 * 26.0,
    "1Hour": 252 * 7.0,  # 6.5h session produces 7 hourly bars (last is short)
    "4Hour": 252 * 2.0,  # 6.5h session produces 2 four-hour bars (last is short)
    "1Day": 252.0,
}

# Engine polling cadence per timeframe.  There is no reason for a 1Day
# strategy to poll every second.
_POLL_INTERVAL_SECONDS: dict[str, float] = {
    "1Min": 15.0,
    "5Min": 30.0,
    "15Min": 60.0,
    "1Hour": 120.0,
    "4Hour": 180.0,
    "1Day": 300.0,
}

SUPPORTED_DATA_FEEDS = ("iex", "sip")

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class AccountMode(str, Enum):
    """Alpaca trading environment selected for a frozen engine session."""

    PAPER = "paper"
    LIVE = "live"


def _environment_flag(name: str, default: bool = False) -> bool:
    """Parse a strict boolean environment flag.

    Only conventional explicit true values enable a dangerous capability;
    typos and arbitrary non-empty strings remain disabled.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def live_trading_enabled(dotenv_path: Optional[str] = None) -> bool:
    """Whether this process is allowed to expose the live-account unlock."""

    load_dotenv(dotenv_path=dotenv_path, override=False)
    return _environment_flag("ENABLE_LIVE_TRADING", default=False)


def timeframe_to_alpaca(timeframe: str):
    """Map a UI timeframe string to an alpaca-py TimeFrame object."""
    # Imported lazily so config.py stays importable without alpaca-py in
    # pure-analysis contexts (tests import this module heavily).
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. Supported: {TIMEFRAMES}")
    return mapping[timeframe]


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Duration of one bar for the given timeframe."""
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. Supported: {TIMEFRAMES}")
    return timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])


def bars_per_year(timeframe: str) -> float:
    """Approximate completed bars per year for Sharpe annualization."""
    if timeframe not in _BARS_PER_YEAR:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. Supported: {TIMEFRAMES}")
    return _BARS_PER_YEAR[timeframe]


def default_poll_interval(timeframe: str) -> float:
    """Reasonable engine polling interval for the given timeframe."""
    if timeframe not in _POLL_INTERVAL_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. Supported: {TIMEFRAMES}")
    return _POLL_INTERVAL_SECONDS[timeframe]


def validate_symbol(symbol: str) -> Optional[str]:
    """Return an error message for an invalid symbol string, else None.

    Conservative: uppercase letters, digits, ``.`` and ``-``, max 10 chars,
    must start with a letter.  Whether the symbol actually exists and is
    tradable is verified against Alpaca's asset endpoint before a live
    session starts.
    """
    if not symbol or not isinstance(symbol, str):
        return "Symbol is required."
    if not _SYMBOL_RE.match(symbol):
        return (
            f"'{symbol}' is not a valid symbol. Use 1-10 uppercase letters/digits "
            "(e.g. AAPL, SPY, BRK.B)."
        )
    return None


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass
class AlpacaCredentials:
    """Alpaca credentials for exactly one account environment.

    The secret is never logged, persisted, or rendered — only
    ``masked_key`` may appear in the UI.
    """

    api_key: str = ""
    secret_key: str = ""
    account_mode: AccountMode = AccountMode.PAPER

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)

    @property
    def masked_key(self) -> str:
        if not self.api_key:
            return "(not set)"
        if len(self.api_key) <= 4:
            return "****"
        return f"{self.api_key[:4]}…{'*' * 4}"

    @property
    def fingerprint(self) -> str:
        """Non-reversible cache key that changes when either credential changes."""
        material = (
            f"{self.account_mode.value}\0{self.api_key}\0{self.secret_key}"
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    def __repr__(self) -> str:  # never leak secrets through repr
        return (
            "AlpacaCredentials("
            f"account_mode={self.account_mode.value!r}, "
            f"api_key={self.masked_key!r}, secret_key='****')"
        )


def load_credentials(
    dotenv_path: Optional[str] = None,
    account_mode: AccountMode | str = AccountMode.PAPER,
) -> AlpacaCredentials:
    """Load credentials for one explicit Alpaca environment.

    Paper mode prefers the namespaced paper variables and retains the legacy
    variables for backwards compatibility. Live mode *only* reads the separate
    live variables; paper credentials can never be silently promoted.
    """

    load_dotenv(dotenv_path=dotenv_path, override=False)
    mode = AccountMode(account_mode)
    if mode is AccountMode.LIVE:
        key_name = "ALPACA_LIVE_API_KEY"
        secret_name = "ALPACA_LIVE_SECRET_KEY"
        api_key = os.environ.get(key_name, "").strip()
        secret_key = os.environ.get(secret_name, "").strip()
    else:
        api_key = (
            os.environ.get("ALPACA_PAPER_API_KEY", "").strip()
            or os.environ.get("ALPACA_API_KEY", "").strip()
        )
        secret_key = (
            os.environ.get("ALPACA_PAPER_SECRET_KEY", "").strip()
            or os.environ.get("ALPACA_SECRET_KEY", "").strip()
        )
    return AlpacaCredentials(
        api_key=api_key,
        secret_key=secret_key,
        account_mode=mode,
    )


# ---------------------------------------------------------------------------
# Typed configuration objects
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    """Parameters of the trend-following strategy (shared by backtest & live)."""

    symbol: str = "SPY"
    timeframe: str = "1Day"
    short_ma_period: int = 20
    long_ma_period: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    atr_multiplier: float = 2.0
    volume_period: int = 20
    min_volume_ratio: float = 1.0
    enable_volatility_filter: bool = True
    atr_percentile_lookback: int = 100
    max_atr_percentile: float = 90.0
    # Optional production-quality filters.  They default off at the typed
    # configuration layer for backwards compatibility; the Streamlit UI's
    # "Robust trend" preset enables them explicitly.
    enable_rsi_floor: bool = False
    rsi_entry_floor: float = 45.0
    enable_adx_filter: bool = False
    adx_period: int = 14
    min_adx: float = 20.0
    enable_efficiency_filter: bool = False
    efficiency_period: int = 20
    min_efficiency_ratio: float = 0.20
    enable_market_regime_filter: bool = False
    regime_symbol: str = "SPY"
    regime_timeframe: str = "1Day"
    regime_ma_period: int = 200
    regime_slope_period: int = 20
    exit_on_regime_break: bool = True
    enable_relative_strength_filter: bool = False
    relative_strength_period: int = 63
    min_relative_strength: float = 0.0
    enable_trailing_stop: bool = False
    trailing_atr_multiplier: float = 3.0
    enable_break_even_stop: bool = False
    break_even_trigger_r: float = 1.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        finite_fields = {
            "RSI overbought": self.rsi_overbought,
            "RSI oversold": self.rsi_oversold,
            "ATR multiplier": self.atr_multiplier,
            "minimum volume ratio": self.min_volume_ratio,
            "maximum ATR percentile": self.max_atr_percentile,
            "RSI entry floor": self.rsi_entry_floor,
            "minimum ADX": self.min_adx,
            "minimum efficiency ratio": self.min_efficiency_ratio,
            "minimum relative strength": self.min_relative_strength,
            "trailing ATR multiplier": self.trailing_atr_multiplier,
            "break-even trigger": self.break_even_trigger_r,
        }
        for label, value in finite_fields.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        symbol_error = validate_symbol(self.symbol)
        if symbol_error:
            errors.append(symbol_error)
        if self.timeframe not in TIMEFRAMES:
            errors.append(f"Timeframe must be one of {TIMEFRAMES}.")
        if self.short_ma_period < 2:
            errors.append("Short MA period must be at least 2.")
        if self.long_ma_period < 2:
            errors.append("Long MA period must be at least 2.")
        if self.short_ma_period >= self.long_ma_period:
            errors.append("Short MA period must be smaller than long MA period.")
        if self.rsi_period < 2:
            errors.append("RSI period must be at least 2.")
        if not (0.0 < self.rsi_overbought <= 100.0):
            errors.append("RSI overbought threshold must be between 0 and 100.")
        if not (0.0 <= self.rsi_oversold < 100.0):
            errors.append("RSI oversold threshold must be between 0 and 100.")
        if self.rsi_oversold >= self.rsi_overbought:
            errors.append("RSI oversold threshold must be below the overbought threshold.")
        if self.enable_rsi_floor:
            if not (0.0 <= self.rsi_entry_floor < 100.0):
                errors.append("RSI entry floor must be between 0 and 100.")
            if self.rsi_entry_floor >= self.rsi_overbought:
                errors.append("RSI entry floor must be below the overbought threshold.")
        if self.atr_period < 2:
            errors.append("ATR period must be at least 2.")
        if self.atr_multiplier <= 0:
            errors.append("ATR stop multiplier must be greater than 0.")
        if self.volume_period < 1:
            errors.append("Volume average period must be at least 1.")
        if self.min_volume_ratio <= 0:
            errors.append("Volume confirmation multiplier must be greater than 0.")
        if self.enable_volatility_filter:
            if self.atr_percentile_lookback < 10:
                errors.append("ATR percentile lookback must be at least 10 bars.")
            if not (0.0 < self.max_atr_percentile <= 100.0):
                errors.append("Maximum ATR percentile must be between 0 and 100.")
        if self.adx_period < 2:
            errors.append("ADX period must be at least 2.")
        if self.enable_adx_filter:
            if not (0.0 <= self.min_adx <= 100.0):
                errors.append("Minimum ADX must be between 0 and 100.")
        if self.efficiency_period < 2:
            errors.append("Efficiency-ratio period must be at least 2.")
        if self.enable_efficiency_filter:
            if not (0.0 <= self.min_efficiency_ratio <= 1.0):
                errors.append("Minimum efficiency ratio must be between 0 and 1.")
        regime_symbol_error = validate_symbol(self.regime_symbol)
        if self.enable_market_regime_filter or self.enable_relative_strength_filter:
            if regime_symbol_error:
                errors.append("Regime symbol: " + regime_symbol_error)
            if self.regime_timeframe not in TIMEFRAMES:
                errors.append(f"Regime timeframe must be one of {TIMEFRAMES}.")
        if self.enable_market_regime_filter:
            if self.regime_ma_period < 2:
                errors.append("Regime MA period must be at least 2.")
            if self.regime_slope_period < 1:
                errors.append("Regime slope period must be at least 1.")
        if self.enable_relative_strength_filter:
            if self.relative_strength_period < 2:
                errors.append("Relative-strength period must be at least 2.")
            if self.regime_timeframe != self.timeframe:
                errors.append(
                    "Relative strength requires the regime and strategy timeframes to match."
                )
            if not (-1.0 <= self.min_relative_strength <= 10.0):
                errors.append("Minimum relative strength must be between -1 and 10.")
        if self.enable_trailing_stop and self.trailing_atr_multiplier <= 0:
            errors.append("Trailing ATR multiplier must be greater than 0.")
        if self.enable_break_even_stop and self.break_even_trigger_r <= 0:
            errors.append("Break-even trigger must be greater than 0R.")
        return errors

    def warmup_bars(self) -> int:
        """Bars required before every indicator is fully initialized.

        The longest dependency wins: long MA, RSI, ATR, volume average and
        the ATR percentile lookback (which itself sits on top of the ATR
        warmup).  A small safety pad is added for robustness.
        """
        atr_chain = self.atr_period + (
            self.atr_percentile_lookback if self.enable_volatility_filter else 0
        )
        longest = max(
            self.long_ma_period,
            self.rsi_period + 1,  # RSI needs one extra bar for the first diff
            self.atr_period + 1,  # TR needs the previous close
            self.volume_period,
            atr_chain,
            (self.adx_period * 2 + 1) if self.enable_adx_filter else 0,
            (self.efficiency_period + 1) if self.enable_efficiency_filter else 0,
            (self.relative_strength_period + 1)
            if self.enable_relative_strength_filter
            else 0,
            self.regime_warmup_bars()
            if (
                (self.enable_market_regime_filter or self.enable_relative_strength_filter)
                and self.regime_symbol == self.symbol
                and self.regime_timeframe == self.timeframe
            )
            else 0,
        )
        # +1 because crossover detection needs the previous bar's MAs too.
        return longest + 1

    def regime_warmup_bars(self) -> int:
        """Completed benchmark bars required by market-context filters."""
        if not (self.enable_market_regime_filter or self.enable_relative_strength_filter):
            return 0
        return max(
            self.regime_ma_period + self.regime_slope_period + 1
            if self.enable_market_regime_filter
            else 0,
            self.relative_strength_period + 1
            if self.enable_relative_strength_filter
            else 0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskConfig:
    """Risk limits (shared by backtest & live)."""

    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0
    max_daily_loss_pct: float = 3.0
    min_position_value: float = 100.0
    min_shares: int = 1
    # On by default and matching the dashboard default: drawdown throttling
    # and volatility scaling apply unless an operator deliberately disables
    # them. Programmatic configs and UI configs now agree.
    enable_adaptive_risk: bool = True
    min_risk_scale: float = 0.25
    drawdown_risk_reduction_start_pct: float = 1.0
    volatility_target_percentile: float = 50.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in {
            "Risk per trade": self.risk_per_trade_pct,
            "Maximum position size": self.max_position_pct,
            "Maximum session loss": self.max_daily_loss_pct,
            "Minimum position value": self.min_position_value,
            "Minimum adaptive risk scale": self.min_risk_scale,
            "Drawdown risk-reduction threshold": self.drawdown_risk_reduction_start_pct,
            "Volatility target percentile": self.volatility_target_percentile,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.risk_per_trade_pct <= 0:
            errors.append("Risk per trade must be greater than 0%.")
        if self.risk_per_trade_pct > 10:
            errors.append("Risk per trade above 10% is not allowed — this bot is conservative.")
        if self.max_position_pct <= 0:
            errors.append("Maximum position size must be greater than 0% of equity.")
        if self.max_position_pct > 100:
            errors.append("Maximum position size cannot exceed 100% of equity.")
        if self.max_daily_loss_pct <= 0:
            errors.append("Maximum session loss must be greater than 0%.")
        if self.max_daily_loss_pct > 100:
            errors.append("Maximum session loss cannot exceed 100%.")
        if self.min_position_value < 0:
            errors.append("Minimum position value cannot be negative.")
        if self.min_shares < 1:
            errors.append("Minimum shares must be at least 1 (whole shares only).")
        if not (0.0 < self.min_risk_scale <= 1.0):
            errors.append("Minimum adaptive risk scale must be in (0, 1].")
        if self.enable_adaptive_risk:
            if self.drawdown_risk_reduction_start_pct < 0:
                errors.append("Drawdown risk-reduction threshold cannot be negative.")
            if self.drawdown_risk_reduction_start_pct >= self.max_daily_loss_pct:
                errors.append(
                    "Drawdown risk-reduction threshold must be below the maximum session loss."
                )
            if not (0.0 < self.volatility_target_percentile <= 100.0):
                errors.append("Volatility target percentile must be between 0 and 100.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineConfig:
    """Live engine behavior.  ``poll_interval_seconds=None`` derives the
    cadence from the timeframe via ``default_poll_interval``."""

    poll_interval_seconds: Optional[float] = None
    max_runtime_seconds: Optional[float] = None
    data_feed: str = "iex"
    block_entries_near_close: bool = True
    close_buffer_minutes: int = 30
    flat_before_close: bool = False
    flat_before_close_minutes: int = 10
    bar_completion_tolerance_seconds: float = 5.0
    clock_skew_threshold_seconds: float = 30.0
    staleness_bars: int = 3

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in {
            "Polling interval": self.poll_interval_seconds,
            "Run duration": self.max_runtime_seconds,
        }.items():
            if value is not None and not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        for label, value in {
            "Bar completion tolerance": self.bar_completion_tolerance_seconds,
            "Clock skew threshold": self.clock_skew_threshold_seconds,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.poll_interval_seconds is not None and self.poll_interval_seconds < 5:
            errors.append("Polling interval must be at least 5 seconds.")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            errors.append("Run duration must be greater than 0.")
        if self.data_feed not in SUPPORTED_DATA_FEEDS:
            errors.append(f"Data feed must be one of {SUPPORTED_DATA_FEEDS}.")
        if self.close_buffer_minutes < 0:
            errors.append("Entry block buffer before close cannot be negative.")
        if self.flat_before_close_minutes < 1:
            errors.append("Flat-before-close buffer must be at least 1 minute.")
        if self.bar_completion_tolerance_seconds < 0:
            errors.append("Bar completion tolerance cannot be negative.")
        if self.clock_skew_threshold_seconds <= 0:
            errors.append("Clock skew threshold must be greater than 0 seconds.")
        if self.staleness_bars < 1:
            errors.append("Staleness window must be at least 1 bar.")
        return errors

    def resolved_poll_interval(self, timeframe: str) -> float:
        if self.poll_interval_seconds is not None:
            return self.poll_interval_seconds
        return default_poll_interval(timeframe)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestConfig:
    """Historical simulation settings."""

    start_date: date = field(default_factory=lambda: date(2022, 1, 1))
    end_date: date = field(default_factory=lambda: date(2023, 12, 31))
    initial_capital: float = 100_000.0
    slippage_bps: float = 0.0
    commission_per_trade: float = 0.0
    risk_free_rate: float = 0.0
    adjustment: str = "split"
    liquidate_at_end: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            errors.append("Start and end dates must be valid dates.")
            return errors
        for label, value in {
            "Starting capital": self.initial_capital,
            "Slippage": self.slippage_bps,
            "Commission": self.commission_per_trade,
            "Risk-free rate": self.risk_free_rate,
        }.items():
            if not math.isfinite(value):
                errors.append(f"{label} must be finite.")
        if self.end_date <= self.start_date:
            errors.append("End date must be after start date.")
        if self.initial_capital <= 0:
            errors.append("Starting capital must be greater than 0.")
        if self.slippage_bps < 0:
            errors.append("Slippage cannot be negative.")
        if self.commission_per_trade < 0:
            errors.append("Commission cannot be negative.")
        if not (-0.5 <= self.risk_free_rate <= 0.5):
            errors.append("Risk-free rate must be between -50% and 50% (as a decimal).")
        if self.adjustment not in ("raw", "split", "dividend", "all"):
            errors.append("Bar adjustment must be one of raw/split/dividend/all.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d


def result_context_fingerprint(
    strategy: StrategyConfig,
    risk: RiskConfig,
    backtest: BacktestConfig,
    *,
    data_feed: str,
    experiment: Optional[dict[str, Any]] = None,
) -> str:
    """Return a non-secret identity for a displayed simulation result.

    Streamlit retains session state across reruns. Binding a result to this
    fingerprint prevents an old backtest or walk-forward report from being
    shown under newly edited controls. Credentials are intentionally absent;
    callers must not place secrets in ``experiment``.
    """
    payload = {
        "strategy": strategy.to_dict(),
        "risk": risk.to_dict(),
        "backtest": backtest.to_dict(),
        "data_feed": data_feed,
        "experiment": experiment or {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
