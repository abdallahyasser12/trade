"""Configuration: environment loading, typed config objects, validation,
timeframe mapping, and application constants.

This module never performs network requests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

APP_NAME = "Trend Bot (Paper)"
CLIENT_ORDER_PREFIX = "trendbot"
DEFAULT_DB_PATH = str(Path("data") / "trading_bot.db")

# UI-facing timeframe strings.  This is the single mapping layer between the
# UI and Alpaca timeframe objects — do not convert timeframes anywhere else.
TIMEFRAMES: list[str] = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
INTRADAY_TIMEFRAMES = frozenset({"1Min", "5Min", "15Min", "1Hour"})

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1Min": 60,
    "5Min": 5 * 60,
    "15Min": 15 * 60,
    "1Hour": 60 * 60,
    "1Day": 24 * 60 * 60,
}

# Approximate completed regular-session bars per year, used only for Sharpe
# annualization.  252 trading days; 6.5 regular-session hours per day.
_BARS_PER_YEAR: dict[str, float] = {
    "1Min": 252 * 390.0,
    "5Min": 252 * 78.0,
    "15Min": 252 * 26.0,
    "1Hour": 252 * 7.0,  # 6.5h session produces 7 hourly bars (last is short)
    "1Day": 252.0,
}

# Engine polling cadence per timeframe.  There is no reason for a 1Day
# strategy to poll every second.
_POLL_INTERVAL_SECONDS: dict[str, float] = {
    "1Min": 15.0,
    "5Min": 30.0,
    "15Min": 60.0,
    "1Hour": 120.0,
    "1Day": 300.0,
}

SUPPORTED_DATA_FEEDS = ("iex", "sip")

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


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
    """Alpaca paper API credentials loaded from the environment.

    The secret is never logged, persisted, or rendered — only
    ``masked_key`` may appear in the UI.
    """

    api_key: str = ""
    secret_key: str = ""

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

    def __repr__(self) -> str:  # never leak secrets through repr
        return f"AlpacaCredentials(api_key={self.masked_key!r}, secret_key='****')"


def load_credentials(dotenv_path: Optional[str] = None) -> AlpacaCredentials:
    """Load Alpaca paper credentials from .env / environment variables."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return AlpacaCredentials(
        api_key=os.environ.get("ALPACA_API_KEY", "").strip(),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", "").strip(),
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

    def validate(self) -> list[str]:
        errors: list[str] = []
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
        )
        # +1 because crossover detection needs the previous bar's MAs too.
        return longest + 1

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

    def validate(self) -> list[str]:
        errors: list[str] = []
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

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            errors.append("Start and end dates must be valid dates.")
            return errors
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
