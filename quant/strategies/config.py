"""Typed configuration for market-agnostic quant-desk strategies.

``CryptoStrategyConfig`` hard-validates BASE/USD pairs, which is correct
for the crypto engine but wrong for stocks.  The intraday strategy family
therefore gets its own frozen config carrying the same structural fields
the ``UniverseStrategy`` protocol consumers rely on (``symbols``,
``timeframe``, ``benchmark_symbol``, ``warmup_bars()``) with per-market
symbol validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from config import TIMEFRAMES, validate_symbol
from crypto_config import validate_crypto_symbol
from quant.contracts import Market
from quant.features import SESSION_ANCHOR_US_EQUITY, SESSION_ANCHOR_UTC


def validate_market_symbol(symbol: str, market: Market) -> str | None:
    """Market-aware symbol validation, delegating to the existing rules."""
    if market is Market.CRYPTO:
        return validate_crypto_symbol(symbol)
    return validate_symbol(symbol)


def default_session_anchor(market: Market) -> str:
    return SESSION_ANCHOR_US_EQUITY if market is Market.US_EQUITY else SESSION_ANCHOR_UTC


@dataclass(frozen=True)
class IntradayStrategyConfig:
    """Shared fields for the intraday/systematic strategy family.

    Subclasses add family-specific parameters and extend ``validate``.
    ``benchmark_symbol`` must be a member of ``symbols`` so relative
    measurements never require data outside the configured universe.
    """

    symbols: tuple[str, ...]
    market: Market
    timeframe: str = "15Min"
    benchmark_symbol: str = ""
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    trailing_atr_multiplier: float = 3.0
    enable_trailing_stop: bool = True
    volume_period: int = 20
    min_volume_ratio: float = 1.2
    atr_percentile_lookback: int = 100
    max_atr_percentile: float = 95.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.symbols:
            errors.append("Universe needs at least one symbol.")
        if len(self.symbols) > 64:
            errors.append("Universe is limited to 64 symbols per strategy instance.")
        if len(set(self.symbols)) != len(self.symbols):
            errors.append("Universe cannot contain duplicate symbols.")
        try:
            market = Market(self.market)
        except ValueError:
            errors.append("Market must be us_equity or crypto.")
            return errors
        for symbol in self.symbols:
            error = validate_market_symbol(symbol, market)
            if error:
                errors.append(error)
        benchmark = self.benchmark_symbol or (self.symbols[0] if self.symbols else "")
        if benchmark and benchmark not in self.symbols:
            errors.append("Benchmark symbol must be included in the universe.")
        if self.timeframe not in TIMEFRAMES:
            errors.append(f"Timeframe must be one of {TIMEFRAMES}.")
        if self.atr_period < 2:
            errors.append("ATR period must be at least 2.")
        for label, value in {
            "ATR stop multiplier": self.atr_stop_multiplier,
            "trailing ATR multiplier": self.trailing_atr_multiplier,
            "minimum volume ratio": self.min_volume_ratio,
            "maximum ATR percentile": self.max_atr_percentile,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and greater than zero.")
        if self.volume_period < 1:
            errors.append("Volume period must be at least 1.")
        if self.atr_percentile_lookback < 20:
            errors.append("ATR percentile lookback must be at least 20 bars.")
        if not (0 < self.max_atr_percentile <= 100):
            errors.append("Maximum ATR percentile must be in (0, 100].")
        return errors

    @property
    def resolved_benchmark(self) -> str:
        return self.benchmark_symbol or self.symbols[0]

    @property
    def session_anchor(self) -> str:
        return default_session_anchor(Market(self.market))

    def warmup_bars(self) -> int:
        return max(
            self.atr_period + self.atr_percentile_lookback + 2,
            self.volume_period + 2,
            30,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        data["market"] = Market(self.market).value
        return data
