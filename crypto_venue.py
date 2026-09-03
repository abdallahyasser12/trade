"""Venue-neutral contracts and canonical crypto instrument mapping.

The legacy strategy surface intentionally keeps its historical ``BASE/USD`` symbols.
Venue adapters translate that compatibility symbol exactly once at the boundary; no
strategy, Quant Desk component, or risk model may concatenate/extract venue symbols.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import pandas as pd

from crypto_models import CryptoQuote
from models import AccountInfo, AssetInfo, BrokerPosition, OrderRecord


class CryptoVenue(str, Enum):
    ALPACA = "alpaca"
    BYBIT = "bybit"


class CryptoExecutionEnvironment(str, Enum):
    ALPACA_PAPER = "ALPACA_PAPER"
    ALPACA_LIVE = "ALPACA_LIVE"
    BYBIT_DEMO = "BYBIT_DEMO"
    BYBIT_LIVE = "BYBIT_LIVE"

    @property
    def is_real_money(self) -> bool:
        return self in {self.ALPACA_LIVE, self.BYBIT_LIVE}


@dataclass(frozen=True)
class CanonicalInstrument:
    base_asset: str
    quote_asset: str
    canonical_symbol: str
    venue_symbol: str
    venue: CryptoVenue
    # Backwards-compatible symbol consumed by the existing strategy/risk stack.
    strategy_symbol: str

    def __post_init__(self) -> None:
        for name in (
            "base_asset",
            "quote_asset",
            "canonical_symbol",
            "venue_symbol",
            "strategy_symbol",
        ):
            value = getattr(self, name)
            if not value or value != value.upper():
                raise ValueError(f"{name} must be non-empty uppercase text")


@dataclass(frozen=True)
class InstrumentMetadata:
    instrument: CanonicalInstrument
    tick_size: float
    quantity_step: float
    min_order_qty: float
    min_order_amount: float
    max_order_qty: float
    max_market_order_qty: float
    fetched_at_epoch_ms: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in (
            "tick_size",
            "quantity_step",
            "min_order_qty",
            "min_order_amount",
            "max_order_qty",
            "max_market_order_qty",
        ):
            if float(getattr(self, name)) <= 0:
                errors.append(f"{name} must be positive")
        if self.min_order_qty > self.max_order_qty:
            errors.append("min_order_qty exceeds max_order_qty")
        return errors

    def as_asset_info(self) -> AssetInfo:
        """Compatibility shape used by the existing crypto risk engine."""
        return AssetInfo(
            symbol=self.instrument.strategy_symbol,
            name=self.instrument.canonical_symbol,
            tradable=True,
            active=True,
            fractionable=True,
            asset_class="crypto",
            min_order_size=self.min_order_qty,
            min_trade_increment=self.quantity_step,
            price_increment=self.tick_size,
            marginable=False,
            shortable=False,
        )


_BYBIT_DEFAULTS: Mapping[str, CanonicalInstrument] = {
    "BTC/USD": CanonicalInstrument(
        "BTC", "USDT", "BTC/USDT", "BTCUSDT", CryptoVenue.BYBIT, "BTC/USD"
    ),
    "ETH/USD": CanonicalInstrument(
        "ETH", "USDT", "ETH/USDT", "ETHUSDT", CryptoVenue.BYBIT, "ETH/USD"
    ),
    "SOL/USD": CanonicalInstrument(
        "SOL", "USDT", "SOL/USDT", "SOLUSDT", CryptoVenue.BYBIT, "SOL/USD"
    ),
}

DEFAULT_BYBIT_SPOT_UNIVERSE = tuple(_BYBIT_DEFAULTS)


def bybit_instrument(symbol: str) -> CanonicalInstrument:
    token = str(symbol).upper().strip()
    if token in _BYBIT_DEFAULTS:
        return _BYBIT_DEFAULTS[token]
    for instrument in _BYBIT_DEFAULTS.values():
        if token in {instrument.canonical_symbol, instrument.venue_symbol}:
            return instrument
    raise ValueError(
        f"Unsupported Bybit Spot instrument {symbol!r}. Configure the permitted universe explicitly."
    )


def build_bybit_instrument(base_asset: str, quote_asset: str = "USDT") -> CanonicalInstrument:
    """Build an explicitly permitted instrument without string replacements elsewhere."""
    base = base_asset.upper().strip()
    quote = quote_asset.upper().strip()
    if not base.isalnum() or not quote.isalnum() or base == quote:
        raise ValueError("Invalid base/quote assets")
    strategy = f"{base}/USD" if quote == "USDT" else f"{base}/{quote}"
    return CanonicalInstrument(
        base_asset=base,
        quote_asset=quote,
        canonical_symbol=f"{base}/{quote}",
        venue_symbol=f"{base}{quote}",
        venue=CryptoVenue.BYBIT,
        strategy_symbol=strategy,
    )


@runtime_checkable
class InstrumentMetadataProvider(Protocol):
    def get_instrument_metadata(
        self, symbol: str, *, force_refresh: bool = False
    ) -> InstrumentMetadata: ...


@runtime_checkable
class CryptoBroker(Protocol):
    """Minimum broker contract required by broker-backed crypto engines.

    The method names intentionally match the established engine surface so Alpaca can
    remain backward-compatible while Bybit implements the same canonical contract.
    """

    environment: CryptoExecutionEnvironment
    instrument_provider: InstrumentMetadataProvider

    def verify_environment(self) -> dict[str, Any]: ...
    def verify_account(self) -> AccountInfo: ...
    def get_account(self) -> AccountInfo: ...
    def verify_crypto_assets(
        self, symbols: list[str] | tuple[str, ...]
    ) -> dict[str, AssetInfo]: ...
    def get_asset(self, symbol: str) -> AssetInfo: ...

    def submit_crypto_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        *,
        trade_increment: float,
        min_order_size: float,
        reference_price: Optional[float] = None,
    ) -> OrderRecord: ...

    def submit_crypto_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_price: float,
        client_order_id: str,
        *,
        trade_increment: float,
        min_order_size: float,
        price_increment: float,
    ) -> OrderRecord: ...

    def amend_protection(
        self,
        client_order_id: str,
        *,
        qty: float,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
    ) -> OrderRecord: ...

    def get_position(self, symbol: str) -> Optional[BrokerPosition]: ...
    def list_positions(self) -> list[BrokerPosition]: ...
    def list_open_orders(self, symbol: Optional[str] = None) -> list[OrderRecord]: ...
    def get_order(self, broker_order_id: str) -> OrderRecord: ...
    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]: ...
    def cancel_order(self, broker_order_id: str) -> None: ...

    def sync_bot_executions(self, symbol: Optional[str] = None) -> int: ...
    def bot_owned_quantity(self, symbol: str) -> float: ...
    def broker_health(self) -> dict[str, Any]: ...


@runtime_checkable
class CryptoMarketDataFeed(Protocol):
    """Canonical market-data contract above the venue provider boundary."""

    def set_interrupt_event(self, event) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame: ...

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        *,
        now=None,
        tolerance_seconds: float = 5.0,
    ) -> pd.DataFrame: ...

    def get_latest_quotes(
        self, symbols: tuple[str, ...] | list[str]
    ) -> dict[str, CryptoQuote]: ...

    def market_context(self, symbol: str) -> Any: ...
    def recent_trades(self, symbol: str, limit: int = 60) -> list[dict[str, Any]]: ...
    def market_health(self) -> dict[str, Any]: ...
