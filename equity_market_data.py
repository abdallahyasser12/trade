"""Read-only Alpaca adapter for market-wide US-equity discovery.

This module deliberately owns no order methods.  It exists to make the full active
US-equity catalogue, latest quotes and batched completed bars available to the
market-wide opportunity engine without reaching through ``AlpacaBroker`` internals.
All network calls use the same bounded retry classifier as ``data_feed.py`` and the
normalization path is shared with the existing audited stock feed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

import pandas as pd

from config import timeframe_to_alpaca, timeframe_to_timedelta
from data_feed import drop_incomplete_bars, normalize_bars_df, with_retries


@dataclass(frozen=True)
class EquityAssetSnapshot:
    symbol: str
    name: str
    exchange: str
    tradable: bool
    active: bool
    fractionable: bool
    marginable: bool
    shortable: bool


@dataclass(frozen=True)
class EquityMarketDataConfig:
    feed: str = "iex"
    quote_batch_size: int = 200
    bar_batch_size: int = 100
    max_attempts: int = 3

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.feed not in {"iex", "sip"}:
            errors.append("Equity discovery feed must be 'iex' or 'sip'.")
        if not 1 <= self.quote_batch_size <= 500:
            errors.append("quote_batch_size must be between 1 and 500.")
        if not 1 <= self.bar_batch_size <= 250:
            errors.append("bar_batch_size must be between 1 and 250.")
        if not 1 <= self.max_attempts <= 8:
            errors.append("max_attempts must be between 1 and 8.")
        return errors


def _chunks(values: Iterable[str], size: int):
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _asset_enum_value(value) -> str:
    return str(getattr(value, "value", value) or "")


class AlpacaEquityMarketData:
    """Batched, read-only market-wide data boundary.

    ``TradingClient`` is used only for GET /assets.  Order submission is intentionally
    absent from this class, so discovery cannot accidentally acquire execution
    privilege even when live credentials are supplied elsewhere in the process.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        config: EquityMarketDataConfig = EquityMarketDataConfig(),
        *,
        sleep_fn: Callable[[float], object] = time.sleep,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid equity market-data config: " + "; ".join(errors))
        if not api_key or not secret_key:
            raise ValueError("Alpaca credentials are required for US-equity discovery.")

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.config = config
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        # Assets are environment-independent market metadata.  paper=True keeps this
        # auxiliary client on the non-live trading endpoint by construction.
        self._asset_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )
        self._sleep_fn = sleep_fn
        self._rate_limit_events = 0
        self._last_error: Optional[str] = None

    @property
    def rate_limit_events(self) -> int:
        return self._rate_limit_events

    def health(self) -> tuple[bool, Optional[str]]:
        return self._last_error is None, self._last_error

    def _note_rate_limit(self, _delay: float) -> None:
        self._rate_limit_events += 1

    def _call(self, fn):
        try:
            result = with_retries(
                fn,
                max_attempts=self.config.max_attempts,
                on_rate_limit=self._note_rate_limit,
                sleep_fn=self._sleep_fn,
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def list_active_us_equities(self) -> list[EquityAssetSnapshot]:
        """Return every active US equity Alpaca currently exposes.

        No ranking or historical performance criterion is applied here.  The method is
        intentionally the broad discovery boundary; filtering happens later and is
        journalable.
        """

        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        request = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
        )
        assets = self._call(lambda: self._asset_client.get_all_assets(request))
        snapshots: list[EquityAssetSnapshot] = []
        for asset in assets:
            status = _asset_enum_value(getattr(asset, "status", ""))
            asset_class = _asset_enum_value(getattr(asset, "asset_class", ""))
            if asset_class != "us_equity":
                continue
            snapshots.append(
                EquityAssetSnapshot(
                    symbol=str(asset.symbol),
                    name=str(getattr(asset, "name", "") or asset.symbol),
                    exchange=_asset_enum_value(getattr(asset, "exchange", "")),
                    tradable=bool(getattr(asset, "tradable", False)),
                    active=status == "active",
                    fractionable=bool(getattr(asset, "fractionable", False)),
                    marginable=bool(getattr(asset, "marginable", False)),
                    shortable=bool(getattr(asset, "shortable", False)),
                )
            )
        snapshots.sort(key=lambda item: item.symbol)
        return snapshots

    def get_latest_quotes(self, symbols: list[str] | tuple[str, ...]) -> dict[str, object]:
        """Fetch latest level-1 quotes in bounded batches."""

        if not symbols:
            return {}
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        feed = DataFeed(self.config.feed)
        out: dict[str, object] = {}
        for batch in _chunks(dict.fromkeys(symbols), self.config.quote_batch_size):
            request = StockLatestQuoteRequest(
                symbol_or_symbols=batch,
                feed=feed,
            )
            response = self._call(
                lambda request=request: self._data_client.get_stock_latest_quote(request)
            )
            for symbol in batch:
                quote = response.get(symbol)
                if quote is not None:
                    out[symbol] = quote
        return out

    def get_recent_bars_many(
        self,
        symbols: list[str] | tuple[str, ...],
        timeframe: str,
        lookback_bars: int,
        *,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
        adjustment: str = "split",
    ) -> dict[str, pd.DataFrame]:
        """Return up to ``lookback_bars`` completed OHLCV rows per symbol.

        The request is multi-symbol and chunked, so scanning hundreds of liquid names
        does not become one HTTP request per ticker.  Each symbol is normalized through
        the same function used by ``AlpacaDataFeed``.
        """

        if lookback_bars < 1:
            raise ValueError("lookback_bars must be positive.")
        if not symbols:
            return {}
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware.")

        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        duration = timeframe_to_timedelta(timeframe)
        if timeframe == "1Day":
            window = timedelta(days=int(lookback_bars * 1.8) + 14)
        else:
            bars_per_day = max(1, int(timedelta(hours=6.5) / duration))
            trading_days = lookback_bars / bars_per_day
            window = timedelta(days=trading_days * 1.8 + 7)
        start = now - window
        feed = DataFeed(self.config.feed)
        out: dict[str, pd.DataFrame] = {}

        for batch in _chunks(dict.fromkeys(symbols), self.config.bar_batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=timeframe_to_alpaca(timeframe),
                start=start,
                end=now,
                adjustment=Adjustment(adjustment),
                feed=feed,
            )
            raw = self._call(lambda request=request: self._data_client.get_stock_bars(request).df)
            for symbol in batch:
                frame = normalize_bars_df(raw, symbol)
                frame = drop_incomplete_bars(
                    frame,
                    timeframe,
                    now,
                    tolerance_seconds,
                )
                if len(frame) > lookback_bars:
                    frame = frame.iloc[-lookback_bars:]
                out[symbol] = frame
        return out

    @staticmethod
    def quote_spread_bps(quote) -> Optional[float]:
        try:
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (bid, ask)) or bid <= 0 or ask <= bid:
            return None
        midpoint = (bid + ask) / 2.0
        return (ask - bid) / midpoint * 10_000.0
