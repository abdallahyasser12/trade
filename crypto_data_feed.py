"""Alpaca 24/7 crypto bars and quote snapshots with fail-closed freshness rules."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config import timeframe_to_alpaca, timeframe_to_timedelta
from crypto_config import validate_crypto_symbol
from crypto_models import CryptoQuote
from data_feed import DataFeedError, drop_incomplete_bars, normalize_bars_df, with_retries


def crypto_max_staleness(timeframe: str, staleness_bars: int = 3) -> timedelta:
    """Maximum age after a completed bar's end in a continuously open market."""

    if staleness_bars < 1:
        raise ValueError("staleness_bars must be at least one")
    return timeframe_to_timedelta(timeframe) * staleness_bars + timedelta(minutes=2)


def is_crypto_data_stale(
    last_bar_start: datetime,
    timeframe: str,
    now: datetime,
    staleness_bars: int = 3,
) -> bool:
    if last_bar_start.tzinfo is None or now.tzinfo is None:
        raise ValueError("staleness timestamps must be timezone-aware")
    bar_end = last_bar_start + timeframe_to_timedelta(timeframe)
    return now - bar_end > crypto_max_staleness(timeframe, staleness_bars)


def quote_is_safe(
    quote: CryptoQuote,
    now: datetime,
    *,
    max_age_seconds: float,
    max_spread_bps: float,
) -> tuple[bool, str]:
    """Validate quote age, ordering, and spread before a new market entry."""

    if now.tzinfo is None or quote.timestamp.tzinfo is None:
        return False, "quote timestamp is not timezone-aware"
    age = (now - quote.timestamp).total_seconds()
    if age < -5:
        return False, "quote timestamp is in the future"
    if age > max_age_seconds:
        return False, f"quote is stale ({age:.1f}s old)"
    if quote.bid_price <= 0 or quote.ask_price <= 0 or quote.ask_price < quote.bid_price:
        return False, "quote prices are invalid or crossed"
    if not math.isfinite(quote.spread_bps) or quote.spread_bps > max_spread_bps:
        return False, f"spread {quote.spread_bps:.1f} bps exceeds {max_spread_bps:.1f} bps"
    return True, "quote passed freshness and spread checks"


class AlpacaCryptoDataFeed:
    """Historical completed bars plus latest crypto quotes from Alpaca's US feed."""

    def __init__(self, api_key: str = "", secret_key: str = ""):
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient

        self._client = CryptoHistoricalDataClient(
            api_key=api_key or None,
            secret_key=secret_key or None,
        )
        self._interrupt_event = None
        self._rate_limit_events = 0

    def set_interrupt_event(self, event) -> None:
        self._interrupt_event = event

    @property
    def rate_limit_events(self) -> int:
        return self._rate_limit_events

    def _note_rate_limit(self, _delay: float) -> None:
        self._rate_limit_events += 1

    @property
    def _sleep_fn(self):
        return self._interrupt_event.wait if self._interrupt_event is not None else time.sleep

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        from alpaca.data.enums import CryptoFeed
        from alpaca.data.requests import CryptoBarsRequest

        error = validate_crypto_symbol(symbol)
        if error:
            raise ValueError(error)
        if start.tzinfo is None or (end is not None and end.tzinfo is None):
            raise ValueError("Crypto bar request timestamps must be timezone-aware.")
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_to_alpaca(timeframe),
            start=start,
            end=end,
            limit=limit,
        )

        def fetch():
            return self._client.get_crypto_bars(request, feed=CryptoFeed.US).df

        raw = with_retries(
            fetch,
            on_rate_limit=self._note_rate_limit,
            sleep_fn=self._sleep_fn,
        )
        return normalize_bars_df(raw, symbol)

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        *,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
    ) -> pd.DataFrame:
        if lookback_bars < 1:
            raise ValueError("lookback_bars must be positive")
        now = now or datetime.now(timezone.utc)
        duration = timeframe_to_timedelta(timeframe)
        start = now - duration * max(lookback_bars + 5, int(lookback_bars * 1.15))
        bars = self.get_bars(symbol, timeframe, start=start, end=now)
        bars = drop_incomplete_bars(bars, timeframe, now, tolerance_seconds)
        return bars.iloc[-lookback_bars:]

    def get_latest_quotes(self, symbols: tuple[str, ...] | list[str]) -> dict[str, CryptoQuote]:
        from alpaca.data.enums import CryptoFeed
        from alpaca.data.requests import CryptoLatestQuoteRequest

        if not symbols:
            return {}
        for symbol in symbols:
            error = validate_crypto_symbol(symbol)
            if error:
                raise ValueError(error)
        request = CryptoLatestQuoteRequest(symbol_or_symbols=list(symbols))

        def fetch():
            return self._client.get_crypto_latest_quote(request, feed=CryptoFeed.US)

        raw = with_retries(
            fetch,
            on_rate_limit=self._note_rate_limit,
            sleep_fn=self._sleep_fn,
        )
        quotes: dict[str, CryptoQuote] = {}
        for symbol in symbols:
            item = raw.get(symbol) if isinstance(raw, dict) else None
            if item is None:
                continue
            timestamp = pd.Timestamp(item.timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            bid = float(item.bid_price)
            ask = float(item.ask_price)
            if not all(math.isfinite(value) for value in (bid, ask)):
                continue
            quotes[symbol] = CryptoQuote(
                symbol=symbol,
                timestamp=timestamp.to_pydatetime(),
                bid_price=bid,
                ask_price=ask,
                bid_size=float(item.bid_size) if item.bid_size is not None else None,
                ask_size=float(item.ask_size) if item.ask_size is not None else None,
            )
        if not quotes:
            raise DataFeedError("Alpaca returned no valid latest crypto quotes.")
        return quotes
