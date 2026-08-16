"""Provider-agnostic data interfaces with graceful degradation (spec §§10, 61-62).

Strategy and feature code depends on these Protocols, never on alpaca-py
directly, so venues and vendors can be swapped without touching research
logic.  The one iron rule: a missing provider marks its source
UNAVAILABLE and disables dependent strategies — nothing anywhere may
substitute fabricated values for data it does not have.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from quant.contracts import Market, ProviderAvailability


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_bars(
        self, symbols: list[str], timeframe: str, start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]: ...

    def get_latest_quote(self, symbol: str) -> Optional[object]: ...


@runtime_checkable
class FundamentalProvider(Protocol):
    def get_fundamentals(self, symbol: str, asof: datetime) -> Optional[dict]: ...


@runtime_checkable
class NewsProvider(Protocol):
    def get_events(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> list[dict]: ...


@runtime_checkable
class MacroProvider(Protocol):
    def get_series(self, name: str, asof: datetime) -> Optional[pd.Series]: ...


@runtime_checkable
class SentimentProvider(Protocol):
    def get_sentiment(self, symbol: str, asof: datetime) -> Optional[dict]: ...


@runtime_checkable
class DerivativesProvider(Protocol):
    def get_derivatives(self, symbol: str, asof: datetime) -> Optional[dict]: ...


class ProviderRegistry:
    """Named data sources plus honest availability reporting.

    A provider may expose ``health() -> tuple[bool, Optional[str]]``; the
    registry consults it so a configured-but-broken source reports
    unavailable with its reason instead of silently serving nothing.
    """

    def __init__(self) -> None:
        self._providers: dict[str, object] = {}

    def register(self, source: str, provider: object) -> None:
        if not source:
            raise ValueError("Provider source name is required.")
        self._providers[source] = provider

    def get(self, source: str) -> Optional[object]:
        return self._providers.get(source)

    def available_sources(self) -> list[str]:
        return sorted(self._providers)

    def availability(self, source: str) -> ProviderAvailability:
        now = datetime.now(timezone.utc)
        provider = self._providers.get(source)
        if provider is None:
            return ProviderAvailability(
                source=source, available=False, reason="not_configured", as_of=now
            )
        health = getattr(provider, "health", None)
        if callable(health):
            try:
                ok, reason = health()
            except Exception as error:
                return ProviderAvailability(
                    source=source,
                    available=False,
                    reason=f"health_check_failed: {error}",
                    as_of=now,
                )
            if not ok:
                return ProviderAvailability(
                    source=source, available=False, reason=reason or "unhealthy", as_of=now
                )
        return ProviderAvailability(source=source, available=True, as_of=now)

    def required_sources_met(
        self, required: tuple[str, ...]
    ) -> tuple[bool, list[ProviderAvailability]]:
        """The strategy-disabling hook: every required source must be
        available or the strategy sits out, with the reasons logged."""
        reports = [self.availability(source) for source in required]
        return all(report.available for report in reports), reports


class AlpacaMarketDataProviderAdapter:
    """Adapts the existing Alpaca feeds to the ``MarketDataProvider`` shape.

    Wraps an ``AlpacaDataFeed`` (stocks) or ``AlpacaCryptoDataFeed``
    (crypto) INSTANCE — construction stays with the caller, so this module
    never imports alpaca-py and stays importable in analysis contexts.
    Errors are captured, not raised: a broken feed yields empty results
    and a failing ``health()``, which the registry surfaces as
    unavailable.
    """

    def __init__(self, feed: object, market: Market):
        self._feed = feed
        self._market = market
        self._last_error: Optional[str] = None

    def health(self) -> tuple[bool, Optional[str]]:
        if self._last_error is not None:
            return False, self._last_error
        return True, None

    def get_bars(
        self, symbols: list[str], timeframe: str, start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                out[symbol] = self._feed.get_bars(symbol, timeframe, start, end)
                self._last_error = None
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"
        return out

    def get_latest_quote(self, symbol: str) -> Optional[object]:
        getter = getattr(self._feed, "get_latest_quotes", None)
        if getter is None:
            self._last_error = "feed_has_no_quote_endpoint"
            return None
        try:
            quotes = getter([symbol])
            self._last_error = None
            return quotes.get(symbol)
        except Exception as error:
            self._last_error = f"{type(error).__name__}: {error}"
            return None
