"""Point-in-time universe construction and provider registry tests."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quant.contracts import Market
from quant.providers import (
    AlpacaMarketDataProviderAdapter,
    MarketDataProvider,
    ProviderRegistry,
)
from quant.universe import SymbolStats, UniverseFilterConfig, UniverseSelection, build_universe

NOW = datetime(2026, 1, 5, tzinfo=timezone.utc)


def stat(symbol, **overrides):
    base = dict(
        symbol=symbol,
        market=Market.US_EQUITY,
        asof=NOW,
        price=50.0,
        avg_dollar_volume=20_000_000.0,
        history_bars=400,
        avg_spread_bps=5.0,
    )
    base.update(overrides)
    return SymbolStats(**base)


def test_each_rejection_reason():
    stats = [
        stat("GOOD"),
        stat("PENNY", price=2.0),
        stat("THIN", avg_dollar_volume=100.0),
        stat("NEW", history_bars=10),
        stat("WIDE", avg_spread_bps=90.0),
        stat("HALT", halted=True),
        stat("DEAD", tradable=False),
        stat("NOSPREAD", avg_spread_bps=None),
    ]
    selection = build_universe(stats, UniverseFilterConfig(), NOW)
    assert selection.selected == ["GOOD"]
    assert selection.rejected == {
        "PENNY": "below_minimum_price",
        "THIN": "insufficient_dollar_volume",
        "NEW": "insufficient_history",
        "WIDE": "spread_too_wide",
        "HALT": "halted",
        "DEAD": "not_tradable",
        "NOSPREAD": "spread_unknown",
    }


def test_future_dated_stats_refused():
    stats = [stat("GOOD"), stat("FUTURE", asof=NOW + timedelta(days=1))]
    with pytest.raises(ValueError, match="Point-in-time violation"):
        build_universe(stats, UniverseFilterConfig(), NOW)


def test_ranking_and_cap_are_deterministic():
    stats = [
        stat("AAA", avg_dollar_volume=30_000_000.0),
        stat("BBB", avg_dollar_volume=30_000_000.0),
        stat("CCC", avg_dollar_volume=50_000_000.0),
    ]
    selection = build_universe(stats, UniverseFilterConfig(max_symbols=2), NOW)
    assert selection.selected == ["CCC", "AAA"]
    assert selection.rejected == {"BBB": "beyond_symbol_cap"}
    assert isinstance(selection, UniverseSelection)


def test_missing_spread_allowed_when_configured():
    stats = [stat("NOSPREAD", avg_spread_bps=None)]
    selection = build_universe(
        stats, UniverseFilterConfig(allow_missing_spread=True), NOW
    )
    assert selection.selected == ["NOSPREAD"]


def test_config_validation():
    assert UniverseFilterConfig(min_price=-1.0).validate()
    assert UniverseFilterConfig(max_symbols=0).validate()
    assert not UniverseFilterConfig().validate()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class StubFeed:
    def get_bars(self, symbol, timeframe, start, end):
        return pd.DataFrame({"close": [1.0]})

    def get_latest_quotes(self, symbols):
        return {symbol: object() for symbol in symbols}


class BrokenFeed:
    def get_bars(self, symbol, timeframe, start, end):
        raise RuntimeError("feed exploded")


def test_adapter_satisfies_protocol_and_captures_errors():
    good = AlpacaMarketDataProviderAdapter(StubFeed(), Market.US_EQUITY)
    assert isinstance(good, MarketDataProvider)
    bars = good.get_bars(["AAPL"], "1Day", NOW - timedelta(days=5), NOW)
    assert "AAPL" in bars
    assert good.health() == (True, None)

    broken = AlpacaMarketDataProviderAdapter(BrokenFeed(), Market.US_EQUITY)
    bars = broken.get_bars(["AAPL"], "1Day", NOW - timedelta(days=5), NOW)
    assert bars == {}
    ok, reason = broken.health()
    assert not ok and "feed exploded" in reason


def test_registry_availability_states():
    registry = ProviderRegistry()
    missing = registry.availability("news")
    assert not missing.available
    assert missing.reason == "not_configured"

    registry.register("price", AlpacaMarketDataProviderAdapter(StubFeed(), Market.CRYPTO))
    assert registry.availability("price").available

    broken = AlpacaMarketDataProviderAdapter(BrokenFeed(), Market.CRYPTO)
    broken.get_bars(["BTC/USD"], "1Hour", NOW - timedelta(days=1), NOW)
    registry.register("derivatives", broken)
    report = registry.availability("derivatives")
    assert not report.available


def test_required_sources_gate():
    registry = ProviderRegistry()
    registry.register("price", AlpacaMarketDataProviderAdapter(StubFeed(), Market.CRYPTO))
    met, reports = registry.required_sources_met(("price",))
    assert met
    met, reports = registry.required_sources_met(("price", "sentiment"))
    assert not met
    unavailable = [r for r in reports if not r.available]
    assert unavailable[0].source == "sentiment"
