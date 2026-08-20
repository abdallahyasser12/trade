from datetime import datetime, timedelta, timezone

import pandas as pd

from equity_market_data import EquityAssetSnapshot
from quant.contracts import Market
from quant.equity_marketwide import (
    MARKET_BENCHMARK,
    MarketWideEquityConfig,
    MarketWideEquitySelector,
    OwnershipAwareStrategy,
    build_equity_scan_strategies,
)

NOW = datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc)


class Quote:
    def __init__(self, bid, ask, *, age_seconds=1):
        self.bid_price = bid
        self.ask_price = ask
        self.timestamp = NOW - timedelta(seconds=age_seconds)
        self.bid_size = 1000
        self.ask_size = 1000


def asset(symbol, **overrides):
    values = dict(
        symbol=symbol,
        name=symbol,
        exchange="NASDAQ",
        tradable=True,
        active=True,
        fractionable=True,
        marginable=True,
        shortable=True,
    )
    values.update(overrides)
    return EquityAssetSnapshot(**values)


def daily(price=50.0, volume=500_000.0, rows=20):
    index = pd.date_range("2026-07-20", periods=rows, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [price] * rows,
            "high": [price * 1.01] * rows,
            "low": [price * 0.99] * rows,
            "close": [price] * rows,
            "volume": [volume] * rows,
        },
        index=index,
    )


def test_marketwide_config_is_bounded():
    assert not MarketWideEquityConfig().validate()
    assert MarketWideEquityConfig().timeframe == "15Min"
    assert MarketWideEquityConfig().max_scan_symbols == 48
    assert MarketWideEquityConfig(timeframe="1Day").validate()
    assert MarketWideEquityConfig(shard_size=65).validate()
    assert MarketWideEquityConfig(risk_per_trade_pct=1.01).validate()
    assert MarketWideEquityConfig(max_total_exposure_pct=101).validate()


def test_selector_has_named_fail_closed_rejections():
    assets = [
        asset("GOOD"),
        asset("HALT", tradable=False),
        asset("PENNY"),
        asset("STALE"),
        asset("WIDE"),
        asset("THIN"),
        asset("NEW"),
    ]
    quotes = {
        "GOOD": Quote(49.99, 50.01),
        "HALT": Quote(49.99, 50.01),
        "PENNY": Quote(1.99, 2.01),
        "STALE": Quote(49.99, 50.01, age_seconds=60),
        "WIDE": Quote(49.0, 51.0),
        "THIN": Quote(49.99, 50.01),
        "NEW": Quote(49.99, 50.01),
    }
    bars = {
        "GOOD": daily(),
        "HALT": daily(),
        "PENNY": daily(price=2.0),
        "STALE": daily(),
        "WIDE": daily(),
        "THIN": daily(volume=1_000.0),
        "NEW": daily(rows=5),
    }
    report = MarketWideEquitySelector().select(assets, quotes, bars, now=NOW)
    assert report.selected == ("GOOD",)
    assert report.rejected["HALT"] == "not_tradable"
    assert report.rejected["PENNY"] == "below_min_price"
    assert report.rejected["STALE"] == "stale_quote"
    assert report.rejected["WIDE"] == "spread_too_wide"
    assert report.rejected["THIN"] == "insufficient_dollar_volume"
    assert report.rejected["NEW"] == "insufficient_daily_history"


def test_selector_cap_is_liquidity_only_and_deterministic():
    assets = [asset("AAA"), asset("BBB"), asset("CCC")]
    quotes = {symbol: Quote(49.99, 50.01) for symbol in ("AAA", "BBB", "CCC")}
    bars = {
        "AAA": daily(volume=600_000),
        "BBB": daily(volume=600_000),
        "CCC": daily(volume=900_000),
    }
    cfg = MarketWideEquityConfig(max_scan_symbols=2, shard_size=2)
    report = MarketWideEquitySelector(cfg).select(assets, quotes, bars, now=NOW)
    assert report.selected == ("CCC", "AAA")
    assert report.rejected["BBB"] == "beyond_scan_cap"


def test_selector_reserves_executable_spy_as_stable_benchmark_anchor():
    assets = [asset("AAA"), asset(MARKET_BENCHMARK), asset("CCC")]
    quotes = {
        symbol: Quote(49.99, 50.01)
        for symbol in ("AAA", MARKET_BENCHMARK, "CCC")
    }
    bars = {
        "AAA": daily(volume=900_000),
        MARKET_BENCHMARK: daily(volume=200_000),
        "CCC": daily(volume=800_000),
    }
    report = MarketWideEquitySelector(
        MarketWideEquityConfig(max_scan_symbols=2, shard_size=2)
    ).select(assets, quotes, bars, now=NOW)
    assert report.selected == (MARKET_BENCHMARK, "AAA")
    assert report.rejected["CCC"] == "beyond_scan_cap"


def test_strategy_builder_shards_without_changing_strategy_identity():
    symbols = tuple(f"S{index:03d}" for index in range(110))
    cfg = MarketWideEquityConfig(max_scan_symbols=192, shard_size=48, timeframe="15Min")
    entries = build_equity_scan_strategies(symbols, config=cfg)
    assert len(entries) == 9  # 3 strategy families x 3 compute shards
    assert {entry.strategy_id for entry in entries} == {
        "opening_range_breakout",
        "vwap_reversion",
        "volatility_squeeze",
    }
    assert all(entry.market is Market.US_EQUITY for entry in entries)
    assert all(len(entry.strategy.config.symbols) <= 48 for entry in entries)
    for strategy_id in {entry.strategy_id for entry in entries}:
        family_symbols = [
            symbol
            for entry in entries
            if entry.strategy_id == strategy_id
            for symbol in entry.strategy.config.symbols
        ]
        assert family_symbols == list(symbols)


def test_strategy_builder_uses_spy_when_it_is_in_the_shard():
    symbols = (MARKET_BENCHMARK, "AAPL", "MSFT")
    entries = build_equity_scan_strategies(symbols, config=MarketWideEquityConfig())
    assert entries
    assert all(entry.strategy.config.benchmark_symbol == MARKET_BENCHMARK for entry in entries)


def test_ownership_wrapper_filters_real_owned_context_but_preserves_shadow_context():
    class Delegate:
        config = object()

        def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
            return held_symbols

        def evaluate(self, bars, held_symbols=frozenset()):
            return held_symbols

        def required_bars(self):
            return 1

        def prepare_symbol(self, bars):
            return bars

        def prepare_universe(self, bars):
            return bars

        def ratchet_stop(self, *args):
            return args[0], args[1]

    paper_owned = OwnershipAwareStrategy(Delegate(), {"AAPL"})
    assert paper_owned.evaluate_prepared({}, NOW, {"AAPL", "MSFT"}) == frozenset({"AAPL"})
    assert paper_owned.evaluate({}, {"AAPL", "MSFT"}) == frozenset({"AAPL"})

    shadow_no_real_position = OwnershipAwareStrategy(Delegate(), frozenset())
    assert shadow_no_real_position.evaluate_prepared({}, NOW, {"MSFT"}) == frozenset({"MSFT"})
    assert shadow_no_real_position.evaluate({}, {"MSFT"}) == frozenset({"MSFT"})
