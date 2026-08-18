"""Market-wide US-equity discovery and intraday strategy orchestration.

The full Alpaca asset catalogue is considered at the discovery stage. Only hard,
point-in-time executability filters (active/tradable status, price, quote freshness,
spread, history and dollar liquidity) may shrink that catalogue before alpha logic.
Historical return is deliberately NOT a universe-selection input: using recent winners
to decide what the backtest was allowed to see would introduce selection bias.

The surviving liquid universe is sharded into bounded strategy instances because the
shared ``IntradayStrategyConfig`` intentionally caps one strategy instance at 64
symbols. The QuantScanner still ranks all resulting candidates globally.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional

import pandas as pd

from equity_market_data import EquityAssetSnapshot
from quant.contracts import Market
from quant.scanner import ScanStrategy
from quant.strategies.opening_range import (
    OpeningRangeBreakoutStrategy,
    OpeningRangeConfig,
)
from quant.strategies.volatility_squeeze import (
    VolatilitySqueezeConfig,
    VolatilitySqueezeStrategy,
)
from quant.strategies.vwap_reversion import (
    VwapReversionConfig,
    VwapReversionStrategy,
)

MARKET_BENCHMARK = "SPY"


@dataclass(frozen=True)
class MarketWideEquityConfig:
    timeframe: str = "15Min"
    min_price: float = 5.0
    max_price: Optional[float] = 1_500.0
    min_avg_dollar_volume: float = 5_000_000.0
    liquidity_lookback_days: int = 20
    min_daily_history_bars: int = 15
    max_quote_age_seconds: float = 30.0
    max_spread_bps: float = 35.0
    max_scan_symbols: int = 48
    shard_size: int = 48
    universe_refresh_seconds: int = 3_600
    intraday_lookback_bars: int = 220
    max_positions: int = 6
    risk_per_trade_pct: float = 0.15
    max_position_pct: float = 10.0
    max_total_exposure_pct: float = 35.0
    max_session_loss_pct: float = 2.0
    near_close_entry_block_minutes: int = 20

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.timeframe not in {"5Min", "15Min"}:
            errors.append("timeframe must be 5Min or 15Min.")
        for label, value in {
            "min_price": self.min_price,
            "min_avg_dollar_volume": self.min_avg_dollar_volume,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "max_spread_bps": self.max_spread_bps,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_position_pct": self.max_position_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_session_loss_pct": self.max_session_loss_pct,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and positive.")
        if self.max_price is not None and (
            not math.isfinite(self.max_price) or self.max_price <= self.min_price
        ):
            errors.append("max_price must be finite and above min_price when set.")
        if not 5 <= self.liquidity_lookback_days <= 120:
            errors.append("liquidity_lookback_days must be between 5 and 120.")
        if not 5 <= self.min_daily_history_bars <= self.liquidity_lookback_days:
            errors.append("min_daily_history_bars must be within the liquidity lookback.")
        if not 1 <= self.max_scan_symbols <= 512:
            errors.append("max_scan_symbols must be between 1 and 512.")
        if not 1 <= self.shard_size <= 64:
            errors.append("shard_size must be between 1 and 64.")
        if self.shard_size > self.max_scan_symbols:
            errors.append("shard_size cannot exceed max_scan_symbols.")
        if self.universe_refresh_seconds < 300:
            errors.append("universe_refresh_seconds must be at least 300.")
        if self.intraday_lookback_bars < 150:
            errors.append("intraday_lookback_bars must be at least 150.")
        if not 1 <= self.max_positions <= 20:
            errors.append("max_positions must be between 1 and 20.")
        if self.risk_per_trade_pct > 1.0:
            errors.append("risk_per_trade_pct must not exceed 1%.")
        if self.max_position_pct > 25.0:
            errors.append("max_position_pct must not exceed 25%.")
        if self.max_total_exposure_pct > 100.0:
            errors.append("max_total_exposure_pct must not exceed 100%.")
        if self.max_session_loss_pct > 5.0:
            errors.append("max_session_loss_pct must not exceed 5%.")
        if not 0 <= self.near_close_entry_block_minutes <= 90:
            errors.append("near_close_entry_block_minutes must be in [0,90].")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EquityUniverseMetric:
    symbol: str
    eligible: bool
    reason: str
    price: Optional[float]
    avg_dollar_volume: Optional[float]
    spread_bps: Optional[float]
    quote_age_seconds: Optional[float]
    history_bars: int
    liquidity_score: float


@dataclass
class EquityUniverseReport:
    asof: datetime
    discovered: int
    selected: tuple[str, ...]
    metrics: list[EquityUniverseMetric] = field(default_factory=list)

    @property
    def rejected(self) -> dict[str, str]:
        return {metric.symbol: metric.reason for metric in self.metrics if not metric.eligible}

    def to_dict(self) -> dict:
        return {
            "asof": self.asof.isoformat(),
            "discovered": self.discovered,
            "selected": list(self.selected),
            "rejected": self.rejected,
            "metrics": [asdict(metric) for metric in self.metrics],
        }


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quote_age_seconds(quote, now: datetime) -> Optional[float]:
    raw = getattr(quote, "timestamp", None)
    if raw is None:
        return None
    try:
        stamp = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        return None
    age = (pd.Timestamp(now).tz_convert("UTC") - stamp.tz_convert("UTC")).total_seconds()
    return max(0.0, float(age))


def _quote_values(quote, now: datetime) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if quote is None:
        return None, None, None
    bid = _finite(getattr(quote, "bid_price", None))
    ask = _finite(getattr(quote, "ask_price", None))
    age = _quote_age_seconds(quote, now)
    if bid is None or ask is None or bid <= 0 or ask <= bid:
        return None, None, age
    midpoint = (bid + ask) / 2.0
    spread_bps = (ask - bid) / midpoint * 10_000.0
    return midpoint, spread_bps, age


def _daily_liquidity(frame: Optional[pd.DataFrame], lookback: int) -> tuple[Optional[float], int]:
    if frame is None or frame.empty or not {"close", "volume"}.issubset(frame.columns):
        return None, 0
    usable = frame[["close", "volume"]].tail(lookback).astype(float)
    valid = (
        usable["close"].map(math.isfinite)
        & usable["volume"].map(math.isfinite)
        & usable["close"].gt(0)
        & usable["volume"].ge(0)
    )
    usable = usable[valid]
    if usable.empty:
        return None, 0
    dollar_volume = usable["close"] * usable["volume"]
    return float(dollar_volume.mean()), int(len(usable))


class MarketWideEquitySelector:
    """Hard-filter the complete active asset catalogue into a liquid scan set."""

    def __init__(self, config: MarketWideEquityConfig = MarketWideEquityConfig()):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid market-wide equity config: " + "; ".join(errors))
        self.config = config

    def select(
        self,
        assets: list[EquityAssetSnapshot],
        quotes: Mapping[str, object],
        daily_bars: Mapping[str, pd.DataFrame],
        *,
        now: Optional[datetime] = None,
    ) -> EquityUniverseReport:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("Universe selection time must be timezone-aware.")
        cfg = self.config
        metrics: list[EquityUniverseMetric] = []

        for asset in sorted(assets, key=lambda item: item.symbol):
            symbol = asset.symbol
            midpoint, spread_bps, age = _quote_values(quotes.get(symbol), now)
            avg_dollar_volume, history_bars = _daily_liquidity(
                daily_bars.get(symbol), cfg.liquidity_lookback_days
            )
            reason = "eligible"
            if not asset.active:
                reason = "inactive"
            elif not asset.tradable:
                reason = "not_tradable"
            elif midpoint is None:
                reason = "invalid_or_missing_quote"
            elif age is None:
                reason = "quote_timestamp_missing"
            elif age > cfg.max_quote_age_seconds:
                reason = "stale_quote"
            elif spread_bps is None or spread_bps > cfg.max_spread_bps:
                reason = "spread_too_wide"
            elif midpoint < cfg.min_price:
                reason = "below_min_price"
            elif cfg.max_price is not None and midpoint > cfg.max_price:
                reason = "above_max_price"
            elif history_bars < cfg.min_daily_history_bars:
                reason = "insufficient_daily_history"
            elif avg_dollar_volume is None or avg_dollar_volume < cfg.min_avg_dollar_volume:
                reason = "insufficient_dollar_volume"

            eligible = reason == "eligible"
            liquidity_score = 0.0
            if eligible:
                volume_quality = math.log10(max(avg_dollar_volume or 1.0, 1.0))
                spread_quality = max(0.0, 1.0 - float(spread_bps) / cfg.max_spread_bps)
                freshness = max(0.0, 1.0 - float(age) / cfg.max_quote_age_seconds)
                liquidity_score = volume_quality + 0.75 * spread_quality + 0.25 * freshness

            metrics.append(
                EquityUniverseMetric(
                    symbol=symbol,
                    eligible=eligible,
                    reason=reason,
                    price=midpoint,
                    avg_dollar_volume=avg_dollar_volume,
                    spread_bps=spread_bps,
                    quote_age_seconds=age,
                    history_bars=history_bars,
                    liquidity_score=liquidity_score,
                )
            )

        eligible = sorted(
            (metric for metric in metrics if metric.eligible),
            key=lambda metric: (-metric.liquidity_score, metric.symbol),
        )
        # SPY is a benchmark anchor, not an alpha winner. If it passes the exact same
        # executability gates, reserve one slot and put it first so every normal
        # one-shard production run uses a stable market benchmark. No return metric is
        # consulted and SPY is not forced in when it is itself unexecutable.
        benchmark = next((metric for metric in eligible if metric.symbol == MARKET_BENCHMARK), None)
        ordered = (
            [benchmark] + [metric for metric in eligible if metric.symbol != MARKET_BENCHMARK]
            if benchmark is not None
            else eligible
        )
        selected = tuple(metric.symbol for metric in ordered[: cfg.max_scan_symbols])
        selected_set = set(selected)
        metrics = [
            EquityUniverseMetric(
                **{
                    **asdict(metric),
                    "eligible": metric.symbol in selected_set if metric.eligible else False,
                    "reason": (
                        "eligible"
                        if metric.symbol in selected_set
                        else "beyond_scan_cap"
                        if metric.eligible
                        else metric.reason
                    ),
                }
            )
            for metric in metrics
        ]
        return EquityUniverseReport(
            asof=now,
            discovered=len(assets),
            selected=selected,
            metrics=metrics,
        )


class OwnershipAwareStrategy:
    """Keep real strategy ownership while preserving SHADOW virtual-position context.

    ``QuantScanner`` passes a strategy-specific held set in SHADOW but the whole broker
    book in PAPER. When this strategy already owns real positions, its explicit durable
    ownership set is authoritative. When it owns none, the scanner-provided held set is
    passed through so SHADOW positions can generate strategy exits. PAPER execution has
    a second ownership check immediately before any sell, so a pass-through symbol can
    never cause one strategy to liquidate another strategy's or a manual position.
    """

    def __init__(self, delegate, owned_symbols: set[str] | frozenset[str]):
        self.delegate = delegate
        self.owned_symbols = frozenset(owned_symbols)
        self.config = delegate.config

    def required_bars(self):
        return self.delegate.required_bars()

    def prepare_symbol(self, bars):
        return self.delegate.prepare_symbol(bars)

    def prepare_universe(self, bars_by_symbol):
        return self.delegate.prepare_universe(bars_by_symbol)

    def _effective_held(self, held_symbols):
        return self.owned_symbols if self.owned_symbols else frozenset(held_symbols)

    def evaluate(self, bars_by_symbol, held_symbols=frozenset()):
        return self.delegate.evaluate(bars_by_symbol, self._effective_held(held_symbols))

    def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
        return self.delegate.evaluate_prepared(
            prepared,
            timestamp,
            self._effective_held(held_symbols),
        )

    def ratchet_stop(self, current_stop, highest_completed_close, completed_close, current_atr):
        return self.delegate.ratchet_stop(
            current_stop,
            highest_completed_close,
            completed_close,
            current_atr,
        )


def _shards(symbols: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [tuple(symbols[index : index + size]) for index in range(0, len(symbols), size)]


def build_equity_scan_strategies(
    symbols: tuple[str, ...],
    *,
    config: MarketWideEquityConfig,
    owned_by_strategy: Optional[Mapping[str, set[str] | frozenset[str]]] = None,
) -> list[ScanStrategy]:
    """Build all enabled US-equity intraday hypotheses over bounded shards.

    Strategy ids/versions stay identical across shards: a shard is only a data/compute
    partition, not a different hypothesis. Symbols occur in exactly one shard, so
    ``strategy_id:symbol`` remains a unique scanner key.
    """

    owned_by_strategy = owned_by_strategy or {}
    entries: list[ScanStrategy] = []
    for shard in _shards(symbols, config.shard_size):
        if not shard:
            continue
        benchmark = MARKET_BENCHMARK if MARKET_BENCHMARK in shard else shard[0]
        families = (
            (
                "opening_range_breakout",
                "1.0.0",
                OpeningRangeBreakoutStrategy(
                    OpeningRangeConfig(
                        symbols=shard,
                        market=Market.US_EQUITY,
                        timeframe=config.timeframe,
                        benchmark_symbol=benchmark,
                        min_volume_ratio=1.0,
                        min_or_volume_ratio=1.4,
                        max_atr_percentile=97.0,
                    )
                ),
            ),
            (
                "vwap_reversion",
                "1.0.0",
                VwapReversionStrategy(
                    VwapReversionConfig(
                        symbols=shard,
                        market=Market.US_EQUITY,
                        timeframe=config.timeframe,
                        benchmark_symbol=benchmark,
                        min_volume_ratio=0.6,
                        max_atr_percentile=97.0,
                    )
                ),
            ),
            (
                "volatility_squeeze",
                "1.0.0",
                VolatilitySqueezeStrategy(
                    VolatilitySqueezeConfig(
                        symbols=shard,
                        market=Market.US_EQUITY,
                        timeframe=config.timeframe,
                        benchmark_symbol=benchmark,
                        min_volume_ratio=0.8,
                        min_expansion_volume_ratio=1.4,
                        max_atr_percentile=97.0,
                    )
                ),
            ),
        )
        for strategy_id, version, strategy in families:
            wrapped = OwnershipAwareStrategy(
                strategy,
                owned_by_strategy.get(strategy_id, frozenset()),
            )
            entries.append(
                ScanStrategy(
                    strategy_id=strategy_id,
                    strategy=wrapped,
                    market=Market.US_EQUITY,
                    timeframe=config.timeframe,
                    version=version,
                    base_risk_pct=config.risk_per_trade_pct,
                )
            )
    return entries


def required_intraday_bars(strategies: list[ScanStrategy], configured_floor: int) -> int:
    required = configured_floor
    for entry in strategies:
        required = max(required, int(entry.strategy.required_bars()) + 5)
    return required
