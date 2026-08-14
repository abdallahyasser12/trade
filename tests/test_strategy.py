"""Unit tests for strategy.TrendStrategy.

``generate_signal`` uses indicator columns directly when they are already
present on the input frame, so these tests hand-craft exact indicator
values row by row.  Only the last two rows drive the decision; earlier
rows are identical padding to satisfy ``required_bars()``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from models import BotAction, SignalDecision
from strategy import TrendStrategy


def make_config(**overrides) -> StrategyConfig:
    """Small, valid strategy config (required_bars() == 6)."""
    params = dict(
        symbol="TEST",
        timeframe="1Day",
        short_ma_period=3,
        long_ma_period=5,
        rsi_period=3,
        atr_period=3,
        volume_period=3,
        enable_volatility_filter=False,
    )
    params.update(overrides)
    return StrategyConfig(**params)


def make_volatility_config(**overrides) -> StrategyConfig:
    """Config with the ATR-percentile filter enabled (required_bars() == 14)."""
    params = dict(
        enable_volatility_filter=True,
        atr_percentile_lookback=10,
        max_atr_percentile=90.0,
    )
    params.update(overrides)
    return make_config(**params)


# Neutral bar used to pad the frame: no crossover, all filters passing.
BASE_ROW = dict(
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1_500.0,
    short_ma=10.0,
    long_ma=10.0,
    rsi=50.0,
    atr=1.0,
    avg_volume=1_000.0,
    volume_ratio=1.5,
    atr_percentile=50.0,
    adx=30.0,
    efficiency_ratio=0.50,
)


def make_frame(n_rows: int, prev_row: dict | None = None, last_row: dict | None = None) -> pd.DataFrame:
    """Build a frame of ``n_rows`` identical bars, overriding the last two.

    ``prev_row``/``last_row`` are partial overrides applied to the
    second-to-last and last rows respectively.  Index is tz-aware UTC.
    """
    rows = [dict(BASE_ROW) for _ in range(n_rows)]
    if n_rows >= 2 and prev_row:
        rows[-2].update(prev_row)
    if n_rows >= 1 and last_row:
        rows[-1].update(last_row)
    index = pd.date_range("2024-01-01", periods=n_rows, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def bullish_frame(n_rows: int = 6, prev_extra: dict | None = None, last_extra: dict | None = None) -> pd.DataFrame:
    """Frame ending in a fresh bullish crossover with every filter passing."""
    prev = dict(short_ma=9.5, long_ma=10.0)
    last = dict(short_ma=10.5, long_ma=10.0, rsi=55.0)
    prev.update(prev_extra or {})
    last.update(last_extra or {})
    return make_frame(n_rows, prev_row=prev, last_row=last)


class TestBullishCrossover:
    def test_buy_on_confirmed_bullish_crossover(self):
        """prev short<=long, curr short>long, all filters pass -> BUY."""
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(bullish_frame())

        assert decision.action == BotAction.BUY
        assert decision.reason == "bullish_crossover_confirmed"
        assert decision.bullish_crossover is True
        assert decision.bearish_crossover is False
        assert decision.filters == {
            "trend": True,
            "rsi": True,
            "volume": True,
            "volatility": True,
        }

    def test_buy_decision_echoes_indicator_values(self):
        """The decision carries the exact indicator values it was based on."""
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(bullish_frame())

        assert decision.short_ma == 10.5
        assert decision.long_ma == 10.0
        assert decision.rsi == 55.0
        assert decision.atr == 1.0
        assert decision.volume == 1_500.0
        assert decision.average_volume == 1_000.0
        assert decision.volume_ratio == 1.5
        assert decision.atr_percentile == 50.0
        assert decision.details["prev_short_ma"] == 9.5
        assert decision.details["prev_long_ma"] == 10.0
        assert decision.details["close"] == 100.0

    def test_buy_timestamp_is_last_bar_utc(self):
        """The decision timestamp is the last completed bar's tz-aware time."""
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(bullish_frame(n_rows=6))

        assert decision.timestamp is not None
        assert decision.timestamp.tzinfo is not None
        assert decision.timestamp == datetime(2024, 1, 6, tzinfo=timezone.utc)

    def test_exact_touch_crossover_counts_as_bullish(self):
        """prev short exactly equal to prev long still qualifies as a cross."""
        strategy = TrendStrategy(make_config())
        frame = bullish_frame(prev_extra=dict(short_ma=10.0, long_ma=10.0))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.BUY
        assert decision.reason == "bullish_crossover_confirmed"


class TestNoRepeatedBuy:
    def test_hold_when_already_in_bullish_state(self):
        """short>long on both bars is a trend, not a fresh cross -> HOLD."""
        strategy = TrendStrategy(make_config())
        frame = make_frame(
            6,
            prev_row=dict(short_ma=10.5, long_ma=10.0),
            last_row=dict(short_ma=10.6, long_ma=10.0),
        )
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "no_crossover"
        assert decision.bullish_crossover is False
        assert decision.filters["trend"] is False

    def test_ten_bars_remaining_bullish_emit_no_repeated_buy(self):
        strategy = TrendStrategy(make_config())
        rows = [dict(BASE_ROW) for _ in range(15)]
        rows[4].update(short_ma=9.5, long_ma=10.0)
        for i in range(5, 15):
            rows[i].update(short_ma=10.5 + i / 100.0, long_ma=10.0)
        frame = pd.DataFrame(
            rows,
            index=pd.date_range("2024-01-01", periods=15, freq="D", tz="UTC"),
        )
        actions = [strategy.generate_signal(frame.iloc[:i]).action for i in range(6, 16)]
        assert actions == [BotAction.BUY] + [BotAction.HOLD] * 9


class TestBearishCrossover:
    def test_exit_on_bearish_crossover(self):
        """prev short>=long, curr short<long -> EXIT."""
        strategy = TrendStrategy(make_config())
        frame = make_frame(
            6,
            prev_row=dict(short_ma=10.5, long_ma=10.0),
            last_row=dict(short_ma=9.5, long_ma=10.0),
        )
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.EXIT
        assert decision.reason == "bearish_crossover"
        assert decision.bearish_crossover is True
        assert decision.bullish_crossover is False

    def test_exit_ignores_entry_filters(self):
        """A bearish cross exits even when entry filters would fail."""
        strategy = TrendStrategy(make_config())
        frame = make_frame(
            6,
            prev_row=dict(short_ma=10.5, long_ma=10.0),
            last_row=dict(short_ma=9.5, long_ma=10.0, rsi=95.0, volume_ratio=0.1),
        )
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.EXIT
        assert decision.reason == "bearish_crossover"


class TestEntryFilters:
    def test_rsi_overbought_blocks_entry(self):
        """RSI at/above the overbought threshold blocks an otherwise-valid BUY."""
        cfg = make_config()
        strategy = TrendStrategy(cfg)
        frame = bullish_frame(last_extra=dict(rsi=cfg.rsi_overbought))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.filters["rsi"] is False
        assert decision.filters["trend"] is True
        assert decision.filters["volume"] is True
        assert decision.filters["volatility"] is True
        assert "rsi" in decision.reason


class TestRobustTrendFilters:
    @staticmethod
    def _smart_config(**overrides):
        params = dict(
            enable_rsi_floor=True,
            rsi_entry_floor=45.0,
            enable_adx_filter=True,
            adx_period=2,
            min_adx=20.0,
            enable_efficiency_filter=True,
            efficiency_period=3,
            min_efficiency_ratio=0.2,
            enable_market_regime_filter=True,
            regime_symbol="MKT",
            regime_timeframe="1Day",
            regime_ma_period=3,
            regime_slope_period=1,
        )
        params.update(overrides)
        return make_config(**params)

    @staticmethod
    def _market(closes, end="2024-01-05"):
        index = pd.date_range(end=end, periods=len(closes), freq="D", tz="UTC")
        return pd.DataFrame({"close": closes}, index=index)

    def test_all_robust_filters_pass_on_clean_bullish_regime(self):
        strategy = TrendStrategy(self._smart_config())
        frame = bullish_frame(n_rows=strategy.required_bars())
        market = self._market([90, 91, 92, 93, 94, 95])
        decision = strategy.generate_signal(frame, market_data=market)

        assert decision.action == BotAction.BUY
        assert decision.filters["adx"] is True
        assert decision.filters["efficiency"] is True
        assert decision.filters["market_regime"] is True
        assert decision.market_regime_bullish is True

    def test_weak_adx_blocks_otherwise_valid_crossover(self):
        strategy = TrendStrategy(self._smart_config())
        frame = bullish_frame(
            n_rows=strategy.required_bars(), last_extra={"adx": 19.99}
        )
        market = self._market([90, 91, 92, 93, 94, 95])
        decision = strategy.generate_signal(frame, market_data=market)

        assert decision.action == BotAction.HOLD
        assert decision.filters["adx"] is False
        assert "adx" in decision.reason

    def test_missing_benchmark_fails_closed(self):
        strategy = TrendStrategy(self._smart_config())
        frame = bullish_frame(n_rows=strategy.required_bars())
        decision = strategy.generate_signal(frame, market_data=None)
        assert decision.action == BotAction.HOLD
        assert decision.reason == "market_context_unavailable"
        assert decision.details["detail"] == "benchmark_bars_missing"

    def test_new_bearish_market_regime_transition_exits(self):
        strategy = TrendStrategy(self._smart_config())
        frame = make_frame(
            strategy.required_bars(),
            prev_row={"short_ma": 10.5, "long_ma": 10.0},
            last_row={"short_ma": 10.6, "long_ma": 10.0},
        )
        market = self._market([90, 91, 92, 93, 94, 80])
        decision = strategy.generate_signal(frame, market_data=market)
        assert decision.action == BotAction.EXIT
        assert decision.reason == "market_regime_break"

    def test_future_benchmark_bar_cannot_change_current_decision(self):
        strategy = TrendStrategy(self._smart_config())
        frame = bullish_frame(n_rows=strategy.required_bars())
        available = self._market([90, 91, 92, 93, 94, 95])
        future = pd.DataFrame(
            {"close": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-07", tz="UTC")]),
        )
        with_future = pd.concat([available, future])
        first = strategy.generate_signal(frame, market_data=available)
        second = strategy.generate_signal(frame, market_data=with_future)
        assert second.action == first.action == BotAction.BUY
        assert second.regime_close == first.regime_close

    def test_equal_timeframe_benchmark_bar_is_visible_after_signal_bar_closes(self):
        strategy = TrendStrategy(self._smart_config())
        frame = bullish_frame(n_rows=strategy.required_bars())
        market = self._market([90, 91, 92, 93, 94, 80], end="2024-01-06")

        decision = strategy.generate_signal(frame, market_data=market)

        assert decision.action == BotAction.EXIT
        assert decision.reason == "market_regime_break"
        assert decision.regime_close == 80.0

    def test_relative_strength_requires_timestamp_aligned_benchmark_bars(self):
        strategy = TrendStrategy(
            self._smart_config(
                enable_market_regime_filter=False,
                enable_relative_strength_filter=True,
                relative_strength_period=2,
            )
        )
        frame = bullish_frame(n_rows=strategy.required_bars())
        market = pd.DataFrame(
            {"close": [90, 91, 92, 93, 94]},
            index=frame.index[:-1],
        )

        decision = strategy.generate_signal(frame, market_data=market)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "market_context_unavailable"
        assert decision.details["detail"] == "relative_strength_bars_not_aligned"

    def test_low_volume_ratio_blocks_entry(self):
        """volume_ratio below the minimum blocks an otherwise-valid BUY."""
        cfg = make_config()
        strategy = TrendStrategy(cfg)
        frame = bullish_frame(last_extra=dict(volume_ratio=cfg.min_volume_ratio - 0.01))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.filters["volume"] is False
        assert decision.filters["trend"] is True
        assert decision.filters["rsi"] is True
        assert decision.filters["volatility"] is True
        assert "volume" in decision.reason

    def test_high_atr_percentile_blocks_entry_when_filter_enabled(self):
        """atr_percentile above the maximum blocks entry with the filter on."""
        cfg = make_volatility_config()
        strategy = TrendStrategy(cfg)
        frame = bullish_frame(
            n_rows=strategy.required_bars(),
            last_extra=dict(atr_percentile=cfg.max_atr_percentile + 5.0),
        )
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.filters["volatility"] is False
        assert decision.filters["trend"] is True
        assert decision.filters["rsi"] is True
        assert decision.filters["volume"] is True
        assert "volatility" in decision.reason

    def test_high_atr_percentile_ignored_when_filter_disabled(self):
        """With the volatility filter off, extreme ATR percentile still BUYs."""
        strategy = TrendStrategy(make_config())
        frame = bullish_frame(last_extra=dict(atr_percentile=99.9))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.BUY
        assert decision.filters["volatility"] is True


class TestWarmup:
    def test_hold_when_fewer_rows_than_required(self):
        """One bar short of required_bars() -> HOLD insufficient_data."""
        strategy = TrendStrategy(make_config())
        n = strategy.required_bars() - 1
        decision = strategy.generate_signal(bullish_frame(n_rows=n))

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"
        assert decision.details["bars_available"] == n
        assert decision.details["bars_required"] == strategy.required_bars()
        assert decision.filters == {
            "trend": False,
            "rsi": False,
            "volume": False,
            "volatility": False,
        }

    def test_zero_atr_is_invalid_and_never_emits_buy(self):
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(bullish_frame(last_extra={"atr": 0.0}))

        assert decision.action == BotAction.HOLD
        assert decision.reason == "invalid_indicator_values"
        assert decision.details["detail"] == "indicator_out_of_range"

    def test_hold_on_empty_frame(self):
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(pd.DataFrame())

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"
        assert decision.timestamp is None
        assert decision.details["bars_available"] == 0

    def test_hold_on_none_input(self):
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(None)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"


class TestNaNIndicators:
    def test_nan_indicator_on_last_row_holds(self):
        """A NaN indicator on the decision bar must never produce a trade."""
        strategy = TrendStrategy(make_config())
        frame = bullish_frame(last_extra=dict(rsi=np.nan))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"
        assert decision.details == {"detail": "indicators_not_initialized"}
        assert decision.rsi is None

    def test_nan_ma_on_prev_row_holds(self):
        """A NaN previous-bar MA also blocks (crossover needs both bars)."""
        strategy = TrendStrategy(make_config())
        frame = bullish_frame(prev_extra=dict(long_ma=np.nan))
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"

    def test_nan_atr_percentile_holds_when_filter_enabled(self):
        """With the volatility filter on, NaN atr_percentile blocks trading."""
        strategy = TrendStrategy(make_volatility_config())
        frame = bullish_frame(
            n_rows=strategy.required_bars(),
            last_extra=dict(atr_percentile=np.nan),
        )
        decision = strategy.generate_signal(frame)

        assert decision.action == BotAction.HOLD
        assert decision.reason == "insufficient_data"


class TestConstructorValidation:
    def test_short_ma_equal_to_long_ma_raises(self):
        with pytest.raises(ValueError, match="Short MA period must be smaller"):
            TrendStrategy(make_config(short_ma_period=5, long_ma_period=5))

    def test_short_ma_greater_than_long_ma_raises(self):
        with pytest.raises(ValueError):
            TrendStrategy(make_config(short_ma_period=10, long_ma_period=5))

    def test_valid_config_constructs(self):
        strategy = TrendStrategy(make_config())
        assert strategy.required_bars() == 6


class TestSignalDecisionSerialization:
    def test_buy_decision_round_trips_through_dict(self):
        """to_dict()/from_dict() preserves every field of a real decision."""
        strategy = TrendStrategy(make_config())
        decision = strategy.generate_signal(bullish_frame())
        assert decision.action == BotAction.BUY  # meaningful payload

        restored = SignalDecision.from_dict(decision.to_dict())

        assert restored == decision
        assert restored.action is BotAction.BUY
        assert restored.timestamp == decision.timestamp
        assert restored.timestamp.tzinfo is not None
        assert restored.filters == decision.filters
        assert restored.details == decision.details

    def test_hold_decision_with_none_timestamp_round_trips(self):
        decision = SignalDecision(action=BotAction.HOLD, timestamp=None, reason="insufficient_data")
        restored = SignalDecision.from_dict(decision.to_dict())

        assert restored == decision
        assert restored.timestamp is None

    def test_to_dict_uses_plain_serializable_values(self):
        """Serialized action is the raw string and timestamp is ISO-8601."""
        ts = datetime(2024, 3, 1, 15, 30, tzinfo=timezone.utc)
        decision = SignalDecision(action=BotAction.EXIT, timestamp=ts, reason="bearish_crossover")
        d = decision.to_dict()

        assert d["action"] == "EXIT"
        assert d["timestamp"] == "2024-03-01T15:30:00+00:00"
        assert datetime.fromisoformat(d["timestamp"]) == ts
