"""Tests for backtest.py: run_backtest, compute_max_drawdown, compute_sharpe.

All price data is small, synthetic and deterministic.  The strategy config
uses tiny indicator periods (short MA 2 / long MA 3, RSI 2, ATR 2, volume 2)
with the volatility filter disabled, so warmup_bars() == 4 and the whole
simulation fits in ~10 bars.

Hand-verified fixture arithmetic (used throughout):

* closes decline 100 -> 97 -> 94 -> 91 then jump to 97.5, so the 2-bar SMA
  crosses above the 3-bar SMA exactly on bar index 4 (the SIGNAL bar);
* RSI(2) on bar 4 is ~68.42 (< 70 overbought), volume ratio is 1.0, so the
  BUY is confirmed;
* Wilder ATR(2) on bar 4 is exactly 5.75 (highs/lows are close +/- 1);
* the order therefore fills on bar index 5 (the ENTRY bar) at that bar's
  open, 97.75, and the protective stop is 97.75 - 2.0 * 5.75 = 86.25;
* position size: risk budget 1% of 100_000 = 1000, risk/share 11.5,
  floor(1000 / 11.5) = 86 shares (the binding cap).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import backtest as backtest_mod
from backtest import MIN_SHARPE_OBSERVATIONS, compute_max_drawdown, compute_sharpe, run_backtest
from config import BacktestConfig, RiskConfig, StrategyConfig, bars_per_year

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

SIGNAL_BAR = 4  # bar index where the bullish crossover is detected
ENTRY_BAR = 5  # bar index where the resulting order fills (next-bar open)
ENTRY_OPEN = 97.75  # raw open of the entry bar
ATR_AT_SIGNAL = 5.75  # Wilder ATR(2) on the signal bar
STOP_PRICE = ENTRY_OPEN - 2.0 * ATR_AT_SIGNAL  # 86.25 with the default 2x multiplier
ENTRY_QTY = 86  # floor(1000 risk budget / 11.5 risk per share)


def make_strategy_config(**overrides) -> StrategyConfig:
    params = dict(
        symbol="TEST",
        timeframe="1Day",
        short_ma_period=2,
        long_ma_period=3,
        rsi_period=2,
        atr_period=2,
        volume_period=2,
        min_volume_ratio=0.1,
        enable_volatility_filter=False,
    )
    params.update(overrides)
    return StrategyConfig(**params)


def make_backtest_config(**overrides) -> BacktestConfig:
    params = dict(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        initial_capital=100_000.0,
        slippage_bps=0.0,
        commission_per_trade=0.0,
    )
    params.update(overrides)
    return BacktestConfig(**params)


def make_df(closes, opens, highs=None, lows=None) -> pd.DataFrame:
    """OHLCV frame on a tz-aware daily UTC index with generous, constant volume."""
    n = len(closes)
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    index = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [float(o) for o in opens],
            "high": [float(h) for h in highs],
            "low": [float(l) for l in lows],
            "close": [float(c) for c in closes],
            "volume": [1_000_000.0] * n,
        },
        index=index,
    )


def entry_only_frame() -> pd.DataFrame:
    """Decline then steady rise: one BUY on bar 4, fill on bar 5, never exits."""
    closes = [100.0, 97.0, 94.0, 91.0, 97.5, 98.0, 99.0, 100.0, 101.0, 102.0]
    opens = [100.0, 97.5, 94.5, 91.5, 96.75, ENTRY_OPEN, 98.5, 99.5, 100.5, 101.5]
    return make_df(closes, opens)


def bearish_exit_frame() -> pd.DataFrame:
    """Entry on bar 5, bearish crossover on bar 6, exit fill on bar 7's open (91.8).

    Lows never reach the 86.25 stop, so the only exit path is the crossover.
    """
    closes = [100.0, 97.0, 94.0, 91.0, 97.5, 98.0, 92.0, 91.5, 91.0, 91.0]
    opens = [100.0, 97.5, 94.5, 91.5, 96.75, ENTRY_OPEN, 92.5, 91.8, 91.2, 91.0]
    return make_df(closes, opens)


def stop_frame(gap_open: bool) -> pd.DataFrame:
    """Entry on bar 5, then bar 6 crashes through the 86.25 stop.

    gap_open=False: bar 6 opens at 95 (above the stop) and its low is 80,
    so the fill is the stop price itself.  gap_open=True: bar 6 opens at 84,
    already below the stop, so the fill is that open (gap rule).
    """
    closes = [100.0, 97.0, 94.0, 91.0, 97.5, 98.0, 85.0, 85.0]
    opens = [100.0, 97.5, 94.5, 91.5, 96.75, ENTRY_OPEN, 95.0, 85.0]
    highs = [101.0, 98.0, 95.0, 92.0, 98.5, 99.0, 95.5, 86.0]
    lows = [99.0, 96.0, 93.0, 90.0, 96.5, 97.0, 80.0, 84.0]
    if gap_open:
        opens[6] = 84.0
    return make_df(closes, opens, highs, lows)


def run(df: pd.DataFrame, *, strategy=None, risk=None, bt=None):
    return run_backtest(
        df,
        strategy or make_strategy_config(),
        risk or RiskConfig(),
        bt or make_backtest_config(),
    )


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


class TestComputeMaxDrawdown:
    def test_known_curve(self):
        """[100,120,90,130]: worst point is 90 against peak 120 -> 25%."""
        equity = pd.Series([100.0, 120.0, 90.0, 130.0])
        assert compute_max_drawdown(equity) == pytest.approx(25.0)

    def test_empty_series_is_zero(self):
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_constant_series_is_zero(self):
        assert compute_max_drawdown(pd.Series([100.0] * 6)) == 0.0

    def test_monotonic_rise_is_zero(self):
        """Equity always at a fresh peak has no drawdown, and the result is
        clamped so it never comes back as a negative zero-ish value."""
        assert compute_max_drawdown(pd.Series([100.0, 101.0, 105.0, 110.0])) == 0.0


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------


def drifting_equity(n_points: int) -> pd.Series:
    """Deterministic positive-drift equity with nonzero return volatility:
    per-bar returns alternate +0.2% / +0.05%."""
    values = [100_000.0]
    for i in range(n_points - 1):
        r = 0.002 if i % 2 == 0 else 0.0005
        values.append(values[-1] * (1.0 + r))
    index = pd.date_range("2024-01-02", periods=n_points, freq="D", tz="UTC")
    return pd.Series(values, index=index)


class TestComputeSharpe:
    def test_too_few_observations_returns_none_with_factor(self):
        """A 10-point curve has 9 returns, below MIN_SHARPE_OBSERVATIONS,
        so the ratio is None but the annualization factor still comes back."""
        sharpe, factor = compute_sharpe(drifting_equity(10), "1Day")
        assert sharpe is None
        assert factor == bars_per_year("1Day") == 252.0

    def test_boundary_observation_count(self):
        """Exactly MIN_SHARPE_OBSERVATIONS returns is enough; one fewer is not."""
        just_enough = drifting_equity(MIN_SHARPE_OBSERVATIONS + 1)
        one_short = drifting_equity(MIN_SHARPE_OBSERVATIONS)
        assert compute_sharpe(just_enough, "1Day")[0] is not None
        assert compute_sharpe(one_short, "1Day")[0] is None

    def test_zero_volatility_returns_none(self):
        flat = pd.Series(
            [100_000.0] * 60,
            index=pd.date_range("2024-01-02", periods=60, freq="D", tz="UTC"),
        )
        sharpe, factor = compute_sharpe(flat, "1Day")
        assert sharpe is None
        assert factor == 252.0

    def test_factor_matches_config_for_5min(self):
        sharpe, factor = compute_sharpe(drifting_equity(5), "5Min")
        assert sharpe is None
        assert factor == bars_per_year("5Min") == 252 * 78.0

    def test_positive_drift_gives_positive_float(self):
        sharpe, factor = compute_sharpe(drifting_equity(300), "1Day")
        assert isinstance(sharpe, float)
        assert sharpe > 0.0
        assert factor == 252.0


# ---------------------------------------------------------------------------
# run_backtest: execution model
# ---------------------------------------------------------------------------


class TestNextBarExecution:
    def test_entry_fills_at_next_bar_open(self):
        """The crossover is detected on bar SIGNAL_BAR; the fill must land on
        bar SIGNAL_BAR+1 at that bar's raw open (slippage 0), never on the
        signal bar's close (98.0 here, vs. the 97.75 open)."""
        df = bearish_exit_frame()
        result = run(df)
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_time == df.index[ENTRY_BAR]
        assert trade.entry_price == pytest.approx(ENTRY_OPEN)
        assert trade.entry_price != pytest.approx(float(df["close"].iloc[SIGNAL_BAR]))
        assert trade.qty == ENTRY_QTY

    def test_altering_future_bars_does_not_change_entry(self):
        """No look-ahead: the three scenario frames are bar-for-bar identical
        through the entry bar and differ only AFTERWARDS (crossover exit,
        stop through the low, gap-open stop, and even different lengths).
        The entry bar, entry fill and size must be identical in all of them,
        as must every equity point up to and including the entry bar."""
        frames = [bearish_exit_frame(), stop_frame(gap_open=False), stop_frame(gap_open=True)]
        for f in frames[1:]:
            pd.testing.assert_frame_equal(frames[0].iloc[: ENTRY_BAR + 1], f.iloc[: ENTRY_BAR + 1])

        results = [run(f) for f in frames]
        reference = results[0].trades[0]
        for r in results[1:]:
            trade = r.trades[0]
            assert trade.entry_time == reference.entry_time
            assert trade.entry_price == reference.entry_price
            assert trade.qty == reference.qty
            pd.testing.assert_series_equal(
                r.equity_curve.iloc[: ENTRY_BAR + 1],
                results[0].equity_curve.iloc[: ENTRY_BAR + 1],
            )


class TestStopExecution:
    def test_stop_fills_at_stop_price_when_open_above(self):
        """Bar 6 opens at 95 (above the 86.25 stop) and trades down through
        it -> the simulated fill is exactly the stop price."""
        df = stop_frame(gap_open=False)
        result = run(df)
        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == "atr_stop"
        assert trade.exit_time == df.index[ENTRY_BAR + 1]
        assert trade.exit_price == pytest.approx(STOP_PRICE)

    def test_gap_open_below_stop_fills_at_open(self):
        """Bar 6 opens at 84, already below the 86.25 stop -> you do not get
        the stop price, you get the (worse) opening print."""
        df = stop_frame(gap_open=True)
        result = run(df)
        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == "atr_stop"
        assert trade.exit_price == pytest.approx(84.0)
        assert trade.exit_price < STOP_PRICE


class TestSlippage:
    def test_slippage_worsens_both_sides(self):
        """With 100 bps slippage a BUY fills above the raw open and the
        bearish-crossover SELL fills below the raw open."""
        df = bearish_exit_frame()
        result = run(df, bt=make_backtest_config(slippage_bps=100.0))
        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == "bearish_crossover"
        assert trade.entry_price == pytest.approx(ENTRY_OPEN * 1.01)
        assert trade.entry_price > ENTRY_OPEN
        raw_exit_open = float(df["open"].iloc[7])
        assert trade.exit_price == pytest.approx(raw_exit_open * 0.99)
        assert trade.exit_price < raw_exit_open


class TestCommission:
    def test_round_trip_costs_two_commissions(self):
        """Same data, same fills: the only difference between the free run
        and the commissioned run is 2 * commission_per_trade of ending
        equity (entry order + exit order)."""
        commission = 7.5
        r_free = run(bearish_exit_frame())
        r_paid = run(
            bearish_exit_frame(), bt=make_backtest_config(commission_per_trade=commission)
        )
        assert r_free.num_trades == r_paid.num_trades == 1
        assert r_free.trades[0].qty == r_paid.trades[0].qty
        assert r_free.ending_equity - r_paid.ending_equity == pytest.approx(2 * commission)
        assert r_free.trades[0].pnl - r_paid.trades[0].pnl == pytest.approx(2 * commission)


# ---------------------------------------------------------------------------
# run_backtest: aggregates
# ---------------------------------------------------------------------------


class TestFlatMarket:
    def test_no_signals_means_flat_equity_and_zero_trades(self):
        """Constant closes never produce a crossover, so the equity curve is
        pinned at initial_capital for every bar."""
        closes = [100.0] * 12
        opens = [100.0] * 12
        df = make_df(closes, opens)
        result = run(df)
        assert result.num_trades == 0
        assert result.trades == []
        assert result.win_rate is None
        assert result.total_return_pct == pytest.approx(0.0)
        assert (result.equity_curve == 100_000.0).all()
        assert result.max_drawdown_pct == 0.0


class TestBenchmark:
    def test_benchmark_normalized_to_initial_capital(self):
        df = entry_only_frame()
        result = run(df)
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        assert result.benchmark_curve.iloc[0] == pytest.approx(100_000.0)
        assert result.benchmark_curve.iloc[-1] == pytest.approx(
            100_000.0 * last_close / first_close
        )
        pd.testing.assert_index_equal(result.benchmark_curve.index, df.index)


class TestWinRate:
    def test_open_position_excluded_from_num_trades_and_win_rate(self):
        """A position still open at the end of the data must not count toward
        num_trades or win_rate, yet its mark-to-market value must be in the
        equity curve: cash after the fill is 100000 - 86 * 97.75 = 91593.50,
        and the last close is 102, so ending equity is 91593.50 + 86 * 102."""
        result = run(entry_only_frame())
        assert result.num_trades == 0
        assert result.win_rate is None
        expected_cash = 100_000.0 - ENTRY_QTY * ENTRY_OPEN
        assert result.ending_equity == pytest.approx(expected_cash + ENTRY_QTY * 102.0)

    def test_open_position_recorded_as_open_trade(self):
        """A position still open when the data ends must appear in ``trades``
        as an open TradeRecord (exit fields None) and be counted in the
        ``open_trade_at_end`` snapshot field.

        models.TradeRecord has optional exit fields for this case; the open
        trade remains excluded from completed-trade statistics."""
        df = entry_only_frame()
        result = run(df)
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert not trade.is_closed
        assert trade.entry_time == df.index[ENTRY_BAR]
        assert trade.entry_price == pytest.approx(ENTRY_OPEN)
        assert trade.exit_time is None
        assert trade.exit_price is None
        assert result.config_snapshot["open_trade_at_end"] == 1
        # The open trade still must not count as a completed round trip.
        assert result.num_trades == 0
        assert result.win_rate is None

    def test_completed_losing_trade_counts(self):
        """The stop scenario closes one losing round trip: num_trades 1,
        win_rate 0%."""
        result = run(stop_frame(gap_open=False))
        assert result.num_trades == 1
        assert result.win_rate == pytest.approx(0.0)
        assert result.trades[0].pnl < 0
        assert result.realized_pnl == pytest.approx(result.trades[0].pnl)


# ---------------------------------------------------------------------------
# run_backtest: validation and determinism
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("mutation", ["reverse", "duplicate", "infinite"])
    def test_rejects_non_normalized_price_frames(self, mutation):
        df = entry_only_frame()
        if mutation == "reverse":
            df = df.iloc[::-1]
        elif mutation == "duplicate":
            df = pd.concat([df, df.iloc[[-1]]])
        else:
            df.loc[df.index[3], "high"] = float("inf")
        with pytest.raises(ValueError):
            run(df)

    def test_insufficient_history_raises(self):
        """warmup_bars() is 4 and run_backtest needs warmup + 2 bars, so a
        5-bar frame must be rejected."""
        closes = [100.0] * 5
        df = make_df(closes, closes)
        with pytest.raises(ValueError, match="Not enough historical data"):
            run(df)

    def test_invalid_backtest_config_raises(self):
        """end_date == start_date fails BacktestConfig.validate and must be
        rejected before any simulation happens."""
        bad = make_backtest_config(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 1)
        )
        with pytest.raises(ValueError, match="Invalid backtest configuration"):
            run(entry_only_frame(), bt=bad)


class TestDeterminism:
    def test_identical_inputs_identical_outputs(self):
        """Two runs over freshly built but identical inputs must agree on
        every equity point and every trade field."""
        r1 = run(bearish_exit_frame())
        r2 = run(bearish_exit_frame())
        pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
        pd.testing.assert_series_equal(r1.benchmark_curve, r2.benchmark_curve)
        assert [t.to_dict() for t in r1.trades] == [t.to_dict() for t in r2.trades]
        assert r1.total_return_pct == r2.total_return_pct
        assert r1.max_drawdown_pct == r2.max_drawdown_pct
        assert r1.sharpe_ratio == r2.sharpe_ratio

    def test_small_sample_constant_exposed(self):
        """The module exposes the observation floor the UI relies on."""
        assert MIN_SHARPE_OBSERVATIONS == 20
        assert backtest_mod.SMALL_SAMPLE_TRADE_THRESHOLD == 30


def test_backtest_sizes_at_signal_time_like_live_even_when_next_open_gaps():
    df = entry_only_frame()
    df.loc[df.index[ENTRY_BAR], ["open", "high", "low"]] = [120.0, 121.0, 97.0]
    result = run(
        df,
        risk=RiskConfig(risk_per_trade_pct=10.0, max_position_pct=10.0),
    )
    # Signal close is 97.50, so live sends floor(10,000 / 97.50)=102
    # shares. The backtest must retain that submitted quantity at N+1.
    assert result.trades[0].qty == 102


def test_trade_return_percentage_is_net_of_both_commissions():
    result = run(
        bearish_exit_frame(),
        bt=make_backtest_config(commission_per_trade=7.5),
    )
    trade = result.trades[0]
    invested = trade.entry_price * trade.qty + 7.5
    assert trade.return_pct == pytest.approx(trade.pnl / invested * 100.0)
