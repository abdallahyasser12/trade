"""Hand-calculable no-look-ahead, stop-gap, fee, and equity-accounting tests."""

import pandas as pd
import pytest

from crypto_backtest import run_crypto_backtest
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig
from tests.crypto_helpers import breakout_bars, compact_crypto_config


def backtest_config(bars, **overrides):
    values = dict(
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date() + pd.Timedelta(days=1),
        initial_capital=10_000.0,
        slippage_bps=0.0,
        taker_fee_bps=0.0,
        spread_bps=0.0,
        liquidate_at_end=True,
    )
    values.update(overrides)
    return CryptoBacktestConfig(**values)


def test_signal_bar_executes_only_at_next_bar_open():
    bars = breakout_bars(periods=33)
    result = run_crypto_backtest(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest_config(bars),
    )
    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.entry_time == bars.index[31].to_pydatetime()
    assert trade.entry_price == pytest.approx(bars.iloc[31]["open"])
    assert trade.entry_time > bars.index[30].to_pydatetime()


def test_gap_below_atr_stop_fills_at_adverse_gap_open_not_stop():
    bars = breakout_bars(periods=35, gap_at=32)
    result = run_crypto_backtest(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest_config(bars),
    )
    trade = result.trades[0]
    assert trade.exit_reason == "atr_stop_gap"
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.exit_price < trade.entry_price


def test_normal_intrabar_stop_uses_stop_not_bar_low():
    bars = breakout_bars(periods=35)
    bars.iloc[32, bars.columns.get_loc("open")] = 111.0
    bars.iloc[32, bars.columns.get_loc("close")] = 110.0
    bars.iloc[32, bars.columns.get_loc("high")] = 112.0
    bars.iloc[32, bars.columns.get_loc("low")] = 90.0
    result = run_crypto_backtest(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest_config(bars),
    )
    trade = result.trades[0]
    assert trade.exit_reason == "atr_stop"
    assert 90.0 < trade.exit_price < trade.entry_price


def test_cash_plus_market_value_accounting_has_no_double_counting():
    bars = breakout_bars(periods=33)
    result = run_crypto_backtest(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest_config(bars),
    )
    assert result.ending_equity == pytest.approx(
        result.starting_equity + result.realized_pnl
    )
    assert result.total_return_pct == pytest.approx(
        result.realized_pnl / result.starting_equity * 100.0
    )


def test_percentage_fees_are_debited_once_per_entry_and_exit():
    bars = breakout_bars(periods=33)
    result = run_crypto_backtest(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest_config(bars, taker_fee_bps=100.0),
    )
    trade = result.trades[0]
    assert result.total_fees == pytest.approx(trade.entry_fee + trade.exit_fee)
    gross_pnl = (trade.exit_price - trade.entry_price) * trade.qty
    assert trade.pnl == pytest.approx(gross_pnl - result.total_fees)


def test_unsorted_duplicate_or_malformed_bars_are_rejected():
    bars = breakout_bars()
    duplicate = pd.concat([bars, bars.iloc[[-1]]])
    with pytest.raises(ValueError, match="de-duplicated"):
        run_crypto_backtest(
            {"BTC/USD": duplicate},
            compact_crypto_config(),
            CryptoRiskConfig(max_positions=1),
            backtest_config(bars),
        )


def test_multi_asset_backtest_rejects_timestamp_misalignment_instead_of_hiding_gaps():
    btc = breakout_bars()
    eth = btc.iloc[1:].copy()
    config = compact_crypto_config(symbols=("BTC/USD", "ETH/USD"))
    with pytest.raises(ValueError, match="exactly timestamp-aligned"):
        run_crypto_backtest(
            {"BTC/USD": btc, "ETH/USD": eth},
            config,
            CryptoRiskConfig(max_positions=2),
            backtest_config(btc),
        )
