"""Streamlit dashboard for the Alpaca PAPER trading bot.

This file orchestrates presentation only — all trading logic lives in
strategy.py / risk_manager.py / trade_engine.py / backtest.py.  Run with:

    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import hashlib
import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest as backtest_module
import config as cfg
from backtest import run_backtest
from broker import AlpacaLiveBroker, AlpacaPaperBroker, BrokerError
from config import (
    AccountMode,
    BacktestConfig,
    EngineConfig,
    RiskConfig,
    StrategyConfig,
    load_credentials,
    live_trading_enabled,
    result_context_fingerprint,
)
from advanced_crypto_backtest import run_regime_ensemble_backtest
from advanced_crypto_engine import make_regime_aware_crypto_engine
from advanced_crypto_research import run_regime_ensemble_walk_forward
from cost_model import break_even_win_rate
from crypto_backtest import run_crypto_backtest
from crypto_config import (
    DEFAULT_CRYPTO_UNIVERSE,
    CryptoBacktestConfig,
    CryptoEngineConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)
from crypto_data_feed import AlpacaCryptoDataFeed
from crypto_engine import (
    CRYPTO_ENGINE_STATUS_KEY,
    CRYPTO_KILL_SWITCH_KEY,
    CryptoTradeEngine,
)
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    run_crypto_walk_forward,
)
from data_feed import AlpacaDataFeed, DataFeedError, DataSubscriptionError
from regime_ensemble import EnsembleConfig
from models import BotAction, EngineState, SignalDecision, StopReason
from risk_manager import RiskManager
from research import (
    PromotionCriteria,
    WalkForwardConfig,
    run_walk_forward,
)
from state_store import StateStore
from strategy import TrendStrategy
from trade_engine import (
    ENGINE_STATUS_KEY,
    KILL_SWITCH_KEY,
    LAST_SIGNAL_KEY,
    PDT_EQUITY_THRESHOLD,
    POSITION_SNAPSHOT_KEY_PREFIX,
    TradeEngine,
    get_controller,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title=cfg.APP_NAME, page_icon="📈", layout="wide")

PAPER_BANNER = "🧪 PAPER TRADING — NO REAL MONEY"

BACKTEST_DISCLAIMER = (
    "Historical simulation only. Backtest performance does not guarantee future "
    "results. Paper trading can behave differently from historical simulation."
)

SENSITIVITY_CAPTION = (
    "Results are specific to the exact parameters and date range chosen; small "
    "parameter changes can produce large result changes. This is a historical "
    "simulation of one parameter set over one period — not a forecast."
)


# ---------------------------------------------------------------------------
# Cached resources (survive Streamlit reruns; one per process)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_store() -> StateStore:
    return StateStore(cfg.DEFAULT_DB_PATH)


@st.cache_resource
def get_credentials():
    return load_credentials()


@st.cache_resource
def get_connection_check(api_key_fingerprint: str) -> dict:
    """One-time (per key fingerprint) verification that credentials reach a
    healthy PAPER account.  Returns plain data so reruns never re-hit Alpaca."""
    creds = get_credentials()
    if not creds.is_configured:
        return {"ok": False, "error": "credentials_missing", "account": None}
    try:
        broker = AlpacaPaperBroker(creds.api_key, creds.secret_key)
        account = broker.verify_paper_account()
        return {
            "ok": True,
            "error": None,
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "status": account.status,
                "currency": account.currency,
                "pattern_day_trader": account.pattern_day_trader,
                "daytrade_count": account.daytrade_count,
            },
        }
    except BrokerError as exc:
        return {"ok": False, "error": str(exc), "account": None}
    except Exception:  # pragma: no cover — defensive UI boundary
        return {
            "ok": False,
            "error": "Unexpected paper-account verification error.",
            "account": None,
        }


def make_broker() -> AlpacaPaperBroker:
    creds = get_credentials()
    return AlpacaPaperBroker(creds.api_key, creds.secret_key)


def make_data_feed(feed: str) -> AlpacaDataFeed:
    creds = get_credentials()
    return AlpacaDataFeed(creds.api_key, creds.secret_key, feed=feed)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def fmt_money(value, none_text: str = "—") -> str:
    if value is None:
        return none_text
    return f"${value:,.2f}"


def fmt_ts(ts, none_text: str = "—") -> str:
    if ts is None:
        return none_text
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pass_fail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def read_engine_status_kv(store: StateStore) -> dict | None:
    raw = store.get_kv(ENGINE_STATUS_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def read_last_signal(store: StateStore) -> SignalDecision | None:
    raw = store.get_kv(LAST_SIGNAL_KEY)
    if not raw:
        return None
    try:
        return SignalDecision.from_dict(json.loads(raw))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def read_kill_switch(store: StateStore) -> dict | None:
    raw = store.get_kv(KILL_SWITCH_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"reason": "UNKNOWN", "message": raw}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def build_sidebar(store: StateStore, conn: dict, strategy_config: StrategyConfig,
                  risk_config: RiskConfig, feed: str) -> None:
    with st.sidebar:
        st.error(PAPER_BANNER)
        st.caption(
            "The US-stock workspace connects exclusively to Alpaca paper trading. "
            "Crypto account mode is isolated in its own guarded tab."
        )

        creds = get_credentials()
        if creds.is_configured:
            if conn["ok"]:
                st.success(f"Alpaca paper account connected · key {creds.masked_key}")
            else:
                st.warning(f"Credentials found (key {creds.masked_key}) but not verified.")
        else:
            st.warning("Alpaca paper credentials not configured.")
        st.caption("Credentials live in the local `.env` file and are never displayed.")

        st.divider()
        st.markdown(f"**Symbol:** {strategy_config.symbol} · **Timeframe:** {strategy_config.timeframe}")
        st.markdown(
            f"**Data feed:** `{feed}`"
            + (" (IEX — free tier, partial market volume)" if feed == "iex" else " (SIP)")
        )
        st.markdown(
            f"**MAs:** {strategy_config.short_ma_period}/{strategy_config.long_ma_period} · "
            f"**RSI:** {strategy_config.rsi_period} (<{strategy_config.rsi_overbought:.0f}) · "
            f"**ATR:** {strategy_config.atr_period}×{strategy_config.atr_multiplier}"
        )
        st.markdown(
            f"**Risk/trade:** {risk_config.risk_per_trade_pct}% · "
            f"**Max position:** {risk_config.max_position_pct}% · "
            f"**Max session loss:** {risk_config.max_daily_loss_pct}%"
        )

        st.divider()
        controller = get_controller()
        if st.button(
            "🛑 EMERGENCY STOP",
            type="primary",
            use_container_width=True,
            help="Signals the engine to terminate immediately. Bot-owned protective "
            "stop orders are left in place at the broker so an open position keeps "
            "its protection.",
        ):
            ok, msg = controller.emergency_stop()
            if ok:
                st.toast(msg, icon="🛑")
            else:
                st.toast(msg, icon="ℹ️")


# ---------------------------------------------------------------------------
# Configuration inputs (Configure & Backtest tab)
# ---------------------------------------------------------------------------


def build_config_inputs() -> tuple[StrategyConfig, RiskConfig, BacktestConfig, EngineConfig]:
    st.subheader("Market")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        symbol = st.text_input("Symbol", value="SPY", help="One US equity ticker, e.g. SPY or AAPL.").strip().upper()
    with c2:
        timeframe = st.selectbox("Timeframe", cfg.TIMEFRAMES,
                                 index=cfg.TIMEFRAMES.index("1Day"),
                                 help="Bar size used for every decision. Intraday timeframes can trigger the pattern-day-trading rule on small accounts.")
    with c3:
        start_date = st.date_input(
            "Backtest start", value=date.today() - timedelta(days=365 * 5 + 30)
        )
    with c4:
        end_date = st.date_input("Backtest end", value=date.today() - timedelta(days=1))
    with c5:
        initial_capital = st.number_input("Starting capital ($)", min_value=1000.0,
                                          value=100_000.0, step=1000.0)

    st.subheader("Trend")
    t1, t2 = st.columns(2)
    with t1:
        short_ma = st.number_input("Short MA period", min_value=2, max_value=500, value=20,
                                   help="Bars in the fast simple moving average.")
    with t2:
        long_ma = st.number_input("Long MA period", min_value=3, max_value=1000, value=50,
                                  help="Bars in the slow simple moving average. Must be larger than the short MA.")

    st.subheader("Momentum (RSI)")
    m1, m2, m3 = st.columns(3)
    with m1:
        rsi_period = st.number_input("RSI period", min_value=2, max_value=100, value=14)
    with m2:
        rsi_overbought = st.number_input("RSI overbought threshold", min_value=1.0, max_value=100.0,
                                         value=70.0,
                                         help="Entries are skipped when RSI is at or above this — the move is already stretched.")
    with m3:
        rsi_oversold = st.number_input("RSI oversold threshold", min_value=0.0, max_value=99.0,
                                       value=30.0,
                                       help="Displayed for reference; the strategy has no RSI-based exit rule.")

    st.subheader("Volatility (ATR)")
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        atr_period = st.number_input("ATR period", min_value=2, max_value=100, value=14)
    with v2:
        atr_multiplier = st.number_input("ATR stop multiplier", min_value=0.1, max_value=10.0,
                                         value=2.0, step=0.1,
                                         help="Initial protective stop distance = ATR × this multiplier, below the actual fill price.")
    with v3:
        enable_vol_filter = st.checkbox("Volatility percentile filter", value=True,
                                        help="Skip entries when current ATR is unusually high versus its own recent history.")
    with v4:
        atr_lookback = st.number_input("ATR percentile lookback", min_value=10, max_value=1000,
                                       value=100, disabled=not enable_vol_filter)
    with v5:
        max_atr_pct = st.number_input("Maximum ATR percentile", min_value=1.0, max_value=100.0,
                                      value=90.0, disabled=not enable_vol_filter)

    st.subheader("Volume confirmation")
    vol1, vol2 = st.columns(2)
    with vol1:
        volume_period = st.number_input("Volume average period", min_value=1, max_value=500, value=20)
    with vol2:
        min_volume_ratio = st.number_input("Volume multiplier", min_value=0.1, max_value=10.0,
                                           value=1.0, step=0.1,
                                           help="Entry requires bar volume ≥ average volume × this multiplier.")
    st.caption(
        "On Alpaca's free IEX feed, volume covers only the IEX exchange — a fraction "
        "of total market volume — so the volume ratio is not comparable to figures on "
        "other platforms and this multiplier may need adjusting."
    )

    st.subheader("Robust trend filters")
    st.caption(
        "These filters reduce exposure to weak, noisy, or broad-market bearish "
        "crossovers. They are deterministic and use completed bars only."
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        enable_rsi_floor = st.checkbox("RSI momentum floor", value=True)
        rsi_entry_floor = st.number_input(
            "Minimum entry RSI",
            min_value=0.0,
            max_value=99.0,
            value=45.0,
            disabled=not enable_rsi_floor,
        )
    with q2:
        enable_adx = st.checkbox("ADX trend-strength filter", value=True)
        adx_period = st.number_input(
            "ADX period", min_value=2, max_value=100, value=14, disabled=not enable_adx
        )
        min_adx = st.number_input(
            "Minimum ADX", min_value=0.0, max_value=100.0, value=20.0,
            disabled=not enable_adx,
        )
    with q3:
        enable_efficiency = st.checkbox("Price-efficiency filter", value=True)
        efficiency_period = st.number_input(
            "Efficiency period", min_value=2, max_value=200, value=20,
            disabled=not enable_efficiency,
        )
        min_efficiency = st.number_input(
            "Minimum efficiency", min_value=0.0, max_value=1.0, value=0.20,
            step=0.05, disabled=not enable_efficiency,
        )
    with q4:
        enable_regime = st.checkbox("Broad-market regime filter", value=True)
        regime_symbol = st.text_input("Regime benchmark", value="SPY").strip().upper()
        regime_timeframe = st.selectbox(
            "Regime timeframe", cfg.TIMEFRAMES, index=4, disabled=not enable_regime
        )

    q5, q6, q7, q8 = st.columns(4)
    with q5:
        regime_ma_period = st.number_input(
            "Regime MA period", min_value=2, max_value=1000, value=200,
            disabled=not enable_regime,
        )
    with q6:
        regime_slope_period = st.number_input(
            "Regime slope lookback", min_value=1, max_value=250, value=20,
            disabled=not enable_regime,
        )
    with q7:
        exit_on_regime_break = st.checkbox(
            "Exit on new regime break", value=True, disabled=not enable_regime
        )
    with q8:
        relative_compatible = regime_timeframe == timeframe
        enable_relative_strength = st.checkbox(
            "Relative-strength filter",
            value=bool(symbol != regime_symbol and relative_compatible),
            disabled=not relative_compatible,
            help="Requires strategy and benchmark bars to use the same timeframe.",
        )
        relative_strength_period = st.number_input(
            "Relative-strength period", min_value=2, max_value=500, value=63,
            disabled=not enable_relative_strength,
        )
        min_relative_strength = st.number_input(
            "Minimum excess return", min_value=-1.0, max_value=10.0, value=0.0,
            step=0.01, disabled=not enable_relative_strength,
        )

    st.subheader("Risk and exits")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        risk_per_trade = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0,
                                         value=1.0, step=0.1,
                                         help="Equity fraction lost if the protective stop fills at its price.")
    with r2:
        max_position = st.number_input("Maximum position size (% of equity)", min_value=1.0,
                                       max_value=100.0, value=20.0, step=1.0)
    with r3:
        max_daily_loss = st.number_input("Maximum session loss (%)", min_value=0.5, max_value=100.0,
                                         value=3.0, step=0.5,
                                         help="Live: the session stops (no new entries; restart requires acknowledgement) "
                                              "once equity falls this far below the session start. "
                                              "Backtest: entries pause for the rest of that day, measured from the day's first equity.")
    with r4:
        min_position_value = st.number_input("Minimum position value ($)", min_value=0.0,
                                             value=100.0, step=50.0,
                                             help="Entries sized below this are rejected rather than rounded up.")

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        enable_adaptive_risk = st.checkbox(
            "Adaptive risk reduction", value=True,
            help="Can only reduce the configured risk; it never leverages above it.",
        )
        min_risk_scale = st.number_input(
            "Minimum risk scale", min_value=0.05, max_value=1.0, value=0.25,
            step=0.05, disabled=not enable_adaptive_risk,
        )
    with r6:
        drawdown_reduction_start = st.number_input(
            "Reduce risk after drawdown (%)", min_value=0.0,
            max_value=max(0.0, float(max_daily_loss) - 0.1), value=min(1.0, max(0.0, float(max_daily_loss) - 0.1)),
            step=0.1, disabled=not enable_adaptive_risk,
        )
        volatility_target = st.number_input(
            "ATR-percentile risk target", min_value=1.0, max_value=100.0,
            value=50.0, disabled=not enable_adaptive_risk,
        )
    with r7:
        enable_trailing_stop = st.checkbox("ATR trailing stop", value=True)
        trailing_atr_multiplier = st.number_input(
            "Trailing ATR multiplier", min_value=0.1, max_value=10.0, value=3.0,
            step=0.1, disabled=not enable_trailing_stop,
        )
    with r8:
        enable_break_even_stop = st.checkbox(
            "Break-even ratchet", value=True, disabled=not enable_trailing_stop
        )
        break_even_trigger_r = st.number_input(
            "Break-even trigger (R)", min_value=0.1, max_value=10.0, value=1.0,
            step=0.1, disabled=not enable_break_even_stop,
        )

    st.subheader("Simulation & data assumptions")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=0.0,
                                       step=0.5, help="Applied against you on every simulated fill.")
    with s2:
        commission = st.number_input("Commission per order ($)", min_value=0.0, value=0.0, step=0.05)
    with s3:
        risk_free_rate = st.number_input("Risk-free rate (annual, decimal)", min_value=-0.5,
                                         max_value=0.5, value=0.0, step=0.01,
                                         help="Used only in the Sharpe ratio.")
    with s4:
        feed = st.selectbox("Data feed", list(cfg.SUPPORTED_DATA_FEEDS), index=0,
                            help="IEX is the free tier (partial market volume). SIP needs a paid data subscription.")

    st.subheader("Live session safety")
    ls1, ls2, ls3, ls4 = st.columns(4)
    with ls1:
        block_near_close = st.checkbox("Block entries near close", value=True,
                                       help="No new entries in the final minutes of the regular session.")
    with ls2:
        close_buffer = st.number_input("Entry block buffer (min before close)", min_value=0,
                                       max_value=120, value=30, disabled=not block_near_close)
    with ls3:
        flat_before_close = st.checkbox("Exit position before close", value=False,
                                        help="OFF by default: positions are held overnight and exposed to opening-gap risk. "
                                             "ON: the bot sells its position shortly before the close.")
    with ls4:
        flat_minutes = st.number_input("Exit buffer (min before close)", min_value=1, max_value=60,
                                       value=10, disabled=not flat_before_close)

    strategy_config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        short_ma_period=int(short_ma),
        long_ma_period=int(long_ma),
        rsi_period=int(rsi_period),
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        atr_period=int(atr_period),
        atr_multiplier=float(atr_multiplier),
        volume_period=int(volume_period),
        min_volume_ratio=float(min_volume_ratio),
        enable_volatility_filter=bool(enable_vol_filter),
        atr_percentile_lookback=int(atr_lookback),
        max_atr_percentile=float(max_atr_pct),
        enable_rsi_floor=bool(enable_rsi_floor),
        rsi_entry_floor=float(rsi_entry_floor),
        enable_adx_filter=bool(enable_adx),
        adx_period=int(adx_period),
        min_adx=float(min_adx),
        enable_efficiency_filter=bool(enable_efficiency),
        efficiency_period=int(efficiency_period),
        min_efficiency_ratio=float(min_efficiency),
        enable_market_regime_filter=bool(enable_regime),
        regime_symbol=regime_symbol,
        regime_timeframe=regime_timeframe,
        regime_ma_period=int(regime_ma_period),
        regime_slope_period=int(regime_slope_period),
        exit_on_regime_break=bool(exit_on_regime_break),
        enable_relative_strength_filter=bool(
            enable_relative_strength and relative_compatible
        ),
        relative_strength_period=int(relative_strength_period),
        min_relative_strength=float(min_relative_strength),
        enable_trailing_stop=bool(enable_trailing_stop),
        trailing_atr_multiplier=float(trailing_atr_multiplier),
        enable_break_even_stop=bool(enable_break_even_stop),
        break_even_trigger_r=float(break_even_trigger_r),
    )
    risk_config = RiskConfig(
        risk_per_trade_pct=float(risk_per_trade),
        max_position_pct=float(max_position),
        max_daily_loss_pct=float(max_daily_loss),
        min_position_value=float(min_position_value),
        enable_adaptive_risk=bool(enable_adaptive_risk),
        min_risk_scale=float(min_risk_scale),
        drawdown_risk_reduction_start_pct=float(drawdown_reduction_start),
        volatility_target_percentile=float(volatility_target),
    )
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        slippage_bps=float(slippage_bps),
        commission_per_trade=float(commission),
        risk_free_rate=float(risk_free_rate),
        adjustment="split",
    )
    engine_config = EngineConfig(
        data_feed=feed,
        block_entries_near_close=bool(block_near_close),
        close_buffer_minutes=int(close_buffer),
        flat_before_close=bool(flat_before_close),
        flat_before_close_minutes=int(flat_minutes),
    )
    return strategy_config, risk_config, backtest_config, engine_config


def validate_all(strategy_config: StrategyConfig, risk_config: RiskConfig,
                 backtest_config: BacktestConfig | None,
                 engine_config: EngineConfig | None) -> list[str]:
    errors = strategy_config.validate() + risk_config.validate()
    if backtest_config is not None:
        errors += backtest_config.validate()
    if engine_config is not None:
        errors += engine_config.validate()
    return errors


# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------


def render_backtest_tab(strategy_config: StrategyConfig, risk_config: RiskConfig,
                        backtest_config: BacktestConfig, engine_config: EngineConfig) -> None:
    creds = get_credentials()
    context_key = result_context_fingerprint(
        strategy_config,
        risk_config,
        backtest_config,
        data_feed=engine_config.data_feed,
    )
    st.markdown("---")
    if not creds.is_configured:
        st.info(
            "Backtesting downloads historical bars from Alpaca, which requires "
            "paper credentials. Set up `.env` first (see the setup box above)."
        )

    if st.button("Run Backtest", type="primary", disabled=not creds.is_configured):
        errors = validate_all(strategy_config, risk_config, backtest_config, engine_config)
        if errors:
            for err in errors:
                st.error(err)
            return
        try:
            with st.spinner("Downloading history and simulating…"):
                feed = make_data_feed(engine_config.data_feed)
                start_dt = datetime.combine(backtest_config.start_date, datetime.min.time(),
                                            tzinfo=timezone.utc)
                end_dt = datetime.combine(backtest_config.end_date, datetime.max.time(),
                                          tzinfo=timezone.utc)
                bars = feed.get_bars(
                    strategy_config.symbol,
                    strategy_config.timeframe,
                    start=start_dt,
                    end=end_dt,
                    adjustment=backtest_config.adjustment,
                )
                market_bars = None
                needs_separate_market_data = (
                    strategy_config.enable_market_regime_filter
                    or strategy_config.enable_relative_strength_filter
                ) and not (
                    strategy_config.regime_symbol == strategy_config.symbol
                    and strategy_config.regime_timeframe == strategy_config.timeframe
                )
                if needs_separate_market_data:
                    market_bars = feed.get_bars(
                        strategy_config.regime_symbol,
                        strategy_config.regime_timeframe,
                        start=start_dt,
                        end=end_dt,
                        adjustment=backtest_config.adjustment,
                    )
                result = run_backtest(
                    bars,
                    strategy_config,
                    risk_config,
                    backtest_config,
                    market_data=market_bars,
                )
            st.session_state["backtest_result"] = result
            st.session_state["backtest_result_context"] = context_key
        except DataSubscriptionError as exc:
            st.error(str(exc))
            return
        except DataFeedError as exc:
            st.error(f"Could not download historical data: {exc}")
            return
        except ValueError as exc:
            st.error(str(exc))
            return
        except BrokerError as exc:
            st.error(str(exc))
            return

    result = st.session_state.get("backtest_result")
    if result is not None:
        if st.session_state.get("backtest_result_context") != context_key:
            st.warning(
                "Configuration changed after this backtest ran. Run it again before "
                "interpreting the result; stale results are intentionally hidden."
            )
        else:
            render_backtest_result(result)


def render_backtest_result(result) -> None:
    st.markdown("### Results")

    row1 = st.columns(4)
    row1[0].metric("Strategy return", f"{result.total_return_pct:+.2f}%")
    row1[1].metric("Buy & hold return", f"{result.benchmark_return_pct:+.2f}%")
    row1[2].metric("Ending equity", fmt_money(result.ending_equity))
    row1[3].metric("Realized P&L", fmt_money(result.realized_pnl))
    row2 = st.columns(4)
    row2[0].metric("Win rate", f"{result.win_rate:.1f}%" if result.win_rate is not None else "n/a")
    row2[1].metric("Max drawdown", f"-{result.max_drawdown_pct:.2f}%")
    row2[2].metric(
        "Sharpe ratio",
        f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "n/a",
        help="Annualized from per-bar equity returns.",
    )
    row2[3].metric("Completed trades", str(result.num_trades))

    st.caption(
        f"Sharpe assumptions: risk-free rate {result.risk_free_rate:.2%} (annual), "
        f"annualization factor {result.sharpe_annualization_factor:,.0f} bars/year "
        f"(derived from the {result.config_snapshot.get('timeframe', '?')} timeframe); "
        f"'n/a' means fewer than {backtest_module.MIN_SHARPE_OBSERVATIONS} return "
        "observations or zero volatility."
    )
    if result.num_trades < backtest_module.SMALL_SAMPLE_TRADE_THRESHOLD:
        st.warning(
            f"Only {result.num_trades} completed trade(s) — far below the "
            f"~{backtest_module.SMALL_SAMPLE_TRADE_THRESHOLD} needed for the win rate and "
            "Sharpe ratio to be statistically meaningful. Treat these numbers as noise."
        )
    st.caption(BACKTEST_DISCLAIMER)
    st.caption(SENSITIVITY_CAPTION)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values,
                             mode="lines", name="Strategy"))
    fig.add_trace(go.Scatter(x=result.benchmark_curve.index, y=result.benchmark_curve.values,
                             mode="lines", name="Buy & Hold"))
    fig.update_layout(
        title="Equity: strategy vs buy & hold (same starting capital)",
        xaxis_title="Time", yaxis_title="Equity ($)", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    extra = st.columns(4)
    extra[0].metric("Average win", fmt_money(result.avg_win, "n/a"))
    extra[1].metric("Average loss", fmt_money(result.avg_loss, "n/a"))
    extra[2].metric(
        "Profit factor",
        f"{result.profit_factor:.2f}" if result.profit_factor is not None else "n/a",
    )
    extra[3].metric("Best / worst trade",
                    f"{fmt_money(result.best_trade, 'n/a')} / {fmt_money(result.worst_trade, 'n/a')}")

    st.markdown("### Trades")
    if result.trades:
        rows = []
        for t in result.trades:
            rows.append({
                "Entry time": fmt_ts(t.entry_time),
                "Exit time": fmt_ts(t.exit_time, "(open)"),
                "Qty": t.qty,
                "Entry price": round(t.entry_price, 4),
                "Exit price": round(t.exit_price, 4) if t.exit_price is not None else None,
                "P&L ($)": round(t.pnl, 2) if t.pnl is not None else None,
                "Return (%)": round(t.return_pct, 2) if t.return_pct is not None else None,
                "Exit reason": t.exit_reason or "(open)",
                "Entry reason": t.entry_reason,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("The strategy generated no trades over this period with these parameters.")

    with st.expander("Simulation assumptions & configuration snapshot"):
        snap = result.config_snapshot
        st.markdown(
            f"- **Execution:** {snap.get('execution_model')}\n"
            f"- **Stop fills:** {snap.get('stop_model')}\n"
            f"- **Account model:** {snap.get('account_model')}\n"
            f"- **Bar adjustment:** {snap.get('bar_adjustment')} (corporate actions)\n"
            f"- **Slippage:** {snap.get('backtest', {}).get('slippage_bps', 0)} bps · "
            f"**Commission:** ${snap.get('backtest', {}).get('commission_per_trade', 0)}/order\n"
            f"- **Bars used:** {snap.get('bars_used')} (warmup {snap.get('warmup_bars')})"
        )
        st.json(snap)


def render_research_tab(
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    backtest_config: BacktestConfig,
    engine_config: EngineConfig,
) -> None:
    """Run a bounded walk-forward experiment; never auto-promote to live."""
    st.error(PAPER_BANNER)
    st.subheader("Chronological walk-forward validation")
    st.write(
        "Each fold chooses one configuration using only its training window, skips an "
        "embargo bar, and evaluates the choice on the following untouched window. "
        "The same out-of-sample window is also run with stressed costs and the fixed "
        "current configuration as a baseline. A PASS only permits supervised paper "
        "testing—it is not evidence of guaranteed profit."
    )
    creds = get_credentials()
    w1, w2, w3, w4 = st.columns(4)
    minimum_train = max(50, strategy_config.warmup_bars() + 25)
    with w1:
        train_bars = st.number_input(
            "Training bars", min_value=minimum_train, value=max(756, minimum_train), step=10
        )
    with w2:
        test_bars = st.number_input("Test bars", min_value=10, value=126, step=5)
    with w3:
        embargo_bars = st.number_input("Embargo bars", min_value=1, value=1, step=1)
    with w4:
        min_train_trades = st.number_input(
            "Minimum training trades", min_value=1, value=5, step=1
        )

    short_values = sorted(
        {max(2, round(strategy_config.short_ma_period * factor)) for factor in (0.8, 1.0, 1.2)}
    )
    long_values = sorted(
        {max(3, round(strategy_config.long_ma_period * factor)) for factor in (0.8, 1.0, 1.2)}
    )
    atr_values = sorted(
        {max(0.1, round(strategy_config.atr_multiplier + delta, 2)) for delta in (-0.5, 0.0, 0.5)}
    )
    candidate_count = len(short_values) * len(long_values) * len(atr_values)
    experiment = {
        "train_bars": int(train_bars),
        "test_bars": int(test_bars),
        "step_bars": int(test_bars),
        "embargo_bars": int(embargo_bars),
        "min_train_trades": int(min_train_trades),
        "short_ma_period": short_values,
        "long_ma_period": long_values,
        "atr_multiplier": atr_values,
    }
    context_key = result_context_fingerprint(
        strategy_config,
        risk_config,
        backtest_config,
        data_feed=engine_config.data_feed,
        experiment=experiment,
    )
    st.caption(
        f"Bounded grid: short MA {short_values}; long MA {long_values}; "
        f"initial ATR stop {atr_values}. Up to {candidate_count} combinations before "
        "invalid MA pairs are removed. Filter structure is held fixed to reduce data mining."
    )

    if st.button(
        "Run Walk-Forward Validation",
        type="primary",
        disabled=not creds.is_configured,
    ):
        errors = validate_all(strategy_config, risk_config, backtest_config, engine_config)
        if errors:
            for error in errors:
                st.error(error)
            return
        try:
            with st.spinner("Downloading history and running chronological folds…"):
                feed = make_data_feed(engine_config.data_feed)
                start_dt = datetime.combine(
                    backtest_config.start_date, datetime.min.time(), tzinfo=timezone.utc
                )
                end_dt = datetime.combine(
                    backtest_config.end_date, datetime.max.time(), tzinfo=timezone.utc
                )
                bars = feed.get_bars(
                    strategy_config.symbol,
                    strategy_config.timeframe,
                    start=start_dt,
                    end=end_dt,
                    adjustment=backtest_config.adjustment,
                )
                market_bars = None
                if (
                    strategy_config.enable_market_regime_filter
                    or strategy_config.enable_relative_strength_filter
                ) and not (
                    strategy_config.regime_symbol == strategy_config.symbol
                    and strategy_config.regime_timeframe == strategy_config.timeframe
                ):
                    market_bars = feed.get_bars(
                        strategy_config.regime_symbol,
                        strategy_config.regime_timeframe,
                        start=start_dt,
                        end=end_dt,
                        adjustment=backtest_config.adjustment,
                    )
                walk = WalkForwardConfig(
                    train_bars=int(train_bars),
                    test_bars=int(test_bars),
                    step_bars=int(test_bars),
                    embargo_bars=int(embargo_bars),
                    min_train_trades=int(min_train_trades),
                    max_candidates=64,
                )
                result = run_walk_forward(
                    bars,
                    strategy_config,
                    risk_config,
                    backtest_config,
                    {
                        "short_ma_period": short_values,
                        "long_ma_period": long_values,
                        "atr_multiplier": atr_values,
                    },
                    walk,
                    PromotionCriteria(),
                    market_data=market_bars,
                )
            st.session_state["walk_forward_result"] = result
            st.session_state["walk_forward_result_context"] = context_key
        except (DataFeedError, DataSubscriptionError, BrokerError, ValueError) as exc:
            st.error(str(exc))
            return

    result = st.session_state.get("walk_forward_result")
    if result is None:
        if not creds.is_configured:
            st.info("Add Alpaca paper credentials to download historical bars.")
        return
    if st.session_state.get("walk_forward_result_context") != context_key:
        st.warning(
            "Configuration or protocol controls changed after this experiment ran. "
            "Run it again; stale promotion evidence is intentionally hidden."
        )
        return

    if result.promotion.promoted:
        st.success("PROMOTION GATE: PASS — eligible for controlled forward paper testing.")
    else:
        st.warning("PROMOTION GATE: FAIL — do not promote this candidate configuration.")
        for failure in result.promotion.failures:
            st.markdown(f"- `{failure}`")

    metrics = st.columns(6)
    metrics[0].metric("OOS return", f"{result.compounded_return_pct:+.2f}%")
    metrics[1].metric(
        "Baseline OOS", f"{result.baseline_compounded_return_pct:+.2f}%"
    )
    metrics[2].metric("Stressed OOS", f"{result.stressed_compounded_return_pct:+.2f}%")
    metrics[3].metric(
        "Aggregate Sharpe",
        f"{result.aggregate_sharpe:.2f}" if result.aggregate_sharpe is not None else "n/a",
    )
    metrics[4].metric("OOS trades", str(result.total_oos_trades))
    metrics[5].metric("Positive folds", f"{result.positive_fold_ratio:.0%}")
    gate_metrics = st.columns(6)
    gate_metrics[0].metric(
        "Profit factor",
        f"{result.profit_factor:.2f}" if result.profit_factor is not None else "n/a",
    )
    gate_metrics[1].metric(
        "Worst fold drawdown", f"-{result.worst_fold_drawdown_pct:.2f}%"
    )
    gate_metrics[2].metric(
        "Excess vs baseline",
        f"{result.compounded_return_pct - result.baseline_compounded_return_pct:+.2f}%",
    )
    gate_metrics[3].metric(
        "Buy & hold OOS", f"{result.benchmark_compounded_return_pct:+.2f}%"
    )
    gate_metrics[4].metric(
        "Excess vs buy & hold",
        f"{result.benchmark_excess_return_pct:+.2f}%",
        help=(
            "The alternative a long-only book always has. Negative means "
            "holding the benchmark outperformed the strategy out of sample."
        ),
    )
    gate_metrics[5].metric(
        "Evidence", f"{len(result.folds)} folds · {result.candidate_count} candidates"
    )

    rows = []
    for number, fold in enumerate(result.folds, start=1):
        rows.append(
            {
                "Fold": number,
                "Train": f"{fold.train_start.date()} → {fold.train_end.date()}",
                "Test": f"{fold.test_start.date()} → {fold.test_end.date()}",
                "Short/long MA": (
                    f"{fold.selected_config.short_ma_period}/"
                    f"{fold.selected_config.long_ma_period}"
                ),
                "ATR stop": fold.selected_config.atr_multiplier,
                "OOS return %": round(fold.test_result.total_return_pct, 3),
                "Baseline %": round(fold.baseline_result.total_return_pct, 3),
                "Stressed %": round(fold.stressed_result.total_return_pct, 3),
                "Trades": fold.test_result.num_trades,
                "Max DD %": round(fold.test_result.max_drawdown_pct, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Protocol and latest training-selected configuration"):
        st.json(
            {
                "protocol": result.protocol,
                "latest_selected_strategy": result.latest_selected_config.to_dict(),
                "promotion_failures": result.promotion.failures,
            }
        )


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------


def render_live_tab(store: StateStore, conn: dict, strategy_config: StrategyConfig,
                    risk_config: RiskConfig, engine_config: EngineConfig) -> None:
    st.error(PAPER_BANNER)
    controller = get_controller()
    creds = get_credentials()

    # --- connection / account panel -----------------------------------
    a1, a2, a3, a4, a5 = st.columns(5)
    account = conn.get("account")
    a1.metric("Credentials", "Connected" if conn["ok"] else "Not connected")
    a2.metric("Account status", account["status"] if account else "—")
    a3.metric("Equity", fmt_money(account["equity"]) if account else "—")
    a4.metric("Cash", fmt_money(account["cash"]) if account else "—")
    a5.metric("Buying power", fmt_money(account["buying_power"]) if account else "—")

    if account:
        p1, p2 = st.columns(2)
        p1.metric("Pattern day trader flag", "YES" if account["pattern_day_trader"] else "no")
        p2.metric("Day trades used (5 business days)", str(account["daytrade_count"]))
        if (strategy_config.timeframe in cfg.INTRADAY_TIMEFRAMES
                and account["equity"] < PDT_EQUITY_THRESHOLD):
            st.warning(
                "Intraday timeframe on an account under $25,000: FINRA's pattern-day-trading "
                "rule limits day trades to 3 per 5 business days (Alpaca's paper environment "
                "simulates this). The bot blocks entries that would breach the limit "
                "(reason: pdt_restriction)."
            )

    if not conn["ok"]:
        if conn["error"] == "credentials_missing":
            st.info("Add Alpaca paper credentials to `.env` to enable stock paper trading.")
        else:
            st.error(f"Paper account verification failed: {conn['error']}")
        st.button("Start Bot", disabled=True,
                  help="Disabled until the paper account is verified.")
        return

    # --- kill switch (unclean stop acknowledgement) --------------------
    kill = read_kill_switch(store)
    if kill:
        st.error(
            f"The last session ended with **{kill.get('reason')}**"
            + (f": {kill.get('message')}" if kill.get("message") else "")
            + ". Review the state below; the bot cannot start again until you acknowledge."
        )
        if st.button("I understand — clear and allow restart"):
            store.delete_kv(KILL_SWITCH_KEY)
            st.rerun()

    # --- run duration + start/stop controls ----------------------------
    running = controller.is_running
    st.markdown("### Session")
    d1, d2, d3 = st.columns([2, 1, 1])
    with d1:
        duration_mode = st.radio("Run duration", ["Until I stop it", "Fixed duration"],
                                 horizontal=True, disabled=running)
    with d2:
        duration_value = st.number_input("Duration", min_value=1, value=1, disabled=(
            running or duration_mode == "Until I stop it"))
    with d3:
        duration_unit = st.selectbox("Unit", ["minutes", "hours", "days"], index=1,
                                     disabled=(running or duration_mode == "Until I stop it"))

    max_runtime = None
    if duration_mode == "Fixed duration":
        seconds_per_unit = {"minutes": 60, "hours": 3600, "days": 86400}
        max_runtime = float(duration_value * seconds_per_unit[duration_unit])

    errors = validate_all(strategy_config, risk_config, None, engine_config)
    start_blocked_reason = None
    if errors:
        start_blocked_reason = "Fix the configuration errors shown below first."
    elif kill:
        start_blocked_reason = "Acknowledge the previous unclean stop above first."

    b1, b2 = st.columns(2)
    with b1:
        start_clicked = st.button(
            "▶️ Start Bot", type="primary", disabled=running or start_blocked_reason is not None,
            use_container_width=True,
        )
    with b2:
        stop_clicked = st.button("⏹ Stop Bot", disabled=not running, use_container_width=True)

    if start_blocked_reason and not running:
        st.caption(f"Start disabled: {start_blocked_reason}")
    for err in errors:
        st.error(err)

    if start_clicked and not running:
        run_cfg = EngineConfig(
            poll_interval_seconds=engine_config.poll_interval_seconds,
            max_runtime_seconds=max_runtime,
            data_feed=engine_config.data_feed,
            block_entries_near_close=engine_config.block_entries_near_close,
            close_buffer_minutes=engine_config.close_buffer_minutes,
            flat_before_close=engine_config.flat_before_close,
            flat_before_close_minutes=engine_config.flat_before_close_minutes,
        )

        def factory() -> TradeEngine:
            return TradeEngine(
                broker=make_broker(),
                data_feed=make_data_feed(run_cfg.data_feed),
                strategy=TrendStrategy(strategy_config),
                risk_manager=RiskManager(risk_config),
                state_store=store,
                strategy_config=strategy_config,
                risk_config=risk_config,
                engine_config=run_cfg,
            )

        ok, msg = controller.start_engine(factory)
        (st.success if ok else st.warning)(msg)
        if ok:
            st.caption(
                "Configuration is frozen for this session — stop and restart the bot "
                "to apply parameter changes."
            )
        st.rerun()

    if stop_clicked and running:
        ok, msg = controller.stop_engine(StopReason.USER_STOP)
        (st.info if ok else st.warning)(msg)

    st.markdown("---")
    render_live_status(store)


@st.fragment(run_every=3)
def render_live_status(store: StateStore) -> None:
    """Live-updating panel.  Reads engine memory + SQLite only — it never
    calls Alpaca, so Streamlit reruns don't add API load."""
    controller = get_controller()
    status = controller.status
    status_dict = status.to_dict() if status is not None else read_engine_status_kv(store)
    now = datetime.now(timezone.utc)

    if status_dict is None:
        st.info("The bot has not run yet in this app session.")
        return

    state = status_dict.get("state", "STOPPED")
    session_id = status_dict.get("session_id")

    state_icon = {
        "RUNNING": "🟢", "STARTING": "🟡", "STOPPING": "🟡",
        "STOPPED": "⚪", "HALTED": "🔴", "ERROR": "🔴",
    }.get(state, "⚪")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Bot", f"{state_icon} {state}")
    s2.metric("Session", (session_id or "—")[:8])
    market_open = status_dict.get("market_open")
    s3.metric("Market", "Open" if market_open else ("Closed" if market_open is not None else "—"))
    heartbeat = status_dict.get("last_heartbeat")
    hb_dt = datetime.fromisoformat(heartbeat) if heartbeat else None
    s4.metric("Last heartbeat", fmt_ts(hb_dt))

    timeframe = status_dict.get("timeframe") or "1Day"
    heartbeat_limit = max(120.0, 2.0 * cfg.default_poll_interval(timeframe) + 30.0)
    if (
        state == "RUNNING"
        and hb_dt is not None
        and (now - hb_dt).total_seconds() > heartbeat_limit
    ):
        st.warning(
            f"Engine heartbeat is stale (no update for over {heartbeat_limit / 60:.1f} "
            "minutes) — the bot may be "
            "stuck or the process may have been restarted. Treat the status below as "
            "out of date."
        )

    t1, t2, t3, t4 = st.columns(4)
    started = status_dict.get("started_at")
    intended = status_dict.get("intended_stop_at")
    t1.metric("Started", fmt_ts(datetime.fromisoformat(started)) if started else "—")
    t2.metric("Scheduled stop", fmt_ts(datetime.fromisoformat(intended)) if intended else "runs until stopped")
    if intended:
        remaining = (datetime.fromisoformat(intended) - now).total_seconds()
        t3.metric("Time remaining", f"{max(0, remaining) / 60:.0f} min")
    else:
        t3.metric("Time remaining", "—")
    t4.metric("Last stop reason", status_dict.get("stop_reason") or "—")

    if status_dict.get("halt_reason"):
        st.error(f"HALTED: {status_dict['halt_reason']}")
    if status_dict.get("error"):
        st.error(f"Engine error: {status_dict['error']}")
    for warning in status_dict.get("warnings", []):
        st.warning(warning)

    # --- signal transparency ------------------------------------------
    st.markdown("#### Latest signal")
    signal = read_last_signal(store)
    if signal is None:
        st.caption("No signal evaluated yet — the bot needs a new completed bar first.")
    else:
        render_signal_panel(signal)

    bar_time = status_dict.get("last_bar_time")
    order_status = status_dict.get("last_order_status")
    st.caption(
        f"Last completed bar processed: {fmt_ts(datetime.fromisoformat(bar_time)) if bar_time else '—'}"
        f" · Last order status: {order_status or '—'}"
    )

    # --- position + P&L ------------------------------------------------
    st.markdown("#### Position & P&L")
    render_position_and_pnl(store, session_id)

    # --- equity curve ---------------------------------------------------
    if session_id:
        curve = store.get_equity_curve(session_id)
        if len(curve) >= 2:
            eq = pd.DataFrame(curve)
            fig = go.Figure(go.Scatter(x=eq["timestamp"], y=eq["equity"], mode="lines",
                                       name="Equity"))
            fig.update_layout(title="Session equity", xaxis_title="Time (UTC)",
                              yaxis_title="Equity ($)", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # --- recent events --------------------------------------------------
    st.markdown("#### Recent trades & orders")
    render_event_log(store, session_id)


def render_signal_panel(signal: SignalDecision) -> None:
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Short MA", f"{signal.short_ma:.2f}" if signal.short_ma is not None else "—")
    i2.metric("Long MA", f"{signal.long_ma:.2f}" if signal.long_ma is not None else "—")
    i3.metric("RSI", f"{signal.rsi:.1f}" if signal.rsi is not None else "—")
    i4.metric("ATR", f"{signal.atr:.3f}" if signal.atr is not None else "—")
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Volume", f"{signal.volume:,.0f}" if signal.volume is not None else "—")
    j2.metric("Avg volume", f"{signal.average_volume:,.0f}" if signal.average_volume is not None else "—")
    j3.metric("Volume ratio", f"{signal.volume_ratio:.2f}×" if signal.volume_ratio is not None else "—")
    j4.metric("ATR percentile", f"{signal.atr_percentile:.0f}" if signal.atr_percentile is not None else "n/a")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ADX", f"{signal.adx:.1f}" if signal.adx is not None else "n/a")
    k2.metric(
        "Price efficiency",
        f"{signal.efficiency_ratio:.2f}" if signal.efficiency_ratio is not None else "n/a",
    )
    k3.metric(
        "Market regime",
        "Bullish" if signal.market_regime_bullish is True else (
            "Bearish" if signal.market_regime_bullish is False else "n/a"
        ),
    )
    k4.metric(
        "Risk scale",
        f"{signal.risk_scale:.2f}×" if signal.risk_scale is not None else "n/a",
    )

    f = signal.filters
    st.markdown(
        f"- Trend (new bullish crossover): **{pass_fail(f.get('trend', False))}**\n"
        f"- RSI below overbought: **{pass_fail(f.get('rsi', False))}**\n"
        f"- Volume confirmation: **{pass_fail(f.get('volume', False))}**\n"
        f"- Volatility filter: **{pass_fail(f.get('volatility', False))}**"
    )
    optional_labels = {
        "adx": "ADX trend strength",
        "efficiency": "Price efficiency",
        "market_regime": "Broad-market regime",
        "relative_strength": "Relative strength",
    }
    optional_lines = [
        f"- {label}: **{pass_fail(f[key])}**"
        for key, label in optional_labels.items()
        if key in f
    ]
    if optional_lines:
        st.markdown("\n".join(optional_lines))
    decision_line = f"**Decision: {signal.action.value}** — reason: `{signal.reason}`"
    block = signal.details.get("entry_block")
    if block:
        decision_line += f" · entry blocked: `{block}`"
    st.markdown(decision_line)
    st.caption(f"Evaluated on completed bar {fmt_ts(signal.timestamp)}")


def render_position_and_pnl(store: StateStore, session_id: str | None) -> None:
    latest_session = store.get_session(session_id) if session_id else store.get_latest_session()
    realized = store.session_realized_pnl(session_id) if session_id else 0.0

    symbol = latest_session["symbol"] if latest_session else None
    bot_pos = store.get_bot_position(symbol) if symbol else None

    if bot_pos:
        qty = float(bot_pos.get("qty", 0))
        entry_price = float(bot_pos.get("entry_price") or 0)
        stop_price = bot_pos.get("stop_price")
        st.markdown(
            f"**{symbol}** · {qty:g} shares @ {fmt_money(entry_price)} · "
            f"protective stop: {fmt_money(stop_price, 'not yet placed')} · bot-owned ✓"
        )
        snapshot = None
        raw_snapshot = store.get_kv(POSITION_SNAPSHOT_KEY_PREFIX + symbol) if symbol else None
        if raw_snapshot:
            try:
                snapshot = json.loads(raw_snapshot)
            except json.JSONDecodeError:
                snapshot = None
        if snapshot:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Current price", fmt_money(snapshot.get("current_price")))
            p2.metric("Market value", fmt_money(snapshot.get("market_value")))
            p3.metric("Unrealized P&L", fmt_money(snapshot.get("unrealized_pl")))
            plpc = snapshot.get("unrealized_plpc")
            p4.metric("Unrealized P&L %", f"{plpc * 100:+.2f}%" if plpc is not None else "—")
            st.caption(
                f"Broker snapshot from {fmt_ts(snapshot.get('timestamp'))} — written by the "
                "engine each cycle; this live panel reads local state only."
            )
        if stop_price and entry_price:
            risk_amount = (entry_price - float(stop_price)) * qty
            start_eq = latest_session.get("starting_equity") if latest_session else None
            risk_pct = (risk_amount / start_eq * 100.0) if start_eq else None
            st.caption(
                f"Risk if stop fills: {fmt_money(risk_amount)}"
                + (f" ({risk_pct:.2f}% of session-start equity)" if risk_pct is not None else "")
            )
    else:
        st.info("No active bot-managed position.")

    start_eq = latest_session.get("starting_equity") if latest_session else None
    curve = store.get_equity_curve(session_id) if session_id else []
    last_equity = curve[-1]["equity"] if curve else None
    equity_change = (last_equity - start_eq) if (last_equity is not None and start_eq) else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Session realized P&L", fmt_money(realized))
    m2.metric(
        "Session equity change",
        fmt_money(equity_change, "—"),
        delta=f"{(equity_change / start_eq * 100):+.2f}%" if equity_change is not None and start_eq else None,
    )
    m3.metric("Latest equity", fmt_money(last_equity, "—"))
    if bot_pos and equity_change is not None:
        unrealized = equity_change - realized
        unrealized_pct = (unrealized / start_eq * 100.0) if start_eq else None
        st.caption(
            f"Current unrealized P&L ≈ {fmt_money(unrealized)}"
            + (f" ({unrealized_pct:+.2f}%)" if unrealized_pct is not None else "")
            + " (session equity change minus realized P&L, from the latest engine snapshot)"
        )


def render_event_log(store: StateStore, session_id: str | None) -> None:
    trades = store.get_trades(session_id, limit=50) if session_id else store.get_trades(limit=50)
    orders = store.get_orders(session_id, limit=50) if session_id else store.get_orders(limit=50)

    rows = []
    for t in trades:
        snapshot = {}
        if t.get("signal_json"):
            try:
                snapshot = json.loads(t["signal_json"])
            except json.JSONDecodeError:
                snapshot = {}
        rows.append({
            "Time": t["timestamp"],
            "Event": {"BUY": "Fill (entry)", "SELL": "Fill (exit)"}.get(t["action"], t["action"]),
            "Side": "buy" if t["action"] == "BUY" else "sell",
            "Qty": t["qty"],
            "Price": t["fill_price"],
            "Status": "filled",
            "Reason": t["reason"],
            "Short MA": snapshot.get("short_ma"),
            "Long MA": snapshot.get("long_ma"),
            "RSI": snapshot.get("rsi"),
            "ATR": snapshot.get("atr"),
            "Vol ratio": snapshot.get("volume_ratio"),
        })
    for o in orders:
        label = {
            "entry": "Order (entry)", "exit": "Order (exit)", "stop": "Protective stop",
        }.get(o["purpose"], f"Order ({o['purpose']})")
        status = o["status"]
        if status == "rejected":
            label = "Rejected " + label.lower()
        elif status == "partially_filled":
            label = "Partial fill " + label.lower()
        rows.append({
            "Time": o["updated_at"] or o["submitted_at"],
            "Event": label,
            "Side": o["side"],
            "Qty": o["requested_qty"],
            "Price": o["fill_price"] if o["fill_price"] is not None else o["stop_price"],
            "Status": status,
            "Reason": o["client_order_id"],
            "Short MA": None, "Long MA": None, "RSI": None, "ATR": None, "Vol ratio": None,
        })

    if not rows:
        st.caption("No trades or orders recorded yet.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("Time", ascending=False, na_position="last")
    st.dataframe(df, use_container_width=True, hide_index=True, height=280)


# ---------------------------------------------------------------------------
# Logs / State tab
# ---------------------------------------------------------------------------


@st.fragment(run_every=5)
def render_logs_tab(store: StateStore) -> None:
    controller = get_controller()
    status = controller.status
    status_dict = status.to_dict() if status is not None else read_engine_status_kv(store)

    st.markdown("#### Engine state")
    if status_dict:
        st.json(status_dict)
    else:
        st.caption("No engine state recorded yet.")

    session = store.get_active_session() or store.get_latest_session()
    st.markdown("#### Session")
    if session:
        display = dict(session)
        for key in ("started_at", "intended_stop_at", "stopped_at"):
            display[key] = fmt_ts(display.get(key))
        st.json(display)
        session_id = session["session_id"]
    else:
        st.caption("No sessions yet.")
        session_id = None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Recent processed bars")
        bars = store.get_processed_bars(limit=20, session_id=session_id)
        if bars:
            st.dataframe(pd.DataFrame(bars), use_container_width=True, hide_index=True, height=240)
        else:
            st.caption("None recorded.")
    with c2:
        st.markdown("#### Bot orders")
        orders = store.get_orders(session_id=session_id, limit=20)
        if orders:
            st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True, height=240)
        else:
            st.caption("None recorded.")

    st.markdown("#### Logs")
    severity_filter = st.multiselect("Severity", ["INFO", "WARNING", "ERROR"],
                                     default=["INFO", "WARNING", "ERROR"])
    logs = store.get_logs(limit=200)
    logs = [l for l in logs if l["severity"] in severity_filter]
    if logs:
        log_df = pd.DataFrame(
            [{"Time": l["timestamp"], "Severity": l["severity"], "Message": l["message"]}
             for l in logs]
        )
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=320)
    else:
        st.caption("No log entries.")

    if session_id:
        trades = store.get_trades(session_id, limit=1000)
        if trades:
            csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(
                "Download session trade log (CSV)", data=csv,
                file_name=f"trades_{session_id[:8]}.csv", mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render_missing_credentials_box() -> None:
    st.warning(
        "**Alpaca paper credentials are not configured yet.**\n\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Add `ALPACA_PAPER_API_KEY` (your paper API key)\n"
        "3. Add `ALPACA_PAPER_SECRET_KEY`\n"
        "4. Restart Streamlit\n\n"
        "Get paper keys from the Alpaca dashboard (Paper Trading section). "
        "The stock workspace is always paper-only; crypto live mode uses separate "
        "credentials and remains disabled by default."
    )


# ---------------------------------------------------------------------------
# Crypto workspace
# ---------------------------------------------------------------------------


def _crypto_context_fingerprint(
    credentials,
    strategy: CryptoStrategyConfig,
    risk: CryptoRiskConfig,
    engine: CryptoEngineConfig,
    backtest: CryptoBacktestConfig,
    decision_engine: str = "base",
) -> str:
    payload = {
        "credential_fingerprint": credentials.fingerprint,
        "strategy": strategy.to_dict(),
        "risk": risk.to_dict(),
        "engine": engine.to_dict(),
        "backtest": backtest.to_dict(),
        "decision_engine": decision_engine,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _render_crypto_backtest_result(result) -> None:
    cols = st.columns(6)
    cols[0].metric("Total return", f"{result.total_return_pct:.2f}%")
    cols[1].metric("BTC benchmark", f"{result.benchmark_return_pct:.2f}%")
    cols[2].metric("Max drawdown", f"{result.max_drawdown_pct:.2f}%")
    cols[3].metric(
        "Sharpe", "—" if result.sharpe_ratio is None else f"{result.sharpe_ratio:.2f}"
    )
    cols[4].metric("Closed trades", str(result.num_trades))
    cols[5].metric("Modeled fees", fmt_money(result.total_fees))

    cost_cols = st.columns(3)
    cost_cols[0].metric(
        "Round-trip cost", f"{result.round_trip_cost_bps:.1f} bps",
        help="Both legs: slippage, half the spread, and the taker fee.",
    )
    if result.median_cost_to_risk is None:
        cost_cols[1].metric("Cost per trade", "—")
        cost_cols[2].metric("Break-even win rate", "—")
    else:
        ratio = result.median_cost_to_risk
        cost_cols[1].metric(
            "Cost per trade", f"{ratio:.2f} R",
            help=(
                "Median friction as a fraction of the trade's own risk unit "
                "(entry minus stop). Above ~0.25R the strategy is paying a "
                "large share of what it risks just to transact."
            ),
        )
        win_rate = break_even_win_rate(ratio, reward_to_risk=2.0)
        cost_cols[2].metric(
            "Break-even win rate",
            "unreachable" if math.isnan(win_rate) else f"{win_rate:.1%}",
            help=(
                "Win rate needed at 2R reward:risk just to break even after "
                "costs. Trend systems typically win 35-45%."
            ),
        )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name="Crypto portfolio",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.benchmark_curve.index,
            y=result.benchmark_curve.values,
            name="BTC buy & hold",
        )
    )
    figure.update_layout(
        height=420,
        yaxis_title="Equity (USD)",
        xaxis_title="Completed bar",
        legend_orientation="h",
    )
    st.plotly_chart(figure, width="stretch")
    if result.trades:
        st.dataframe(
            pd.DataFrame([trade.to_dict() for trade in result.trades]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No completed trades occurred under this configuration and period.")
    with st.expander("Crypto simulation assumptions", expanded=False):
        st.json(result.config_snapshot["execution"])
        st.caption(
            "Signals use completed bar N and execute at N+1 open. Gap stops fill at "
            "the adverse gap open. Costs include spread, slippage, and percentage fee."
        )


def _crypto_broker_for(credentials, risk: CryptoRiskConfig):
    if credentials.account_mode is AccountMode.LIVE:
        return AlpacaLiveBroker(
            credentials.api_key,
            credentials.secret_key,
            max_live_order_notional=risk.max_live_order_notional,
        )
    return AlpacaPaperBroker(credentials.api_key, credentials.secret_key)


def render_crypto_workspace(store: StateStore) -> None:
    st.header("₿ 24/7 Crypto Research & Execution")
    st.warning(
        "No algorithm is guaranteed to make money. Crypto is volatile and can gap through "
        "a stop-limit. Paper validation and promotion gates are evidence—not a forecast."
    )
    controller = get_controller()
    engine_active = controller.is_running

    decision_engine_label = st.radio(
        "Decision engine",
        ["Regime ensemble (default)", "Base breakout/trend"],
        horizontal=True,
        disabled=engine_active,
        key="crypto_decision_engine",
        help=(
            "Both engines run through the identical execution, sizing, cost-gate, "
            "and promotion machinery. The ensemble blends breakout, pullback, and "
            "mean-reversion sleeves weighted by market regime; the base engine is "
            "the original breakout/trend model. The selection is part of the result "
            "fingerprint, so cached research never mixes engines."
        ),
    )
    use_ensemble = decision_engine_label.startswith("Regime")
    ensemble_config = EnsembleConfig() if use_ensemble else None

    mode_label = st.radio(
        "Alpaca crypto account",
        ["Paper", "Live"],
        horizontal=True,
        disabled=engine_active,
        key="crypto_account_mode",
        help="Account mode is frozen for the entire engine session.",
    )
    previous_mode = st.session_state.get("crypto_previous_account_mode")
    if previous_mode != mode_label:
        st.session_state["crypto_live_unlocked"] = False
        st.session_state.pop("crypto_verified", None)
        st.session_state["crypto_previous_account_mode"] = mode_label
    account_mode = AccountMode.LIVE if mode_label == "Live" else AccountMode.PAPER
    live_feature_enabled = live_trading_enabled()

    if account_mode is AccountMode.LIVE:
        st.error("🔴 LIVE ACCOUNT — ORDERS USE REAL MONEY")
        if not live_feature_enabled:
            st.warning(
                "Live mode is deployment-disabled. Set `ENABLE_LIVE_TRADING=true`, add "
                "separate live credentials, and restart Streamlit."
            )
        live_ack = st.checkbox(
            "I understand this can lose real money and a stop-limit may not fill.",
            key="crypto_live_ack",
            disabled=not live_feature_enabled or engine_active,
        )
        live_phrase = st.text_input(
            "Type LIVE to unlock this process",
            key="crypto_live_phrase",
            disabled=not live_feature_enabled or engine_active,
        )
        if st.button(
            "Unlock LIVE Trading",
            type="primary",
            disabled=(
                not live_feature_enabled
                or not live_ack
                or live_phrase.strip() != "LIVE"
                or engine_active
            ),
        ):
            st.session_state["crypto_live_unlocked"] = True
        live_unlocked = bool(st.session_state.get("crypto_live_unlocked"))
        if live_unlocked:
            st.error("LIVE mode is unlocked for this process. Verify the account before Start.")
    else:
        st.error("🧪 CRYPTO PAPER TRADING — NO REAL MONEY")
        live_unlocked = True

    universe_text = st.text_input(
        "Crypto universe (comma separated USD pairs)",
        value=", ".join(DEFAULT_CRYPTO_UNIVERSE),
        disabled=engine_active,
        key="crypto_universe",
    )
    symbols = tuple(
        dict.fromkeys(
            token.strip().upper()
            for token in universe_text.split(",")
            if token.strip()
        )
    )
    config_left, config_right = st.columns(2)
    with config_left:
        timeframe = st.selectbox(
            "Crypto timeframe",
            ["15Min", "1Hour", "4Hour", "1Day"],
            index=1,
            disabled=engine_active,
            key="crypto_timeframe",
        )
        fast_ma = int(
            st.number_input(
                "Fast trend MA", 5, 100, 20, 1, disabled=engine_active, key="crypto_fast_ma"
            )
        )
        slow_ma = int(
            st.number_input(
                "Slow trend MA", 20, 300, 100, 1, disabled=engine_active, key="crypto_slow_ma"
            )
        )
        breakout = int(
            st.number_input(
                "Prior-bar breakout lookback",
                5,
                100,
                20,
                1,
                disabled=engine_active,
                key="crypto_breakout",
            )
        )
    with config_right:
        risk_per_trade = float(
            st.number_input(
                "Risk per crypto trade (%)",
                0.05,
                2.0,
                0.35,
                0.05,
                disabled=engine_active,
                key="crypto_risk_trade",
            )
        )
        max_position = float(
            st.number_input(
                "Maximum per-position exposure (%)",
                1.0,
                25.0,
                10.0,
                1.0,
                disabled=engine_active,
                key="crypto_max_position",
            )
        )
        max_exposure = float(
            st.number_input(
                "Maximum crypto portfolio exposure (%)",
                1.0,
                60.0,
                30.0,
                1.0,
                disabled=engine_active,
                key="crypto_max_exposure",
            )
        )
        max_session_loss = float(
            st.number_input(
                "Session loss halt (%)",
                0.25,
                10.0,
                2.0,
                0.25,
                disabled=engine_active,
                key="crypto_loss_halt",
            )
        )

    with st.expander("Advanced crypto analyst and execution controls", expanded=False):
        a, b, c = st.columns(3)
        regime_ma = int(a.number_input("BTC regime MA", 50, 400, 200, 5, key="crypto_regime_ma"))
        min_adx = float(b.number_input("Minimum ADX", 0.0, 60.0, 20.0, 1.0, key="crypto_min_adx"))
        min_efficiency = float(
            c.number_input("Minimum path efficiency", 0.0, 1.0, 0.25, 0.05, key="crypto_min_er")
        )
        min_rsi = float(a.number_input("Minimum RSI", 0.0, 90.0, 50.0, 1.0, key="crypto_min_rsi"))
        max_rsi = float(b.number_input("Maximum RSI", 10.0, 100.0, 72.0, 1.0, key="crypto_max_rsi"))
        max_atr_pct = float(
            c.number_input("Maximum ATR percentile", 10.0, 100.0, 90.0, 1.0, key="crypto_max_atr_pct")
        )
        atr_stop = float(a.number_input("Initial ATR stop", 0.5, 10.0, 2.5, 0.25, key="crypto_atr_stop"))
        trailing_atr = float(
            b.number_input("Trailing ATR stop", 0.5, 12.0, 3.5, 0.25, key="crypto_trailing_atr")
        )
        max_spread = float(c.number_input("Entry spread cap (bps)", 1.0, 500.0, 40.0, 1.0, key="crypto_spread_cap"))
        max_live_order = float(
            a.number_input("Hard live order cap ($)", 10.0, 10_000.0, 1_000.0, 10.0, key="crypto_live_order_cap")
        )
        max_live_exposure = float(
            b.number_input("Hard live total exposure (%)", 0.5, 20.0, 10.0, 0.5, key="crypto_live_exposure_cap")
        )
        stop_buffer = float(
            c.number_input("Stop-limit buffer (bps)", 1.0, 1_000.0, 50.0, 1.0, key="crypto_stop_buffer")
        )

    benchmark = "BTC/USD"
    strategy_config = CryptoStrategyConfig(
        symbols=symbols,
        timeframe=timeframe,
        benchmark_symbol=benchmark,
        fast_ma_period=fast_ma,
        slow_ma_period=slow_ma,
        regime_ma_period=regime_ma,
        breakout_period=breakout,
        min_rsi=min_rsi,
        max_rsi=max_rsi,
        min_adx=min_adx,
        min_efficiency_ratio=min_efficiency,
        max_atr_percentile=max_atr_pct,
        atr_stop_multiplier=atr_stop,
        trailing_atr_multiplier=trailing_atr,
    )
    risk_config = CryptoRiskConfig(
        risk_per_trade_pct=risk_per_trade,
        max_position_pct=max_position,
        max_total_exposure_pct=max_exposure,
        max_session_loss_pct=max_session_loss,
        max_live_order_notional=max_live_order,
        max_live_total_exposure_pct=max_live_exposure,
    )
    engine_config = CryptoEngineConfig(
        account_mode=account_mode,
        max_spread_bps=max_spread,
        stop_limit_buffer_bps=stop_buffer,
    )

    st.subheader("Historical portfolio simulation")
    date_cols = st.columns(4)
    yesterday = date.today() - timedelta(days=1)
    start_date = date_cols[0].date_input(
        "Crypto start", value=yesterday - timedelta(days=365), key="crypto_start"
    )
    end_date = date_cols[1].date_input("Crypto end", value=yesterday, key="crypto_end")
    capital = float(
        date_cols[2].number_input(
            "Crypto starting capital", 100.0, 10_000_000.0, 100_000.0, 1_000.0,
            key="crypto_capital",
        )
    )
    fee_bps = float(
        date_cols[3].number_input("Taker fee (bps)", 0.0, 500.0, 25.0, 1.0, key="crypto_fee")
    )
    cost_cols = st.columns(2)
    slippage_bps = float(
        cost_cols[0].number_input("Crypto slippage (bps)", 0.0, 500.0, 10.0, 1.0, key="crypto_slippage")
    )
    modeled_spread = float(
        cost_cols[1].number_input("Modeled full spread (bps)", 0.0, 500.0, 5.0, 1.0, key="crypto_modeled_spread")
    )
    backtest_config = CryptoBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        slippage_bps=slippage_bps,
        taker_fee_bps=fee_bps,
        spread_bps=modeled_spread,
    )
    credentials = load_credentials(account_mode=account_mode)
    context_key = _crypto_context_fingerprint(
        credentials,
        strategy_config,
        risk_config,
        engine_config,
        backtest_config,
        decision_engine="ensemble" if use_ensemble else "base",
    )
    config_errors = (
        strategy_config.validate()
        + risk_config.validate()
        + engine_config.validate()
        + backtest_config.validate()
    )
    if end_date > yesterday:
        config_errors.append("Crypto backtests must end on yesterday or earlier (completed bars only).")
    duration = cfg.timeframe_to_timedelta(timeframe)
    estimated_bars = (
        (end_date - start_date).total_seconds() / duration.total_seconds()
        + strategy_config.warmup_bars()
        + 5
    ) * max(1, len(symbols))
    if estimated_bars > 300_000:
        config_errors.append(
            "Requested crypto history exceeds the 300,000 aligned-bar safety limit; "
            "shorten the date range or use a slower timeframe."
        )
    if config_errors:
        st.error("Crypto configuration is invalid:\n- " + "\n- ".join(config_errors))

    if st.button("Run Crypto Backtest", type="primary", disabled=bool(config_errors)):
        try:
            feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
            activation = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
            fetch_start = activation - duration * (strategy_config.warmup_bars() + 3)
            fetch_end = datetime.combine(
                end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
            )
            with st.spinner("Downloading completed Alpaca crypto bars and simulating…"):
                bars = {
                    symbol: feed.get_bars(
                        symbol, timeframe, start=fetch_start, end=fetch_end
                    )
                    for symbol in symbols
                }
                if use_ensemble:
                    result = run_regime_ensemble_backtest(
                        bars,
                        strategy_config,
                        risk_config,
                        backtest_config,
                        active_start=pd.Timestamp(activation),
                        ensemble_config=ensemble_config,
                    )
                else:
                    result = run_crypto_backtest(
                        bars,
                        strategy_config,
                        risk_config,
                        backtest_config,
                        active_start=pd.Timestamp(activation),
                    )
            st.session_state["crypto_backtest_result"] = result
            st.session_state["crypto_backtest_bars"] = bars
            st.session_state["crypto_backtest_context"] = context_key
        except Exception as exc:  # safe UI boundary; no credentials rendered
            st.error(f"Crypto backtest failed: {exc}")
    result = st.session_state.get("crypto_backtest_result")
    if result is not None:
        if st.session_state.get("crypto_backtest_context") == context_key:
            _render_crypto_backtest_result(result)
        else:
            st.info("Crypto controls changed; rerun the backtest to refresh the result.")

    with st.expander("Chronological crypto walk-forward promotion", expanded=False):
        st.caption(
            "Training-only selection, one-bar embargo, non-overlapping tests, fixed "
            "baseline, and stressed costs. A pass never auto-enables trading."
        )
        wcols = st.columns(3)
        train_bars = int(wcols[0].number_input("Training bars", 300, 20_000, 1_500, 100, key="crypto_wf_train"))
        test_bars = int(wcols[1].number_input("Test bars", 50, 5_000, 360, 10, key="crypto_wf_test"))
        min_trades = int(wcols[2].number_input("Minimum training trades", 1, 100, 4, 1, key="crypto_wf_trades"))
        if st.button(
            "Run Crypto Walk-Forward",
            disabled=(
                st.session_state.get("crypto_backtest_context") != context_key
                or bool(config_errors)
            ),
        ):
            try:
                walk = CryptoWalkForwardConfig(
                    train_bars=train_bars,
                    test_bars=test_bars,
                    step_bars=test_bars,
                    min_train_trades=min_trades,
                )
                grid = {
                    "breakout_period": sorted({breakout, max(5, breakout + 10)}),
                    "min_adx": sorted({min_adx, min(60.0, min_adx + 5.0)}),
                }
                research_start = pd.Timestamp(
                    datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
                )
                research_bars = {
                    symbol: frame.loc[frame.index >= research_start]
                    for symbol, frame in st.session_state["crypto_backtest_bars"].items()
                }
                with st.spinner("Running bounded chronological crypto research…"):
                    if use_ensemble:
                        report = run_regime_ensemble_walk_forward(
                            research_bars,
                            strategy_config,
                            risk_config,
                            backtest_config,
                            grid,
                            walk,
                            CryptoPromotionCriteria(),
                            ensemble_config=ensemble_config,
                        )
                    else:
                        report = run_crypto_walk_forward(
                            research_bars,
                            strategy_config,
                            risk_config,
                            backtest_config,
                            grid,
                            walk,
                            CryptoPromotionCriteria(),
                        )
                st.session_state["crypto_walk_result"] = report
                st.session_state["crypto_walk_context"] = context_key
            except Exception as exc:
                st.error(f"Crypto walk-forward failed: {exc}")
        report = st.session_state.get("crypto_walk_result")
        if report is not None and st.session_state.get("crypto_walk_context") == context_key:
            if report.promotion.promoted:
                st.success("PROMOTION GATES PASSED — eligible for controlled paper testing only.")
            else:
                st.error("PROMOTION GATES FAILED — do not promote this configuration.")
                st.write(report.promotion.failures)
            rcols = st.columns(5)
            rcols[0].metric("OOS return", f"{report.compounded_return_pct:.2f}%")
            rcols[1].metric("Baseline", f"{report.baseline_compounded_return_pct:.2f}%")
            rcols[2].metric("Stressed", f"{report.stressed_compounded_return_pct:.2f}%")
            rcols[3].metric("OOS trades", str(report.total_oos_trades))
            rcols[4].metric("Folds", str(len(report.folds)))

    st.subheader("Alpaca account verification & 24/7 engine")
    if not credentials.is_configured:
        variable_prefix = "ALPACA_LIVE" if account_mode is AccountMode.LIVE else "ALPACA_PAPER"
        st.warning(
            f"{mode_label} credentials are missing. Configure `{variable_prefix}_API_KEY` "
            f"and `{variable_prefix}_SECRET_KEY` (legacy paper names remain supported)."
        )
    verification_key = hashlib.sha256(
        json.dumps(
            {
                "credential": credentials.fingerprint,
                "mode": account_mode.value,
                "symbols": symbols,
                "live_order_cap": risk_config.max_live_order_notional,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    can_verify = (
        credentials.is_configured
        and not config_errors
        and live_unlocked
        and (account_mode is AccountMode.PAPER or live_feature_enabled)
        and not engine_active
    )
    if st.button("Verify Crypto Account & Assets", disabled=not can_verify):
        try:
            broker = _crypto_broker_for(credentials, risk_config)
            account = broker.verify_account()
            assets = broker.verify_crypto_assets(symbols)
            st.session_state["crypto_verified"] = {
                "key": verification_key,
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "assets": list(assets),
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            st.session_state.pop("crypto_verified", None)
            st.error(f"Crypto account verification failed: {exc}")
    verified = st.session_state.get("crypto_verified")
    verified_current = bool(verified and verified.get("key") == verification_key)
    if verified_current:
        st.success(
            f"Verified {mode_label.upper()} account and {len(verified['assets'])} assets. "
            f"Equity {fmt_money(verified['equity'])}; cash {fmt_money(verified['cash'])}."
        )

    kill_raw = store.get_kv(CRYPTO_KILL_SWITCH_KEY)
    kill_armed = False
    if kill_raw:
        try:
            kill_armed = bool(json.loads(kill_raw).get("armed"))
        except json.JSONDecodeError:
            kill_armed = True
    if kill_armed:
        st.error("Crypto kill switch is armed; Start is blocked.")
        if st.button("Acknowledge Crypto Kill Switch", disabled=engine_active):
            store.delete_kv(CRYPTO_KILL_SWITCH_KEY)
            st.success("Crypto kill switch acknowledged. Verify the account again before Start.")
            st.session_state.pop("crypto_verified", None)

    start_disabled = (
        engine_active
        or not verified_current
        or bool(config_errors)
        or kill_armed
        or not live_unlocked
        or (account_mode is AccountMode.LIVE and not live_feature_enabled)
    )
    action_cols = st.columns(3)
    if action_cols[0].button("Start Crypto Bot", type="primary", disabled=start_disabled):
        def factory():
            broker = _crypto_broker_for(credentials, risk_config)
            data_feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
            if use_ensemble:
                # Same production engine underneath: broker verification, mode
                # isolation, loss latch, stop watchdog, and recovery are all
                # owned by the parent class. Only the decision layer differs.
                return make_regime_aware_crypto_engine(
                    broker,
                    data_feed,
                    store,
                    strategy_config,
                    risk_config,
                    engine_config,
                    ensemble_config=ensemble_config,
                )
            return CryptoTradeEngine(
                broker,
                data_feed,
                store,
                strategy_config,
                risk_config,
                engine_config,
            )

        started, message = controller.start_engine(factory)
        (st.success if started else st.warning)(message)
    if action_cols[1].button("Stop Active Bot", disabled=not engine_active):
        stopped, message = controller.stop_engine(StopReason.USER_STOP)
        (st.success if stopped else st.warning)(message)
    if action_cols[2].button(
        "Emergency Stop Active Bot", disabled=not engine_active, type="secondary"
    ):
        stopped, message = controller.emergency_stop()
        (st.error if stopped else st.warning)(message)
    active_status = controller.status
    if active_status is not None:
        st.json(active_status.to_dict())
    persisted_status = store.get_kv(CRYPTO_ENGINE_STATUS_KEY)
    if persisted_status and not engine_active:
        with st.expander("Last persisted crypto-engine status", expanded=False):
            try:
                st.json(json.loads(persisted_status))
            except json.JSONDecodeError:
                st.error("Persisted crypto status is corrupt.")


def main() -> None:
    store = get_store()
    creds = get_credentials()
    conn = get_connection_check(creds.fingerprint)
    if not conn["ok"] and conn["error"] != "credentials_missing":
        # A failed verification (network outage, Alpaca hiccup, bad key)
        # must not be pinned for the process lifetime: drop the cached
        # result so the next rerun retries. Start stays disabled until a
        # verification actually succeeds (fail-closed either way).
        get_connection_check.clear()

    st.title("📈 Trend & Crypto Bot — Alpaca")
    st.caption(
        "A deterministic, regime-aware trend strategy with causal robustness filters, "
        "adaptive portfolio risk, and walk-forward validation. Stocks remain paper-only; "
        "crypto defaults to paper and exposes a separately gated live-account workflow. "
        "No result implies future performance."
    )

    if not creds.is_configured:
        render_missing_credentials_box()

    tab_backtest, tab_research, tab_live, tab_crypto, tab_logs = st.tabs(
        [
            "⚙️ Configure & Backtest",
            "🧪 Walk-Forward Research",
            "🟢 Run Live (Paper)",
            "₿ Crypto Research & Trading",
            "🧾 Logs / State",
        ]
    )

    with tab_backtest:
        strategy_config, risk_config, backtest_config, engine_config = build_config_inputs()
        render_backtest_tab(strategy_config, risk_config, backtest_config, engine_config)

    build_sidebar(store, conn, strategy_config, risk_config, engine_config.data_feed)

    with tab_research:
        render_research_tab(
            strategy_config, risk_config, backtest_config, engine_config
        )

    with tab_live:
        render_live_tab(store, conn, strategy_config, risk_config, engine_config)

    with tab_crypto:
        render_crypto_workspace(store)

    with tab_logs:
        render_logs_tab(store)


if __name__ == "__main__":
    main()
