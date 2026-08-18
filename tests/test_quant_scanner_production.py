"""Production regressions for QuantScanner defects found in the deep audit."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from crypto_models import CryptoQuote, CryptoSignalDecision
from models import BotAction
from quant.contracts import AttributionRecord, DecisionOutcome, Market, StrategyState
from quant.execution import ExecutionQualityConfig, ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
from quant.marketplace import MIN_SAMPLE
from quant.portfolio import BookPosition, PortfolioBook
from quant.scanner import QuantScanner, ScanStrategy
from quant.store import QuantStore

NOW = datetime(2026, 8, 18, 4, 0, tzinfo=timezone.utc)
SID = "scanner_prod"


class ScriptedStrategy:
    def __init__(self, symbol="ETH/USD", action=BotAction.BUY):
        self.symbol = symbol
        self.action = action
        self.config = type(
            "Cfg", (), {"symbols": (symbol,), "benchmark_symbol": symbol}
        )()

    def required_bars(self):
        return 5

    def prepare_symbol(self, bars):
        out = bars.copy()
        out["atr"] = 2.0
        return out

    def prepare_universe(self, bars_by_symbol):
        return {symbol: self.prepare_symbol(frame) for symbol, frame in bars_by_symbol.items()}

    def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
        close = float(prepared[self.symbol].loc[pd.Timestamp(timestamp), "close"])
        action = self.action
        if action is BotAction.BUY and self.symbol in held_symbols:
            action = BotAction.HOLD
        if action is BotAction.EXIT and self.symbol not in held_symbols:
            action = BotAction.HOLD
        return [
            CryptoSignalDecision(
                symbol=self.symbol,
                action=action,
                timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                reason="scripted",
                close=close,
                atr=2.0,
                atr_percentile=50.0,
                stop_price=close - 4.0 if action is BotAction.BUY else None,
            )
        ]

    def ratchet_stop(self, current_stop, highest, close, atr_value):
        return current_stop, max(highest, close)


def bars(symbol_seed=1, periods=120, base=100.0):
    rng = np.random.default_rng(symbol_seed)
    index = pd.date_range("2026-08-17", periods=periods, freq="5min", tz="UTC")
    close = base * np.cumprod(1.0 + rng.normal(0.0001, 0.001, periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(periods, 1_000_000.0),
        },
        index=index,
    )


def seed_edge(store, strategy_id=SID, n=MIN_SAMPLE + 5):
    for i in range(n):
        r = 1.0 if i % 3 else -0.4
        store.record_attribution(
            AttributionRecord(
                strategy_id=strategy_id,
                strategy_version="1.0.0",
                symbol="ETH/USD",
                market=Market.CRYPTO,
                timeframe="5Min",
                session_id=None,
                signal_timestamp=NOW - timedelta(days=2),
                entry_timestamp=NOW - timedelta(days=2),
                exit_timestamp=NOW - timedelta(days=2, minutes=-30),
                qty=1.0,
                entry_price=100.0,
                exit_price=100.0 + 4.0 * r,
                initial_stop_price=96.0,
                fees=0.0,
                expected_cost_bps=None,
                realized_slippage_bps=None,
                gross_pnl=4.0 * r,
                net_pnl=4.0 * r,
                r_multiple=r,
                max_favorable_excursion_pct=4.0,
                max_adverse_excursion_pct=1.0,
                entry_reason="scripted",
                exit_reason="scripted",
                regime_at_entry="RANGE",
                shadow=True,
            )
        )


def scanner(store):
    lifecycle = LifecycleManager(store)
    return (
        QuantScanner(
            store=store,
            lifecycle=lifecycle,
            execution_engine=ExecutionQualityEngine(
                ExecutionQualityConfig(
                    uncertainty_margin_bps=0.0,
                    assumed_slippage_bps=1.0,
                    crypto_assumed_spread_bps=1.0,
                    crypto_taker_fee_bps=0.0,
                )
            ),
        ),
        lifecycle,
    )


def test_scanner_uses_real_quote_age_not_zero():
    store = QuantStore(":memory:")
    try:
        scan, _ = scanner(store)
        store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
        seed_edge(store)
        quote = CryptoQuote(
            symbol="ETH/USD",
            timestamp=NOW - timedelta(seconds=37),
            bid_price=100.0,
            ask_price=100.01,
        )
        report = scan.scan(
            strategies=[ScanStrategy(SID, ScriptedStrategy(), Market.CRYPTO, "5Min")],
            bars_by_symbol={"ETH/USD": bars()},
            equity=100_000.0,
            book=PortfolioBook(equity=100_000.0),
            quotes={"ETH/USD": quote},
            now=NOW,
        )
        candidates = [d for d in report.decisions if d.candidate.action.value == "BUY"]
        assert candidates
        assert candidates[0].candidate.details["quote_age_seconds"] == 37.0
    finally:
        store.close()


def test_exit_is_never_dropped_by_scanner():
    store = QuantStore(":memory:")
    try:
        scan, _ = scanner(store)
        store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
        frame = bars()
        book = PortfolioBook(
            equity=100_000.0,
            positions=[
                BookPosition.from_stop(
                    "ETH/USD",
                    Market.CRYPTO,
                    SID,
                    qty=1.0,
                    price=float(frame["close"].iloc[-1]),
                    stop_price=float(frame["close"].iloc[-1]) - 4.0,
                )
            ],
        )
        report = scan.scan(
            strategies=[
                ScanStrategy(SID, ScriptedStrategy(action=BotAction.EXIT), Market.CRYPTO, "5Min")
            ],
            bars_by_symbol={"ETH/USD": frame},
            equity=100_000.0,
            book=book,
            now=NOW,
        )
        assert len(report.exit_signals) == 1
        exit_rows = [
            d
            for d in report.decisions
            if d.candidate.action.value == "EXIT" and d.outcome is DecisionOutcome.EXECUTE
        ]
        assert exit_rows
    finally:
        store.close()


def test_existing_book_correlation_is_enforced():
    store = QuantStore(":memory:")
    try:
        scan, _ = scanner(store)
        store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
        seed_edge(store)
        candidate = bars(symbol_seed=7)
        # Identical close path -> correlation exactly 1 with the candidate.
        held = candidate.copy()
        book = PortfolioBook(
            equity=100_000.0,
            positions=[
                BookPosition.from_stop(
                    "BTC/USD",
                    Market.CRYPTO,
                    "other_strategy",
                    qty=1.0,
                    price=float(held["close"].iloc[-1]),
                    stop_price=float(held["close"].iloc[-1]) - 5.0,
                )
            ],
        )
        report = scan.scan(
            strategies=[ScanStrategy(SID, ScriptedStrategy(), Market.CRYPTO, "5Min")],
            bars_by_symbol={"ETH/USD": candidate, "BTC/USD": held},
            equity=100_000.0,
            book=book,
            now=NOW,
        )
        assert not report.executable
        assert any("correlated_exposure" in d.rejection_reasons for d in report.decisions)
    finally:
        store.close()


def test_lifecycle_risk_multiplier_can_only_reduce_base_risk():
    store = QuantStore(":memory:")
    try:
        scan, _ = scanner(store)
        store.set_strategy_state(
            SID, StrategyState.RESEARCH, StrategyState.LIVE_CANARY, reason="test"
        )
        seed_edge(store)
        report = scan.scan(
            strategies=[
                ScanStrategy(
                    SID,
                    ScriptedStrategy(),
                    Market.CRYPTO,
                    "5Min",
                    base_risk_pct=0.20,
                )
            ],
            bars_by_symbol={"ETH/USD": bars()},
            equity=100_000.0,
            book=PortfolioBook(equity=100_000.0),
            now=NOW,
        )
        if report.executable:
            assert report.executable[0].risk_allocation_pct <= 0.05 + 1e-12
    finally:
        store.close()
