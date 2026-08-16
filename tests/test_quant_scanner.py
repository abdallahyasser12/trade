"""Scanner integration tests: shadow flow, measured-alpha gate, ranking,
portfolio grants, and journaling — the full decision pipeline in memory."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.contracts import (
    AttributionRecord,
    DecisionOutcome,
    Market,
    StrategyState,
)
from quant.execution import ExecutionQualityConfig, ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
from quant.marketplace import MIN_SAMPLE
from quant.portfolio import PortfolioBook
from quant.regime import RegimeConfig, RegimeEngine
from quant.scanner import QuantScanner, ScanStrategy
from quant.shadow import ShadowConfig, ShadowExecutor
from quant.store import QuantStore

NOW = datetime(2026, 1, 10, tzinfo=timezone.utc)
SID = "stub_strategy"


class StubStrategy:
    """Protocol stub emitting a scripted action on the LAST aligned bar."""

    def __init__(self, action=BotAction.BUY, symbols=("BTC/USD",)):
        self.action = action
        self.config = type("Cfg", (), {"symbols": tuple(symbols), "benchmark_symbol": symbols[0]})()

    def required_bars(self):
        return 5

    def prepare_symbol(self, bars):
        out = bars.copy()
        out["atr"] = 2.0
        return out

    def prepare_universe(self, bars_by_symbol):
        return {s: self.prepare_symbol(f) for s, f in bars_by_symbol.items()}

    def evaluate(self, bars_by_symbol, held_symbols=frozenset()):
        prepared = self.prepare_universe(bars_by_symbol)
        common = None
        for frame in prepared.values():
            common = frame.index if common is None else common.intersection(frame.index)
        return self.evaluate_prepared(prepared, common[-1], held_symbols)

    def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
        decisions = []
        for symbol, frame in prepared.items():
            close = float(frame.loc[pd.Timestamp(timestamp)]["close"])
            action = self.action
            if action is BotAction.BUY and symbol in held_symbols:
                action = BotAction.HOLD
            if action is BotAction.EXIT and symbol not in held_symbols:
                action = BotAction.HOLD
            decisions.append(
                CryptoSignalDecision(
                    symbol=symbol,
                    action=action,
                    timestamp=timestamp.to_pydatetime(),
                    reason="scripted",
                    close=close,
                    atr=2.0,
                    atr_percentile=50.0,
                    stop_price=close - 4.0 if action is BotAction.BUY else None,
                )
            )
        return decisions

    def ratchet_stop(self, current_stop, highest, close, atr_value):
        return current_stop, max(highest, close)


def make_bars(periods=120, seed=1, start_price=100.0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-12-01", periods=periods, freq="1h", tz="UTC")
    close = start_price * np.cumprod(1.0 + 0.0005 + rng.normal(0, 0.002, periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            # Liquid enough that a 0.35%-risk candidate stays far below the
            # participation cap — the cap itself is tested in the execution
            # module's own suite.
            "volume": np.full(periods, 100_000.0),
        },
        index=index,
    )


@pytest.fixture()
def store():
    s = QuantStore(":memory:")
    yield s
    s.close()


def make_scanner(store):
    lifecycle = LifecycleManager(store)
    return QuantScanner(
        store=store,
        lifecycle=lifecycle,
        execution_engine=ExecutionQualityEngine(
            ExecutionQualityConfig(uncertainty_margin_bps=1.0)
        ),
        regime_engine=RegimeEngine(
            RegimeConfig(
                trend_ma_period=30,
                trend_slope_period=5,
                fast_ma_period=10,
                momentum_period=20,
                vol_short_period=10,
                vol_long_period=40,
                vol_percentile_lookback=60,
                drawdown_lookback=60,
                correlation_window=20,
            )
        ),
        shadow_executor=ShadowExecutor(store, ShadowConfig()),
    ), lifecycle


def seed_attributions(store, n=MIN_SAMPLE + 2, r=1.0):
    for i in range(n):
        store.record_attribution(
            AttributionRecord(
                strategy_id=SID,
                strategy_version="1.0.0",
                symbol="BTC/USD",
                market=Market.CRYPTO,
                timeframe="1Hour",
                session_id=None,
                signal_timestamp=NOW - timedelta(days=1),
                entry_timestamp=NOW - timedelta(days=1),
                exit_timestamp=NOW - timedelta(hours=20),
                qty=1.0,
                entry_price=100.0,
                exit_price=100.0 + r * 4,
                initial_stop_price=96.0,
                fees=0.1,
                expected_cost_bps=None,
                realized_slippage_bps=None,
                gross_pnl=r * 4,
                net_pnl=r * 4,
                r_multiple=r if i % 3 else -0.5,
                max_favorable_excursion_pct=5.0,
                max_adverse_excursion_pct=1.0,
                entry_reason="scripted",
                exit_reason="stop",
                regime_at_entry="STRONG_BULL",
                shadow=True,
            )
        )


def test_research_state_strategies_are_not_scanned(store):
    scanner, _ = make_scanner(store)
    report = scanner.scan(
        strategies=[
            ScanStrategy(SID, StubStrategy(), Market.CRYPTO, "1Hour")
        ],
        bars_by_symbol={"BTC/USD": make_bars()},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    assert SID in report.skipped_strategies
    assert "research" in report.skipped_strategies[SID]
    assert report.decisions == []


def test_shadow_strategy_gets_virtual_fills_and_attribution(store):
    scanner, _ = make_scanner(store)
    store.set_strategy_state(
        SID, StrategyState.RESEARCH, StrategyState.SHADOW, reason="test"
    )
    bars = make_bars()
    entry_report = scanner.scan(
        strategies=[ScanStrategy(SID, StubStrategy(BotAction.BUY), Market.CRYPTO, "1Hour")],
        bars_by_symbol={"BTC/USD": bars},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    assert entry_report.shadow_counts[SID]["entries"] == 1
    positions = store.get_shadow_positions(SID)
    assert len(positions) == 1
    journaled = store.get_decisions(strategy_id=SID)
    assert journaled and journaled[0]["outcome"] == "SHADOW"

    exit_report = scanner.scan(
        strategies=[ScanStrategy(SID, StubStrategy(BotAction.EXIT), Market.CRYPTO, "1Hour")],
        bars_by_symbol={"BTC/USD": bars},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    assert exit_report.shadow_counts[SID]["exits"] == 1
    assert store.get_shadow_positions(SID) == []
    attributions = store.get_attributions(strategy_id=SID)
    assert len(attributions) == 1
    assert attributions[0]["shadow"] == 1


def test_paper_strategy_without_measured_expectancy_is_rejected(store):
    scanner, _ = make_scanner(store)
    store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
    report = scanner.scan(
        strategies=[ScanStrategy(SID, StubStrategy(BotAction.BUY), Market.CRYPTO, "1Hour")],
        bars_by_symbol={"BTC/USD": make_bars()},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    rejections = [
        d for d in report.decisions if "no_measured_expectancy" in d.rejection_reasons
    ]
    assert rejections, "unmeasured strategy must be journaled as rejected"
    assert not report.executable


def test_paper_strategy_with_measured_edge_executes_with_grant(store):
    scanner, _ = make_scanner(store)
    store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
    seed_attributions(store)
    report = scanner.scan(
        strategies=[ScanStrategy(SID, StubStrategy(BotAction.BUY), Market.CRYPTO, "1Hour")],
        bars_by_symbol={"BTC/USD": make_bars()},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    executable = report.executable
    assert executable, [d.to_dict() for d in report.decisions]
    decision = executable[0]
    assert decision.rank == 1
    assert decision.risk_allocation_pct > 0
    assert decision.expected_net_alpha_bps is not None
    assert decision.candidate.details["measured_trades"] >= MIN_SAMPLE
    # Journal contains the executed decision.
    journaled = store.get_decisions(strategy_id=SID, outcome="EXECUTE")
    assert journaled


def test_regime_is_assessed_and_attached(store):
    scanner, _ = make_scanner(store)
    store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
    seed_attributions(store)
    report = scanner.scan(
        strategies=[ScanStrategy(SID, StubStrategy(BotAction.BUY), Market.CRYPTO, "1Hour")],
        bars_by_symbol={"BTC/USD": make_bars()},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    assert report.regime is not None
    assert report.regime.confidence > 0


def test_scan_with_no_aligned_bars_returns_empty_report(store):
    scanner, _ = make_scanner(store)
    report = scanner.scan(
        strategies=[],
        bars_by_symbol={},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
    )
    assert report.timestamp is None
    assert report.decisions == []
