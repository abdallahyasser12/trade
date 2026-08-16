"""Portfolio heat, cap reasons, and greedy allocator tests."""

import numpy as np
import pandas as pd
import pytest

from quant.contracts import Market
from quant.portfolio import (
    BookPosition,
    HeatCheckResult,
    PortfolioBook,
    PortfolioLimitsConfig,
    PortfolioRiskGovernor,
    sector_for,
)


def make_book(equity=100_000.0, positions=None):
    return PortfolioBook(equity=equity, positions=positions or [])


def position(symbol="AAPL", market=Market.US_EQUITY, strategy="s1", notional=10_000.0, risk=500.0):
    return BookPosition(
        symbol=symbol,
        market=market,
        sector=sector_for(symbol, market),
        strategy_id=strategy,
        notional=notional,
        risk_at_stop=risk,
    )


def series(seed, periods=80):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq="1D", tz="UTC")
    return pd.Series(rng.normal(0, 0.01, periods), index=index)


def test_book_arithmetic():
    book = make_book(positions=[position()])
    assert book.heat_pct == pytest.approx(0.5)
    assert book.gross_exposure_pct == pytest.approx(10.0)
    assert book.market_exposure_pct(Market.US_EQUITY) == pytest.approx(10.0)
    assert book.sector_exposure_pct("technology") == pytest.approx(10.0)
    assert book.strategy_heat_pct("s1") == pytest.approx(0.5)


def test_each_cap_produces_named_reason():
    governor = PortfolioRiskGovernor(
        PortfolioLimitsConfig(
            max_portfolio_heat_pct=1.0,
            max_gross_exposure_pct=15.0,
            max_market_exposure_pct=12.0,
            max_sector_exposure_pct=11.0,
            max_strategy_heat_pct=0.6,
        )
    )
    book = make_book(positions=[position()])
    check = governor.check(
        book,
        candidate_symbol="MSFT",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s1",
        add_notional=10_000.0,
        add_risk=600.0,
    )
    assert not check.approved
    assert {
        "portfolio_heat_cap",
        "gross_exposure_cap",
        "market_exposure_cap",
        "sector_exposure_cap",
        "strategy_heat_cap",
    }.issubset(set(check.reasons))


def test_many_small_trades_blocked_by_portfolio_heat():
    # Five individually fine 0.4% risk positions; the sixth breaches the
    # 2% portfolio heat cap even though it is individually acceptable.
    governor = PortfolioRiskGovernor(
        PortfolioLimitsConfig(max_portfolio_heat_pct=2.0, max_strategy_heat_pct=2.0)
    )
    positions = [
        position(symbol=s, notional=5_000.0, risk=400.0)
        for s in ("AAPL", "MSFT", "JPM", "XOM", "UNH")
    ]
    book = make_book(positions=positions)
    check = governor.check(
        book,
        candidate_symbol="WMT",
        market=Market.US_EQUITY,
        sector="consumer_staples",
        strategy_id="s1",
        add_notional=5_000.0,
        add_risk=400.0,
    )
    assert not check.approved
    assert "portfolio_heat_cap" in check.reasons


def test_correlation_gate():
    governor = PortfolioRiskGovernor(
        PortfolioLimitsConfig(max_pairwise_correlation=0.8, min_correlation_observations=30)
    )
    book = make_book(positions=[position(symbol="AAPL")])
    base = series(1)
    correlated = base * 1.02 + 0.0001
    uncorrelated = series(2)
    blocked = governor.check(
        book,
        candidate_symbol="MSFT",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s2",
        add_notional=1_000.0,
        add_risk=50.0,
        candidate_returns=correlated,
        held_returns={"AAPL": base},
    )
    assert "correlated_exposure" in blocked.reasons
    passed = governor.check(
        book,
        candidate_symbol="MSFT",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s2",
        add_notional=1_000.0,
        add_risk=50.0,
        candidate_returns=uncorrelated,
        held_returns={"AAPL": base},
    )
    assert "correlated_exposure" not in passed.reasons


def test_unknown_correlation_default_allows():
    governor = PortfolioRiskGovernor()
    book = make_book(positions=[position()])
    check = governor.check(
        book,
        candidate_symbol="MSFT",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s2",
        add_notional=1_000.0,
        add_risk=50.0,
        candidate_returns=series(1, periods=5),
        held_returns={"AAPL": series(2, periods=5)},
    )
    assert "correlation_unknown" not in check.reasons
    strict = PortfolioRiskGovernor(PortfolioLimitsConfig(reject_unknown_correlation=True))
    check = strict.check(
        book,
        candidate_symbol="MSFT",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s2",
        add_notional=1_000.0,
        add_risk=50.0,
        candidate_returns=series(1, periods=5),
        held_returns={"AAPL": series(2, periods=5)},
    )
    assert "correlation_unknown" in check.reasons


def test_zero_equity_fails_closed():
    governor = PortfolioRiskGovernor()
    check = governor.check(
        make_book(equity=0.0),
        candidate_symbol="AAPL",
        market=Market.US_EQUITY,
        sector="technology",
        strategy_id="s1",
        add_notional=1_000.0,
        add_risk=50.0,
    )
    assert check == HeatCheckResult(approved=False, reasons=["non_positive_equity"])


def test_allocator_grants_down_the_queue_within_budget():
    governor = PortfolioRiskGovernor(
        PortfolioLimitsConfig(
            max_portfolio_heat_pct=2.0,
            max_strategy_heat_pct=2.0,
            max_new_risk_per_cycle_pct=0.8,
        )
    )
    book = make_book()
    ranked = [
        {
            "key": f"s1:{symbol}",
            "requested_risk_pct": 0.35,
            "notional": 5_000.0,
            "symbol": symbol,
            "market": "us_equity",
            "sector": sector_for(symbol, Market.US_EQUITY),
            "strategy_id": "s1",
        }
        for symbol in ("AAPL", "JPM", "XOM")
    ]
    grants = governor.allocate(book, ranked)
    assert grants["s1:AAPL"] == pytest.approx(0.35)
    assert grants["s1:JPM"] == pytest.approx(0.35)
    # Cycle budget 0.8 leaves only 0.10 for the third candidate.
    assert grants["s1:XOM"] == pytest.approx(0.10)


def test_allocator_deterministic_and_respects_caps():
    governor = PortfolioRiskGovernor(
        PortfolioLimitsConfig(max_sector_exposure_pct=6.0)
    )
    book = make_book()
    ranked = [
        {
            "key": f"s1:{symbol}",
            "requested_risk_pct": 0.2,
            "notional": 5_000.0,
            "symbol": symbol,
            "market": "us_equity",
            "sector": "technology",
            "strategy_id": "s1",
        }
        for symbol in ("AAPL", "MSFT", "NVDA")
    ]
    first = governor.allocate(book, ranked)
    second = governor.allocate(book, ranked)
    assert first == second
    # 5k notional each on 100k equity = 5% per position; the sector cap of
    # 6% admits exactly one.
    granted = [key for key, value in first.items() if value > 0]
    assert granted == ["s1:AAPL"]


def test_config_validation():
    assert PortfolioLimitsConfig(max_portfolio_heat_pct=0.0).validate()
    assert PortfolioLimitsConfig(
        max_strategy_heat_pct=5.0, max_portfolio_heat_pct=2.0
    ).validate()
    assert not PortfolioLimitsConfig().validate()
