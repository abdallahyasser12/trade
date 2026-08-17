"""The equity research CLI.

Before this there was no equity research entry point at all: the only
quant-desk research CLI hardcodes Market.CRYPTO in every strategy factory,
fetches from AlpacaCryptoDataFeed, and annualizes with
crypto_bars_per_year. opening_range_breakout — the one equities-only
strategy in the registry — could not be researched by any command.

These tests run the CLI in-process. The real-data path is exercised
through a fixture bar source, so nothing here needs credentials or a
network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from config import bars_per_year
from crypto_config import crypto_bars_per_year
from quant.contracts import Market
from quant.store import QuantStore
from scripts.run_equity_research import (
    DEFAULT_UNIVERSE,
    ENGINES,
    _quality_problems,
    _strategy_factory,
    _synthetic_equity_bars,
    build_parser,
    main,
)


# ---------------------------------------------------------------------------
# Annualization — the error that would manufacture a passing Sharpe
# ---------------------------------------------------------------------------


def test_equity_annualization_is_not_the_crypto_factor():
    """A 6.5-hour session is not a 24-hour day.

    Annualizing hourly equity returns with crypto's 8766 bars/yr instead of
    252*7 overstates Sharpe by sqrt(8766/1764) ~ 2.2x — straight through
    the min_aggregate_sharpe gate.
    """
    assert bars_per_year("1Hour") == 252 * 7
    assert crypto_bars_per_year("1Hour") > 4 * bars_per_year("1Hour")
    assert bars_per_year("15Min") == 252 * 26
    assert bars_per_year("1Day") == 252.0


def test_the_cli_passes_the_equity_annualization_factor(monkeypatch):
    captured = {}
    import scripts.run_equity_research as cli

    real = cli.run_quant_walk_forward

    def spy(**kwargs):
        captured["annualization"] = kwargs["sim_config"].annualization_factor
        return real(**kwargs)

    monkeypatch.setattr(cli, "run_quant_walk_forward", spy)
    main([
        "--engine", "orb", "--timeframe", "15Min", "--synthetic",
        "--synthetic-sessions", "40", "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", ":memory:", "--no-record",
    ])
    assert captured["annualization"] == bars_per_year("15Min")


# ---------------------------------------------------------------------------
# Strategy construction — all four engines, bound to US_EQUITY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_builds_a_us_equity_strategy(engine):
    build, grid = _strategy_factory(engine, ("SPY", "QQQ"), "15Min", "SPY")
    assert grid, "every engine needs a candidate grid"
    strategy = build(grid[0])
    assert Market(strategy.config.market) is Market.US_EQUITY
    assert strategy.config.timeframe == "15Min"
    assert strategy.config.resolved_benchmark == "SPY"


def test_opening_range_breakout_finally_has_a_research_path():
    # It is registered as equities-only and had no CLI able to run it.
    assert "orb" in ENGINES
    build, grid = _strategy_factory("orb", ("SPY",), "15Min", "SPY")
    strategy = build(grid[0])
    assert type(strategy).__name__ == "OpeningRangeBreakoutStrategy"


def test_grids_are_small_because_candidates_cost_multiplicity():
    # Every candidate multiplies trials_this_run by the fold count.
    for engine in ENGINES:
        _, grid = _strategy_factory(engine, ("SPY",), "15Min", "SPY")
        assert len(grid) <= 6, f"{engine} grid is too wide for its evidence budget"


# ---------------------------------------------------------------------------
# Synthetic bars are laid out on REAL sessions
# ---------------------------------------------------------------------------


def test_synthetic_bars_fall_only_on_trading_days():
    from quant.data.calendar import MARKET_TZ, is_trading_day

    bars, _ = _synthetic_equity_bars(("SPY",), "15Min", 40)
    days = {ts.date() for ts in bars["SPY"].index.tz_convert(MARKET_TZ)}
    assert days
    assert all(is_trading_day(day) for day in days), (
        "a flat date_range would include weekends and holidays, and an "
        "opening-range strategy on a series with no sessions is meaningless"
    )


def test_synthetic_bars_respect_regular_session_hours():
    from quant.data.calendar import MARKET_TZ

    bars, _ = _synthetic_equity_bars(("SPY",), "15Min", 20)
    times = {ts.strftime("%H:%M") for ts in bars["SPY"].index.tz_convert(MARKET_TZ)}
    assert min(times) >= "09:30"
    assert max(times) < "16:00"


def test_synthetic_daily_bars_are_one_per_session():
    bars, _ = _synthetic_equity_bars(("SPY",), "1Day", 30)
    assert len(bars["SPY"]) == 30


def test_synthetic_is_deterministic():
    first, _ = _synthetic_equity_bars(("SPY", "QQQ"), "15Min", 50)
    second, _ = _synthetic_equity_bars(("SPY", "QQQ"), "15Min", 50)
    for symbol in first:
        assert first[symbol].equals(second[symbol])


# ---------------------------------------------------------------------------
# Data-quality gating
# ---------------------------------------------------------------------------


def test_quality_problems_flags_missing_and_incomplete():
    provenance = {
        "SPY": {"quality": "OK", "completeness": {"coverage": 1.0}},
        "QQQ": {"quality": "INCOMPLETE", "completeness": {"coverage": 0.7}},
        "GONE": {"quality": "MISSING", "completeness": {"coverage": 0.0}},
    }
    problems = _quality_problems(provenance)
    assert len(problems) == 2
    assert any("QQQ" in p for p in problems)
    assert any("GONE" in p for p in problems)


def test_a_historical_window_marked_stale_is_not_a_problem():
    # STALE only means "not current", which a backtest window always is.
    provenance = {"SPY": {"quality": "STALE", "completeness": {"coverage": 1.0}}}
    assert _quality_problems(provenance) == []


def test_an_incomplete_universe_fails_closed(tmp_path, monkeypatch):
    """Researching around missing history makes a strategy look like it had
    no signals when it simply had no data."""
    import scripts.run_equity_research as cli

    def fake_fetch(symbols, timeframe, start, end, **kwargs):
        bars, _ = _synthetic_equity_bars(tuple(symbols), timeframe, 40)
        provenance = {
            s: {"quality": "INCOMPLETE", "completeness": {"coverage": 0.42}}
            for s in symbols
        }
        return bars, provenance

    monkeypatch.setattr(cli, "_real_equity_bars", fake_fetch)
    code = main([
        "--engine", "orb", "--symbols", "SPY", "QQQ",
        "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / "q.db"), "--no-record",
    ])
    assert code == 2


def test_allow_incomplete_lets_it_proceed(tmp_path, monkeypatch):
    import scripts.run_equity_research as cli

    def fake_fetch(symbols, timeframe, start, end, **kwargs):
        bars, _ = _synthetic_equity_bars(tuple(symbols), timeframe, 40)
        provenance = {
            s: {"quality": "INCOMPLETE", "completeness": {"coverage": 0.9}}
            for s in symbols
        }
        return bars, provenance

    monkeypatch.setattr(cli, "_real_equity_bars", fake_fetch)
    code = main([
        "--engine", "orb", "--symbols", "SPY", "QQQ", "--allow-incomplete",
        "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / "q.db"), "--no-record",
    ])
    assert code in (0, 1), "should reach a verdict, not bail out"


# ---------------------------------------------------------------------------
# Input validation and exit codes
# ---------------------------------------------------------------------------


def test_missing_credentials_is_exit_two_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert main(["--engine", "orb", "--db", str(tmp_path / "q.db"), "--no-record"]) == 2


def test_an_invalid_symbol_is_rejected_before_any_fetch(tmp_path):
    assert main([
        "--symbols", "not a symbol", "--db", str(tmp_path / "q.db"), "--no-record",
    ]) == 2


def test_a_benchmark_outside_the_universe_is_rejected(tmp_path):
    assert main([
        "--symbols", "SPY", "QQQ", "--benchmark", "IWM",
        "--db", str(tmp_path / "q.db"), "--no-record",
    ]) == 2


def test_a_reversed_date_range_is_rejected(tmp_path):
    assert main([
        "--start", "2024-06-01", "--end", "2024-01-01",
        "--db", str(tmp_path / "q.db"), "--no-record",
    ]) == 2


def test_four_hour_equities_is_not_offered():
    # A 6.5h session does not divide into 4-hour buckets.
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--timeframe", "4Hour"])


def test_duplicate_symbols_are_collapsed(tmp_path):
    code = main([
        "--engine", "orb", "--symbols", "SPY", "SPY", "QQQ", "--synthetic",
        "--synthetic-sessions", "40", "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / "q.db"),
        "--no-record",
    ])
    assert code in (0, 1)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_synthetic_run_reaches_a_verdict_and_records_lineage(tmp_path):
    db = str(tmp_path / "desk.db")
    code = main([
        "--engine", "orb", "--timeframe", "15Min", "--synthetic",
        "--synthetic-sessions", "40", "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", db, "--seed", "7",
    ])
    assert code in (0, 1)

    store = QuantStore(db)
    runs = store.get_research_runs("opening_range_breakout_synthetic")
    assert len(runs) == 1
    # Synthetic results are quarantined under their own strategy id so they
    # can never satisfy a real promotion request.
    assert store.total_trials("opening_range_breakout") == 0
    assert store.total_trials("opening_range_breakout_synthetic") > 0
    store.close()


def test_trials_recorded_are_candidates_times_folds(tmp_path):
    db = str(tmp_path / "desk.db")
    main([
        "--engine", "orb", "--synthetic", "--synthetic-sessions", "40",
        "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", db, "--json", str(tmp_path / "r.json"),
    ])
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["trials_this_run"] == 6 * payload["folds"]
    store = QuantStore(db)
    assert store.total_trials("opening_range_breakout_synthetic") == payload[
        "trials_this_run"
    ]
    store.close()


def test_the_json_report_carries_provenance_and_the_benchmark_caveat(tmp_path):
    out = tmp_path / "r.json"
    main([
        "--engine", "vwap", "--synthetic", "--synthetic-sessions", "40",
        "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / "d.db"), "--no-record", "--json", str(out),
    ])
    payload = json.loads(out.read_text())
    assert payload["market"] == Market.US_EQUITY.value
    assert payload["synthetic"] is True
    assert "data_provenance" in payload
    assert payload["protocol"]["benchmark_is_cost_free"] is True
    assert set(payload["universe"]) <= set(DEFAULT_UNIVERSE)


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_runs_end_to_end_on_synthetic_data(tmp_path, engine):
    # 300 training bars clears the widest candidate warmup (the squeeze
    # family looks back 100 bars for its percentile on top of the shared
    # ATR warmup); a narrower window is a starved run, not a failing one.
    code = main([
        "--engine", engine, "--synthetic", "--synthetic-sessions", "40",
        "--train-bars", "300", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / f"{engine}.db"), "--no-record",
    ])
    assert code in (0, 1), f"{engine} could not complete a run"


def test_a_training_window_narrower_than_the_warmup_says_so(tmp_path):
    """The old message blamed missing history for a window problem.

    With plenty of bars but train_bars below the candidate warmup, every
    window is rejected and folds ends up empty — which used to raise "Not
    enough aligned history", sending the operator to fetch more data when
    the actual fix is a wider training window.
    """
    from quant.research.factory import (
        QuantPromotionCriteria,
        QuantWalkForwardConfig,
        run_quant_walk_forward,
    )
    from quant.research.simulator import SimulationConfig

    bars, _ = _synthetic_equity_bars(("SPY",), "15Min", 40)
    build, grid = _strategy_factory("squeeze", ("SPY",), "15Min", "SPY")

    with pytest.raises(ValueError) as excinfo:
        run_quant_walk_forward(
            strategy_id="squeeze_probe",
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(annualization_factor=bars_per_year("15Min")),
            walk_config=QuantWalkForwardConfig(
                train_bars=150, test_bars=60, step_bars=400
            ),
            criteria=QuantPromotionCriteria(),
        )

    message = str(excinfo.value)
    # The distinction that matters: enough bars existed, so "go fetch more
    # history" is the wrong advice. Here every window was rejected because
    # no candidate cleared min_train_trades in a 150-bar window.
    assert "is enough history" in message, message
    assert "min_train_trades" in message, message
    assert "starved run, not a passing one" in message, message
    assert "Not enough aligned history" not in message

    # And the CLI surfaces it as could-not-run rather than as a verdict.
    assert main([
        "--engine", "squeeze", "--synthetic", "--synthetic-sessions", "40",
        "--train-bars", "150", "--test-bars", "60", "--step-bars", "400",
        "--db", str(tmp_path / "s.db"), "--no-record",
    ]) == 2
