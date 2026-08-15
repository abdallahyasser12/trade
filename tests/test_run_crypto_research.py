"""The research CLI must survive to its verdict, not crash after the download."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_crypto_research.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_crypto_research", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crypto_research"] = module
    spec.loader.exec_module(module)
    return module


def _bars(periods=2_000, seed=0, slope=0.06, cycle=45):
    """Uptrend with regular pullbacks, so a breakout strategy actually trades.

    A plain random walk produces too few entries to exercise the report.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    base = 100.0 * np.exp(slope * t / 100.0)
    pullback = 1.0 - 0.045 * (np.sin(2 * np.pi * t / cycle) < 0) * np.abs(
        np.sin(2 * np.pi * t / cycle)
    )
    close = base * pullback * (1 + rng.normal(0, 0.002, periods))
    open_price = np.r_[close[0], close[:-1]]
    index = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.004,
            "low": np.minimum(open_price, close) * 0.996,
            "close": close,
            "volume": rng.uniform(900.0, 1_100.0, periods),
        },
        index=index,
    )


def test_missing_credentials_exit_with_the_configuration_code(monkeypatch, capsys):
    module = _load()
    for name in (
        "ALPACA_PAPER_API_KEY",
        "ALPACA_PAPER_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "load_credentials", lambda *a, **k: type(
        "C", (), {"api_key": "", "secret_key": ""}
    )())
    assert module.main([]) == 2
    assert "credentials" in capsys.readouterr().err


def test_full_report_renders_and_returns_a_promotion_exit_code(monkeypatch, capsys):
    """Runs everything except the network: fetch is stubbed, the rest is real."""

    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )

    frames = {
        "BTC/USD": _bars(seed=1),
        "ETH/USD": _bars(seed=2),
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)

    code = module.main(
        [
            "--symbols", "BTC/USD,ETH/USD",
            "--benchmark", "BTC/USD",
            "--train-bars", "700",
            "--test-bars", "300",
            "--min-train-trades", "1",
        ]
    )
    out = capsys.readouterr().out

    # The verdict must be reachable regardless of which way it goes.
    assert code in (0, 1)
    assert "COST" in out
    assert "Round-trip friction" in out
    assert "VERSUS BUY-AND-HOLD" in out
    assert "PROMOTION VERDICT" in out
    assert ("PASS" in out) or ("FAIL" in out)


def test_ensemble_engine_runs_and_reports_a_verdict(monkeypatch, capsys):
    """The ensemble engine must produce a structurally valid report too."""

    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )

    frames = {
        "BTC/USD": _bars(seed=1),
        "ETH/USD": _bars(seed=2),
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)

    code = module.main(
        [
            "--symbols", "BTC/USD,ETH/USD",
            "--benchmark", "BTC/USD",
            "--train-bars", "700",
            "--test-bars", "300",
            "--min-train-trades", "1",
            "--engine", "ensemble",
        ]
    )
    out = capsys.readouterr().out

    # The verdict must be reachable regardless of which way it goes.
    assert code in (0, 1)
    assert "COST" in out
    assert "Round-trip friction" in out
    assert "VERSUS BUY-AND-HOLD" in out
    assert "PROMOTION VERDICT" in out
    assert ("PASS" in out) or ("FAIL" in out)


def test_default_engine_is_base(monkeypatch):
    module = _load()
    args = module._parse_args(["--symbols", "BTC/USD"])
    assert args.engine == "base"


def test_json_output_is_written_and_parsable(monkeypatch, tmp_path):
    import json

    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )
    frames = {"BTC/USD": _bars(seed=3), "ETH/USD": _bars(seed=4)}

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
    target = tmp_path / "verdict.json"
    module.main(
        [
            "--symbols", "BTC/USD,ETH/USD",
            "--train-bars", "700",
            "--test-bars", "300",
            "--min-train-trades", "1",
            "--json-out", str(target),
        ]
    )
    payload = json.loads(target.read_text())
    assert "cost" in payload and "benchmark" in payload and "promotion" in payload
    assert payload["config"]["timeframe"] == "1Hour"
    assert isinstance(payload["promotion"]["promoted"], bool)


def test_empty_vendor_response_is_reported_not_crashed(monkeypatch, capsys):
    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )

    class EmptyFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz="UTC"),
            )

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", EmptyFeed)
    assert module.main(["--symbols", "BTC/USD"]) == 2
    assert "no bars" in capsys.readouterr().err


def test_slow_timeframes_search_a_scaled_grid_and_fast_ones_keep_the_original():
    """Slow bars need shorter breakouts; hourly keeps its audited grid."""

    from crypto_research import TUNABLE_CRYPTO_PARAMETERS

    module = _load()
    strategy_config = module.CryptoStrategyConfig(symbols=("BTC/USD",))
    daily = module._parameter_grid(strategy_config, "1Day")
    hourly = module._parameter_grid(strategy_config, "1Hour")
    assert 10 in daily["breakout_period"]
    assert 15.0 in daily["min_adx"]
    assert set(daily) <= TUNABLE_CRYPTO_PARAMETERS
    assert set(hourly) <= TUNABLE_CRYPTO_PARAMETERS
    candidates = 1
    for values in daily.values():
        candidates *= len(values)
    assert candidates <= 32
    assert hourly["breakout_period"] == [
        strategy_config.breakout_period, strategy_config.breakout_period + 10
    ]


def test_embargo_defaults_to_the_strategy_warmup(monkeypatch, capsys):
    """A 1-bar embargo against a 226-bar warmup is silent train/test leakage."""

    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )
    frames = {"BTC/USD": _bars(seed=5), "ETH/USD": _bars(seed=6)}

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
    code = module.main(
        ["--symbols", "BTC/USD,ETH/USD", "--train-bars", "700",
         "--test-bars", "300", "--min-train-trades", "1"]
    )
    out = capsys.readouterr().out
    assert code in (0, 1)
    warmup = module.CryptoStrategyConfig(
        symbols=("BTC/USD", "ETH/USD")
    ).warmup_bars()
    assert f"embargo {warmup} (warmup-derived)" in out


def test_cross_engine_searches_its_own_parameters_not_the_breakout_grid():
    """The breakout grid varies keys the cross engine never reads."""

    module = _load()
    from crypto_config import CryptoStrategyConfig

    strategy = CryptoStrategyConfig(symbols=("BTC/USD", "ETH/USD", "SOL/USD"))
    grid = module._cross_parameter_grid(strategy)
    assert "cross" in grid
    assert not {"breakout_period", "min_adx", "min_efficiency_ratio"} & set(grid)
    # The stop axis must not collapse to a single value against the default.
    assert len(grid["atr_stop_multiplier"]) == 2

    from cross_sectional import CrossSectionalConfig
    from cross_sectional_research import _candidate_builder

    candidates = _candidate_builder(CrossSectionalConfig())(
        strategy, grid, max_candidates=32
    )
    identities = {repr(sorted(c.to_dict().items())) for c in candidates}
    assert len(candidates) == 24
    assert len(identities) == 24, "candidates collapsed: the search would be a no-op"
    # Both regime settings must be present, or a bear-market training window
    # has nothing to select and the whole run aborts on its first fold.
    assert {c.cross.require_benchmark_bull for c in candidates} == {True, False}


def test_cross_engine_runs_end_to_end_and_reports_what_each_fold_selected(
    monkeypatch, capsys
):
    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )
    frames = {
        "BTC/USD": _bars(seed=1),
        "ETH/USD": _bars(seed=2, slope=0.09),
        "SOL/USD": _bars(seed=3, slope=0.03),
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
    code = module.main(
        [
            "--symbols", "BTC/USD,ETH/USD,SOL/USD",
            "--benchmark", "BTC/USD",
            "--engine", "cross",
            "--train-bars", "700",
            "--test-bars", "300",
            "--min-train-trades", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "PROMOTION VERDICT" in out
    assert "Candidates per fold        24" in out
    assert "Selected per fold (in-sample):" in out
    assert "lookback" in out


def test_a_late_listing_is_named_rather_than_silently_truncating_everything(
    monkeypatch, capsys
):
    """The failure this reproduces: one 2023 listing ate 65% of the window."""

    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )
    full = _bars(periods=2_000, seed=1)
    frames = {
        "BTC/USD": full,
        "ETH/USD": _bars(periods=2_000, seed=2),
        "LATE/USD": _bars(periods=2_000, seed=3).iloc[-700:],
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
    code = module.main(
        [
            "--symbols", "BTC/USD,ETH/USD,LATE/USD",
            "--benchmark", "BTC/USD",
            "--train-bars", "700",
            "--test-bars", "300",
        ]
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    # Whatever the outcome, the symbol responsible must be named.
    assert "LATE/USD" in out
    assert "alignment cost" in out
    assert "--drop-late-listings" in out
    # 700 aligned bars cannot serve 700 train + 226 embargo + 300 test.
    assert code == 2
    assert "too few for one fold" in out


def test_too_little_aligned_history_fails_before_the_folds_not_during():
    """A vague 'walk-forward could not complete' is a worse error than this."""

    module = _load()
    frames = {
        "BTC/USD": _bars(periods=2_000, seed=1),
        "ETH/USD": _bars(periods=2_000, seed=2).iloc[-400:],
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            return frames[symbol]

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            module, "load_credentials",
            lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
        )
        mp.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
        code = module.main(
            [
                "--symbols", "BTC/USD,ETH/USD",
                "--benchmark", "BTC/USD",
                "--train-bars", "700",
                "--test-bars", "300",
            ]
        )
    assert code == 2


def test_dropping_late_listings_recovers_the_window_and_keeps_the_benchmark():
    module = _load()
    index = pd.date_range("2021-01-01", periods=2_000, freq="h", tz="UTC")
    frames = {
        "BTC/USD": pd.DataFrame({"close": 1.0}, index=index),
        "ETH/USD": pd.DataFrame({"close": 1.0}, index=index),
        "SOL/USD": pd.DataFrame({"close": 1.0}, index=index),
        "LATE/USD": pd.DataFrame({"close": 1.0}, index=index[-500:]),
    }
    kept, dropped = module._drop_late_listings(frames, "BTC/USD")
    assert [symbol for symbol, _ in dropped] == ["LATE/USD"]
    assert set(kept) == {"BTC/USD", "ETH/USD", "SOL/USD"}
    assert module._aligned_length(kept) == 2_000
    assert "recovers 1500 aligned bars" in dropped[0][1]

    # Selection is by measured impact, not start date: a symbol that begins on
    # day one but is missing scattered days limits the window just as much, and
    # a start-date rule would never see it.
    gappy = dict(frames)
    del gappy["LATE/USD"]
    gappy["GAPPY/USD"] = pd.DataFrame({"close": 1.0}, index=index.delete(slice(100, 900)))
    kept, dropped = module._drop_late_listings(gappy, "BTC/USD")
    assert [symbol for symbol, _ in dropped] == ["GAPPY/USD"]
    assert module._aligned_length(kept) == 2_000

    # The benchmark is never dropped, even when it is the late one.
    benchmark_late = dict(frames)
    benchmark_late["BTC/USD"] = pd.DataFrame({"close": 1.0}, index=index[-500:])
    kept, _ = module._drop_late_listings(benchmark_late, "BTC/USD")
    assert "BTC/USD" in kept

    # And a universe already at the floor is left alone.
    three = {k: frames[k] for k in ("BTC/USD", "ETH/USD", "LATE/USD")}
    kept, dropped = module._drop_late_listings(three, "BTC/USD")
    assert dropped == []
    assert set(kept) == set(three)
