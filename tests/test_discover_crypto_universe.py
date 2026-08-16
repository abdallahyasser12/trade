"""The universe probe must rank by depth, cap at 12, and keep the benchmark."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "discover_crypto_universe.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("discover_crypto_universe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["discover_crypto_universe"] = module
    spec.loader.exec_module(module)
    return module


def _daily_bars(periods, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.05, 1.0, periods).cumsum()
    close = np.maximum(close, 5.0)
    open_price = np.r_[close[0], close[:-1]]
    index = pd.date_range("2021-01-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.5,
            "low": np.minimum(open_price, close) - 0.5,
            "close": close,
            "volume": rng.uniform(900.0, 1_100.0, periods),
        },
        index=index,
    )


def _module_with_stub(monkeypatch, depths):
    module = _load()
    monkeypatch.setattr(
        module, "load_credentials",
        lambda *a, **k: type("C", (), {"api_key": "k", "secret_key": "s"})(),
    )
    frames = {
        symbol: _daily_bars(periods, seed=i)
        for i, (symbol, periods) in enumerate(depths.items())
    }

    class StubFeed:
        def __init__(self, *args, **kwargs):
            pass

        def get_bars(self, symbol, timeframe, start=None, end=None):
            if symbol not in frames:
                raise ValueError(f"{symbol} is not served")
            return frames[symbol]

    monkeypatch.setattr(module, "AlpacaCryptoDataFeed", StubFeed)
    return module


def test_recommendation_ranks_by_depth_and_reports_alignment(monkeypatch, capsys):
    module = _module_with_stub(
        monkeypatch,
        {"BTC/USD": 2_000, "ETH/USD": 2_000, "SOL/USD": 1_200, "DOGE/USD": 1_950},
    )
    code = module.main(
        ["--candidates", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,FAKE/USD",
         "--min-bars", "1800"]
    )
    out = capsys.readouterr().out
    assert code == 0
    # Shallow and unserved symbols are reported but excluded.
    assert "below --min-bars" in out
    assert "FAKE/USD" in out
    assert '--symbols "BTC/USD,DOGE/USD,ETH/USD"' in out or \
           '--symbols "BTC/USD,ETH/USD,DOGE/USD"' in out
    assert "Aligned daily bars" in out


def test_recommendation_caps_at_twelve_and_keeps_the_benchmark(monkeypatch, capsys, tmp_path):
    depths = {f"C{i:02d}/USD": 2_050 - i for i in range(14)}
    depths["BTC/USD"] = 1_850  # qualifies, but would fall outside a naive top-12
    module = _module_with_stub(monkeypatch, depths)
    target = tmp_path / "universe.json"
    code = module.main(
        ["--candidates", ",".join(depths), "--min-bars", "1800",
         "--json-out", str(target)]
    )
    assert code == 0
    payload = json.loads(target.read_text())
    recommended = payload["recommended"]
    assert len(recommended) == 12
    assert "BTC/USD" in recommended
    assert payload["aligned_bars"] > 0


def test_all_probes_failing_exits_with_the_configuration_code(monkeypatch, capsys):
    module = _module_with_stub(monkeypatch, {})
    code = module.main(["--candidates", "AAA/USD,BBB/USD"])
    assert code == 2
    assert "No symbol met the depth requirement" in capsys.readouterr().err
