"""The ledger must be read honestly, or it is worse than not reading it.

Every backtest in this repository charges an assumed 10 bps of slippage. These
tests pin the properties that decide whether the measured replacement can be
trusted more than the assumption it replaces.
"""

import math

import numpy as np
import pytest

from execution_calibration import (
    ENTRY_PURPOSES,
    EXIT_PURPOSES,
    MIN_SAMPLES_FOR_RECOMMENDATION,
    Z_95_ONE_SIDED,
    _percentile,
    calibrate_from_executions,
    calibrate_from_store,
    student_t_95,
    summarise,
)
from state_store import StateStore


def _fills(count, slippage=8.0, purpose="entry", price=100.0):
    return [
        {
            "slippage_bps": slippage,
            "purpose": purpose,
            "fill_price": price,
            "filled_qty": 1.0,
            "fees": 0.25,
        }
        for _ in range(count)
    ]


def test_percentile_matches_numpy_across_awkward_lengths():
    """Hand-rolled interpolation is only acceptable if it is actually right."""

    for values in (
        [1.0],
        [1.0, 2.0],
        [3.0, 1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 1.0, 4.0, 2.0, 3.0],
        [-4.0, 0.0, 12.5, 3.25, 7.0, 9.0],
    ):
        for fraction in (0.0, 0.25, 0.5, 0.9, 1.0):
            assert _percentile(values, fraction) == pytest.approx(
                float(np.percentile(values, fraction * 100.0))
            ), (values, fraction)


def test_the_planning_number_actually_covers_the_true_mean_on_skewed_data():
    """The property, not the formula.

    An earlier test asserted `mean + z*stdev/sqrt(n)`, which only checked that
    the code matched itself. It could never have caught the real defect: that
    construction covers the true mean about 74% of the time on heavily skewed
    data while claiming 95%, erring in the one direction that matters. This
    measures coverage instead, so a regression to any under-covering estimator
    fails here.
    """

    rng = np.random.default_rng(20260816)
    for sigma, floor in ((0.5, 0.93), (1.0, 0.93), (1.5, 0.90)):
        true_mean = math.exp(sigma ** 2 / 2.0)
        covered = 0
        trials = 400
        for _ in range(trials):
            sample = rng.lognormal(0.0, sigma, MIN_SAMPLES_FOR_RECOMMENDATION)
            if summarise(sample.tolist()).planning_bps >= true_mean:
                covered += 1
        rate = covered / trials
        assert rate >= floor, f"sigma={sigma} covered only {rate:.1%}"


def test_the_planning_number_is_floored_by_p90_when_the_tail_is_heavy():
    """The p90 floor is what rescues coverage; without it the bound fails."""

    heavy = [1.0] * 40 + [500.0] * 5
    stats = summarise(heavy)
    mean = float(np.mean(heavy))
    stdev = float(np.std(heavy, ddof=1))
    t_bound = mean + student_t_95(len(heavy) - 1) * stdev / math.sqrt(len(heavy))
    assert stats.planning_bps == pytest.approx(max(t_bound, stats.p90_bps))
    assert stats.planning_bps >= stats.p90_bps
    assert stats.skewness > 1.0


def test_student_t_matches_published_quantiles():
    for df, expected in ((5, 2.015), (10, 1.812), (29, 1.699), (100, 1.660)):
        assert student_t_95(df) == pytest.approx(expected, abs=0.002)
    assert student_t_95(10_000_000) == pytest.approx(Z_95_ONE_SIDED, abs=1e-4)
    with pytest.raises(ValueError):
        student_t_95(0)


def test_a_single_reading_yields_no_planning_number_at_all():
    """One observation is not a bound.

    The previous version returned the observation itself and labelled it an
    upper bound - the maximally anti-conservative answer wearing a
    conservative label. On right-skewed data a single draw sits below the true
    mean roughly 70% of the time.
    """

    stats = summarise([7.0])
    assert stats.samples == 1
    assert stats.stdev_bps is None
    assert stats.planning_bps is None
    assert stats.sufficient is False


def test_non_finite_readings_are_dropped_not_propagated():
    """One NaN would otherwise poison every statistic in the population."""

    stats = summarise([1.0, float("nan"), 3.0, float("inf"), None])
    assert stats.samples == 2
    assert stats.mean_bps == pytest.approx(2.0)


def test_an_empty_population_reports_nothing_rather_than_zero():
    stats = summarise([])
    assert stats.samples == 0
    assert stats.mean_bps is None and stats.planning_bps is None
    assert stats.sufficient is False


def test_below_the_sample_floor_there_is_no_recommendation():
    """An average of four fills is an anecdote wearing a number's clothing."""

    calibration = calibrate_from_executions(
        _fills(MIN_SAMPLES_FOR_RECOMMENDATION - 1, slippage=90.0),
        assumed_slippage_bps=10.0,
    )
    assert calibration.sufficient is False
    assert calibration.recommended_slippage_bps is None
    # Even though 90 bps is wildly worse than assumed, it must not claim so.
    assert calibration.understated is None
    assert "Insufficient evidence" in calibration.verdict()


def test_understated_is_none_not_false_when_evidence_is_thin():
    """None and False are different answers and must not be collapsed."""

    thin = calibrate_from_executions(_fills(3), assumed_slippage_bps=10.0)
    assert thin.understated is None
    assert thin.understated is not False


def test_an_optimistic_assumption_is_called_out_with_the_ratio():
    calibration = calibrate_from_executions(
        _fills(60, slippage=30.0), assumed_slippage_bps=10.0
    )
    assert calibration.sufficient
    assert calibration.understated is True
    assert calibration.recommended_slippage_bps == pytest.approx(30.0)
    assert calibration.overall.sufficient
    verdict = calibration.verdict()
    assert "optimistic" in verdict and "3.0x" in verdict


def test_a_holding_assumption_is_confirmed_rather_than_flattered():
    calibration = calibrate_from_executions(
        _fills(60, slippage=4.0), assumed_slippage_bps=10.0
    )
    assert calibration.understated is False
    assert "assumption holds" in calibration.verdict()


def test_the_watchdog_exit_is_classified_as_an_exit():
    """The defect this guards, caught before it shipped.

    The first version omitted "guard" from the exit set, which would have filed
    every watchdog emergency exit - the worst-slippage population by
    construction - as an entry, corrupting both splits in the direction that
    flatters execution.
    """

    assert "guard" in EXIT_PURPOSES
    for purpose in ("exit", "stop", "guard"):
        assert purpose in EXIT_PURPOSES, purpose
    assert "entry" in ENTRY_PURPOSES
    assert not (ENTRY_PURPOSES & EXIT_PURPOSES), "a purpose cannot be both"

    calibration = calibrate_from_executions(
        _fills(2, slippage=5.0, purpose="entry")
        + _fills(2, slippage=50.0, purpose="guard"),
        assumed_slippage_bps=10.0,
    )
    assert calibration.entries.samples == 2
    assert calibration.exits.samples == 2
    assert calibration.exits.mean_bps == pytest.approx(50.0)
    assert calibration.entries.mean_bps == pytest.approx(5.0)


def test_an_unrecognised_purpose_is_reported_never_absorbed():
    """A silent default is what made the guard defect possible."""

    calibration = calibrate_from_executions(
        _fills(2, purpose="entry") + _fills(3, purpose="something_new"),
        assumed_slippage_bps=10.0,
    )
    assert calibration.entries.samples == 2
    assert calibration.exits.samples == 0
    assert calibration.unclassified.samples == 3
    assert calibration.unknown_purposes == ("something_new",)
    warning = calibration.coverage_warning()
    assert warning is not None and "something_new" in warning
    # The overall figure still includes them, so the headline is never wrong.
    assert calibration.overall.samples == 5


def test_clean_classification_produces_no_warning():
    calibration = calibrate_from_executions(
        _fills(2, purpose="entry") + _fills(2, purpose="exit"),
        assumed_slippage_bps=10.0,
    )
    assert calibration.unknown_purposes == ()
    assert calibration.coverage_warning() is None


def test_an_empty_ledger_is_insufficient_not_zero_slippage():
    calibration = calibrate_from_executions([], assumed_slippage_bps=10.0)
    assert calibration.overall.samples == 0
    assert calibration.sufficient is False
    assert calibration.recommended_slippage_bps is None
    assert calibration.realised_fee_bps() is None


def test_fills_without_a_reference_price_do_not_count_as_zero_slippage():
    """Absence of evidence is not evidence of a perfect fill."""

    rows = _fills(5, slippage=20.0) + [
        {"slippage_bps": None, "purpose": "entry", "fill_price": 100.0,
         "filled_qty": 1.0, "fees": 0.25}
        for _ in range(5)
    ]
    calibration = calibrate_from_executions(rows, assumed_slippage_bps=10.0)
    assert calibration.overall.samples == 5
    assert calibration.overall.mean_bps == pytest.approx(20.0)
    # Their notional and fees still count, because those were really paid.
    assert calibration.total_notional == pytest.approx(1_000.0)
    assert calibration.total_fees == pytest.approx(2.5)


def test_a_negative_assumption_is_rejected():
    with pytest.raises(ValueError, match="finite and non-negative"):
        calibrate_from_executions([], assumed_slippage_bps=-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        calibrate_from_executions([], assumed_slippage_bps=float("nan"))


def test_the_sign_convention_survives_a_round_trip_through_the_store(tmp_path):
    """Positive must mean 'worse than reference' for a buy and a sell alike."""

    store = StateStore(str(tmp_path / "s.db"))
    # Buy filled above reference: worse, so positive.
    store.record_execution(
        "s1", "c1", "BTC/USD", "entry", "buy", 100.0, 101.0, 1.0, event_key="k1"
    )
    # Sell filled below reference: also worse, so also positive.
    store.record_execution(
        "s1", "c2", "BTC/USD", "exit", "sell", 100.0, 99.0, 1.0, event_key="k2"
    )
    rows = {row["client_order_id"]: row for row in store.get_executions()}
    assert rows["c1"]["slippage_bps"] == pytest.approx(100.0)
    assert rows["c2"]["slippage_bps"] == pytest.approx(100.0)

    calibration = calibrate_from_store(store, assumed_slippage_bps=10.0)
    assert calibration.overall.mean_bps == pytest.approx(100.0)
    assert calibration.entries.samples == 1 and calibration.exits.samples == 1


def test_price_improvement_is_recorded_as_negative_not_clipped(tmp_path):
    store = StateStore(str(tmp_path / "s.db"))
    store.record_execution(
        "s1", "c1", "BTC/USD", "entry", "buy", 100.0, 99.5, 1.0, event_key="k1"
    )
    calibration = calibrate_from_store(store, assumed_slippage_bps=10.0)
    assert calibration.overall.mean_bps == pytest.approx(-50.0)


def _load_cli():
    import importlib.util
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "run_execution_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("run_execution_calibration", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_execution_calibration"] = module
    spec.loader.exec_module(module)
    return module


def _seed(store, count, slippage_bps, purpose="entry"):
    """Record `count` buys whose fills sit `slippage_bps` worse than reference."""

    reference = 100.0
    fill = reference * (1.0 + slippage_bps / 10_000.0)
    for index in range(count):
        store.record_execution(
            "s1", f"c{purpose}{index}", "BTC/USD", purpose, "buy",
            reference, fill, 1.0, fees=0.25, event_key=f"k{purpose}{index}",
        )


def test_the_cli_exits_zero_when_evidence_is_thin(tmp_path, capsys):
    module = _load_cli()
    path = tmp_path / "s.db"
    _seed(StateStore(str(path)), 5, 40.0)
    assert module.main(["--db", str(path)]) == 0, "thin evidence is not a failure"
    out = capsys.readouterr().out
    assert "Insufficient evidence" in out
    assert "more recorded fills are needed" in out


def test_the_cli_exits_one_when_the_assumption_is_optimistic(tmp_path, capsys):
    module = _load_cli()
    path = tmp_path / "s.db"
    _seed(StateStore(str(path)), 40, 35.0)
    assert module.main(["--db", str(path), "--assumed-slippage-bps", "10"]) == 1
    out = capsys.readouterr().out
    assert "optimistic" in out
    assert "--slippage-bps" in out, "must print the number to re-run research with"


def test_the_cli_exits_zero_when_the_assumption_holds(tmp_path, capsys):
    module = _load_cli()
    path = tmp_path / "s.db"
    _seed(StateStore(str(path)), 40, 3.0)
    assert module.main(["--db", str(path), "--assumed-slippage-bps", "10"]) == 0
    assert "assumption holds" in capsys.readouterr().out


def test_the_cli_reports_a_missing_ledger_rather_than_crashing(tmp_path, capsys):
    module = _load_cli()
    assert module.main(["--db", str(tmp_path / "nope.db")]) == 2
    assert "No state store" in capsys.readouterr().err


def test_the_cli_writes_parsable_json(tmp_path):
    import json

    module = _load_cli()
    path = tmp_path / "s.db"
    _seed(StateStore(str(path)), 40, 22.0)
    target = tmp_path / "out.json"
    module.main(["--db", str(path), "--json-out", str(target)])
    payload = json.loads(target.read_text())
    assert payload["calibration"]["sufficient"] is True
    assert payload["calibration"]["recommended_slippage_bps"] == pytest.approx(22.0)
    assert payload["calibration"]["understated"] is True


def test_research_refuses_to_silently_fall_back_to_the_assumption(tmp_path, capsys):
    """A silent revert would produce a report that looks calibrated and is not."""

    import importlib.util
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_crypto_research.py"
    )
    spec = importlib.util.spec_from_file_location("run_crypto_research_cal", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crypto_research_cal"] = module
    spec.loader.exec_module(module)

    path = tmp_path / "s.db"
    _seed(StateStore(str(path)), 4, 50.0)
    code = module.main(["--symbols", "BTC/USD", "--slippage-from-ledger", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "are needed before an average is a measurement" in err
    assert "Refusing to fall back" in err


def test_research_reports_a_missing_ledger(tmp_path, capsys):
    import importlib.util
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_crypto_research.py"
    )
    spec = importlib.util.spec_from_file_location("run_crypto_research_cal2", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crypto_research_cal2"] = module
    spec.loader.exec_module(module)

    code = module.main(
        ["--symbols", "BTC/USD", "--slippage-from-ledger", str(tmp_path / "nope.db")]
    )
    assert code == 2
    assert "to calibrate slippage from" in capsys.readouterr().err


def test_sufficiency_is_judged_per_population_not_once_overall():
    """The exit split is the one the module argues matters most.

    An earlier version gated only the overall figure, so a two-fill exit
    population published a planning number under a column headed "95% upper"
    while the headline was properly withheld - discipline applied to the
    population it mattered least for.
    """

    calibration = calibrate_from_executions(
        _fills(200, slippage=8.0, purpose="entry")
        + _fills(2, slippage=400.0, purpose="guard"),
        assumed_slippage_bps=10.0,
    )
    assert calibration.sufficient is True
    assert calibration.overall.sufficient is True
    assert calibration.exits.samples == 2
    assert calibration.exits.sufficient is False, "two fills is not a measurement"
    assert calibration.exits.to_dict()["sufficient"] is False


def test_unmeasurable_fills_are_counted_and_flagged_as_non_random():
    """slippage_bps is NULL exactly when quotes were missing - a fast market."""

    rows = _fills(30, slippage=12.0) + [
        {"slippage_bps": None, "purpose": "guard", "fill_price": 100.0,
         "filled_qty": 1.0, "fees": 0.25}
        for _ in range(10)
    ]
    calibration = calibrate_from_executions(rows, assumed_slippage_bps=10.0)
    assert calibration.unmeasurable_fills == 10
    warning = calibration.measurability_warning()
    assert warning is not None
    assert "10 of 40" in warning and "25%" in warning
    assert "above the measured one" in warning
    assert calibration.to_dict()["unmeasurable_fills"] == 10


def test_no_unmeasurable_fills_means_no_warning():
    calibration = calibrate_from_executions(_fills(30), assumed_slippage_bps=10.0)
    assert calibration.unmeasurable_fills == 0
    assert calibration.measurability_warning() is None


def test_skewness_is_reported_so_the_reader_can_judge_the_tail():
    symmetric = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert symmetric.skewness == pytest.approx(0.0, abs=1e-9)
    skewed = summarise([1.0] * 20 + [100.0])
    assert skewed.skewness > 2.0
    assert summarise([5.0, 5.0, 5.0]).skewness == pytest.approx(0.0)


def test_percentile_always_returns_a_float():
    assert isinstance(_percentile([1, 2, 3], 0.5), float)
    assert isinstance(_percentile([1], 0.9), float)
