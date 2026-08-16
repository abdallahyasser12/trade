"""What execution actually cost, measured, versus what the simulator assumed.

Every backtest in this repository charges 10 bps of slippage per side. That
number was never measured — it was chosen. Eleven walk-forward experiments,
every cost-to-risk figure, and the entire "friction is 75 bps a round trip"
argument all rest on it. If real slippage is 30 bps then round-trip friction is
115 bps rather than 75, every cost-to-risk figure is understated by half, and
some rejected configurations were rejected for the wrong reason while any
future accepted one would be accepted for the wrong reason too.

The engine has been recording fills against their decision prices for a while.
Nothing read them. This module closes that loop.

Choosing the statistic
----------------------

Slippage is right-skewed: most fills are near the reference price and a few are
badly worse, because the bad ones happen exactly when the market is moving.
That shape makes the choice of estimator load-bearing.

*The mean is the correct estimator for a cost model.* Total friction over n
trades is n times the mean, not n times the median. A median would systematically
understate the cost of a strategy that trades often, which is precisely the
strategy type this repository keeps testing. The median is reported anyway,
because a large gap between mean and median is itself the signal that the tail
is doing the damage.

*The planning number errs high, and that had to be measured rather than
assumed.* Underestimating friction is the failure mode this whole codebase
exists to prevent — it is what makes an edgeless strategy look tradeable.

The obvious construction, ``mean + 1.645 * stdev / sqrt(n)``, does not deliver
that. It assumes the sample mean is normally distributed, and on right-skewed
data at realistic sample sizes it is not: the sample usually misses the tail
that dominates the true mean, so the "conservative" bound sits *below* true
friction far more often than advertised. Measured by simulation, 60k trials per
cell:

    distribution        skew    n=30     n=60    n=120
    normal               0.0   95.0%    95.1%    95.1%
    lognormal sigma 1.0  6.2   84.6%    87.0%    89.1%
    lognormal sigma 1.5 33.5   74.4%    78.4%    81.7%

At skew 33 it covers three quarters of the time while claiming 95%. Bootstrap
resampling was tried and does not rescue it (84.1% against 84.2% at skew 6.2) —
the sample cannot resample a tail it never drew.

The planning number is therefore ``max(t-bound, p90)``: the Student-t upper
bound on the mean, floored by the 90th percentile of observed slippage. The
t-bound governs symmetric data; the p90 floor takes over exactly when the tail
is heavy, which is when the t-bound fails. Measured coverage of that estimator
never fell below 95.6% in any cell above, including skew 33.5 at n=30. It is
deliberately over-conservative on well-behaved data, because that is the cheap
direction to be wrong in.

Student-t rather than the normal quantile because the deviation is estimated
from the same sample, which matters most at small n.

*Below a minimum sample count there is no recommendation at all.* An average of
four fills is not a measurement, and presenting it as one would be worse than
presenting nothing: it carries the authority of a number without the content of
one. The calibration reports ``sufficient = False`` and refuses to recommend.

Entries and exits are calibrated separately. A market entry chosen at leisure
and a protective exit fired into a falling market are different distributions,
and the exit is both the worse one and the one that decides whether a stop
actually limits a loss.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

# Below this many fills the mean is not stable enough to plan with. Thirty is
# the conventional threshold at which the central limit theorem is usually
# taken to bite for a moderately skewed distribution; it is a convention, not a
# law, which is why the sample count is always reported alongside.
MIN_SAMPLES_FOR_RECOMMENDATION = 30

# One-sided 95% normal quantile, the df -> infinity limit of the t quantile
# below. The bound is one-sided because only the underestimate is dangerous.
Z_95_ONE_SIDED = 1.6448536269514722


def student_t_95(df: int) -> float:
    """One-sided 95% Student-t quantile, Cornish-Fisher expansion.

    Accurate to under 0.001 for df >= 5 against published tables, which is
    ample here: no recommendation is issued below 30 samples anyway, and the
    expansion is only used to avoid a scipy dependency for one number.
    """

    if df < 1:
        raise ValueError("Degrees of freedom must be at least 1.")
    z = Z_95_ONE_SIDED
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3

# The crypto engine records exactly four purposes: "entry" when opening, and
# "exit", "stop" and "guard" when closing. The remaining names are recognised
# because other engines and older rows use them.
#
# Both sets are explicit, and anything matching neither lands in a third
# reported bucket rather than defaulting into one of them. That is not
# fastidiousness: the first version of this file omitted "guard" from the exit
# set, which would have filed every watchdog emergency exit - the worst-slippage
# population there is, by construction - as an *entry*. It would have polluted
# entry statistics with the worst fills and removed those same fills from the
# exit statistics, corrupting both in the direction that flatters execution.
# A silent default is what made that possible, so there is no longer one.
ENTRY_PURPOSES = frozenset({"entry", "buy", "open"})
EXIT_PURPOSES = frozenset({
    "exit", "stop", "guard", "stop_limit", "guard_exit", "liquidation",
    "protective_stop", "close", "sell",
})


@dataclass(frozen=True)
class SlippageStats:
    """Distribution summary for one population of fills."""

    samples: int
    mean_bps: Optional[float]
    median_bps: Optional[float]
    p90_bps: Optional[float]
    worst_bps: Optional[float]
    stdev_bps: Optional[float]
    skewness: Optional[float]
    planning_bps: Optional[float]

    @property
    def sufficient(self) -> bool:
        """Enough fills for the planning number to mean anything.

        Checked per population, not once for the whole ledger. An earlier
        version gated only the overall figure, so the exit split - the
        population the module argues matters most - could publish a planning
        number from two observations while the headline was properly withheld.
        """

        return self.samples >= MIN_SAMPLES_FOR_RECOMMENDATION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sufficient"] = self.sufficient
        return data


@dataclass(frozen=True)
class ExecutionCalibration:
    """Measured execution cost against the value the simulator assumed."""

    assumed_slippage_bps: float
    overall: SlippageStats
    entries: SlippageStats
    exits: SlippageStats
    unclassified: SlippageStats
    unknown_purposes: tuple[str, ...]
    total_fees: float
    total_notional: float
    unmeasurable_fills: int = 0

    @property
    def sufficient(self) -> bool:
        return self.overall.sufficient

    @property
    def recommended_slippage_bps(self) -> Optional[float]:
        """The planning number, or None when the evidence is too thin.

        A caller that wants a number anyway has to decide that for itself
        rather than have this module quietly hand one over.
        """

        return self.overall.planning_bps if self.sufficient else None

    @property
    def understated(self) -> Optional[bool]:
        """Was the simulator's assumption optimistic?

        None when there is not enough evidence to say, which is a different
        answer from "no" and must not be collapsed into it.
        """

        recommended = self.recommended_slippage_bps
        if recommended is None:
            return None
        return recommended > self.assumed_slippage_bps

    def realised_fee_bps(self) -> Optional[float]:
        """Fees as bps of traded notional, to check the fee assumption too."""

        if self.total_notional <= 0 or not math.isfinite(self.total_notional):
            return None
        return self.total_fees / self.total_notional * 10_000.0

    def measurability_warning(self) -> Optional[str]:
        """Flag fills that carried no reference price, because the gap is not random.

        ``slippage_bps`` is NULL exactly when the reference price was missing,
        and the reference goes missing when a quote was unavailable or an order
        was reconciled without a local intent - degraded connectivity and fast
        markets. Those are the high-slippage conditions, so the omission is
        very likely biased toward dropping the worst fills. Silently averaging
        what remains would report execution as better than it was.
        """

        if not self.unmeasurable_fills:
            return None
        total = self.unmeasurable_fills + self.overall.samples
        share = self.unmeasurable_fills / total if total else 0.0
        return (
            f"{self.unmeasurable_fills} of {total} fills ({share:.0%}) carried no "
            "reference price and could not be measured. That gap is unlikely to "
            "be random - a reference is missing when quotes were unavailable or "
            "an order was reconciled without a local intent, which are "
            "fast-market conditions - so the true figure is more likely above "
            "the measured one than below it."
        )

    def coverage_warning(self) -> Optional[str]:
        """Flag fills whose purpose matched neither set, by name.

        The entry/exit split is only as trustworthy as its classification, so
        an unrecognised purpose is surfaced rather than absorbed.
        """

        if not self.unknown_purposes:
            return None
        return (
            f"{self.unclassified.samples} fills had unrecognised purposes "
            f"{list(self.unknown_purposes)} and are counted in the overall "
            "figures but in neither the entry nor the exit split. Add them to "
            "ENTRY_PURPOSES or EXIT_PURPOSES before reading that split."
        )

    def verdict(self) -> str:
        if not self.sufficient:
            return (
                f"Insufficient evidence: {self.overall.samples} fills recorded, "
                f"{MIN_SAMPLES_FOR_RECOMMENDATION} needed before an average is a "
                "measurement rather than an anecdote. The simulator's assumption "
                f"of {self.assumed_slippage_bps:g} bps remains unverified, and "
                "every cost figure derived from it remains provisional."
            )
        recommended = self.recommended_slippage_bps
        assert recommended is not None  # implied by sufficient
        if recommended > self.assumed_slippage_bps:
            ratio = recommended / self.assumed_slippage_bps if self.assumed_slippage_bps else math.inf
            return (
                f"The assumption is optimistic. Measured execution supports "
                f"{recommended:.1f} bps against the assumed "
                f"{self.assumed_slippage_bps:g} bps ({ratio:.1f}x) over "
                f"{self.overall.samples} fills. Every backtest run at the assumed "
                "figure understates friction, so re-run research at the measured "
                "value before trusting any cost-to-risk number."
            )
        return (
            f"The assumption holds. Measured execution supports "
            f"{recommended:.1f} bps against the assumed "
            f"{self.assumed_slippage_bps:g} bps over {self.overall.samples} fills, "
            "so backtest friction is not being understated by slippage."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumed_slippage_bps": self.assumed_slippage_bps,
            "sufficient": self.sufficient,
            "recommended_slippage_bps": self.recommended_slippage_bps,
            "understated": self.understated,
            "realised_fee_bps": self.realised_fee_bps(),
            "total_fees": self.total_fees,
            "total_notional": self.total_notional,
            "overall": self.overall.to_dict(),
            "entries": self.entries.to_dict(),
            "exits": self.exits.to_dict(),
            "unclassified": self.unclassified.to_dict(),
            "unknown_purposes": list(self.unknown_purposes),
            "unmeasurable_fills": self.unmeasurable_fills,
            "coverage_warning": self.coverage_warning(),
            "measurability_warning": self.measurability_warning(),
            "verdict": self.verdict(),
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile, without pulling in numpy for six values."""

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise(values: Iterable[float]) -> SlippageStats:
    """Summarise one population of signed slippage readings.

    Non-finite readings are dropped rather than propagated: a single NaN would
    otherwise poison every statistic, and a fill with no reference price is an
    absence of evidence rather than evidence of zero slippage.
    """

    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return SlippageStats(
            samples=0, mean_bps=None, median_bps=None, p90_bps=None,
            worst_bps=None, stdev_bps=None, skewness=None, planning_bps=None,
        )

    count = len(clean)
    mean = sum(clean) / count
    p90 = _percentile(clean, 0.9)
    skewness: Optional[float] = None
    planning: Optional[float] = None
    stdev: Optional[float] = None
    if count > 1:
        variance = sum((v - mean) ** 2 for v in clean) / (count - 1)
        stdev = math.sqrt(variance)
        if stdev > 0:
            third = sum((v - mean) ** 3 for v in clean) / count
            skewness = third / (stdev ** 3)
        else:
            skewness = 0.0
        t_bound = mean + student_t_95(count - 1) * stdev / math.sqrt(count)
        # The p90 floor is what keeps this conservative when the tail is heavy;
        # the t-bound alone covered only 74% at skew 33. See the module
        # docstring for the measured coverage of both.
        planning = max(t_bound, p90)
    # A single reading yields no planning number at all. The previous version
    # returned the observation itself and called it an upper bound, which is
    # the maximally anti-conservative answer wearing a conservative label: on
    # right-skewed data one draw sits below the true mean about 70% of the time.
    return SlippageStats(
        samples=count,
        mean_bps=mean,
        median_bps=_percentile(clean, 0.5),
        p90_bps=p90,
        worst_bps=max(clean),
        stdev_bps=stdev,
        skewness=skewness,
        planning_bps=planning,
    )


def calibrate_from_executions(
    executions: Iterable[Mapping[str, Any]],
    *,
    assumed_slippage_bps: float,
) -> ExecutionCalibration:
    """Turn recorded fills into a calibration against the assumed slippage.

    Takes rows rather than a store so the same logic serves the live database,
    an exported ledger, and a test fixture without any of them needing to know
    about the others.
    """

    if not math.isfinite(assumed_slippage_bps) or assumed_slippage_bps < 0:
        raise ValueError("The assumed slippage must be finite and non-negative.")

    rows = list(executions)
    overall_values: list[float] = []
    entry_values: list[float] = []
    exit_values: list[float] = []
    other_values: list[float] = []
    unknown: set[str] = set()
    unmeasurable = 0
    total_fees = 0.0
    total_notional = 0.0

    for row in rows:
        slippage = row.get("slippage_bps")
        purpose = str(row.get("purpose") or "").strip().lower()
        if slippage is not None and math.isfinite(float(slippage)):
            value = float(slippage)
            overall_values.append(value)
            if purpose in EXIT_PURPOSES:
                exit_values.append(value)
            elif purpose in ENTRY_PURPOSES:
                entry_values.append(value)
            else:
                # Never guess. An unrecognised purpose is reported, not filed.
                other_values.append(value)
                unknown.add(purpose or "<blank>")
        else:
            # Counted, not silently discarded: the fills that cannot be
            # measured are the ones most likely to have been expensive.
            unmeasurable += 1

        fees = row.get("fees")
        if fees is not None and math.isfinite(float(fees)):
            total_fees += float(fees)
        price = row.get("fill_price")
        qty = row.get("filled_qty")
        if price is not None and qty is not None:
            price, qty = float(price), float(qty)
            if math.isfinite(price) and math.isfinite(qty):
                total_notional += abs(price * qty)

    return ExecutionCalibration(
        assumed_slippage_bps=float(assumed_slippage_bps),
        overall=summarise(overall_values),
        entries=summarise(entry_values),
        exits=summarise(exit_values),
        unclassified=summarise(other_values),
        unknown_purposes=tuple(sorted(unknown)),
        total_fees=total_fees,
        unmeasurable_fills=unmeasurable,
        total_notional=total_notional,
    )


def calibrate_from_store(
    store,
    *,
    assumed_slippage_bps: float,
    session_id: Optional[str] = None,
    limit: int = 100_000,
) -> ExecutionCalibration:
    """Calibrate from a StateStore's execution ledger."""

    return calibrate_from_executions(
        store.get_executions(session_id=session_id, limit=limit),
        assumed_slippage_bps=assumed_slippage_bps,
    )
