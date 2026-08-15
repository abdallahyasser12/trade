# Field assessment — 15 August 2026

Seven completed out-of-sample experiments against real Alpaca market data, run
through promotion gates that were never loosened. An eighth, testing a
structurally different signal family, is running.

Hosted report: https://claude.ai/code/artifact/ea38f266-e60c-42eb-9af9-12b874dfa2b1

## Verdict

**Not elite — yet.** The engineering is close to professional grade and in two
places ahead of it. The strategy has now been measured seven ways and loses money
in every configuration tested — most decisively in experiment 7, which removed the
cost excuse and the sample-size excuse simultaneously and still lost 9 folds out
of 9. Those are separate findings; conflating them is the mistake this document
exists to prevent.

## Evidence

All figures out-of-sample, walk-forward, net of 75 bps round-trip friction.

| # | Bars | Engine | Universe | Folds | Trades | Strategy | Buy & hold | Sharpe | PF |
|---|------|--------|----------|------:|-------:|---------:|-----------:|-------:|-----:|
| 1 | 1Hour | base | 2 symbols | 31 | 88 | −6.86% | +272.14% | −2.35 | 0.26 |
| 2 | 1Day | ensemble | 2 symbols | 5 | 2 | +0.34% | −36.57% | 0.70 | 3.32 |
| 3 | 1Day | ensemble | 8 symbols | 5 | 2 | −0.16% | −36.57% | −0.93 | 0.00 |
| 4 | 1Day | base | 12 symbols | 3 | 1 | −0.30% | −44.61% | −1.11 | 0.00 |
| 5 | 1Day | base | 9 symbols | 2 | 4 | −0.07% | +19.80% | −0.14 | 0.82 |
| 6 | 4Hour | base | 8 symbols | 9 | 82 | −2.48% | +207.71% | −0.40 | 0.76 |
| 7 | 4Hour | ensemble | 8 symbols | 9 | 164 | −11.50% | +207.71% | −1.80 | 0.35 |

Runs 2 and 4 appear to "beat the benchmark" only because the strategy sat in cash
through a crash. With 1–2 trades they carry no statistical weight, and the
promotion gate rejected them on exactly that basis.

**Experiment 6 settles it.** 82 trades, 9 leakage-free folds, ~4.1 years, on the
timeframe where friction is cheapest (0.07R per trade). Profit factor 0.76.

**The gradient is the finding.** As friction fell from 0.46R per trade (hourly) to
0.07R (4-hour), profit factor climbed 0.26 → 0.76 and stopped short of 1.0. Costs
were a real drag and are now largely removed. What remains underneath is a signal
with negative expectancy. The entries are not predictive.

**Experiment 7 closes the cost argument for good.** 164 trades across 9 folds at
0.13R per trade — friction is no longer a plausible explanation for anything. The
break-even win rate at 2R is 37.8%, comfortably inside the 35–45% that trend
systems achieve. Both defects previously diagnosed are now fixed: trade frequency
went from 2 to 164, and cost per unit of risk fell by an order of magnitude. The
strategy still lost in **9 folds out of 9**.

The fold-by-fold shape is the real evidence, more than any aggregate:

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|
| Strategy | −1.33% | −1.09% | −2.03% | −1.28% | −1.26% | −0.26% | −1.32% | −2.11% | −1.44% |
| Benchmark | −15.95% | +57.22% | +30.75% | +83.26% | −4.66% | +38.07% | +30.21% | −38.11% | −8.38% |

Nine folds, a tight band of small losses, across markets ranging from +83% to
−38%. A strategy with a real but noisy edge scatters around zero and correlates
with something. This one bleeds a near-constant amount independent of what the
market does — the signature of a zero-expectancy signal paying friction, not of an
edge that needs better conditions. No amount of further tuning of this signal
family is indicated. That is why the next hypothesis is structurally different
rather than a reparameterisation.

## Predicted real-world behaviour

Extrapolated from experiments 6 and 7 — a central expectation with wide error bars.

**Paper trading**
- ~20 trades/year across 8 symbols; long quiet stretches are normal.
- −0.5% to −1%/year at modelled costs; ~−1.8%/year stressed.
- Shallow drawdowns; worst observed fold −1.46%.
- ~1/3 of periods green — enough to sustain false hope indefinitely.

Experiment 7 tightens this and makes it worse. At 164 trades over 9 folds the rate
is closer to ~40 trades/year, the annualised drag is roughly −2.5%/year at modelled
costs and −3.4%/year stressed, and **zero** of nine periods were green. The wider
sample removed the "~1/3 green" consolation: that figure came from small-sample
noise, not from the strategy having good stretches.

**Live money**
- Expect worse than paper. No execution-quality ledger exists, so the gap between
  assumed 10 bps slippage and reality is currently unmeasurable.
- The opportunity cost dominates: −2.5% versus +207.7% for holding BTC.
- It will not blow up. Sizing, session-loss latch, heat cap, correlation gate and
  hard live notional caps are enforced and tested.

**Failure modes**
- The seductive flat period: sitting in cash during a crash looks like skill.
- Untested market stress: no liquidity event, venue outage or unfilled stop-limit
  appears in any experiment.
- Data-window bias: Alpaca crypto history starts 2021-01-01. One cycle is not a sample.

## What is genuinely elite

- **It cannot fool itself about costs.** One friction formula feeds simulator and
  live engine, expressed in units of the risk each trade takes.
- **Its gates do not bend.** Seven runs, seven honest verdicts, zero adjustments.

That machinery is the durable asset — it is what makes the negative result
trustworthy, and what any future strategy would be judged by.

## What would have to change

Every lever the codebase exposes has been pulled — three timeframes, four universe
sizes, both engines, two grid scales, a corrected embargo — and they converge.

1. A different signal family, through the same unmodified gates.
2. Build the execution-quality ledger first; until realized slippage is measured,
   no live result can be attributed to signal versus execution.
3. Or stop: keep the measurement machine, run the ensemble in paper as a
   capital-preservation overlay, and stop paying the opportunity cost.

### Item 1 is now in progress

`cross_sectional.py` is the different signal family, chosen to attack the two
diagnosed failures rather than to reparameterise around them:

- **Relative, not absolute.** Rather than asking each symbol "are ten conditions
  true at once?", it ranks the universe at each timestamp and holds the leaders.
  Something is always ranked first, so trade frequency is a property of `top_k`
  rather than an accident of filter alignment. Measured: 54 trades where the
  conjunction produced 2.
- **Dimensionally coherent.** `(close/close[-L] − 1) / (σ·√L)` is unitless, so a
  move in DOGE and a move in BTC are actually comparable. The old score summed a
  raw fractional return with a normalised ADX and therefore ranked by volatility.
- **Asymmetric turnover.** Exit requires falling `top_k + buffer` places, because
  turnover is the one cost lever repeatedly shown to matter.
- **Six parameters, not fifteen.** Every parameter is overfitting surface.

It runs through the same unmodified simulator, cost gate, benchmark comparison and
promotion criteria — only the decision layer differs, injected into the audited
code objects so the mechanics stay byte-for-byte identical.

Two things learned while wiring it up, both worth recording because both would
have produced a dishonest test:

- The engine was initially pointed at the breakout parameter grid, which varies
  keys it never reads. All candidates per fold were behaviourally identical, so
  the argmax was decided by tie-break order and the reported candidate count sized
  a multiple-testing penalty for a search that never happened. The grid now varies
  what the engine actually reads.
- With the benchmark-regime gate pinned on, a training window inside the 2022 bear
  market gives every candidate zero trades, so no candidate can be selected and
  the run aborts. Skipping such folds would have biased `positive_fold_ratio` to
  bull folds only — the strategy would score better the more often it declined to
  trade. The gate is therefore searched rather than assumed.

Whether it has edge is an empirical question. Nothing above answers it; it only
makes the question askable through gates that have never been loosened.

## Limits

Walk-forward out-of-sample on Alpaca data from 2021-01-01 — one market cycle.
Experiment 7 was still computing at publication. Nothing here is investment advice.
A promotion PASS would have meant only "eligible for supervised paper testing".
The system has never traded real money, and nothing here recommends that it should.
