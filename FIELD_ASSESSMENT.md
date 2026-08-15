# Field assessment — 15 August 2026

Seven out-of-sample experiments against real Alpaca market data, run through
promotion gates that were never loosened.

Hosted report: https://claude.ai/code/artifact/ea38f266-e60c-42eb-9af9-12b874dfa2b1

## Verdict

**Not elite — yet.** The engineering is close to professional grade and in two
places ahead of it. The strategy has now been measured seven ways and loses money
in every configuration tested. Those are separate findings; conflating them is the
mistake this document exists to prevent.

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
| 7 | 4Hour | ensemble | 8 symbols | — | — | pending | | | |

Runs 2 and 4 appear to "beat the benchmark" only because the strategy sat in cash
through a crash. With 1–2 trades they carry no statistical weight, and the
promotion gate rejected them on exactly that basis.

**Experiment 6 settles it.** 82 trades, 9 leakage-free folds, ~4.1 years, on the
timeframe where friction is cheapest (0.07R per trade). Profit factor 0.76.

**The gradient is the finding.** As friction fell from 0.46R per trade (hourly) to
0.07R (4-hour), profit factor climbed 0.26 → 0.76 and stopped short of 1.0. Costs
were a real drag and are now largely removed. What remains underneath is a signal
with negative expectancy. The entries are not predictive.

## Predicted real-world behaviour

Extrapolated from experiment 6 — a central expectation with wide error bars.

**Paper trading**
- ~20 trades/year across 8 symbols; long quiet stretches are normal.
- −0.5% to −1%/year at modelled costs; ~−1.8%/year stressed.
- Shallow drawdowns; worst observed fold −1.46%.
- ~1/3 of periods green — enough to sustain false hope indefinitely.

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

## Limits

Walk-forward out-of-sample on Alpaca data from 2021-01-01 — one market cycle.
Experiment 7 was still computing at publication. Nothing here is investment advice.
A promotion PASS would have meant only "eligible for supervised paper testing".
The system has never traded real money, and nothing here recommends that it should.
