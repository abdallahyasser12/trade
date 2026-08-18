# Adaptive Intraday v2 — Production Path

`crypto_intraday_adaptive` v2.0.0 is the preferred high-velocity research path.
The older `crypto_intraday_momentum` v1 remains available for comparison, but its
research, lifecycle state and execution database are intentionally isolated from v2.

The objective is many **independent, positive-expectancy** intraday opportunities.
The system does not force a target return into historical data. The requested 2%/day
number is reported as an untouched OOS diagnostic only.

## Decision architecture

Completed 5-minute or 15-minute bars feed four causal sleeves:

1. cross-sectional momentum continuation;
2. volatility-expansion / prior-range breakout;
3. trend pullback continuation;
4. range mean reversion with reversal confirmation.

A cross-sectional router estimates bull/range/stress/risk-off conditions from the
benchmark, market breadth and volatility. Stress/risk-off can block entries. The
bar-driven strategy does not use future data or live order-book state.

Immediately before a broker entry, Quant Desk independently applies:

- current quote freshness and spread;
- measured/assumed round-trip cost and net-alpha threshold;
- statistical expectancy confidence from real attribution history;
- adverse order-book depth penalty (negative depth can reduce quality; positive
  depth never creates alpha or boosts expected return);
- regime compatibility and abstention;
- current-book/new-candidate correlation;
- portfolio heat/exposure/position caps;
- lifecycle risk multiplier;
- original broker sizing, idempotency, reconciliation and protective-stop controls.

Quotes are refreshed every five seconds while flat and every one second while
exposed. OHLCV history is cached until a newly completed bar, so protection stays
responsive without repeatedly downloading the same history. Fresh quote/depth
telemetry is persisted to Quant Desk; telemetry failure cannot break protection.

## 1. Real-data adaptive research

Default full 12-pair USD universe, 120 days, 5-minute bars:

```bash
python scripts/run_adaptive_intraday_research.py --days 120
```

The bounded research grid contains 18 structural hypotheses. Candidate selection is
training-only inside chronological walk-forward folds; OOS folds do not overlap and
are separated by an embargo. Cumulative historical trial count is included in the
deflated-Sharpe correction.

A research PASS requires all generic and adaptive gates, including:

- at least 8 OOS folds;
- at least 300 OOS trades;
- at least 65% positive OOS folds;
- positive compounded OOS return;
- positive excess return over the benchmark;
- aggregate Sharpe >= 0.85;
- profit factor >= 1.18;
- worst OOS-fold drawdown <= 8%;
- positive return under harsher execution costs;
- deflated-Sharpe probability >= 0.80;
- Monte Carlo ruin probability <= 1%;
- at least 6 OOS closed trades/day;
- no more than 50 OOS closed trades/day (anti-churn ceiling);
- OOS expectancy >= +0.03R/trade;
- broad symbol participation (normally >= 6 symbols on the default universe);
- no single symbol contributing more than 55% of positive symbol PnL;
- no single OOS fold contributing more than 45% of positive fold return.

The output separately reports geometric OOS return/day and whether 2%/day was
observed. That diagnostic never changes candidate selection or promotion.

## 2. Promote RESEARCH -> SHADOW

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive

python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive
```

No passing real-data research run means no promotion.

## 3. Continuous SHADOW forward test

```bash
python scripts/run_adaptive_intraday_runtime.py --mode shadow
```

SHADOW receives current Alpaca bars and quotes but submits no broker orders. Only
forward attributions recorded since the SHADOW state began count. PAPER requires:

- >= 100 SHADOW trades;
- expectancy >= +0.03R/trade;
- profit factor >= 1.10;
- max R-drawdown <= 5R.

Then run the lifecycle `promote` command again.

## 4. PAPER execution

```bash
python scripts/run_adaptive_intraday_runtime.py --mode paper
```

The adaptive PAPER path uses a separate recovery database:

`data/adaptive_intraday_trading.db`

Every exit fill is attributed back to Quant Desk. Automatic degradation can send the
strategy back to SHADOW or quarantine it when real expectancy, PF, drawdown or
execution costs deteriorate.

LIVE_CANARY eligibility requires, from the current PAPER state only:

- >= 200 PAPER trades;
- expectancy >= +0.03R/trade;
- PF >= 1.10;
- drawdown <= 6R;
- >= 50 measured execution samples;
- zero open operational defects;
- a new passing research run whose modeled slippage is at least measured PAPER
  slippage.

## 5. Re-run with measured execution costs

Read the current execution quality:

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive
```

Then rerun research using assumptions no better than observed PAPER execution. For
example, if average measured slippage is 14.2 bps:

```bash
python scripts/run_adaptive_intraday_research.py \
  --days 120 \
  --slippage-bps 14.2 \
  --spread-bps 8 \
  --fee-bps 25
```

The stress run automatically uses costs at least 1.5x the configured base assumptions
(and never below its conservative stress floors).

## 6. PAPER -> LIVE_CANARY

Use the ID of that passing measured-cost research run:

```bash
python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive \
  --measured-research-run-id <PASSING_RUN_ID> \
  --operational-defects 0 \
  --live-ack I_ACCEPT_LIVE_RISK
```

The canary lifecycle multiplier is 25%. With the 0.15% base strategy budget, maximum
risk starts at **0.0375% equity/trade** before correlation, drawdown, volatility and
portfolio reductions.

## 7. Guarded LIVE canary

Configure separate Alpaca live credentials plus the repository live-deployment gate,
then:

```bash
python scripts/run_adaptive_intraday_runtime.py \
  --mode live \
  --live-ack I_ACCEPT_LIVE_RISK
```

LIVE uses its own state database:

`data/adaptive_intraday_live_trading.db`

The base crypto LIVE configuration cannot mint the Quant-controlled process-local
authorization, so the old UI/base engine cannot bypass this path.

## 8. LIVE_CANARY -> LIVE

Full LIVE still requires a new explicit acknowledgement and current-canary evidence:

- >= 100 real canary trades;
- expectancy >= +0.01R/trade;
- PF >= 1.05;
- drawdown <= 4R.

Automatic health supervision remains active after promotion.

## What “iterate” means here

A failed research run remains a failed hypothesis. Do not keep expanding parameter
ranges on the same sample until a desired percentage appears. Any materially new
signal idea receives a new hypothesis/version, cumulative trial accounting and the
same untouched-OOS/forward ladder.

That is the production iteration loop:

**hypothesis -> chronological research -> stressed OOS -> SHADOW -> PAPER -> measured
cost rerun -> CANARY -> LIVE -> continuous degradation monitoring**.

The system is designed to maximize the chance that an observed edge is real, not to
manufacture a backtest that says 2%/day.
