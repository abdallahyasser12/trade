# High-Velocity Intraday Production Path

This path is designed to create **many independent intraday opportunities** without
turning trade count or a desired return into an overfitting target.

## What it is

- Market: Alpaca USD crypto pairs, 24/7.
- Default universe: BTC, ETH, SOL, XRP, DOGE, AVAX, LINK, LTC, BCH, ADA, UNI, AAVE.
- Decision timeframe: 5 minutes (15 minutes is also supported by the strategy).
- Market-state refresh: quotes every 5 seconds while flat; broker protection loop
  every 1 second while exposed.
- Historical-bar cache: one OHLCV refresh per newly completed bar; quotes are never
  cached.
- Base risk/trade: 0.15% of equity before lifecycle/drawdown/volatility/correlation
  reductions.
- Gross exposure cap: 30% paper, 10% live.
- Session loss latch: 2%.
- Live order notional cap: $1,000 per order by default.
- Target activity diagnostic: 6-40 closed trades/day.
- Aspirational return diagnostic: 2%/day. **This is never a candidate-selection or
  promotion gate.** If untouched OOS/forward data does not produce it, the report
  says NOT OBSERVED; the optimizer is not allowed to tune until it appears.

## 1. Run real-data walk-forward research

```bash
python scripts/run_quant_research.py --engine intraday --days 90
```

The default run evaluates 18 structural candidates over the full 12-pair universe.
For this high-turnover strategy a research PASS requires, among other gates:

- at least 5 non-overlapping OOS folds;
- at least 100 OOS trades;
- at least 60% positive folds;
- positive compounded and benchmark-excess return;
- aggregate Sharpe >= 0.75;
- profit factor >= 1.15;
- positive stressed return under worse spread/slippage and 50 bps modeled fees;
- deflated-Sharpe probability >= 0.75;
- stress ruin probability <= 2%.

A PASS is only permission to enter SHADOW. It is not permission to trade a broker.

## 2. Promote RESEARCH -> SHADOW

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_momentum

python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_momentum
```

The manager reads the latest passing real research run and refuses promotion if the
research evidence is missing or failed.

## 3. Run forward SHADOW continuously

```bash
python scripts/run_intraday_runtime.py --mode shadow
```

SHADOW receives current Alpaca quotes and completed 5-minute bars but places no
broker orders. To reach PAPER, the current SHADOW state must accumulate at least:

- 100 trades;
- expectancy >= +0.03R/trade;
- PF >= 1.10;
- max R-drawdown <= 5R.

Then run the same lifecycle `promote` command again. Only rows recorded since the
SHADOW state began are eligible evidence.

## 4. Run broker PAPER

```bash
python scripts/run_intraday_runtime.py --mode paper
```

PAPER entries pass through QuantScanner before the audited broker engine:

1. measured expectancy/evidence confidence;
2. actual current quote age and spread;
3. expected alpha after measured/assumed execution cost;
4. regime/abstention checks;
5. existing-book and new-candidate correlation;
6. portfolio heat/exposure/strategy caps;
7. lifecycle risk multiplier;
8. original CryptoRiskManager sizing/cost-to-risk/asset increment checks;
9. idempotent broker order submission, fill reconciliation and protective stops.

Every real PAPER exit is attributed back into `quant_desk.db`. The high-velocity
strategy needs at least 200 current-state PAPER trades with expectancy >= +0.03R,
PF >= 1.10 and max drawdown <= 6R, plus at least 50 measured execution samples,
before a live canary is even eligible.

If forward performance deteriorates, the runtime can automatically demote or
quarantine the strategy instead of waiting for the operator to notice.

## 5. Re-run research using measured execution costs

Before PAPER -> LIVE_CANARY, run the research protocol again using cost assumptions
that are **at least as adverse as the measured broker slippage**. The lifecycle CLI
requires the ID of that passing run and refuses it if its modeled slippage is below
what PAPER actually measured.

Inspect status and research run IDs:

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_momentum
```

## 6. Explicit PAPER -> LIVE_CANARY promotion

```bash
python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_momentum \
  --measured-research-run-id <PASSING_RUN_ID> \
  --operational-defects 0 \
  --live-ack I_ACCEPT_LIVE_RISK
```

The canary starts at only 25% of strategy risk. With the default 0.15% base risk,
that is **at most 0.0375% equity risk/trade** before other reductions.

## 7. Run the separate live canary path

Configure separate Alpaca live credentials and the broker deployment flag described
in `.env.example`, then:

```bash
python scripts/run_intraday_runtime.py \
  --mode live \
  --live-ack I_ACCEPT_LIVE_RISK
```

LIVE uses a separate recovery/state database from PAPER. The runtime independently
verifies lifecycle state and historical operator acknowledgement; `AlpacaLiveBroker`
also independently verifies live deployment is enabled and that the endpoint is not
paper.

The legacy Streamlit/base crypto LIVE path does not possess the process-local Quant
live capability and therefore fails configuration validation.

## 8. LIVE_CANARY -> LIVE

Full LIVE is not eligible until the canary itself has at least:

- 100 current-state real trades;
- expectancy >= +0.01R;
- PF >= 1.05;
- max drawdown <= 4R;
- a second explicit `I_ACCEPT_LIVE_RISK` promotion acknowledgement.

## Interpretation of the 2%/day objective

2% compounded every day would be an extraordinarily high long-run return. The
platform deliberately refuses to make that number a tuning objective because doing
so would reward repeated hypothesis searching and backtest overfit. The research
CLI prints the realized geometric OOS daily return and whether 2%/day happened on
untouched data. If it did not, the correct result is **NOT OBSERVED**, not another
round of tuning the same data until it appears.

The strategy is allowed to earn promotion because it has robust positive expectancy
and execution quality—not because a desired headline percentage was fitted into the
backtest.
