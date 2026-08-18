# Market-wide US-equity intraday desk

This path turns the existing Quant Desk into a market-wide US-equity opportunity
scanner with guarded Alpaca PAPER execution. It is intentionally **not** a promise of
profitability and it does not force a trade quota. The objective is to surface and
trade multiple independent opportunities when — and only when — the strategy,
execution-cost, evidence, regime, portfolio and lifecycle gates all agree.

## What “market-wide” means

Every universe refresh starts from the complete active US-equity catalogue returned by
Alpaca. Current hard executability fields then remove names that cannot be responsibly
scanned: inactive/non-tradable assets, missing or stale quotes, wide spreads, prices
outside the configured range, insufficient completed daily history and insufficient
average dollar volume.

The default production research universe contains **48** executable names. If SPY
passes exactly the same hard executability/liquidity gates, it reserves one slot as the
stable market benchmark; the remaining slots are filled by a deterministic
liquidity/freshness score. SPY is not inserted when it is itself unexecutable, and no
historical return, strategy P&L, Sharpe, hit rate or future information chooses the
universe. That bounded universe is required because research must be computationally
reproducible and because one strategy instance is capped at 64 symbols.

After a strategy advances beyond RESEARCH, the runtime may temporarily **remove** a
locked symbol from fresh-entry eligibility when it is no longer executable, but it may
never add a symbol that was not in the promoted research manifest. A currently held
SHADOW symbol remains in a separate management-only set so its virtual stops/exits keep
receiving completed bars; that management exception does not restore permission for a
new SHADOW or PAPER BUY. To change code, timeframe, universe order or any runtime
configuration, rerun research on the new final commit.

## Strategies and ranking

The common 15-minute universe is evaluated by three existing versioned hypotheses:

1. `opening_range_breakout`
2. `vwap_reversion`
3. `volatility_squeeze`

When executable, SPY is the shared benchmark anchor used for market/regime and relative
context rather than an arbitrary first-ranked stock. It is still eligible for ordinary
strategy evaluation only under the same rules as every other symbol.

The existing Quant Desk remains the decision authority. It applies strategy evidence,
regime/abstention, current quote quality, modeled execution cost, expected post-cost
alpha, portfolio heat, gross/sector exposure, correlation and lifecycle risk scaling.
The execution engine consumes only final `EXECUTE` BUY grants and risk-reducing EXITs.

Multiple candidates can be entered from one completed bar, subject to the default
limits of 6 bot positions, 0.15% equity risk per trade, 10% equity per position, 35%
gross bot exposure and the account's available cash/buying power. The engine never
creates trades just to hit a daily count.

## Research protocol

Run research only from the exact commit you intend to forward-test:

```bash
python scripts/run_marketwide_equity_research.py \
  --strategy all \
  --timeframe 15Min \
  --universe-size 48 \
  --history-bars 6500
```

The runner:

- discovers the current full active/tradable US-equity catalogue;
- chooses the research universe using only current executability/liquidity, with SPY
  reserving the benchmark slot only when it passes those same gates;
- downloads completed historical bars for that fixed ordered universe;
- requires at least 12 names with at least 5,750 exactly aligned completed bars;
- runs chronological train -> embargo -> untouched OOS folds;
- uses one fixed structural hypothesis per strategy rather than optimizing a parameter
  grid and silently deploying a different winner;
- stresses transaction costs and applies return, benchmark-excess, Sharpe, profit
  factor, drawdown, deflated-Sharpe and ruin-probability gates;
- requires at least 5 OOS folds and at least 100 OOS trades;
- records every result plus an immutable manifest containing the full git SHA,
  strategy/version, timeframe, ordered universe and full runtime configuration.

A failed strategy stays in RESEARCH. There is no automatic fallback to an older or
synthetic passing result.

## RESEARCH -> SHADOW

Inspect each strategy:

```bash
python scripts/manage_marketwide_equity_lifecycle.py status \
  --strategy opening_range_breakout
```

Promote a passing run explicitly (repeat for each strategy that passed on the same
commit/config/universe):

```bash
python scripts/manage_marketwide_equity_lifecycle.py promote \
  --strategy opening_range_breakout \
  --research-run-id <PASSING_RUN_ID>
```

Promotion fails closed unless the research row passed, its immutable manifest exists,
the checked-out full git SHA matches it, and every already-forward market-wide strategy
has the same code/config/timeframe/ordered universe. The manifest is durably locked
before lifecycle privilege increases, so a crash cannot leave SHADOW without identity.

## SHADOW forward evidence

Start the same production runtime:

```bash
python scripts/run_marketwide_equity_runtime.py
```

A SHADOW strategy receives real current bars and quotes but no broker allocation. Its
virtual positions pay modeled slippage/spread/fees, maintain virtual stops/trailing
stops and write shadow-only attribution. Fresh SHADOW BUYs are subjected to the same
current entry-eligible universe as PAPER; management-only held symbols may produce
EXIT/stop evidence but may not create new virtual risk.

SHADOW -> PAPER requires **all** of:

- at least 100 current-SHADOW closed trades;
- expectancy >= +0.03R/trade;
- profit factor >= 1.10;
- maximum R-drawdown <= 5R.

Then run the lifecycle promotion command again for that strategy. A strategy can remain
SHADOW while another qualified strategy is PAPER; their real and virtual ownership are
kept separate.

## PAPER execution

Run the same manifest-locked runtime:

```bash
python scripts/run_marketwide_equity_runtime.py
```

For PAPER strategies the engine:

- uses a dedicated crash-safe execution DB at
  `data/equity_marketwide_trading.db` by default;
- writes a durable order intent before broker submission;
- uses deterministic Alpaca client order IDs and broker lookup for restart idempotency;
- reconciles non-terminal orders from the current and prior market-wide sessions;
- manages only positions carrying this engine's durable strategy ownership record;
- counts unrelated/manual US-equity positions toward account exposure but never exits
  them;
- submits a protective GTC stop for each bot-owned position and ratchets it upward;
- cancels and confirms the protective stop before a strategy market exit;
- rechecks quote timestamp/spread immediately before every new order;
- reconstructs incremental partial-fill prices from cumulative broker averages;
- records real completed fills into QuantStore exactly once across crashes so PAPER
  lifecycle health is broker-backed evidence;
- treats a fill through the planned stop as an invalidated setup, adopts the position
  first, then immediately flattens it with conservative fallback stop protection;
- blocks fresh entries near the close, when cash/buying power is unavailable, at the
  session-loss halt, or whenever lifecycle/portfolio/cost/evidence gates reject;
- leaves protective stops in place on graceful process shutdown.

The market-wide runner uses `AlpacaIntradayMarginPaperBroker`, which does not depend on
legacy PDT/day-trade-count account fields removed from Alpaca's Trading API in July
2026. The engine is long-only and sizes new orders by the lesser of available cash and
current buying power; Alpaca's current real-time pre-trade intraday-margin checks remain
the final broker authority.

The default session equity-loss halt is 2%. When it trips, the kill switch is persisted
and new entries stop. After investigating the cause, restart requires explicit operator
acknowledgement:

```bash
python scripts/run_marketwide_equity_runtime.py --acknowledge-kill-switch
```

## Live-money boundary

This implementation deliberately stops at PAPER. The runner constructs the paper-only
`AlpacaIntradayMarginPaperBroker`, the engine independently verifies PAPER mode at
startup, the existing equity broker order boundary rejects LIVE equity orders, and the
market-wide lifecycle CLI refuses promotion beyond PAPER.

Do not weaken those controls to get more trades. A future live-equity path should be a
separate audited change with measured PAPER execution evidence, operational-defect
gates and an explicit live-capital acknowledgement.

## Operational inspection

Useful durable keys in the execution StateStore:

- `equity_marketwide_engine_status`
- `equity_marketwide_universe`
- `equity_marketwide_last_decisions`
- `equity_marketwide_kill_switch`

Use lifecycle status for strategy evidence:

```bash
python scripts/manage_marketwide_equity_lifecycle.py status \
  --strategy opening_range_breakout
python scripts/manage_marketwide_equity_lifecycle.py status \
  --strategy vwap_reversion
python scripts/manage_marketwide_equity_lifecycle.py status \
  --strategy volatility_squeeze
```

## Verification gate

`.github/workflows/marketwide-equity-ci.yml` compiles every new runtime/research module
and runs the current Alpaca account-schema, market-wide selector, manifest,
restart/gap-recovery, SHADOW entry-gating, strict lifecycle and exactly-once attribution
regression tests. The repository's existing full Python CI remains the broader
regression gate.
