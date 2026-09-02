# Bybit Demo burn-in

The 72–120 hour burn-in is an **operational acceptance test plus preliminary forward evidence**. It is not a profitability proof and never unlocks real-money trading automatically.

## Operator flow

Use the primary dashboard:

1. Open **SETTINGS**.
2. Confirm `Bybit / DEMO` and that Live is disabled.
3. Confirm credentials are configured on the server.
4. Confirm the adaptive strategy is already lifecycle-eligible for broker-backed Demo execution.
5. Choose 72 or 120 hours.
6. Set the USDT strategy capital allocation.
7. Choose a non-empty subset of BTC / ETH / SOL.
8. Click **Start Bybit Demo**.
9. Monitor HOME, MARKETS and PERFORMANCE.
10. Use **Stop gracefully** for ordinary shutdown; use **EMERGENCY STOP** only for a fail-safe halt.

The backend remains alive when the browser closes.

## Frozen identity

A run freezes Git SHA, strategy version, configuration hash, universe, start time, planned end time and environment. Restart recovery is permitted only for the exact same identity. Parameter changes require a new run.

## Economic safety

Every Bybit economic order is created with Spot semantics only. The adapter rejects non-Spot categories and nonzero leverage before network order submission. Wallet balances are an upper bound, not proof of ownership.

The bot may sell only the minimum of:

- quantity proven by bot entry/exit executions and durable execution IDs; and
- currently available wallet quantity.

Unexplained excess wallet holdings are not adopted. An unexplained deficit fails closed and requires reconciliation.

## Conservative burn-in defaults

The burn-in uses a strategy capital budget independent of the Demo wallet, risk per trade, position and total-exposure caps, maximum concurrent positions, correlation and portfolio-heat gates, a rolling 24-hour loss limit, a consecutive-loss cooldown, spread/cost gates, maximum five high-quality entries per rolling 24 hours, and zero required trades.

There is no quota to trade. Zero trades is correct when no setup passes.

## Data and execution topology

- Public Spot WebSocket: klines, order book and recent trades.
- Demo REST: account, metadata, economic order submit/amend/cancel/query and reconciliation.
- Private Demo WebSocket: order, execution and wallet-event observation.
- REST reconciliation: authoritative recovery after uncertain disconnects or submission timeouts.

No Demo WebSocket order-entry capability is fabricated.

## Recorded evidence

The durable evidence store records:

- public/private WebSocket health and reconnects;
- gaps, stale quotes/bars, spread, rate-limit events and clock skew;
- scans, decisions and **NO-TRADE decisions**;
- candidate/rejection reasoning and strategy/regime context;
- deterministic intent IDs, `orderLinkId`, Bybit order IDs and execution IDs;
- submissions, acknowledgements, rejections, cancellations and fills;
- bot-owned inventory and protection state;
- entry/exit price, quantity, fees, slippage, P&L and R where measurable;
- MFE/MAE and timestamps for +0.25%, +0.50%, +0.75% and +1.00%;
- portfolio equity/exposure/heat/drawdown evidence;
- operational health state: HEALTHY / DEGRADED / HALTED.

## Smart exit wiring

The production Bybit engine does not implement a second exit policy. It executes the existing `quant.exits.TradeManagementPolicy` and `quant.exit_intelligence` architecture behind explicit gates.

Supported mechanics include initial protection, partial profit taking, break-even transition, time/stagnation exit, smart full exit, adaptive trailing, runner management and emergency market exit.

The existing legacy strategy stop ratchet is neutralized on the Bybit path so `TradeManagementPolicy` is the single policy source. Broker/protection/reconciliation mechanics remain owned by the existing crypto engine.

Partial exits resize live protection first, then submit the released quantity, then reconcile broker truth and resize protection to the actual residual. The system never intentionally leaves a residual unprotected while replacing a stop.

## Reports

Final reports are written to:

```text
reports/bybit_burnin_<RUN_ID>.json
reports/bybit_burnin_<RUN_ID>.md
```

The report separates:

- `OPERATIONAL_ACCEPT` / `OPERATIONAL_REJECT`
- `EDGE_PRELIMINARY_POSITIVE` / `EDGE_INCONCLUSIVE` / `EDGE_PRELIMINARY_NEGATIVE`

and always states that the strategy is **not proven** by a 3–5 day sample.

During an active run the dashboard uses a time-causal report boundary ending at the current time, never the future planned stop time.

## Exact service commands

The dashboard launches these semantics itself, but the exact acceptance commands are:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
```

Optional server-side universe/capital assertions can be supplied, for example:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72 --capital-usdt 1000 --symbols BTC/USD ETH/USD SOL/USD
```

## Demo smoke

The economic smoke is explicitly opt-in and Demo-only:

```bash
python scripts/bybit_demo_smoke.py --acknowledge-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

It reads the account and current instrument metadata, establishes public/private data connectivity, submits a tiny dynamically-valid Spot Demo order inside a hard notional cap, reconciles order/execution truth, closes only proven bot-owned inventory, and fails if residual bot-owned exposure remains.

No automated test or smoke command can select a Live Bybit environment.
