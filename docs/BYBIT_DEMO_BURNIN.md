# Bybit Demo burn-in

The 72–120 hour burn-in is an **operational acceptance test plus preliminary forward evidence**. It is not a profitability proof and never unlocks real-money trading automatically.

## Operator flow

Use the primary dashboard:

1. Open **SETTINGS → Connect Bybit Demo**.
2. Paste a key/secret created under **Bybit Demo Trading** and click **Test & Save**. Do not grant withdrawal permission.
3. Under **Verify tiny Demo order path**, explicitly acknowledge the small economic Demo order and run the smoke. It must PASS for the current credential pair.
4. Confirm the readiness row shows the Demo API ready, Demo smoke PASS, adaptive strategy lifecycle eligible, a BTC/ETH/SOL universe selected, and the emergency latch clear.
5. Choose 72 or 120 hours.
6. Set the USDT strategy capital allocation.
7. Leave **researched smart-exit consumers** enabled for the full management-policy burn-in unless deliberately testing the conservative baseline.
8. Enable/disable phone push notifications as desired; notification delivery never affects trading.
9. Click **Start Bybit Demo**.
10. Monitor HOME, MARKETS and PERFORMANCE.
11. Use **Stop gracefully** for ordinary shutdown; use **EMERGENCY STOP** only for a fail-safe halt.

The backend remains alive when the browser closes.

Dashboard-saved credentials live only in the server-side credential vault (`secrets/bybit_demo_credentials.json`, or `/app/secrets/...` in production), with restrictive file permissions. The vault is Git-ignored and mounted only into the operator dashboard container. Credentials are never written to SQLite, reports, logs, URLs, API responses or subprocess arguments; the detached backend receives them only through its environment.

Changing or clearing dashboard-saved credentials invalidates the stored Demo-smoke PASS. The dashboard refuses to start the burn-in until the smoke passes again for the new credential pair.

Production browser access itself is protected by Caddy Basic Auth over HTTPS. Both `/` and `/research` require username `operator` plus the private `TRADE_DASHBOARD_PASSWORD`; `/api/*` retains its separate bearer-token authentication.

## Frozen identity

A run freezes Git SHA, strategy version, configuration hash, universe, start time, planned end time and environment. Restart recovery is permitted only for the exact same identity. Parameter changes require a new run.

## Economic safety

Every Bybit economic order is created with Spot semantics only. The adapter rejects non-Spot categories and nonzero leverage before network order submission. Wallet balances are an upper bound, not proof of ownership.

The bot may sell only the minimum of:

- quantity proven by bot entry/exit executions and durable execution IDs; and
- currently available wallet quantity.

Unexplained excess wallet holdings are not adopted. An unexplained deficit fails closed and requires reconciliation.

## Production decision path under burn-in

The burn-in exercises the actual broker-backed decision chain:

```text
completed causal market data
  -> adaptive momentum / breakout / pullback / reversion strategy
  -> EvidenceQuantScanner + profile-aware Edge Intelligence
  -> expected spread / slippage / fee / participation gate
  -> Bybit multi-level order-book stale/gap/depth/impact gate
  -> structural + ATR + initial-R + cost target-room gate
  -> global ranking
  -> portfolio / correlation / heat governor
  -> rolling-loss / cooldown / entry ceiling
  -> deterministic Bybit Spot Demo order intent
  -> broker reconciliation and protection
  -> existing TradeManagementPolicy / exit intelligence
```

The Bybit-specific layer is **negative-only**. Favorable order-book state cannot manufacture alpha or bypass research evidence. Stale/gapped books, inadequate depth, excessive impact or insufficient target room can only demote/reject a setup.

A target plan is frozen only after global ranking and portfolio allocation return a positive executable risk grant. The plan combines completed-bar structure, ATR, initial R, recent realized movement and expected round-trip cost. It then feeds into the existing `TradeManagementPolicy`; there is no second exit system.

The +0.25%, +0.50%, +0.75% and +1.00% families remain forward-research measurements. They are not fixed profit orders and never create a trade quota.

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
- target-plan family / target R where captured;
- operational health state: HEALTHY / DEGRADED / HALTED.

## Smart exit wiring

The production Bybit engine does not implement a second exit policy. It executes the existing `quant.exits.TradeManagementPolicy` and `quant.exit_intelligence` architecture behind explicit gates.

Supported mechanics include initial protection, partial profit taking, break-even transition, accepted structure/ATR/R profit target, time/stagnation exit, smart full exit, adaptive trailing, runner management and emergency market exit.

The existing legacy strategy stop ratchet is neutralized on the Bybit path so `TradeManagementPolicy` is the single policy source. Broker/protection/reconciliation mechanics remain owned by the existing crypto engine.

Partial exits resize live protection first, then submit the released quantity, then reconcile broker truth and resize protection to the actual residual. The system never intentionally leaves a residual unprotected while replacing a stop.

## Notifications

If the existing iOS/Expo device registration is configured, the dashboard notification switch controls Bybit fill, safety-lock and engine-state push delivery. Device tokens remain in the existing mobile API database; the observer reads the Bybit engine database and the operator preference database. Push failure is observational only and cannot alter order/protection state.

The notifier uses the latest persisted trade as a restart baseline, so historical fills are not replayed. If the ledger is empty at notifier startup, a zero-id baseline allows the first real fill to generate a notification.

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

## One-command repository acceptance

When GitHub-hosted runners are unavailable, run:

```bash
python scripts/run_bybit_acceptance.py
```

This runs the hermetic Bybit/dashboard safety suite plus the complete legacy regression suite and writes `reports/bybit_acceptance_<SHA>.json` and `.md` evidence.

To append the real Demo smoke after the hermetic checks pass:

```bash
python scripts/run_bybit_acceptance.py \
  --demo-smoke \
  --confirm-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

The optional economic smoke remains Demo-only and requires real Demo credentials.

## Exact service commands

The dashboard launches these semantics itself, but the exact service commands are:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
```

Optional server-side universe/capital assertions can be supplied, for example:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72 --capital-usdt 1000 --symbols BTC/USD ETH/USD SOL/USD
```

The service command remains available for recovery/diagnostics, but normal dashboard operation enforces the credential-bound smoke gate first.

## Demo smoke

The economic smoke is explicitly opt-in and Demo-only. Normal operators run it from SETTINGS; the equivalent command is:

```bash
python scripts/bybit_demo_smoke.py --confirm I_ACCEPT_BYBIT_DEMO_ORDER
```

It reads the account and current instrument metadata, establishes public/private data connectivity, submits a tiny dynamically-valid Spot Demo order inside a hard notional cap, reconciles order/execution truth, closes only proven bot-owned inventory, and fails if residual bot-owned exposure remains.

No automated test or smoke command can select a Live Bybit environment.
