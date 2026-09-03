# Bybit Demo burn-in

The 72–120 hour burn-in is an **operational acceptance test plus preliminary forward evidence**. It is not a profitability proof and never unlocks real-money trading automatically.

## Operator flow

1. Open **SETTINGS → Connect Bybit Demo** and Test & Save credentials created under Bybit Demo Trading. Do not grant withdrawal permission.
2. Under **Verify tiny Demo order path**, acknowledge the tiny economic Demo order and run the smoke. It must PASS for the current credential pair.
3. Confirm readiness: Demo API ready, smoke PASS, lifecycle eligible, BTC/ETH/SOL universe selected, safety latch clear.
4. Choose 72 or 120 hours and set the USDT strategy capital allocation.
5. Leave researched smart-exit consumers enabled for the full management-policy burn-in unless deliberately testing a baseline.
6. Choose phone push preference; notification delivery never affects trading.
7. Click **Start Bybit Demo** and monitor HOME/MARKETS/PERFORMANCE.
8. Use **Stop gracefully** normally; **EMERGENCY STOP** only for fail-safe halt.

The backend remains alive when the browser closes.

Dashboard-saved credentials live only in the server-side credential vault with restrictive permissions and are never written raw to SQLite, reports, logs, URLs, API responses or subprocess arguments. Changing/clearing saved credentials invalidates the Demo-smoke PASS.

Production `/` and `/research` are protected by Caddy Basic Auth over HTTPS using username `operator` and `TRADE_DASHBOARD_PASSWORD`; `/api/*` retains independent bearer-token authentication.

## Frozen identity

A run freezes Git SHA, strategy version, configuration hash, universe, start time, planned end time and environment. Restart recovery is permitted only for exact matching identity; parameter changes require a new run.

## Economic safety

Every economic order is Spot-only. The adapter rejects non-Spot categories and nonzero leverage before network order submission. Wallet balance is an upper bound, not ownership proof. The bot may sell only the minimum of proven bot-owned execution quantity and currently available wallet quantity. Unexplained excess is not adopted; unexplained deficit fails closed.

## Production decision path

```text
completed causal market data
  -> adaptive momentum / breakout / pullback / reversion strategy
  -> EvidenceQuantScanner + profile-aware Edge Intelligence
  -> spread / slippage / fee / participation gate
  -> Bybit stale/gap/depth/impact negative gate
  -> structure + ATR + initial-R + cost target-room gate
  -> global ranking
  -> portfolio / correlation / heat governor
  -> rolling-loss / cooldown / entry ceiling
  -> deterministic Bybit Spot Demo intent
  -> reconciliation and protection
  -> existing TradeManagementPolicy / exit intelligence
```

The Bybit layer is negative-only. Favorable order-book state cannot manufacture alpha or bypass evidence. Target plans freeze only after ranking/allocation return a positive executable risk grant, and feed into the existing `TradeManagementPolicy`. +0.25/+0.50/+0.75/+1.00 remain research measurements, not fixed mandatory exits or trade quotas.

## Conservative defaults

The burn-in uses a strategy capital budget independent of the Demo wallet, risk per trade, position/total-exposure caps, max concurrent positions, correlation/heat gates, rolling 24h loss limit, consecutive-loss cooldown, spread/cost gates, max five high-quality entries per rolling 24h, and zero required trades. Zero trades is valid.

## Data / execution topology

- Public Spot WS: klines, order book, trades
- Demo REST: account, metadata, submit/amend/cancel/query/reconcile
- Private Demo WS: order/execution/wallet observation
- REST reconciliation after uncertain disconnects/timeouts

No Demo WebSocket order entry is fabricated.

## Recorded evidence

Evidence includes public/private WS health/reconnects, gaps/staleness/spread/rate limits/clock skew, every decision including NO TRADE, rejection/setup/regime context, intent/orderLinkId/order/execution IDs, order lifecycle/fills, bot-owned inventory/protection, fees/slippage/P&L/R, MFE/MAE and +0.25/+0.50/+0.75/+1.00 timing, portfolio equity/exposure/heat/drawdown, target-plan family/R, and HEALTHY/DEGRADED/HALTED state.

## Smart exits

The existing `quant.exits.TradeManagementPolicy` and `quant.exit_intelligence` remain the only post-entry policy/intelligence. Supported mechanics include initial protection, partial profit, break-even, accepted structure/ATR/R target, time/stagnation exit, smart full exit, adaptive trailing, runner management and emergency market exit. Partial exits reconcile broker truth before residual protection is finalized.

## Notifications

The existing mobile/Expo observer can watch the Bybit engine DB while device tokens remain in the mobile API DB. Dashboard preference controls delivery. Historical fills are not replayed on restart, while the first new fill after an empty-ledger start can notify. Push failure cannot alter trading/protection.

## Reports

Final reports are:

```text
reports/bybit_burnin_<RUN_ID>.json
reports/bybit_burnin_<RUN_ID>.md
```

They separate `OPERATIONAL_ACCEPT`/`OPERATIONAL_REJECT` from `EDGE_PRELIMINARY_POSITIVE`/`EDGE_INCONCLUSIVE`/`EDGE_PRELIMINARY_NEGATIVE`, and always state the strategy is not proven by 3–5 days. Active dashboard reports are time-causal and never include future planned time.

## Unified repository acceptance

Local and Python CI use:

```bash
python scripts/run_bybit_acceptance.py
```

This executes byte-compilation, dedicated Bybit/dashboard safety tests, the complete Alpaca/Quant/mobile/Bybit regression suite, tracked-secret checks and Docker Compose validation when available, then writes JSON/Markdown receipts. Python CI runs this same command on 3.11/3.13 and uploads receipts, so CI and local fallback cannot drift.

Optional real Demo smoke:

```bash
python scripts/run_bybit_acceptance.py \
  --demo-smoke \
  --confirm-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

## Exact service commands

```bash
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
```

Normal dashboard operation requires the credential-bound Demo smoke PASS first.

Equivalent direct smoke command:

```bash
python scripts/bybit_demo_smoke.py --confirm I_ACCEPT_BYBIT_DEMO_ORDER
```

No automated test or smoke command can select Bybit Live.
