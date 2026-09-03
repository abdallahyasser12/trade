# Bybit Spot Production Integration — Source-of-Truth Audit

**Repository:** `abdallahyasser12/trade`  
**Audited base:** `main@943e9800cd6eb41503d836f4bcb54e588fecf0a0`  
**Mission:** add a supervised Bybit V5 Spot Demo execution path without rebuilding the trading architecture or weakening the Quant Desk/lifecycle/risk boundaries.

## Executive verdict

The repository already had a substantial production-oriented trading architecture: durable SQLite state, deterministic economic intent, broker-truth reconciliation, protective-order recovery, a Quant Desk scanner, lifecycle promotion/degradation, portfolio/risk gating, causal Edge Intelligence, hold/exit intelligence, research simulation, mobile support, and CI definitions. The correct integration seam was the broker/data-provider boundary plus a Bybit-specific service runtime.

The implementation therefore preserves `CryptoTradeEngine`, `QuantControlledCryptoTradeEngine`, `EvidenceQuantScanner`, `LifecycleManager`, `CryptoRiskManager`, `TradeManagementPolicy`, and the existing research stack. Bybit-specific behavior lives below/around those boundaries: venue ownership, normalization, public/private WebSockets, rate-limit/clock safety, Demo isolation, multi-level liquidity/impact gating, target-room planning, burn-in evidence, and the simple operator control plane.

## Audited gaps and closure status

1. **Alpaca-shaped broker/data assumptions — CLOSED.** Venue-neutral contracts and explicit Bybit adapters isolate SDK/venue semantics.
2. **Implicit `BTC/USD` → venue symbol assumptions — CLOSED.** Canonical instruments map strategy, canonical and Bybit symbols explicitly.
3. **Whole-wallet holdings vs bot ownership — CLOSED.** Bybit execution IDs/order mappings reconstruct only proven bot-owned inventory; manual holdings are not adopted/sold.
4. **`TradeManagementPolicy` research/production gap — CLOSED for Bybit.** The Bybit path executes the existing policy; the old strategy ATR ratchet is neutralized there.
5. **REST-only / missing Bybit microstructure — CLOSED.** Public Spot WS + private Demo WS + REST reconciliation are implemented; multi-level book context is a negative-only entry-quality gate.
6. **Missing Demo burn-in identity/report/runtime — CLOSED.** 72h/120h frozen runs, daily snapshots, final reports, health samples and deterministic resume rules are implemented.
7. **Research-heavy UI as operator surface — CLOSED.** Production `/` is the simple Bybit Demo dashboard; `/research` preserves the full existing application.
8. **API-key setup too operationally complex — CLOSED.** SETTINGS provides masked Test / Test & Save / Test saved / Clear controls backed by an isolated server-only credential vault.
9. **Target-room / market-impact layer not in actual execution path — CLOSED.** `bybit_entry_intelligence.py` is wired before service prepare/start. Target plans freeze only after global ranking and a positive portfolio risk grant, then feed into the existing exit policy.
10. **Bybit mobile notification source — CLOSED.** Existing push may observe the Bybit engine DB while device tokens remain in the mobile API DB; dashboard preference controls delivery and never trading.
11. **Economic smoke required CLI/manual coordination — CLOSED.** The production dashboard exposes an explicit tiny Demo-only smoke and blocks the 72/120h start until it passes for the current credential pair.
12. **Privileged production dashboards lacked an access gate — CLOSED.** Caddy requires authenticated operator access for `/` and `/research`, while `/api/*` preserves independent mobile bearer-token authentication.
13. **Hosted-runner outage left no single local evidence path — CLOSED in source.** `scripts/run_bybit_acceptance.py` is the single acceptance command used both locally and by Python CI and emits JSON/Markdown evidence.

## Component classification and implemented disposition

| Component | Disposition |
|---|---|
| `broker.py` | Preserved; canonical normalized models/IDs remain useful to both paths. |
| `crypto_engine.py` | Preserved as economic sequencing/protection/reconciliation authority. |
| `crypto_data_feed.py` | Alpaca path preserved; Bybit adapter added separately. |
| `crypto_models.py` / config | Existing strategy-facing compatibility retained; venue contracts added outside strategy code. |
| adaptive crypto strategy | Preserved. No Bybit SDK object or symbol concatenation leaks into it. |
| `crypto_risk.py` | Preserved; Bybit strategy capital is capped independently of full Demo wallet. |
| `intraday_crypto_engine.py` | Preserved; accepts a supplied scanner and strict lifecycle context. |
| `quant/` | Preserved as evidence/lifecycle/ranking/risk authority. |
| `quant/evidence_scanner.py` | Preserved; Bybit scanner subclasses it with negative-only venue quality gates. |
| `quant/exits.py` / `quant/exit_intelligence.py` | Existing policy/intelligence is broker-backed on the Bybit path. |
| `state_store.py` | Existing engine state preserved; Bybit venue ownership/evidence is kept separately in `bybit_state.py`. |
| `app.py` | Preserved as Advanced / Research at `/research`. |
| `bybit_operator_ui.py` | Reusable HOME/MARKETS/PERFORMANCE and credential UI implementation. |
| `bybit_operator_app.py` | Primary production operator entrypoint with credential-bound Demo-smoke/start acceptance gate. |
| mobile API / push | Existing API/token ownership preserved; Bybit source/preference observation added without secret access. |
| CI/tests | Dedicated Bybit safety/chaos/credential/target/smoke/auth/notification coverage plus full regression are executed through one acceptance runner on Python 3.11/3.13. |

## Architecture preserved

```text
authenticated operator dashboard
        ↓
credential-bound tiny Bybit Demo smoke
        ↓
adaptive intraday strategy hypothesis
        ↓
EvidenceQuantScanner + Edge Intelligence
        ↓
expected cost / execution-quality gate
        ↓
Bybit-only stale/gap/depth/impact/target-room demotion gate
        ↓
global opportunity ranking
        ↓
portfolio correlation / exposure / heat allocation
        ↓
CryptoTradeEngine deterministic economic sequencing
        ↓
venue-neutral crypto broker/feed contract
        ├── Alpaca adapters (preserved)
        └── Bybit V5 Spot adapters
                ├── public Spot WS: kline/book/trades
                ├── Demo REST: submit/amend/cancel/query/reconcile
                ├── private Demo WS: order/execution/wallet
                └── bot-owned execution ledger
        ↓
existing TradeManagementPolicy + exit intelligence
```

Bybit Demo WebSocket order entry is intentionally not implemented. Economic orders are REST-only; private WS is event delivery and REST reconciliation remains authoritative after uncertainty.

## Bybit Spot safety model

Every Bybit economic request must pass the construction invariant immediately before network submission:

- `category == "spot"`
- `isLeverage == 0`
- no derivative-only fields/endpoints
- side is long-only inventory acquisition/reduction (`Buy` entry, `Sell` owned inventory)
- sell quantity <= proven bot-owned quantity and <= available wallet quantity
- instrument metadata is current/trusted and quantity/price/minimums are normalized dynamically
- deterministic `orderLinkId` is persisted before the network call
- timeout-after-submit is reconciled by `orderLinkId`; creation is never blindly retried

The application does not certify religious/Sharia compliance. It only enforces the requested Spot/no-borrow/no-leverage execution boundary and a user-configured permitted universe.

## Bybit entry-quality / target model

The existing adaptive hypothesis still creates the strategy signal. The Bybit layer cannot turn HOLD into BUY and cannot increase expected alpha.

It can reject/demote based on missing/stale/gapped multi-level order-book context, insufficient quote-denominated ask depth, insufficient depth multiple for estimated notional, excessive conservative impact proxy, adverse order-flow/book imbalance, and insufficient target room after expected round-trip costs.

The target planner combines initial risk, ATR, completed prior structure, recent realized movement and expected costs. Fixed +0.50/+0.75/+1.00 prices are retained only as report/research families.

Candidate target plans freeze only after `ScanReport.executable` confirms a positive portfolio risk grant, preventing rejected opportunities from appearing as accepted target plans in the dashboard.

## Credential/control model

Normal setup is Dashboard → SETTINGS → Connect Bybit Demo.

- fields are password-masked and clear after submit;
- Test connection is read-only and Demo-only;
- Test & Save verifies before persisting;
- saved values use a server-only JSON vault with restrictive permissions;
- the vault path is Git-ignored and production mounts it only in the operator dashboard;
- no raw credential is written to operator/engine SQLite, reports, logs, URLs or child command arguments;
- the detached child receives credentials only in its environment;
- Bybit Live credential names are not read by the Demo operator/runner path.

Production browser access is privileged. Caddy authenticates `/` and `/research` over HTTPS with username `operator` and deployment-supplied `TRADE_DASHBOARD_PASSWORD`; it derives the password hash at startup and fails closed if absent. `/api/*` remains on the mobile API's independent bearer-token boundary.

## Demo-smoke acceptance gate

Normal dashboard start requires a real tiny Bybit Demo economic smoke to PASS first. The operator explicitly acknowledges that the smoke will open and close one small Demo Spot order.

The smoke verifies current private account access, instrument metadata, public/private streams, Spot submission, execution reconciliation, bot-owned inventory accounting, close, and zero bot-owned residual. It cannot select Live, margin, leverage, derivatives or shorts.

The persisted PASS is bound to a one-way fingerprint of the currently configured Demo credential pair. Changing or clearing dashboard-saved credentials invalidates the result and blocks burn-in start until a new PASS is obtained.

## Smart-exit closure

`quant/exits.py` remains the single policy object and `quant/exit_intelligence.py` remains the causal hold assessment. The Bybit runtime translates persisted position state into `ManagedPosition`, evaluates completed bars, and executes policy instructions through broker-truth/protection machinery.

The accepted Bybit target plan temporarily specializes the existing policy's profit-target R/fraction for that managed position; after each management call the original policy object is restored. Partial exits reconcile before residual protection is resized.

## Burn-in acceptance boundary

A 72–120 hour Bybit Demo run is an **operational acceptance test plus preliminary forward evidence**. It cannot establish durable profitability and never auto-unlocks live trading. Reports issue two independent verdicts:

- `OPERATIONAL_ACCEPT` / `OPERATIONAL_REJECT`
- `EDGE_PRELIMINARY_POSITIVE` / `EDGE_INCONCLUSIVE` / `EDGE_PRELIMINARY_NEGATIVE`

Live Bybit remains outside this mission.

## Unified repository acceptance evidence

The source-controlled acceptance command is:

```bash
python scripts/run_bybit_acceptance.py
```

It runs byte-compilation, the dedicated Bybit/dashboard safety suite, the complete Alpaca/Quant/mobile/Bybit regression suite, tracked-secret checks and Docker Compose validation when Docker is installed, then writes JSON/Markdown evidence under `reports/`.

`.github/workflows/python-ci.yml` invokes this same command on Python 3.11 and 3.13 and uploads the resulting acceptance reports as artifacts. Thus local and CI acceptance cannot silently drift apart.

An optional real Demo-smoke phase remains explicit:

```bash
python scripts/run_bybit_acceptance.py \
  --demo-smoke \
  --confirm-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

## Evidence still requiring external execution

Implementation gaps above are closed, but three evidence gates cannot be fabricated:

1. A real runner/local acceptance host must execute install/compile/pytest and produce test totals. GitHub-hosted Actions currently creates jobs but fails before any workflow step starts.
2. The Demo economic smoke must be run with real Bybit Demo credentials.
3. A real 72–120h forward burn-in must elapse and produce the final operational/edge report.

These remain acceptance evidence requirements, not code TODOs.
