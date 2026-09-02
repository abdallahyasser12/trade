# Bybit Spot Production Integration — Source-of-Truth Audit

**Repository:** `abdallahyasser12/trade`  
**Audited base:** `main@943e9800cd6eb41503d836f4bcb54e588fecf0a0`  
**Mission:** add a supervised Bybit V5 Spot Demo execution path without rebuilding the trading architecture or weakening the Quant Desk/lifecycle/risk boundaries.

## Executive verdict

The repository already has a substantial production-oriented trading architecture: durable SQLite state, deterministic economic intent, broker-truth reconciliation, protective-order recovery, a Quant Desk scanner, lifecycle promotion/degradation, portfolio/risk gating, causal Edge Intelligence, hold/exit intelligence, research simulation, mobile support, and CI coverage. The correct integration seam is the broker/data-provider boundary plus a Bybit-specific service runtime.

The implementation must **not** replace `CryptoTradeEngine`, `QuantControlledCryptoTradeEngine`, `EvidenceQuantScanner`, `LifecycleManager`, `CryptoRiskManager`, `TradeManagementPolicy`, or the existing research stack. The Bybit runtime should reuse them and add venue-specific ownership, normalization, WebSocket, rate-limit, clock-skew, and burn-in controls below/around those existing boundaries.

The largest production gaps found at the audited base are:

1. Crypto broker/data-feed types and environment checks are explicitly Alpaca-shaped.
2. `BTC/USD`-style strategy symbols are also used as venue symbols; no explicit instrument mapping object exists.
3. Alpaca positions are treated as broker-side inventory. Bybit Spot needs a bot-owned execution ledger so manual wallet holdings are never adopted.
4. The parent production crypto engine executes a plain ATR stop ratchet; the existing `TradeManagementPolicy` / hold intelligence is research-measurable but is not fully production-executable.
5. Market data is REST/poll oriented and lacks Bybit public/private WebSocket supervision and microstructure snapshots.
6. There is no Bybit Demo burn-in identity/event/report store, Demo/live isolation gate, or Bybit operator dashboard.
7. The current Streamlit application is a combined research/operator surface and is too complex for routine burn-in operation.

## Component classification

| Component | Classification | Audit finding / required action |
|---|---|---|
| `broker.py` | GENERALIZE | Keep `AlpacaBroker` intact. Its normalized `OrderRecord`/account/asset semantics and deterministic retry approach are valuable, but client-ID length, URL checks, positions, and Alpaca SDK types must not leak into Bybit. Add a venue-neutral protocol and a Bybit adapter rather than rewriting Alpaca. |
| `crypto_engine.py` | REUSE AS-IS + GENERALIZE AT BOUNDARY | Core economic sequencing is strong: persist intent before submit, reconcile timeout by deterministic ID, protect fills, cancel/confirm before exits, and recover after restart. Bybit should satisfy the canonical broker/feed contract. Bybit-specific runtime may extend management without duplicating entry strategy logic. |
| `crypto_data_feed.py` | REUSE AS-IS / BYBIT ADAPTER REQUIRED | Preserve Alpaca feed. Add a Bybit feed implementing the same completed-bar/latest-quote contract plus public WS order book/trades/kline state. |
| `crypto_config.py` | GENERALIZE | Preserve existing Alpaca configs. Add venue/environment and strategy-capital-budget configuration without weakening legacy validation. Existing canonical `BASE/USD` symbol validation is venue-specific enough that an explicit instrument mapper is required. |
| `crypto_models.py` | GENERALIZE | Existing canonical `CryptoQuote`, decisions, engine status, and position state are reusable. Add venue/instrument/health/microstructure/burn-in structures outside the legacy model where possible. |
| `crypto_strategy.py` | REUSE AS-IS | Strategy should continue to consume canonical bars/quotes and remain unaware of Bybit symbol formatting or SDK objects. |
| `crypto_risk.py` | REUSE AS-IS + EXTEND INPUT CAP | Preserve sizing/correlation/session-risk logic. Bybit runtime must cap sizing equity/cash to `strategy_capital_budget_usdt`, never full Demo wallet balance. |
| `intraday_crypto_engine.py` | REUSE AS-IS + GENERALIZE TYPE HINTS | Strict lifecycle and adaptive strategy adapters are the right production path. Bybit should reuse the adaptive strategy/Quant-controlled execution rather than add a second strategy engine. |
| `advanced_crypto_engine.py` | REUSE AS-IS | Research/advanced strategy orchestration remains available; not the Bybit operator entry point. |
| `cross_sectional_engine.py` | REUSE AS-IS | Existing cross-sectional logic is useful evidence/ranking infrastructure but does not need venue-specific behavior. |
| `quant/` | REUSE AS-IS | Quant Desk remains authoritative for evidence and lifecycle. No replacement architecture. |
| `quant/scanner.py` | REUSE AS-IS | Candidate ranking/portfolio/cost/lifecycle gates must remain in the economic entry path. |
| `quant/production_runtime.py` | REUSE AS-IS + GENERALIZE ENVIRONMENT MAPPING | Already proves the Quant Scanner is in the broker-backed entry path. Bybit Demo must map to a non-live lifecycle execution stage and must not gain live promotion automatically. |
| `quant/intraday.py` | REUSE AS-IS | Existing intraday research profile is retained. |
| `quant/exits.py` | PRODUCTION GAP TO CLOSE | `TradeManagementPolicy` already owns trailing, partial take, break-even, time stop, profit target, smart exit, adaptive trail, and runners. Wire this policy into Bybit production management; do not implement a second exit policy. |
| `quant/edge_intelligence.py` | REUSE AS-IS + EXTEND CONTEXT | Existing causal profile-aware edge assessment remains the setup-quality layer. Add microstructure/cross-asset observations as explicit causal inputs rather than indicator soup. |
| `quant/exit_intelligence.py` | PRODUCTION GAP TO CLOSE | Existing causal hold score is advisory and research-measurable. Bybit runtime must feed it into `TradeManagementPolicy` behind explicit gates. |
| `quant/lifecycle.py` | REUSE AS-IS | Preserve evidence ladder and degradation/quarantine. Bybit Demo burn-in cannot auto-promote to real money. |
| `quant/store.py` | REUSE AS-IS | Keep Quant Desk evidence persistence separate from venue execution ownership. |
| `scripts/run_adaptive_intraday_runtime.py` | REUSE AS-IS | Preserve Alpaca path; do not repurpose it for Bybit. |
| `scripts/run_intraday_runtime.py` | REUSE AS-IS | Preserve existing runtime. Add a separate explicit Bybit Demo burn-in runner. |
| `scripts/run_*research.py` | REUSE AS-IS | Chronological research remains the only route for parameter evidence. Burn-in must not tune toward a desired daily return. |
| `state_store.py` | GENERALIZE / EXTEND | Existing WAL + FULL synchronous persistence and engine lock are strong. Add venue-order mapping, venue executions, bot inventory, burn-in identity/events/metrics while retaining backward-compatible schema migrations. |
| `app.py` | PRODUCTION GAP / KEEP AS ADVANCED | Current Streamlit application is broad and research-heavy. Preserve it as advanced/research functionality; add a simple operator UI as the normal Bybit burn-in surface. |
| `mobile_api.py` | REUSE AS-IS + EXTEND SAFE STATUS | Keep API; only expose sanitized Bybit status/health if added. Never expose secrets. |
| `mobile/` | REUSE AS-IS | Do not break existing mobile application; no secret-bearing venue logic belongs in the client. |
| `tests/` | EXTEND | Preserve all tests and add Bybit unit/contract/integration/chaos tests with injected fake REST/WS clients. No automated live order tests. |
| `.github/workflows/` | EXTEND | Keep existing CI and add Bybit adapter/burn-in tests to Python/crypto jobs. Demo economic smoke remains explicitly opt-in and credential-gated. |
| `README.md` and architecture docs | EXTEND | Add Bybit setup, Demo burn-in, architecture, UI, safety, and exact runner commands. |

## Alpaca assumptions leaking above the broker/data boundary

The following leaks are concrete in the audited source and must be contained rather than copied into Bybit:

- `CryptoTradeEngine` and intraday constructors annotate concrete `AlpacaBroker` / `AlpacaCryptoDataFeed` types.
- Engine startup compares `CryptoEngineConfig.account_mode` to `broker.account_mode` using Alpaca `PAPER/LIVE` semantics.
- Startup error text and account checks name Alpaca directly.
- `crypto_config.py` validates canonical symbols as `BASE/USD`; Bybit venue symbols are `BASEUSDT`.
- `broker.make_client_order_id()` enforces Alpaca's 48-character contract. Bybit requires its own `orderLinkId` generator/length enforcement.
- `AssetInfo` is documented as Alpaca-shaped, although its dynamic increments can be reused as a canonical compatibility shape.
- Parent exposure logic consumes `list_positions()`. A Bybit adapter must define this as bot-owned Spot inventory, not whole-wallet holdings.
- `AlpacaCryptoDataFeed` uses REST latest quotes and Alpaca timeframe conversion; it cannot provide the required Bybit WS order book/tape state.
- Alpaca stop/market-order mechanics and Bybit Spot TP/SL/order-filter mechanics differ and must stay inside venue adapters.

## Architecture to preserve

```text
strategy / adaptive intraday hypothesis
        ↓
EvidenceQuantScanner + Edge Intelligence
        ↓
lifecycle + degradation + portfolio/risk grants
        ↓
CryptoTradeEngine economic sequencing
        ↓
venue-neutral crypto broker/feed contract
        ├── Alpaca adapters (unchanged behavior)
        └── Bybit V5 Spot adapters
                ├── public Spot WS: kline/book/trades
                ├── Demo REST: submit/amend/cancel/query
                ├── private Demo WS: order/execution/wallet
                └── REST reconciliation after uncertainty
```

For Bybit Demo, WebSocket order entry is intentionally not implemented. Economic orders are REST-only; private WS is event delivery and REST reconciliation remains authoritative after gaps or uncertain disconnects.

## Bybit Spot safety model

Every Bybit economic request must pass one construction invariant immediately before network submission:

- `category == "spot"`
- `isLeverage == 0`
- no derivative-only fields/endpoints
- side is long-only inventory acquisition/reduction (`Buy` entry, `Sell` owned inventory)
- sell quantity <= proven bot-owned quantity and <= available wallet quantity
- instrument metadata is current/trusted and quantity/price/minimums are normalized dynamically
- deterministic `orderLinkId` is persisted before the network call
- timeout-after-submit is reconciled by `orderLinkId`; creation is never blindly retried

The application does not certify religious/Sharia compliance. It only enforces the requested Spot/no-borrow/no-leverage execution boundary and a user-configured permitted universe.

## Smart-exit gap

`quant/exits.py` already contains the correct policy object. `quant/exit_intelligence.py` already provides a causal, profile-aware hold assessment. The Bybit production runtime must translate current persisted position state into `ManagedPosition`, evaluate only completed bars, and execute returned instructions through the existing broker-truth/protection machinery. Partial exits must be reconciled before residual protection is resized. Original quantity and remaining quantity must be persisted so a restart cannot forget partial-taking state.

## Burn-in acceptance boundary

A 72–120 hour Bybit Demo run is an **operational acceptance test plus preliminary forward evidence**. It cannot establish durable profitability and must never auto-unlock live trading. Reports must issue two independent verdicts:

- `OPERATIONAL_ACCEPT` / `OPERATIONAL_REJECT`
- `EDGE_PRELIMINARY_POSITIVE` / `EDGE_INCONCLUSIVE` / `EDGE_PRELIMINARY_NEGATIVE`

Live Bybit construction remains disabled by default and requires a separate explicit future gate/review.
