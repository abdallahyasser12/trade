"""Quant desk layer: multi-market research, ranking, and lifecycle control.

This package holds the systematic-desk architecture that sits ON TOP of the
existing, audited execution engines:

    features -> regime -> strategies -> abstention -> opportunity ranking
    -> portfolio heat -> execution-quality gate -> (existing engines)

Design rules, in order of priority:

1. Nothing in this package weakens an existing deterministic safety control.
   Hard risk ceilings live in ``risk_manager``/``crypto_risk`` and remain
   authoritative; this layer may only REDUCE risk, never raise it.
2. Every historical computation is point-in-time safe.  Features, regimes,
   research folds, and model training may only see data whose timestamp
   precedes the decision.
3. No module fabricates data.  A missing provider marks its features
   unavailable and disables dependent strategies; it never inserts
   placeholder values into a decision path.
4. Strategy promotion is evidence-based and never automatic into a live
   state.  ``lifecycle`` enforces the ladder RESEARCH -> SHADOW -> PAPER ->
   LIVE_CANARY -> LIVE, and any transition into a live state additionally
   requires an explicit operator acknowledgement.

The package deliberately has no import-time side effects and performs no
network requests at import time.
"""
