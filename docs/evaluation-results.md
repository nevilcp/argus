# Evaluation results (PR 10)

Produced by `scripts/run_evaluation.py`, run against the one captured
fixture session (`tests/fixtures/`, 6 tickers, session date 2026-05-22)
with real forward closes fetched via `LiveMarketDataProvider`. See
[ADR 0012](adr/0012-pre-registered-evaluation.md) for what these numbers
mean and the success threshold they're judged against — that ADR was
written and committed before this file existed.

## Pre-registered metrics (H=5 calendar days, dead-band=1%)

| Condition | n | rank IC | 95% CI | hit-rate | n scored | 95% CI |
|---|---|---|---|---|---|---|
| Open-loop (reliability fixed at 0.5 prior) | 6 | +0.0911 | [-0.8000, +1.0000] | 0.6000 | 5 | [0.2000, 1.0000] |
| Closed-loop (reliability reads real `chroma_db`) | 6 | +0.0911 | [-0.8000, +1.0000] | 0.6000 | 5 | [0.2000, 1.0000] |

**Verdict: closed-loop did not clear the pre-registered bar.** The bar
(ADR 0012) required the closed-loop rank IC's 95% CI to lie entirely above
0 *and* exceed the open-loop point estimate. Neither condition: the CI
spans essentially the full [-1, 1] range at `n=6`, and the two conditions
are numerically identical.

They are identical because they're expected to be: `chroma_db` held 201
`PENDING` and 0 resolved outcomes at evaluation time (queried directly,
`Counter(m.get('outcome') for m in coll.get()['metadatas'])` →
`{'PENDING': 201}`). `get_agent_accuracy` has no resolved trade history to
shrink away from the 0.5 prior for any of these six tickers' agents, so
closed-loop's reliability weights collapse to exactly the open-loop
baseline (ADR 0011's documented "unchanged behaviour" case). This isn't a
bug in the replay or the metric — it's what a mechanism that "earns its
influence as evidence accumulates" (ADR 0011) looks like before any
evidence has accumulated.

At `n=6`, this evaluation cannot distinguish "reliability weighting
doesn't help" from "not enough decisions have been reconciled yet to see
whether it does." Both readings are consistent with the data. Scaling
past this requires the same thing ADR 0009 already named for the replay
mechanism generally: capturing more fixture sessions from a live pipeline
run over time, and running `scripts/reconcile_outcomes.py` against enough
of them that `get_agent_accuracy` has real per-regime win rates to work
with.

## System-behavior metrics (reported separately, not blended into the above)

| Metric | Value |
|---|---|
| Schema validity | 6/6 (100%) |
| Errors recorded | 0 |
| Constraint violations | 0 — enforced structurally by `RiskAssessment`'s Pydantic validator; 5/6 `REDUCE` verdicts exercised it |
| API calls / decision | 2.00 (12 total) |
| Retries / decision | not instrumented (disclosed gap — no attempt count is surfaced to state by any agent) |
| Tokens / decision | not instrumented (disclosed gap — only pre-call token *estimates* exist, not actual usage) |
| Replay determinism (open-loop) | Confirmed — two independent replays of the same fixture produced bit-identical `signal`/`conviction`/`total_api_calls` for all 6 decisions |

## Reproducing this

```
.venv/bin/python -m scripts.run_evaluation
```

Requires network access (real forward closes for the six tickers). The
fixture session's decision date has long since cleared the 5-day horizon
by the time this file was written, so the run is scoring genuine
historical outcomes, not live-dependent ones.
