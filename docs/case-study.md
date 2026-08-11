# Engineering case study: finding and closing a dead loop

This is a write-up of the investigation behind [issue #1](https://github.com/nevilcp/argus/issues/1),
the 12-PR rebuild this repository's git history documents. It's written as
the investigation happened — a query, then a widening set of questions about
why the query returned what it did — rather than as a tour of the features
that resulted. The ADRs cited throughout are its primary sources; this
document adds nothing they don't already say, it just tells the story in
order.

## The query that started it

ARGUS stores every decision it makes — ticker, regime, aggregated signal,
rationale — as a document in a ChromaDB collection, tagged with an
`outcome` field meant to move from `PENDING` to `SUCCESS`/`FAILED` once
enough time has passed to know whether the decision was right. The
mechanism for making that transition, `cultural.store_trade_outcome`, was
fully implemented: it took a decision ID and a realized outcome, updated
the metadata, and let two other functions —
`retrieve_wisdom`/`retrieve_warnings` and `get_agent_accuracy` — query
against real history instead of an empty collection.

One query against the live collection:

```python
Counter(m.get('outcome') for m in coll.get()['metadatas'])
```

returned `{'PENDING': 159}`. All 159. Zero resolved. `store_trade_outcome`
had never been called from anywhere in the codebase — not a bug in the
function, an absence of any caller. Every mechanism that depended on
resolved outcomes was, by construction, dead: `retrieve_wisdom` and
`retrieve_warnings` always returned `[]`, `get_agent_accuracy` always
returned the 0.5 prior, and the "cultural memory" the README described as
letting the Portfolio agent "recall historical wisdom across similar
market regimes" had never recalled anything, because nothing had ever been
written to recall.

That's a bug. What made it worth a rebuild rather than a patch was what
turned up when the same question — *is anything actually reading this?* —
got asked of the rest of the codebase.

## Eleven dead seams

A full read of the code, cross-checked by running it and querying the live
datastores rather than just inspecting source, found the same shape of
problem repeated eleven times. Each is a mechanism that is written, and in
several cases logged or stored, and consumed by nothing:

| # | Mechanism | Evidence |
|---|---|---|
| 1 | Outcome resolution (`store_trade_outcome`) | 159/159 `PENDING`, confirmed above |
| 2 | Cultural wisdom/warnings retrieval | Downstream of #1 — always `[]` |
| 3 | Agent reliability (`get_agent_accuracy`) | Downstream of #1 — always the 0.5 prior |
| 4 | Backtest technical signals | `_simulate_session_state` returned 2 of the 8 keys `TechnicalStatisticalAgent.analyze` requires; every call returned `None` |
| 5 | Phase 1/2 calibration | A 16-config grid search over a function (#4) that never varied — all 16 configs identical |
| 6 | Multi-year backtest | Structurally impossible: Yahoo Finance's `5m` interval has a hard 60-day lookback (`5m/1mo` → 1,794 rows, `5m/3mo` → 0, `5m/6mo` → 0), and the live system's technical agent requires 5-minute-resolution features that don't exist further back than that |
| 7 | Kill switch | `initialize_kill_switch` never called; `get_kill_switch()` always `None`; both `/analyze` safety guards short-circuited |
| 8 | Bias audit output | `ui/app.py` printed "PASS: Survivorship \| Lookahead \| Data Quality" unconditionally; `BiasAuditor` was constructed once, against an empty returns series |
| 9 | Rate limit enforcement | `governor.py` slept 60s on exhaustion and `return`ed without incrementing or raising — advisory, not enforced |
| 10 | `AggregatedSignal.debate_triggered`/`skip_reason` | Written by `aggregator.aggregate()`, read by nothing; `ARCHITECTURE_WALKTHROUGH.md` described a `_resolve_conflict()` LLM call that was never in the repository |
| 11 | `DecisionLogger` | ~260 lines, fully implemented, never instantiated |

Eleven items, one shape: a value gets computed, sometimes stored, sometimes
even rendered to a user as a claim ("✅ KILL SWITCH ACTIVE" — printed
regardless of whether the switch had ever been initialized) — and nothing
downstream ever reads it back.

## The single root cause

Eleven findings look like eleven bugs. They aren't. Cross-referencing them
turns up one pattern: **the system's architecture was described in prose,
diagrams, and function signatures, but the wiring between components —
the part that turns "component A produces a value" into "component B
consumes it" — was never written.** `store_trade_outcome` isn't buggy; it's
unreachable. The kill switch isn't buggy; it's uninitialized. The bias
audit isn't buggy; it's disconnected from any real input and its output
was fabricated instead. Each mechanism is individually competent code. The
system is not a system, in the sense of parts that affect each other — it
is a set of parts that happen to share a repository.

This reframes what "fixing" means. Patching each of the eleven items in
place (call `store_trade_outcome` somewhere, call `initialize_kill_switch`
somewhere) would produce a codebase that runs without changing what kind
of codebase it is. The rebuild plan (issue #1) instead asked, for each
item: is this worth wiring, or worth deleting? Both are honest answers;
leaving it unwired and undocumented is the only dishonest one. The kill
switch, dual-gate rate limiter, and the decision→outcome loop were wired
([ADR 0002](adr/0002-return-none-not-fabricated-defaults.md)'s sibling
decisions, PR 2, PR 8). The fabricated bias-audit output, the vacuous
backtest, and `DecisionLogger` were deleted
([ADR 0009](adr/0009-no-multiyear-backtest.md), [ADR 0010](adr/0010-closing-the-decision-outcome-loop.md)).
`debate_triggered`/`skip_reason` were deleted rather than given an invented
downstream use, because inventing one would have been a new policy
decision dressed up as a fix ([ADR 0011](adr/0011-reliability-weighting.md)).

## The Yahoo 60-day constraint, and why it mattered more than it looked

Finding #6 looks like the least interesting item on the list — a
third-party API limitation, not a design flaw. It turned out to be the
reason findings #4 and #5 existed at all: the walk-forward backtest
apparatus (`engine.py`, `walk_forward.py`, `phase1_calibration.py`,
`phase2_validation.py`, `bias_auditor.py`, `pit_enforcer.py`) had been
built to validate the system's live signal-generation logic over multi-year
windows, but that logic runs on 5-minute technical indicators the data
source cannot supply beyond 60 days. `_simulate_session_state` didn't fail
to compute those indicators for old dates through a bug — it had no data
to compute them from, and the two keys it did fabricate silently satisfied
the type checker while producing a `None` signal on every call.

There is no code fix for a data source's lookback window. [ADR 0009](adr/0009-no-multiyear-backtest.md)
made the boundary explicit rather than working around it: the six-file
walk-forward apparatus and the `backtrader` dependency were deleted (PR 7),
keeping only `metrics.py` (pure functions of a returns series — Sharpe,
Sortino, max drawdown, IC), and replaced with a much smaller mechanism —
`argus/backtesting/replay.py` — that replays *recorded fixture sessions*
through the real, compiled graph rather than pretending to reconstruct
history it doesn't have. Point-in-time correctness stopped being a runtime
check (`pit_enforcer.py`) and became structural: each fixture session's
data provider is scoped to its own directory, so there is no code path by
which a later session's data could leak into an earlier one. As of this
writing exactly one such session has been captured; scaling this up means
running the live pipeline and recording more sessions over time, not
writing more code.

## The fix: three PRs on the critical path

Everything up to this point was diagnosis. The fix — [ADR 0007](adr/0007-injection-seam.md)
through [ADR 0011](adr/0011-reliability-weighting.md) — followed a critical
path of PR 5 → 6 → 8 → 9 → 10 (everything else in the twelve-PR plan
reorders freely around it):

- **PR 5** gave every network- and LLM-touching agent an injected seam
  (`MarketDataProvider`, `LLMClient`) instead of module-level singletons,
  with fixture-backed implementations captured from real traces.
- **PR 6** used that seam to build a deterministic test suite — golden DAG
  tests, and property tests on the pure scoring functions (RSI
  monotonicity, aggregator conviction bounds) that caught a real scoring
  bug during development, the same way a later property test on the
  ablation function ([ADR 0010](adr/0010-closing-the-decision-outcome-loop.md))
  caught a saturation artifact before it shipped.
- **PR 8** closed the loop itself: `store_trade_outcome` gained a real
  caller (`scripts/reconcile_outcomes.py`, walking the LangGraph
  checkpoint — the audit trail that already existed rather than a new
  archive), and `primary_driver` is now computed by leave-one-out ablation
  on `HybridSignalAggregator.aggregate()` (a pure function), not the
  three-independent-threshold heuristic that used to occupy that name.
  Leave-one-out was chosen over exact Shapley deliberately: with three
  agents they're equivalent up to a constant, and the extra machinery
  ("defend a game-theoretic allocation across nested coalitions nobody will
  inspect," per the ADR) buys nothing a reader would use.
- **PR 9** made the loop's output actually change behavior:
  `get_agent_accuracy` shrinks toward the 0.5 prior with pseudo-count
  `k=10` (so an agent credited on two trades doesn't get trusted like one
  with two hundred), and `aggregate()` scales each agent's vote by
  `win_rate / 0.5` — 1.0 at the prior, unchanged from before this PR
  existed. `cultural_memory["warnings"]`, previously retrieved every
  session and never used, was wired into the Portfolio agent's prompt
  alongside `wisdom`. `debate_triggered` and `skip_reason` were deleted
  rather than wired, for the reason above.
- **PR 10** is the evaluation that follows.

Twelve PRs total, no PR over ~400 lines, and CI green at every step —
process choices, not incidental facts, since the object being rebuilt was
a set of unverified claims, and adding more code without verification at
each step would have just produced a different version of the original
problem.

## The pre-registered evaluation, and where reliability weighting didn't help

[ADR 0012](adr/0012-pre-registered-evaluation.md) fixed the evaluation's
horizon (H = 5 calendar days), dead-band (1%), metric (rank information
coefficient — signed conviction vs. forward return — plus
hit-rate-with-dead-band), and success threshold *before* running it: the
closed-loop condition's rank IC 95% CI had to lie entirely above 0 and
exceed the open-loop point estimate. The result, committed at
[`docs/evaluation-results.md`](evaluation-results.md):

| Condition | n | rank IC | 95% CI | hit-rate |
|---|---|---|---|---|
| Open-loop (reliability fixed at 0.5) | 6 | +0.0911 | [-0.80, +1.00] | 0.60 |
| Closed-loop (reads real `chroma_db`) | 6 | +0.0911 | [-0.80, +1.00] | 0.60 |

**The pre-registered bar was not cleared.** The two conditions are
numerically identical, and the reason is traceable rather than mysterious:
at evaluation time `chroma_db` held 201 `PENDING` outcomes and 0 resolved
ones for these six tickers, so `get_agent_accuracy` had no history to
shrink away from the 0.5 prior — closed-loop's reliability weights
collapsed to exactly the open-loop baseline. That collapse is
[ADR 0011](adr/0011-reliability-weighting.md)'s own documented behavior
for the zero-evidence case, not a bug in the replay: a mechanism designed
to "earn its influence as evidence accumulates" looks exactly like this
before any evidence has accumulated.

At n=6, the evaluation cannot distinguish "reliability weighting doesn't
help" from "not enough decisions have been reconciled yet to see whether
it does" — both readings are consistent with the same numbers, and the
result is reported that way rather than rounded up to a claim the data
doesn't support. Scaling past this needs the same thing ADR 0009 already
named for the replay mechanism generally: more captured fixture sessions
from a live pipeline run over time, and `scripts/reconcile_outcomes.py`
run against enough of them that `get_agent_accuracy` has real per-regime
win rates to weight against. The system-behavior metrics reported
alongside the pre-registered result (schema validity 6/6, 0 constraint
violations, bit-identical replay determinism across two independent runs)
are kept separate from it deliberately — a healthy pipeline and a
statistically inconclusive result are two different claims, and blending
them into one score would have hidden which one this evaluation actually
supports.

## What this demonstrates

Not that the reliability-weighting mechanism works — the evaluation
doesn't show that, and says so. What it demonstrates is a decision loop
that is now real enough to fail an honest test: a decision gets made, its
outcome gets reconciled against real market data, credit gets assigned to
the agent that actually moved the number, and the next decision's agent
weights are a function of that history rather than a constant. Before this
rebuild, none of that chain existed to evaluate — the 159/159 `PENDING`
query wasn't a disappointing result, it was proof there was no result to
be had. The gap between "we couldn't measure whether this works" and "we
measured it and the result was inconclusive at n=6" is the actual work of
these twelve PRs, and the second sentence is a better place to end up than
a fabricated "PASS" ever was.
