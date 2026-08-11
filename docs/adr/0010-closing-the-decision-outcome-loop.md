# ADR 0010: Closing the decision→outcome loop

**Status:** Accepted (predictive — written before the code it justifies).

## Context

The evidence that motivated this whole rebuild: ChromaDB held 159 decision
snapshots, all 159 `PENDING`, 0 with `primary_driver` set. `store_trade_outcome`
(`argus/memory/cultural.py`) was fully implemented and never called from
anywhere. Nothing ever told the system whether a decision was right, so
`retrieve_wisdom`/`retrieve_warnings` always returned `[]` and
`get_agent_accuracy` always returned the 0.5 prior — the reliability loop PR 9
is meant to build has nothing to weight against until this closes.

Three separate problems live under "close the loop":

1. **Where do decisions live between decision time and horizon time?** A
   decision needs to be looked up again once enough time has passed to know
   whether it worked. `node_log_decisions` (`graph.py`) already calls
   `store_decision_snapshot`, but that writes a short summary document +
   metadata (regime, ticker, agg_signal, timestamp, decision_id) — not the
   full `TechnicalSignal`/`FundamentalSignal`/`SentimentSignal` needed to
   redo the aggregation. The obvious alternative, `DecisionLogger`
   (`data/cache.py`), archives the complete decision as JSON in
   `argus_decisions.db` — but is never instantiated.
2. **How is `primary_driver` actually computed?** The dead code in
   `store_trade_outcome` used three independent conviction thresholds
   (`> 0.8` per agent) — not a real measure of which agent's signal moved
   the outcome, just three unrelated boolean checks that happen to look
   like credit assignment.
3. **What counts as "the outcome"?** A realized return needs an entry price,
   an exit price, and a holding period, computed from data the system
   already has a seam for (`MarketDataProvider`).

## Decision

### Decision persistence: reuse the LangGraph checkpoint, not a new store

`build_graph()` already checkpoints full `ARGUSState` — including the
`decisions: list[ARGUSDecision]` field `node_log_decisions` populates — to
`argus_graph.db` via `SqliteSaver`, keyed by `thread_id` per session
(`docs/historical/ARCHITECTURE_WALKTHROUGH.md` already documents this as "the audit
trail"). That is a complete, durable, already-written record of every
decision, including the nested signal objects ablation needs. Building a
second archive (`DecisionLogger` or otherwise) to hold the same information
a second time would be exactly the kind of duplicated, half-wired mechanism
this rebuild exists to remove.

`argus/orchestration/reconciliation.py` adds
`load_decisions_from_checkpoints()`, which walks `SqliteSaver.list(None)`
(newest-first per thread) and keeps each thread's first (= latest, most
complete) checkpoint's `decisions` list. `JsonPlusSerializer` needs an
explicit `allowed_msgpack_modules` allowlist to reconstruct
`argus.schemas.signals.*` `BaseModel` subclasses on the way back out
(confirmed empirically — omitting it either logs a deprecation warning today
or, in a future `langgraph` version, silently degrades reconstructed
decisions to plain dicts). `build_checkpoint_serde()` in `graph.py` is the
single place that allowlist is declared, shared by `build_graph()`'s real
checkpointer and the reconciliation loader, so the two never drift.

**Considered and rejected:** giving `DecisionLogger` a real caller instead of
deleting it. Rejected because it would duplicate exactly the data the
checkpoint already holds, in a second SQLite file, for no consumer that
doesn't already have a better source. Deleting dead code that duplicates a
working mechanism is not the same call as deleting dead code with no
replacement — this one has one.

### Credit assignment: leave-one-out ablation, not exact Shapley

`HybridSignalAggregator.aggregate()` is a pure function of
`(technical, macro, fundamental, sentiment)`. `credit_primary_driver()` reruns
it once per present agent with that agent's signal set to `None`, and scores
each removal as `(did the consensus direction flip, how large was this
agent's own baseline weighted vote)` — the largest-scoring removal is
credited. A plain delta on the ablated call's final `conviction` was tried
first and doesn't work: `aggregate()` normalizes its bull/bear/neutral pools
to a percentage of the total, so whenever ablation leaves exactly one voting
agent, that agent trivially owns 100% of its own pool regardless of size —
the final conviction saturates near `AGGREGATOR.max_conviction` either way,
so a same-direction two-agent decision (the common case) always ties and
silently reduces to "whichever agent is checked first," reproducing almost
exactly the brokenness this PR exists to fix. Direction-flip-first,
magnitude-second avoids that: an agent whose removal flips the consensus was
clearly essential; among removals that don't flip it, `aggregate()`'s pools
are additive sums of independent per-agent votes, so a removed agent's
marginal effect on its pool is exactly its own already-computed
`weighted_votes` entry — no need to re-derive it per ablation, and no
saturation artifact. (Caught by property-testing this function while writing
it, the same way PR 6's RSI monotonicity test caught a real scoring bug —
another instance of the tests finding something the implementation missed.)

With three agents, leave-one-out is also the full Shapley value up to the
constant term — exact Shapley only diverges from leave-one-out once there
are enough coalitions that marginal contribution order matters (2^n - 1
coalitions vs. n). For n = 3 that's 7 coalitions vs. 3 — computable, but the
extra machinery buys nothing here: the goal is to be able to say "the
technical agent's signal, on its own, moved the number here," not to defend
a game-theoretic allocation across nested coalitions nobody will inspect.
Considered and rejected in favor of the simpler, equally-informative,
easier-to-explain version — consistent with this rebuild's stated bias
toward explainability over sophistication.

### Outcome: entry/exit close prices via the existing MarketDataProvider seam

`compute_realized_return()` takes the decision's entry price from
`decision.technical.current_price` (the close price the technical agent
scored against) and looks up the close price on or after
`session_timestamp + horizon_days` via `MarketDataProvider.ohlcv_daily()`.
Decisions with no `technical` signal (no entry price) or no `allocation`
(no position was actually taken — nothing to reconcile) are skipped, not
fabricated a return. If the price series doesn't yet extend past the target
exit date, reconciliation defers the decision rather than guessing — same
rule as ADR 0002.

`horizon_days` is a new `RECONCILIATION` group in `params.py`, tagged
`ARBITRARY` — a placeholder value used so the mechanism has something to
run against before PR 10 fixes the real evaluation horizon H
*before observing any result*, per the plan's pre-registration requirement.
Whatever value ships here is provisional by design; PR 10 either confirms or
replaces it, not the other way around.

### `store_trade_outcome` takes `primary_driver` as a parameter

`cultural.py`'s job (per its own docstring) is persistence, not signal
arbitration — computing credit assignment inline conflated the two. The
three-threshold heuristic is deleted; `store_trade_outcome` now takes
`primary_driver: str`, computed by the caller (`reconcile_decision()`) via
`credit_primary_driver()` before the call.

### `DecisionLogger` is deleted

`argus/data/cache.py`'s `DecisionLogger` (SQLite decision archive) is never
instantiated anywhere in the codebase (confirmed by full-repo search — the
only other reference is a stale docstring in `orchestration/state.py`). Its
role — a durable, queryable decision archive — is already filled by the
LangGraph checkpoint (see above), so this is a straight deletion, not a
replacement.

## Consequences

- `scripts/reconcile_outcomes.py` is the operational entry point: load
  decisions from `argus_graph.db`, reconcile whichever ones have cleared
  `horizon_days`, report how many were stored. It's meant to run
  periodically (e.g. a daily cron) against the live checkpoint database —
  this PR ships the mechanism and the CLI, not a scheduler.
- `argus_decisions.db` (the file `DecisionLogger` used to write) becomes
  stale after this PR; it is not migrated or read by anything new.
- The regression check from the rebuild plan — `Counter(m.get('outcome')
  for m in coll.get()['metadatas'])` moving off all-`PENDING` — now has a
  real code path that can make it true, for the first time.
- `horizon_days` being `ARBITRARY` means any reconciled outcome produced
  before PR 10 lands is provisional — real for testing the mechanism, not
  yet meaningful for reliability weighting (PR 9) or evaluation (PR 10).
