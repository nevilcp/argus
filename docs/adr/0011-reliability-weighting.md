# ADR 0011: Reliability weighting consumes the outcome loop

**Status:** Accepted.

## Context

PR 8 closed the decision→outcome loop: `cultural.store_trade_outcome` is now
actually called, `primary_driver` is populated via leave-one-out ablation
(ADR 0010), and `get_agent_accuracy` has real rows to query instead of an
empty collection. But nothing consumed `get_agent_accuracy`'s output — the
loop terminated at storage. This PR is the one differentiated capability the
rebuild plan is organized around: make the system's future signal weighting
actually depend on whether its past signals were right.

Two smaller, previously-committed-and-forgotten pieces travel with it because
they're the same shape of bug: a value computed and never read.
`cultural_memory["warnings"]` was retrieved every session and only ever
handed to `portfolio_allocation`'s `wisdom` argument — `warnings` sat in
state, unused. And `AggregatedSignal.debate_triggered` / `skip_reason` were
written by `aggregator.aggregate()` and read by nothing, anywhere — the
`docs/historical/ARCHITECTURE_WALKTHROUGH.md` description of a `_resolve_conflict()`
LLM call that spends a Groq request breaking split votes described code that
was never actually in the repository.

## Decision

### Shrinkage toward the 0.5 prior

`get_agent_accuracy(agent_name, regime)` returns `(wins + k * 0.5) / (n + k)`
rather than the raw win rate `wins / n`, with `k = MEMORY.accuracy_shrinkage_k`
(10, tagged `ARBITRARY` — no basis for this specific strength, see
`argus/params.py`). Plain `wins / n` is unusable at small `n`: a technical
agent that has been credited on 2 trades and won both would report 100%
accuracy and get scaled up as if that were as reliable as a 200-trade track
record. Additive (Laplace-style) smoothing toward the neutral prior is the
standard fix and needs no new machinery — `n = 0` already returned exactly
0.5 before this PR (an unweighted special case), and the shrinkage formula
subsumes it without a separate branch, since `(0 + k*0.5) / (0 + k) = 0.5` for
any `k`.

**Considered and rejected:** a fixed minimum-sample-size gate (e.g. "return
0.5 unless n >= 20"). Rejected because it's a step function — the 19th and
20th observation are treated completely differently — where shrinkage
degrades continuously as evidence accumulates, which is the actual claim
being made ("a little evidence should move you a little").

### The reliability multiplier: `win_rate / 0.5`

`HybridSignalAggregator.aggregate()` gains an optional `reliability: dict[str,
float] | None` parameter — agent name → win rate. Each agent's effective vote
weight becomes `base_weight × macro_multiplier × (reliability.get(name, 0.5) /
0.5)`. At the neutral prior (0.5, or missing from the dict, or `reliability=
None` entirely) the multiplier is exactly 1.0 — behaviour is unchanged from
before this PR. Above 0.5 it scales up linearly; below 0.5 it scales down
linearly, floors at 0 for an agent with a 0% shrunk win rate (impossible in
practice once `k=10` pseudo-observations are mixed in, but bounded correctly
regardless).

This keeps `aggregate()` a pure function of its arguments — no new dependency
on `CulturalMemoryManager` inside `orchestration/aggregator.py`, which stays
importable and property-testable (`tests/test_aggregator_properties.py`)
without pulling in ChromaDB or sentence-transformers. `orchestration/graph.py`
computes `reliability` once per session (win rate doesn't depend on ticker,
only agent + regime) via `get_cultural_memory().get_agent_accuracy(name,
regime=macro.macro_regime.value)` and passes it into every `aggregate()` call
for that session.

**Considered and rejected:** giving the aggregator its own `CulturalMemoryManager`
reference and calling `get_agent_accuracy` internally. Rejected for the same
reason PR 5 injects `MarketDataProvider`/`LLMClient` rather than reaching for
module singletons: it would make `aggregate()` untestable without a live or
mocked vector store, for a value the caller already has cheaply.

### `cultural_memory["warnings"]` reaches the Portfolio Manager prompt

`node_portfolio_allocation` now reads `mem.get("warnings", [])` alongside
`wisdom` and passes both into `PortfolioManagerAgent.allocate()`, which adds a
`CULTURAL WARNINGS` section to the LLM prompt, mirroring the existing
`CULTURAL WISDOM` section. `retrieve_warnings()` already scoped its query to
`FAILED` outcomes in the current macro regime (PR 8) — this PR is only the
missing wire from state into the prompt.

### `debate_triggered` and `skip_reason`: deleted, not consumed

Both fields are removed from `AggregatedSignal` and from `aggregate()`. They
are not given a downstream reader. Two reasons this is deletion rather than
wiring:

1. `debate_triggered`'s stated purpose — "flag split or contested votes for
   downstream portfolio handling" — has no downstream handling to hook it
   into within this PR's scope, and inventing one (e.g. suppressing
   allocation on a contested vote) would be a new policy decision this ADR
   isn't the place to make casually.
2. `skip_reason` only ever took one value (`"No agent signals available"`),
   set exactly when `total == 0.0` — a case the caller (`node_signal_aggregation`)
   already handles correctly by simply not producing a meaningful signal.
   The string added no information beyond what `weighted_votes` being all-zero
   already says.

`AGGREGATOR.debate_trigger_margin`, the threshold parameter that only fed
`debate_triggered`'s computation, is deleted from `params.py` alongside it —
keeping an unused provenance-tagged constant around would be the same
dishonesty this rebuild exists to remove, just moved one file over.

`weighted_votes` is untouched: it is genuinely consumed, by
`orchestration/reconciliation.py`'s `credit_primary_driver()` (ADR 0010).

## Consequences

- The reliability-weighting unit test (see `tests/test_aggregator.py`)
  verifies both ends of the claim: an agent with a strong regime-specific
  track record measurably outweighs one with a poor record, and with zero
  history (or a `reliability=None` caller, matching every call site before
  this PR existed) the result is bit-for-bit identical to the old
  no-reliability behaviour.
- `credit_primary_driver()`'s ablation reruns (`orchestration/reconciliation.py`)
  call `aggregate()` without a `reliability` argument, since a completed
  `ARGUSDecision` doesn't carry the win rates that were live at decision
  time. This makes ablation a *baseline* re-aggregation rather than an exact
  replay of the reliability-weighted call that produced `decision.aggregated`
  — acceptable because credit assignment is already documented (ADR 0010) as
  an approximation optimized for explainability, not exact replay.
- Until enough outcomes accumulate per agent per regime, `reliability`
  reports at or near 0.5 for everything and the system's behavior is
  observably unchanged — this is intended, not a bug: the mechanism should
  earn its influence as evidence accumulates, not assume it on day one.
