# ADR 0012: Pre-registered evaluation

**Status:** Accepted (predictive — written before any result below is
observed; see "Pre-registration" for the point at which this ADR was
frozen).

## Context

PR 8 closed the decision→outcome loop; PR 9 made reliability weighting
consume it. Both shipped with `RECONCILIATION.horizon_days = 5` tagged
`ARBITRARY` and a comment reading "provisional; PR 10 pre-registers the
real evaluation horizon H before observing any result" — this ADR is that
pre-registration.

The evaluation question is narrow and falls out of what PR 9 actually
built: does per-regime reliability weighting (ADR 0011) change whether
`aggregated.conviction`'s direction predicts a decision's forward return,
relative to the unweighted baseline? Two things this ADR does **not**
attempt, and why:

- **A multi-year backtest.** Ruled out by ADR 0009 — Yahoo's 60-day
  intraday lookback means historical `session_states` cannot be
  reconstructed beyond ~60 days, and fabricating them violates ADR 0002.
- **A large-sample evaluation.** Exactly one fixture session exists today
  (`tests/fixtures/`, captured in PR 5, 6 tickers). Scaling that up means
  running the live MFT pipeline and capturing more sessions over time
  (ADR 0009's "future work"), not something this PR can manufacture. The
  sample size below is what actually exists, not a target.

Pre-registering with `n=6` is still worth doing honestly: it fixes the
metric and the bar for "worked" *before* looking at the answer, which is
the actual discipline being exercised here — the alternative (deciding
what counts as success after seeing the numbers) is the failure mode this
whole rebuild exists to avoid. The plan already commits to reporting
confidence intervals and disclosing low statistical power rather than
treating six decisions as a large evaluation.

## Decision

### What is measured

For each ticker's decision in a replayed session, pair its **signed
conviction** — `aggregated.conviction` if `aggregated.signal ==
BULLISH`, `-aggregated.conviction` if `BEARISH`, `0.0` if `NEUTRAL` — with
its **realized forward return**, computed by
`orchestration/reconciliation.py:compute_realized_return` (entry =
`technical.current_price` at `session_timestamp`, exit = first close on
or after `session_timestamp + H` days). This is the same function PR 8
already uses to feed `cultural.store_trade_outcome` — the evaluation
reuses the real reconciliation path rather than a separate one.

### Horizon H = 5 calendar days

Kept at its existing value rather than changed, for two reasons: (1) it
already matches `cultural.store_trade_outcome`'s own horizon, so the
evaluation measures exactly the decisions that would actually be
reconciled into memory in production, not a hypothetical alternate
horizon; (2) it is short enough that the one captured session (dated
2026-05-22) has already cleared it against real market data by the time
this ADR was written (today, per repository state, is well past
2026-05-27). Still tagged `ARBITRARY` in `params.py` — pre-registering a
value is not the same as justifying it empirically, and no such
justification exists yet.

### Dead-band = 1% (reuses `RECONCILIATION.min_abs_return_for_storage`)

Rather than invent a second arbitrary threshold, hit-rate scoring reuses
the same 1% band `cultural.store_trade_outcome` already applies before
persisting an outcome. A decision whose forward return falls inside
±1% is excluded from hit-rate scoring (not countable as "right" or
"wrong" — this is what `cultural.store_trade_outcome` already implicitly
decides by not storing it) but still included in rank IC, which measures
ordering rather than a binary win/loss.

### Metrics, reported separately, never blended

1. **Rank information coefficient** — Spearman correlation between signed
   conviction and forward return across all decisions in a session.
   Chosen over Pearson because conviction is not claimed to be linearly
   related to return magnitude, only orderable; rank IC is the standard
   choice for exactly this weaker claim.
2. **Hit-rate-with-dead-band** — fraction of decisions outside the dead
   band where `sign(conviction) == sign(forward_return)`.
3. Both computed **twice per session**: once replaying open-loop
   (`replay_session(..., closed_loop=False)`, reliability weighting fixed
   at the 0.5 prior — today's mocked default) and once closed-loop
   (`closed_loop=True`, reliability weighting reads whatever accuracy
   history actually exists in `chroma_db` via the real
   `get_agent_accuracy`). The comparison is closed-loop vs. open-loop, not
   ARGUS vs. a market benchmark — the question is whether PR 9's
   mechanism helps, not whether the system beats SPY.

Confidence intervals: percentile bootstrap (2,000 resamples, paired
resampling of (conviction, return) tuples, seed fixed for
reproducibility) rather than a closed-form Spearman CI, because closed-form
formulas assume asymptotic normality that does not hold at `n=6`.

### System-behavior metrics — reported separately, never blended into the above

- **Schema validity**: count of tickers for which `node_log_decisions`
  successfully built an `ARGUSDecision` vs. `state["errors"]` entries
  recorded during the session. Every constructed `ARGUSDecision` is valid
  by construction (Pydantic raises before an invalid one exists) — this
  metric reports completion rate, not post-hoc validation.
- **Constraint violations**: `RiskAssessment.approved_weight <=
  proposed_weight` is a Pydantic `model_validator` — literally impossible
  to construct a violating instance, the same structural argument ADR 0009
  makes about point-in-time correctness. Reported as "0 by construction"
  alongside the count of `REDUCE` verdicts that actually exercised the
  constraint (evidence it does something, not just that nothing broke).
- **API calls per decision**: `ARGUSDecision.total_api_calls` already
  exists and is summed here — the honest proxy for "cost per decision"
  that the codebase actually instruments. Not a token count.
- **Retries and tokens/decision**: **not instrumented.** Each agent retries
  up to 3 times internally (`fundamental.py`, `sentiment.py`,
  `portfolio.py`) but no attempt count or token usage is surfaced to
  state. Reported as a disclosed gap rather than fabricated — adding that
  instrumentation is out of this PR's scope (it touches every agent for a
  metric this evaluation does not depend on) and is left as future work.
- **Replay determinism**: `replay_session` invoked twice against the same
  fixture directory with `closed_loop=False`; every decision's `signal`,
  `conviction`, and `total_api_calls` must match exactly. (`closed_loop=True`
  is not required to be deterministic across runs — it reads live
  `chroma_db` state, which the reconciliation loop can mutate between
  runs.)

### Success threshold, fixed before running

Closed-loop is judged to have helped only if **both**: the closed-loop
rank IC's bootstrap 95% CI lies entirely above 0, **and** it is
strictly greater than the open-loop rank IC's point estimate. A CI that
crosses zero, or that overlaps the open-loop estimate, is a negative
result and is reported as one — not reframed as "promising" or
"suggestive." At `n=6` the a priori expectation is that this bar will not
be cleared (an underpowered test correctly failing to reject a true null
looks identical to one incorrectly failing to detect a real effect); that
expectation is stated here, before running, precisely so the result
below cannot be quietly rationalized either way after the fact.

There is also a structural reason to expect open-loop and closed-loop to
be numerically close regardless of power: `chroma_db` held 201 `PENDING`
(zero resolved) outcomes at the time this ADR was written — no
reconciled trade history exists yet for `get_agent_accuracy` to shrink
away from the 0.5 prior, so closed-loop's reliability weights are
themselves expected to sit at or near 0.5 (ADR 0011's "unchanged
behaviour" case), independent of sample size. This is disclosed here, not
discovered after running the numbers.

## Consequences

- `argus/backtesting/evaluation.py` provides `rank_information_coefficient`,
  `hit_rate_with_deadband`, `bootstrap_ci`, `evaluate_decisions`, and
  `system_behavior_report` — pure functions over `ARGUSDecision` lists and
  a `MarketDataProvider`, reusing `reconciliation.compute_realized_return`
  and `metrics.py`'s existing statistical machinery rather than
  duplicating it.
- `replay_session` gains a `closed_loop: bool = False` parameter;
  default behavior (open-loop) is unchanged from PR 7/8/9.
- `scripts/run_evaluation.py` runs both conditions against
  `LiveMarketDataProvider` (forward prices for the 2026-05-22 session are
  now historical, not live-dependent on the day the script runs) and
  prints the report; results are committed to
  `docs/evaluation-results.md` rather than into this ADR, keeping the
  pre-registered decision separate from the observed outcome it is judged
  against.
- Scaling this evaluation to more than one session is the same
  prerequisite ADR 0009 already named: capturing more fixture sessions
  from a live pipeline run over time. Nothing here manufactures that
  history.
