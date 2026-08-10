# ADR 0005: Monotonic risk-verdict downgrade

**Status:** Accepted (retrospective).

## Context

`RiskStatisticalEngine.evaluate` (`argus/agents/risk.py`) is the last
statistical gate before an allocation is returned. It has to reconcile
hard structural rules (single-position cap, diversification floor/ceiling,
VIX blackout), portfolio-level statistical limits (VaR 99%, CVaR, beta,
average pairwise correlation, sector concentration), and the optimizer's
attempt to fix violations by re-weighting (the SLSQP solver at
`risk.py:387-398`). A verdict engine that lets a later, softer check
override an earlier, harder one — e.g. an optimizer that satisfies VaR
producing an APPROVE that overwrites an earlier structural VETO — would
silently reintroduce the violation it was supposed to prevent.

## Decision

`evaluate` checks gates in a fixed order of decreasing severity, and each
stage can only make the outcome more conservative than the stage before it,
never less:

1. **Structural violations** (`risk.py:297-318`: position cap, diversification
   bounds, VIX blackout) → `RiskVerdict.VETO`, `approved_weight = 0.0`. If any
   of these trip, the function returns immediately; nothing downstream runs.
2. **Statistical violations** (`risk.py:407-421`: VaR/CVaR/beta/correlation/
   sector limits, checked only if step 1 passed) → `RiskVerdict.REDUCE`,
   `approved_weight = min(total_weight * 0.5, total_weight)`.
3. **Otherwise** → `RiskVerdict.APPROVE`, `approved_weight = total_weight`.

`approved_weight <= proposed_weight` holds in every branch by construction —
there is no code path that increases the weight relative to what was
proposed. The SLSQP optimizer runs *before* the statistical check and can
only change how weight is distributed across tickers/sectors within the
proposed total, never raise the total past what was proposed, and never
downgrade a VETO back to an APPROVE — it isn't consulted at all once step 1
has already vetoed.

## Consequences

- A REDUCE or VETO from this engine is a floor, not a suggestion the
  downstream `PortfolioManagerAgent` can quietly override — `portfolio.py`
  reads `approved_weight` and `stop_loss` from the `RiskAssessment` and
  applies them (see ADR 0004), it does not re-derive its own risk figure.
- This is exactly the property PR 6's property-based tests are meant to
  pin down permanently: "approved_weight ≤ proposed_weight across the
  RiskAssessment space," via Hypothesis rather than fixed examples, so a
  future change to the ordering or the optimizer's objective can't silently
  reintroduce a case where a later, softer check overrides an earlier,
  harder one.
