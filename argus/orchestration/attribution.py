"""
argus/orchestration/attribution.py

Credits a completed decision's outcome to one specialist agent.

Responsibilities:
  - credit_primary_driver: leave-one-out ablation over HybridSignalAggregator

Not responsible for:
  - Computing the outcome or storing it (see orchestration/reconcile.py)
  - The aggregation itself (see orchestration/aggregator.py)

Dependencies:
  - argus.orchestration.aggregator (HybridSignalAggregator)
"""

from __future__ import annotations

from argus.orchestration.aggregator import SPECIALIST_AGENTS, HybridSignalAggregator
from argus.schemas.signals import ARGUSDecision


def credit_primary_driver(
    decision: ARGUSDecision, aggregator: HybridSignalAggregator | None = None
) -> str:
    """Credits whichever specialist agent's signal moved the aggregated result the most.

    Leave-one-out ablation: reruns HybridSignalAggregator.aggregate() once per
    present agent with that agent's signal removed. Each removal is scored as
    (did the consensus direction flip, how large was this agent's own
    baseline weighted vote) and the largest-scoring removal is credited.

    Direction first, magnitude second, rather than a scalar delta on the
    ablated call's final `conviction`: it is the consensus *direction* an
    ablation settles on that decides whether the resulting trade changes, so an
    agent whose removal flips the direction had a categorically larger effect
    than one whose removal only moved the number. Among removals that don't
    flip the outcome, the agent that cast more raw vote was the bigger
    contributor — read straight off the baseline `weighted_votes`, since the
    pools are additive sums of independent per-agent votes and a removed
    agent's marginal effect on its pool is exactly its own vote. This avoids
    exact Shapley over the 2^3 - 1 coalitions.

    Each ablated rerun replays `decision.aggregated.reliability` — the same
    reliability dict the baseline aggregation used — rather than an
    unweighted aggregate() call. Without it, every ablation would silently
    use reliability_mult=1.0, so the baseline (computed with reliability)
    and every ablated rerun (computed without it) could disagree on
    direction before any agent is even removed, spuriously flipping credit
    to argmax(baseline_votes) for every agent regardless of which one was
    actually removed.

    Args:
        decision: Completed ARGUSDecision with technical/fundamental/sentiment
            signals and its `aggregated` result already populated.
        aggregator: HybridSignalAggregator to re-run; defaults to a fresh
            instance (aggregate() is a pure function of its arguments — there
            is no state for two calls to share).

    Returns:
        'technical', 'fundamental', or 'sentiment'. A sole present signal is
        credited directly, with no ablation, even when the decision has no
        aggregated result. 'unknown' if no agent contributed a signal, or if
        two or more did but the decision has no aggregated result to ablate from.
    """
    present = [name for name in SPECIALIST_AGENTS if getattr(decision, name) is not None]

    if decision.aggregated is None or len(present) < 2:
        return present[0] if len(present) == 1 else "unknown"

    agg = aggregator or HybridSignalAggregator()
    baseline_signal = decision.aggregated.signal
    baseline_votes = decision.aggregated.weighted_votes

    best_name = "unknown"
    best_score = (False, -1.0)
    for name in present:
        result = agg.aggregate(
            None if name == "technical" else decision.technical,
            decision.macro,
            None if name == "fundamental" else decision.fundamental,
            None if name == "sentiment" else decision.sentiment,
            reliability=decision.aggregated.reliability,
        )
        flipped = result.signal != baseline_signal
        score = (flipped, baseline_votes.get(name, 0.0))
        if score > best_score:
            best_score = score
            best_name = name

    return best_name
