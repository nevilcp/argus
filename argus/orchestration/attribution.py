"""Outcome credit from direction-first leave-one-out specialist ablation."""

from __future__ import annotations

from argus.orchestration.aggregator import SPECIALIST_AGENTS, HybridSignalAggregator
from argus.schemas.signals import ARGUSDecision


def credit_primary_driver(decision: ARGUSDecision, aggregator: HybridSignalAggregator | None = None) -> str:
    """Credit the specialist whose removal flips consensus first, then whose baseline vote is largest."""
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
