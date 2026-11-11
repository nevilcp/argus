"""Tests for orchestration/attribution.py: leave-one-out credit assignment."""

from __future__ import annotations

from datetime import datetime

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.attribution import credit_primary_driver
from argus.schemas.signals import (
    ARGUSDecision,
    Signal,
)
from tests.helpers.builders import (
    aggregated_decision,
    fundamental_signal,
    macro_context,
    sentiment_signal,
    technical_signal,
)

# ---------------------------------------------------------------------------
# credit_primary_driver
# ---------------------------------------------------------------------------


def test_credit_primary_driver_returns_unknown_with_no_signals():
    """A decision with no agent signals credits no driver."""
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime.now())
    assert credit_primary_driver(decision) == "unknown"


def test_credit_primary_driver_returns_the_sole_signal_when_only_one_present():
    """With only one agent signal present, that agent is credited by default."""
    decision = aggregated_decision(technical_signal(Signal.BULLISH, 0.8), macro_context())
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_credits_the_dominant_agent():
    """Leave-one-out ablation credits whichever agent's removal swings conviction most."""
    decision = aggregated_decision(
        technical_signal(Signal.BULLISH, 0.9),
        macro_context(),
        fundamental_signal(Signal.BULLISH, 0.1),
    )
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_is_symmetric_under_relabeling():
    """Swapping which agent is dominant should swap the credited driver."""
    decision = aggregated_decision(
        technical_signal(Signal.BULLISH, 0.1),
        macro_context(),
        fundamental_signal(Signal.BULLISH, 0.9),
    )
    assert credit_primary_driver(decision) == "fundamental"


def test_credit_primary_driver_replays_the_baseline_reliability_in_each_ablation():
    """Ablation must reuse the exact reliability dict the baseline aggregation used.

    Regression test for X3: without this, every ablated rerun silently falls
    back to aggregate()'s unweighted reliability_mult=1.0 default, so the
    baseline (computed WITH reliability) and each ablated rerun (computed
    WITHOUT it) can disagree on consensus direction before any agent is even
    removed — flipping credit to argmax(baseline_votes) regardless of which
    agent was actually ablated.
    """
    reliability = {"technical": 0.95, "fundamental": 0.20, "sentiment": 0.50}
    decision = aggregated_decision(
        technical_signal(Signal.BULLISH, 0.9),
        macro_context(),
        fundamental_signal(Signal.BEARISH, 0.85),
        sentiment_signal(Signal.NEUTRAL, 0.3),
        reliability=reliability,
    )

    seen_reliability: list[object] = []

    class _SpyAggregator(HybridSignalAggregator):
        def aggregate(self, *args, **kwargs):
            seen_reliability.append(kwargs.get("reliability"))
            return super().aggregate(*args, **kwargs)

    credit_primary_driver(decision, aggregator=_SpyAggregator())

    assert seen_reliability
    assert all(r == reliability for r in seen_reliability)
