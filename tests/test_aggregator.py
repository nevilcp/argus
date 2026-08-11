"""
tests/test_aggregator.py

Unit tests for HybridSignalAggregator.aggregate()'s reliability-weighting
parameter (see docs/adr/0011). Reuses the signal/macro builders from
test_aggregator_properties.py rather than duplicating them.

Covers the two claims the rebuild plan's verification section makes about
this mechanism: an agent with a strong regime-specific track record
measurably outweighs one with a poor record, and with no history the
result is identical to today's (pre-reliability) behavior.
"""

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import Regime, Signal
from tests.test_aggregator_properties import _fundamental, _macro, _sentiment, _technical


def _split_signals():
    """A genuine three-way split: BULLISH vs. BEARISH vs. NEUTRAL, equal weight."""
    technical = _technical(Signal.BULLISH, 0.8)
    fundamental = _fundamental(Signal.BEARISH, 0.8)
    sentiment = _sentiment(Signal.NEUTRAL, 0.3)
    macro = _macro(Regime.EXPANSION, fund_mult=1.0, tech_mult=1.0, sent_mult=1.0)
    return technical, macro, fundamental, sentiment


def test_no_history_reliability_matches_unweighted_aggregation():
    technical, macro, fundamental, sentiment = _split_signals()
    aggregator = HybridSignalAggregator()

    unweighted = aggregator.aggregate(technical, macro, fundamental, sentiment)
    explicit_prior = aggregator.aggregate(
        technical,
        macro,
        fundamental,
        sentiment,
        reliability={"technical": 0.5, "fundamental": 0.5, "sentiment": 0.5},
    )
    no_reliability_arg = aggregator.aggregate(
        technical, macro, fundamental, sentiment, reliability=None
    )

    assert unweighted.signal == explicit_prior.signal == no_reliability_arg.signal
    assert (
        unweighted.conviction == explicit_prior.conviction == no_reliability_arg.conviction
    )
    assert (
        unweighted.weighted_votes
        == explicit_prior.weighted_votes
        == no_reliability_arg.weighted_votes
    )


def test_strong_track_record_measurably_outweighs_a_poor_one():
    technical, macro, fundamental, sentiment = _split_signals()
    aggregator = HybridSignalAggregator()

    baseline = aggregator.aggregate(technical, macro, fundamental, sentiment)
    # technical (BULLISH) has been reliably right in this regime;
    # fundamental (BEARISH) has been reliably wrong.
    reweighted = aggregator.aggregate(
        technical,
        macro,
        fundamental,
        sentiment,
        reliability={"technical": 0.9, "fundamental": 0.1, "sentiment": 0.5},
    )

    assert reweighted.weighted_votes["technical"] > baseline.weighted_votes["technical"]
    assert reweighted.weighted_votes["fundamental"] < baseline.weighted_votes["fundamental"]
    assert reweighted.signal == Signal.BULLISH
    assert reweighted.conviction > baseline.conviction
