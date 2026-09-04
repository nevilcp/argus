"""
tests/test_aggregator.py

Unit tests for HybridSignalAggregator.aggregate()'s reliability weighting.

Reuses the signal/macro builders from test_aggregator_properties.py rather
than duplicating them. Covers two claims about that mechanism: an agent
with a strong regime-specific track record measurably outweighs one with a
poor record, and with no history the result is identical to today's
(pre-reliability) behavior.
"""

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import (
    FundamentalSignal,
    MacroContext,
    Regime,
    SentimentSignal,
    Signal,
    TechnicalSignal,
)
from tests.test_aggregator_properties import _fundamental, _macro, _sentiment, _technical


def _split_signals() -> tuple[TechnicalSignal, MacroContext, FundamentalSignal, SentimentSignal]:
    """A genuine three-way split: BULLISH vs. BEARISH vs. NEUTRAL, equal weight."""
    technical = _technical(Signal.BULLISH, 0.8)
    fundamental = _fundamental(Signal.BEARISH, 0.8)
    sentiment = _sentiment(Signal.NEUTRAL, 0.3)
    macro = _macro(Regime.EXPANSION, fund_mult=1.0, tech_mult=1.0, sent_mult=1.0)
    return technical, macro, fundamental, sentiment


def test_no_history_reliability_matches_unweighted_aggregation():
    """With no reliability history, aggregation is identical to the unweighted path."""
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

    for other in (explicit_prior, no_reliability_arg):
        assert other.signal == unweighted.signal
        assert other.conviction == unweighted.conviction
        assert other.weighted_votes == unweighted.weighted_votes


def test_strong_track_record_measurably_outweighs_a_poor_one():
    """A high-reliability agent's vote gains weight; a low-reliability one's shrinks."""
    technical, macro, fundamental, sentiment = _split_signals()
    aggregator = HybridSignalAggregator()

    baseline = aggregator.aggregate(technical, macro, fundamental, sentiment)
    # technical (BULLISH) reliably right in this regime, fundamental (BEARISH) reliably wrong
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


def test_aggregate_with_every_input_signal_absent():
    """No specialist signal at all degrades to a neutral, zero-conviction result.

    Every input absent must complete and report a visibly unmeasured
    result — NEUTRAL at zero conviction with no agents present — rather
    than raise or fabricate a directional call.
    """
    aggregator = HybridSignalAggregator()

    result = aggregator.aggregate(None, None, None, None)

    assert result.signal == Signal.NEUTRAL
    assert result.conviction == 0.0
    assert result.agents_present == []
    assert result.weighted_votes == {"fundamental": 0.0, "technical": 0.0, "sentiment": 0.0}
