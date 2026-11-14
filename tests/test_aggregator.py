"""Unit tests for HybridSignalAggregator.aggregate()'s reliability weighting."""

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import (
    FundamentalSignal,
    MacroContext,
    Regime,
    SentimentSignal,
    Signal,
    TechnicalSignal,
)
from tests.helpers.builders import (
    fundamental_signal,
    macro_context,
    sentiment_signal,
    technical_signal,
)


def _split_signals() -> tuple[TechnicalSignal, MacroContext, FundamentalSignal, SentimentSignal]:
    """Build an equally weighted three-way directional split."""
    technical = technical_signal(Signal.BULLISH, 0.8)
    fundamental = fundamental_signal(Signal.BEARISH, 0.8)
    sentiment = sentiment_signal(Signal.NEUTRAL, 0.3)
    macro = macro_context(Regime.EXPANSION)
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
    no_reliability_arg = aggregator.aggregate(technical, macro, fundamental, sentiment, reliability=None)

    for other in (explicit_prior, no_reliability_arg):
        assert other.signal == unweighted.signal
        assert other.conviction == unweighted.conviction
        assert other.weighted_votes == unweighted.weighted_votes


def test_strong_track_record_measurably_outweighs_a_poor_one():
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
    aggregator = HybridSignalAggregator()

    result = aggregator.aggregate(None, None, None, None)

    assert result.signal == Signal.NEUTRAL
    assert result.conviction == 0.0
    assert result.agents_present == []
    assert result.weighted_votes == {"fundamental": 0.0, "technical": 0.0, "sentiment": 0.0}
