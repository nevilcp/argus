"""Aggregation properties for bounded conviction, participation, and monotonicity."""

from hypothesis import assume, given
from hypothesis import strategies as st

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.params import AGGREGATOR
from argus.schemas.signals import (
    Regime,
    Signal,
)
from tests.helpers.builders import (
    fundamental_signal,
    macro_context,
    sentiment_signal,
    technical_signal,
)

_signals = st.sampled_from(list(Signal))
_convictions = st.floats(min_value=0.0, max_value=0.95, allow_nan=False)
_multipliers = st.floats(min_value=0.5, max_value=1.5, allow_nan=False)
_regimes = st.sampled_from(list(Regime))


def _multipliers_for(fund_mult: float, tech_mult: float, sent_mult: float) -> dict[str, float]:
    """Build macro multipliers in the same order as the generated inputs."""
    return {"fundamental": fund_mult, "technical": tech_mult, "sentiment": sent_mult}


@given(
    tech_signal=_signals,
    tech_conv=_convictions,
    fund_signal=_signals,
    fund_conv=_convictions,
    sent_signal=_signals,
    sent_conv=_convictions,
    regime=_regimes,
    fund_mult=_multipliers,
    tech_mult=_multipliers,
    sent_mult=_multipliers,
)
def test_aggregated_conviction_never_exceeds_cap(
    tech_signal,
    tech_conv,
    fund_signal,
    fund_conv,
    sent_signal,
    sent_conv,
    regime,
    fund_mult,
    tech_mult,
    sent_mult,
):
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=technical_signal(tech_signal, tech_conv),
        macro=macro_context(regime, agent_multipliers=_multipliers_for(fund_mult, tech_mult, sent_mult)),
        fundamental=fundamental_signal(fund_signal, fund_conv),
        sentiment=sentiment_signal(sent_signal, sent_conv),
    )
    assert result.conviction >= 0.0
    assert result.conviction <= AGGREGATOR.max_conviction + 1e-9


@given(
    conv1=_convictions,
    conv2=_convictions,
    fund_mult=_multipliers,
    tech_mult=_multipliers,
    sent_mult=_multipliers,
)
def test_lone_agent_conviction_is_monotone_in_its_own_conviction(conv1, conv2, fund_mult, tech_mult, sent_mult):
    assume(conv2 - conv1 > 1e-6)
    aggregator = HybridSignalAggregator()
    macro = macro_context(Regime.EXPANSION, agent_multipliers=_multipliers_for(fund_mult, tech_mult, sent_mult))
    lo = aggregator.aggregate(
        technical=technical_signal(Signal.BULLISH, conv1),
        macro=macro,
        fundamental=None,
        sentiment=None,
    )
    hi = aggregator.aggregate(
        technical=technical_signal(Signal.BULLISH, conv2),
        macro=macro,
        fundamental=None,
        sentiment=None,
    )
    assert hi.conviction > lo.conviction


@given(
    conv=_convictions,
    fund_mult=_multipliers,
    tech_mult=_multipliers,
    sent_mult=_multipliers,
    regime=_regimes,
)
def test_lone_agent_cannot_reach_the_cap(conv, fund_mult, tech_mult, sent_mult, regime):
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=technical_signal(Signal.BULLISH, conv),
        macro=macro_context(regime, agent_multipliers=_multipliers_for(fund_mult, tech_mult, sent_mult)),
        fundamental=None,
        sentiment=None,
    )
    assert result.conviction < AGGREGATOR.max_conviction


def test_exact_tie_between_bull_and_bear_resolves_neutral():
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=technical_signal(Signal.BULLISH, 0.8),
        macro=macro_context(Regime.EXPANSION, agent_multipliers=_multipliers_for(1.0, 1.0, 1.0)),
        fundamental=fundamental_signal(Signal.BEARISH, 0.8),
        sentiment=None,
    )
    assert result.signal == Signal.NEUTRAL
    assert result.conviction == 0.0
    assert result.agents_present == ["fundamental", "technical"]
