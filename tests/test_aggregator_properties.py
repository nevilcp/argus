"""
tests/test_aggregator_properties.py

Property-based tests on HybridSignalAggregator.aggregate() — a pure function
of its four signal/macro inputs. Invariants under test:
  - no matter how strongly technical/fundamental/sentiment agree, and no
    matter what macro multiplier scales their votes, the aggregated
    conviction never claims more certainty than
    argus.params.AGGREGATOR.max_conviction (0.95), kept below 1.0 so no
    aggregate claims certainty
  - a lone voting agent's aggregated conviction is monotone in its own
    conviction (Design Decision A, issue #24)
  - a lone voting agent can never reach the cap, since the denominator is
    the vote mass all three agents could cast, not just the one that voted
"""

from datetime import datetime

from hypothesis import assume, given
from hypothesis import strategies as st

from argus.orchestration.aggregator import HybridSignalAggregator
from argus.params import AGGREGATOR
from argus.schemas.signals import (
    FundamentalSignal,
    MacroContext,
    Regime,
    Signal,
    SectorSignal,
    SentimentSignal,
    TechnicalSignal,
    VixRegime,
    YieldCurve,
)

_signals = st.sampled_from(list(Signal))
_convictions = st.floats(min_value=0.0, max_value=0.95, allow_nan=False)
_multipliers = st.floats(min_value=0.5, max_value=1.5, allow_nan=False)
_regimes = st.sampled_from(list(Regime))


def _technical(signal: Signal, conviction: float) -> TechnicalSignal:
    return TechnicalSignal(
        ticker="TEST",
        current_price=100.0,
        rsi_14=50.0,
        macd_histogram=0.0,
        bb_percent_b=0.5,
        atr_pct=0.02,
        adx_14=20.0,
        vwap_distance=0.0,
        volume_ratio=1.0,
        momentum_30m=0.0,
        momentum_1d=0.0,
        signal=signal,
        conviction=conviction,
        net_score=0.0,
        timestamp=datetime.now(),
    )


def _fundamental(signal: Signal, conviction: float) -> FundamentalSignal:
    return FundamentalSignal(
        ticker="TEST",
        sector="Technology",
        industry="Software",
        signal=signal,
        conviction=conviction,
        moat_score=5.0,
        reasoning="test",
        data_as_of_date=datetime.now().date(),
        timestamp=datetime.now(),
    )


def _sentiment(signal: Signal, conviction: float) -> SentimentSignal:
    return SentimentSignal(
        ticker="TEST",
        finbert_net_score=0.0,
        pct_positive=0.3,
        pct_negative=0.3,
        news_volume_7d=5,
        upcoming_catalyst=False,
        signal=signal,
        conviction=conviction,
        sentiment_decay_risk="LOW",
        reasoning="test",
        timestamp=datetime.now(),
    )


def _macro(regime: Regime, fund_mult: float, tech_mult: float, sent_mult: float) -> MacroContext:
    return MacroContext(
        fed_funds=4.0,
        cpi_yoy=3.0,
        unemployment=4.0,
        t10y2y=0.1,
        consumer_sentiment=60.0,
        vix_level=18.0,
        macro_regime=regime,
        regime_confidence=0.7,
        model_healthy=True,
        interest_rate_trend="STABLE",
        yield_curve_shape=YieldCurve.NORMAL,
        vix_regime=VixRegime.MEDIUM,
        vix_percentile=40.0,
        inflation_trajectory="STABLE",
        sector_rotation_signal=SectorSignal.GROWTH_FAVORED,
        agent_multipliers={
            "fundamental": fund_mult,
            "technical": tech_mult,
            "sentiment": sent_mult,
        },
        timestamp=datetime.now(),
    )


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
    tech_signal, tech_conv, fund_signal, fund_conv, sent_signal, sent_conv,
    regime, fund_mult, tech_mult, sent_mult,
):
    """Aggregated conviction stays within [0, max_conviction] for any input combination."""
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=_technical(tech_signal, tech_conv),
        macro=_macro(regime, fund_mult, tech_mult, sent_mult),
        fundamental=_fundamental(fund_signal, fund_conv),
        sentiment=_sentiment(sent_signal, sent_conv),
    )
    assert result.conviction <= AGGREGATOR.max_conviction + 1e-9
    assert 0.0 <= result.conviction


@given(
    conv1=_convictions,
    conv2=_convictions,
    fund_mult=_multipliers,
    tech_mult=_multipliers,
    sent_mult=_multipliers,
)
def test_lone_agent_conviction_is_monotone_in_its_own_conviction(
    conv1, conv2, fund_mult, tech_mult, sent_mult
):
    """A lone voting agent's aggregated conviction strictly increases with its own conviction.

    Fixed to EXPANSION so the CONTRACTION override — which can zero out a
    lone BULLISH agent's conviction independently of its magnitude — doesn't
    interfere with the property under test.
    """
    assume(conv2 - conv1 > 1e-6)
    aggregator = HybridSignalAggregator()
    macro = _macro(Regime.EXPANSION, fund_mult, tech_mult, sent_mult)
    lo = aggregator.aggregate(
        technical=_technical(Signal.BULLISH, conv1), macro=macro, fundamental=None, sentiment=None
    )
    hi = aggregator.aggregate(
        technical=_technical(Signal.BULLISH, conv2), macro=macro, fundamental=None, sentiment=None
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
    """A single voting agent's conviction stays strictly below the aggregation cap.

    The denominator is the vote mass all three agents could cast, so a lone
    agent's share of it is always < 1 regardless of its own conviction.
    """
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=_technical(Signal.BULLISH, conv),
        macro=_macro(regime, fund_mult, tech_mult, sent_mult),
        fundamental=None,
        sentiment=None,
    )
    assert result.conviction < AGGREGATOR.max_conviction


def test_exact_tie_between_bull_and_bear_resolves_neutral():
    """An exact bull/bear vote tie resolves NEUTRAL rather than defaulting BULLISH."""
    aggregator = HybridSignalAggregator()
    result = aggregator.aggregate(
        technical=_technical(Signal.BULLISH, 0.8),
        macro=_macro(Regime.EXPANSION, 1.0, 1.0, 1.0),
        fundamental=_fundamental(Signal.BEARISH, 0.8),
        sentiment=None,
    )
    assert result.signal == Signal.NEUTRAL
    assert result.conviction == 0.0
    assert result.agents_present == ["fundamental", "technical"]
