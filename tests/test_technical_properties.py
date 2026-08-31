"""
tests/test_technical_properties.py

Property-based tests (Hypothesis) on the pure scoring functions in
argus/agents/technical.py — these have no I/O and a well-defined input
domain, so they're a better fit for property tests than example-based ones:
the invariants below (monotonicity, boundedness, antisymmetry) are what the
functions' docstrings already claim, and a property test checks the claim
across the whole domain rather than at a handful of hand-picked points.
"""

from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from argus.agents.technical import _score_bollinger, _score_rsi


@given(rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_score_rsi_bounded(rsi: float) -> None:
    """RSI score stays within [-1.0, 1.0] across the entire RSI domain."""
    score = _score_rsi({"rsi_14": rsi})
    assert -1.0 <= score <= 1.0


@given(
    rsi=st.floats(min_value=0.0, max_value=99.9, allow_nan=False),
    delta=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
)
def test_score_rsi_monotonically_non_increasing(rsi: float, delta: float) -> None:
    """Higher RSI never scores more bullish than a lower RSI."""
    higher_rsi = min(100.0, rsi + delta)
    assert _score_rsi({"rsi_14": higher_rsi}) <= _score_rsi({"rsi_14": rsi}) + 1e-9


@given(bb=st.floats(min_value=-5.0, max_value=6.0, allow_nan=False))
def test_score_bollinger_bounded(bb: float) -> None:
    """Bollinger %B score stays within [-1.0, 1.0] across the entire %B domain."""
    score = _score_bollinger({"bb_percent_b": bb})
    assert -1.0 <= score <= 1.0


@given(bb=st.floats(min_value=-5.0, max_value=6.0, allow_nan=False))
def test_score_bollinger_antisymmetric_about_half(bb: float) -> None:
    """%B and its mirror image around 0.5 score as exact opposites.

    A breach below the lower band scores the same magnitude, opposite sign, as
    the mirrored breach above the upper band.
    """
    mirrored = 1.0 - bb
    score = _score_bollinger({"bb_percent_b": bb})
    mirrored_score = _score_bollinger({"bb_percent_b": mirrored})
    assert score == approx(-mirrored_score, abs=1e-9)
