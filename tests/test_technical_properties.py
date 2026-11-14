"""Property-based tests (Hypothesis) on the pure scoring functions in technical.py."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from argus.agents.technical import _score_bollinger, _score_rsi


@given(rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_score_rsi_bounded(rsi: float) -> None:
    score = _score_rsi({"rsi_14": rsi})
    assert -1.0 <= score <= 1.0


@given(
    rsi=st.floats(min_value=0.0, max_value=99.9, allow_nan=False),
    delta=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
)
def test_score_rsi_monotonically_non_increasing(rsi: float, delta: float) -> None:
    higher_rsi = min(100.0, rsi + delta)
    assert _score_rsi({"rsi_14": higher_rsi}) <= _score_rsi({"rsi_14": rsi}) + 1e-9


@given(bb=st.floats(min_value=-5.0, max_value=6.0, allow_nan=False))
def test_score_bollinger_bounded(bb: float) -> None:
    score = _score_bollinger({"bb_percent_b": bb})
    assert -1.0 <= score <= 1.0


@given(bb=st.floats(min_value=-5.0, max_value=6.0, allow_nan=False))
def test_score_bollinger_antisymmetric_about_half(bb: float) -> None:
    mirrored = 1.0 - bb
    score = _score_bollinger({"bb_percent_b": bb})
    mirrored_score = _score_bollinger({"bb_percent_b": mirrored})
    assert score == pytest.approx(-mirrored_score, abs=1e-9)
