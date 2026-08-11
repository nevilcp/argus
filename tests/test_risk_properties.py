"""
tests/test_risk_properties.py

Property-based test on RiskStatisticalEngine.evaluate() — regardless of the
proposed position weight or VIX level, the engine must never approve more
than it was asked to approve. RiskAssessment already enforces this as a
Pydantic model validator (argus/schemas/signals.py: "approved_weight cannot
exceed proposed_weight"), so this test's job is to prove the engine's own
decision logic can't construct a RiskAssessment that violates it — i.e. the
validator never actually fires across the input space evaluate() is called
with.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from argus.agents.risk import RiskStatisticalEngine


def _price_history() -> dict[str, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    hist = {}
    for t in ["AAPL", "MSFT", "GOOGL"]:
        returns = np.random.normal(0.001, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        hist[t] = pd.Series(prices, index=dates)
    return hist


@settings(deadline=None)
@given(
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    vix=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
def test_approved_weight_never_exceeds_proposed(weight: float, vix: float) -> None:
    engine = RiskStatisticalEngine()
    positions = [{"ticker": "AAPL", "weight": weight}]

    result = engine.evaluate(positions, _price_history(), current_vix=vix)

    # RiskAssessment's own validator already raises on construction if this is
    # violated; reaching this assertion is itself part of the proof.
    assert result.approved_weight <= result.proposed_weight + 1e-9
