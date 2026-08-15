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
from argus.orchestration.graph import _apply_portfolio_cap
from argus.params import RISK
from argus.schemas.signals import RiskVerdict

_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]


def _price_history() -> dict[str, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    hist = {}
    for t in ["AAPL", "MSFT", "GOOGL"]:
        returns = np.random.normal(0.001, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        hist[t] = pd.Series(prices, index=dates)
    return hist


def _multi_asset_price_history() -> dict[str, pd.Series]:
    np.random.seed(7)
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    hist = {}
    for t in _UNIVERSE:
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
    """Across the full weight/VIX input space, approved weight never exceeds proposed weight."""
    engine = RiskStatisticalEngine()
    positions = [{"ticker": "AAPL", "weight": weight}]

    result = engine.evaluate(positions, _price_history(), current_vix=vix)

    # RiskAssessment's validator would raise on construction if this were violated
    assert result.approved_weight <= result.proposed_weight + 1e-9


@settings(deadline=None, max_examples=50)
@given(
    tickers=st.lists(
        st.sampled_from(_UNIVERSE),
        min_size=RISK.min_positions_diversification,
        max_size=len(_UNIVERSE),
        unique=True,
    ),
    data=st.data(),
)
def test_optimal_weights_never_exceed_total_deployment_budget(tickers: list[str], data) -> None:
    """RE-1 regression: the SLSQP solve never deploys more than the book's capital.

    Before the fix, the optimizer had a sum(w) >= 0.50 floor and no upper bound,
    so a sufficiently bullish conviction vector could solve past 100% deployment.
    """
    engine = RiskStatisticalEngine()
    # Mirrors graph.py's node_risk_evaluation: even split, capped at the single-position limit
    base_weight = min(0.15, 1.0 / len(tickers))
    positions = [{"ticker": t, "weight": base_weight} for t in tickers]
    convictions = {
        t: data.draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)) for t in tickers
    }

    result = engine.evaluate(
        positions, _multi_asset_price_history(), current_vix=20.0, convictions=convictions
    )

    assert sum(result.optimal_weights.values()) <= RISK.slsqp_max_total_deployment + 1e-6


def test_all_negative_convictions_veto_not_reduce_at_zero() -> None:
    """RE-1 guard: a near-zero SLSQP cap becomes VETO, not REDUCE-to-zero.

    REDUCE still counts as an approved ticker downstream (portfolio.py:236), so a
    0%-cap REDUCE would leave the portfolio agent demanding invest_pct deployment
    against a cap with no room for it — this is the bug _apply_portfolio_cap closes.
    """
    engine = RiskStatisticalEngine()
    history = _multi_asset_price_history()
    tickers = _UNIVERSE[:5]
    positions = [{"ticker": t, "weight": 0.15} for t in tickers]
    convictions = {t: -0.9 for t in tickers}

    portfolio_result = engine.evaluate(positions, history, current_vix=20.0, convictions=convictions)

    assert portfolio_result.optimizer_converged is True
    assert sum(portfolio_result.optimal_weights.values()) < RISK.slsqp_zero_cap_epsilon

    for ticker in tickers:
        single = engine.evaluate([{"ticker": ticker, "weight": 0.15}], history, current_vix=20.0)
        updates = _apply_portfolio_cap(single, portfolio_result.optimal_weights[ticker])
        assert updates["verdict"] == RiskVerdict.VETO
        assert updates["approved_weight"] == 0.0
