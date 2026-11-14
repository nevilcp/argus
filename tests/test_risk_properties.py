"""Property-based tests on RiskStatisticalAgent.evaluate()."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from argus.agents.risk import RiskStatisticalAgent
from argus.orchestration.nodes import _apply_portfolio_cap
from argus.params import RISK
from argus.schemas.signals import RiskVerdict
from tests.helpers.market_data import FakeMarketData

_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]


def _price_history(tickers: list[str], seed: int) -> dict[str, pd.Series]:
    """Build a year of seeded geometric-random-walk closes, one series per ticker."""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    history = {}
    for ticker in tickers:
        returns = np.random.normal(0.001, 0.015, len(dates))
        history[ticker] = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    return history


@settings(deadline=None)
@given(
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    vix=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
def test_approved_weight_never_exceeds_proposed(weight: float, vix: float) -> None:
    engine = RiskStatisticalAgent(
        market_data=FakeMarketData(
            ticker_info=lambda ticker: {"sector": ticker},
            ohlcv_daily=OSError("benchmark unavailable"),
        )
    )
    positions = [{"ticker": "AAPL", "weight": weight}]

    result = engine.evaluate(positions, _price_history(["AAPL", "MSFT", "GOOGL"], seed=42), current_vix=vix)

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
    engine = RiskStatisticalAgent(
        market_data=FakeMarketData(
            ticker_info=lambda ticker: {"sector": ticker},
            ohlcv_daily=OSError("benchmark unavailable"),
        )
    )
    base_weight = min(0.15, 1.0 / len(tickers))
    positions = [{"ticker": t, "weight": base_weight} for t in tickers]
    convictions = {t: data.draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)) for t in tickers}

    result = engine.evaluate(positions, _price_history(_UNIVERSE, seed=7), current_vix=20.0, convictions=convictions)

    assert sum(result.optimal_weights.values()) <= RISK.slsqp_max_total_deployment + 1e-6


def test_all_negative_convictions_veto_not_reduce_at_zero() -> None:
    engine = RiskStatisticalAgent(
        market_data=FakeMarketData(
            ticker_info=lambda ticker: {"sector": ticker},
            ohlcv_daily=OSError("benchmark unavailable"),
        )
    )
    history = _price_history(_UNIVERSE, seed=7)
    tickers = _UNIVERSE[:5]
    positions = [{"ticker": t, "weight": 0.15} for t in tickers]
    convictions = dict.fromkeys(tickers, -0.9)

    portfolio_result = engine.evaluate(positions, history, current_vix=20.0, convictions=convictions)

    assert portfolio_result.optimizer_converged is True
    assert sum(portfolio_result.optimal_weights.values()) < RISK.slsqp_zero_cap_epsilon

    for ticker in tickers:
        single = engine.evaluate([{"ticker": ticker, "weight": 0.15}], history, current_vix=20.0)
        updates = _apply_portfolio_cap(single, portfolio_result.optimal_weights[ticker])
        assert updates["verdict"] == RiskVerdict.VETO
        assert updates["approved_weight"] == 0.0
