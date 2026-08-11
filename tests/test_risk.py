"""
Tests for the Risk Statistical Engine (argus/agents/risk.py).
"""

import numpy as np
import pandas as pd
import pytest

from argus.agents.risk import RiskStatisticalEngine
from argus.schemas.signals import RiskVerdict


@pytest.fixture
def price_history():
    """Create a simulated price history dictionary for tests.

    Returns:
        Mapping of ticker (plus "SPY" benchmark) to a one-year daily price series.
    """
    np.random.seed(42)  # Fixed seed keeps returns reproducible across runs
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    hist = {}

    # Moderate, normal volatility: 0.1% daily mean return, 1.5% daily vol
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]
    for t in tickers:
        returns = np.random.normal(0.001, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        hist[t] = pd.Series(prices, index=dates)

    spy_returns = np.random.normal(0.0005, 0.01, len(dates))
    hist["SPY"] = pd.Series(100 * np.exp(np.cumsum(spy_returns)), index=dates)

    return hist


@pytest.fixture
def risk_engine():
    """Returns:
        A fresh RiskStatisticalEngine instance.
    """
    return RiskStatisticalEngine()


def test_vix_blackout(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """VIX above the 35 blackout threshold vetoes the whole portfolio."""
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("blackout" in r.lower() for r in result.veto_reasons)


def test_overweight_position(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """A position above the 0.15 weight limit vetoes the portfolio."""
    positions = [
        {"ticker": "AAPL", "weight": 0.20},  # exceeds the 0.15 limit
        {"ticker": "MSFT", "weight": 0.10},
        {"ticker": "GOOGL", "weight": 0.10},
        {"ticker": "META", "weight": 0.10},
        {"ticker": "AMZN", "weight": 0.10},
    ]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("weight" in r.lower() for r in result.veto_reasons)


def test_high_var_reduce(risk_engine: RiskStatisticalEngine) -> None:
    """Extreme volatility pushes VaR high enough to trigger REDUCE rather than VETO."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=253, freq="B")
    hist = {}

    # 15% daily volatility is well above normal to force VaR past the REDUCE threshold
    for t in ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]:
        returns = np.random.normal(0.0, 0.15, len(dates))
        hist[t] = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    hist["SPY"] = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates)))), index=dates
    )

    positions = [{"ticker": t, "weight": 0.15} for t in ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]]

    result = risk_engine.evaluate(positions, hist, current_vix=20.0)

    assert result.verdict == RiskVerdict.REDUCE
    assert result.var_99 > 0.03


def test_approve_healthy_portfolio(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """A diversified, normally-weighted portfolio with normal vol is approved outright."""
    positions = [
        {"ticker": "AAPL", "weight": 0.1},
        {"ticker": "MSFT", "weight": 0.1},
        {"ticker": "GOOGL", "weight": 0.1},
        {"ticker": "META", "weight": 0.1},
        {"ticker": "AMZN", "weight": 0.1},
        {"ticker": "TSLA", "weight": 0.1},
        {"ticker": "JPM", "weight": 0.1},
        {"ticker": "BAC", "weight": 0.1},
    ]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.APPROVE
    assert result.var_99 <= 0.03
    assert not result.veto_reasons


def test_zero_api_calls(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """The risk engine evaluates purely statistically, never calling an LLM."""
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.api_calls_used == 0
