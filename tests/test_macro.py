"""
Tests for the Macro-Economic Agent (argus/agents/macro.py).
"""

from datetime import datetime

import pandas as pd
import pytest

from argus.agents.macro import MacroStatisticalAgent, RegimeClassifier
from argus.schemas.signals import Regime


class _StubMarketData:
    """Minimal MarketDataProvider stub for macro.analyze()'s data needs."""

    def macro_bundle(self) -> dict:
        """Returns:
            Fixed macro indicator values.
        """
        return {"vix": 15.0, "fed_funds": 2.0, "t10y2y": 1.5, "cpi_yoy": 2.0, "unemployment": 3.5}

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns a fixed close-price series regardless of the ticker or period.

        Args:
            ticker: Ticker symbol (ignored).
            period: Lookback period (ignored).

        Returns:
            Fixed OHLCV close-price frame.
        """
        return pd.DataFrame({"close": [20.0, 20.0, 20.0, 20.0, 10.0]})

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Returns a fixed series regardless of the FRED series ID or start date.

        Args:
            series_id: FRED series identifier (ignored).
            start: Start date (ignored).

        Returns:
            Fixed series values.
        """
        return pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


def test_rule_based_fallback() -> None:
    """An unfitted classifier falls back to the VIX/yield-curve heuristic."""
    classifier = RegimeClassifier()
    assert not classifier.is_fitted

    # High VIX plus an inverted curve is the contraction heuristic's trigger condition
    regime, conf = classifier.predict({"vix": 35.0, "t10y2y": -0.6})
    assert regime == Regime.CONTRACTION.value
    assert conf == 0.6


def test_agent_multipliers_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """EXPANSION regime with low VIX percentile sets fundamental and sentiment multipliers."""
    agent = MacroStatisticalAgent(market_data=_StubMarketData())

    monkeypatch.setattr(
        agent.classifier, "predict", lambda current_values: (Regime.EXPANSION.value, 0.9)
    )

    ctx = agent.analyze()

    assert ctx.macro_regime == Regime.EXPANSION
    # VIX percentile < 40 and EXPANSION together map to a 1.3 fundamental multiplier
    assert ctx.agent_multipliers["fundamental"] == 1.3
    # VIX percentile < 50 maps to a 0.9 sentiment multiplier
    assert ctx.agent_multipliers["sentiment"] == 0.9


def test_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A populated cache is returned as-is, without touching any fetchers."""
    agent = MacroStatisticalAgent()

    dummy_ctx = type("MockCtx", (), {"macro_regime": Regime.EXPANSION})()
    agent._cache = (dummy_ctx, datetime.now())  # type: ignore

    # Fetchers are unmocked, so a cache miss here would raise rather than return stale data
    res1 = agent.analyze()
    res2 = agent.analyze()

    assert res1 is res2
    assert res1.macro_regime == Regime.EXPANSION
