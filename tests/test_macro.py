"""
Tests for the Macro-Economic Agent (argus/agents/macro.py).
"""

from datetime import datetime

import pytest

from argus.agents.macro import MacroStatisticalAgent, RegimeClassifier
from argus.schemas.signals import Regime


def test_rule_based_fallback() -> None:
    """Test that the classifier uses the heuristic fallback when not fitted."""
    classifier = RegimeClassifier()
    assert not classifier.is_fitted

    # Provide contraction-like values: high VIX, inverted curve
    regime, conf = classifier.predict({"vix": 35.0, "t10y2y": -0.6})
    assert regime == Regime.CONTRACTION.value
    assert conf == 0.6


def test_agent_multipliers_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the fetch bundle and prediction to force an EXPANSION regime."""
    agent = MacroStatisticalAgent()

    # Mock data fetch to return benign values
    monkeypatch.setattr(
        "argus.agents.macro.fetch_macro_bundle",
        lambda: {"vix": 15.0, "fed_funds": 2.0, "t10y2y": 1.5, "cpi_yoy": 2.0, "unemployment": 3.5},
    )
    # Mock the classifier to always return EXPANSION
    monkeypatch.setattr(
        agent.classifier, "predict", lambda current_values: (Regime.EXPANSION.value, 0.9)
    )
    import pandas as pd

    # Mock VIX history to return a low percentile
    monkeypatch.setattr(
        "argus.agents.macro.fetch_ohlcv_daily",
        lambda ticker, period: pd.DataFrame({"close": [20.0, 20.0, 20.0, 20.0, 10.0]}),
    )
    # Mock FRED series for trend logic
    monkeypatch.setattr(
        "argus.agents.macro.fetch_fred_series",
        lambda series_id, start=None: pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    )

    ctx = agent.analyze()

    assert ctx.macro_regime == Regime.EXPANSION
    # VIX percentile < 40 and EXPANSION should give 1.3 fundamental multiplier
    assert ctx.agent_multipliers["fundamental"] == 1.3
    # Low VIX percentile < 50 should give 0.9 sentiment
    assert ctx.agent_multipliers["sentiment"] == 0.9


def test_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call analyze() twice and verify the second call returns the cached object."""
    agent = MacroStatisticalAgent()

    # Use a dummy context
    dummy_ctx = type("MockCtx", (), {"macro_regime": Regime.EXPANSION})()
    agent._cache = (dummy_ctx, datetime.now())  # type: ignore

    # The analyze method should return the cache immediately without fetching
    # We can verify this by ensuring it doesn't crash even if fetchers aren't mocked
    res1 = agent.analyze()
    res2 = agent.analyze()

    assert res1 is res2
    assert res1.macro_regime == Regime.EXPANSION
