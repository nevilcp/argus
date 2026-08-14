"""
Tests for the Macro-Economic Agent (argus/agents/macro.py).
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from argus.agents.macro import FEATURE_COLUMNS, MacroStatisticalAgent, RegimeClassifier
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


# Synthetic per-regime cluster means, well separated so a small Gaussian HMM
# fits three cleanly distinguishable states without needing real FRED/VIX data.
_EXPANSION_POINT = {"fed_funds": 1.0, "cpi_yoy": 2.0, "t10y2y": 1.5, "unemployment": 3.5, "vix": 13.0}
_CONTRACTION_POINT = {"fed_funds": 4.0, "cpi_yoy": 1.0, "t10y2y": -1.0, "unemployment": 7.0, "vix": 38.0}
_TRANSITIONAL_POINT = {"fed_funds": 2.5, "cpi_yoy": 2.5, "t10y2y": 0.2, "unemployment": 5.0, "vix": 20.0}


def _fit_synthetic_classifier(seed: int = 7) -> RegimeClassifier:
    """Fits a RegimeClassifier on synthetic, well-separated 3-regime data.

    Blocks (not shuffled) so the HMM also learns a sensible transition structure,
    rather than i.i.d. noise around three means.

    Args:
        seed: RNG seed for the per-point Gaussian noise.

    Returns:
        A fitted RegimeClassifier whose state_to_regime covers all three regimes.
    """
    rng = np.random.default_rng(seed)
    n_per_block = 60

    def block(point: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: point[col] + rng.normal(0, 0.05 if col != "vix" else 0.8, n_per_block)
                for col in FEATURE_COLUMNS
            }
        )

    history = pd.concat(
        [
            block(_EXPANSION_POINT),
            block(_CONTRACTION_POINT),
            block(_TRANSITIONAL_POINT),
            block(_EXPANSION_POINT),
        ],
        ignore_index=True,
    )
    history.index = pd.date_range("2015-01-01", periods=len(history), freq="D")

    # random_state=2 (vs. RegimeClassifier's production default of 42) is picked because
    # it reliably separates this synthetic data into 3 clean clusters; EM's local-optimum
    # sensitivity means not every seed does on data this small.
    classifier = RegimeClassifier(n_components=3, random_state=2)
    classifier.fit(history)

    # Fitting on one single long sequence (rather than several independent ones) gives
    # hmmlearn nothing to average over for the initial-state distribution, so startprob_
    # collapses to a near one-hot pointing at whichever state the training sequence's very
    # first sample happened to land in — an artifact of where the sequence starts, not a
    # real prior. Neutralizing it isolates the emission/transition dynamics windowed
    # inference is actually about.
    classifier.hmm.startprob_ = np.full(3, 1.0 / 3)

    assert classifier.is_fitted
    assert set(classifier.state_to_regime.values()) == {
        Regime.EXPANSION.value,
        Regime.CONTRACTION.value,
        Regime.TRANSITIONAL.value,
    }
    return classifier


def test_save_load_round_trip(tmp_path: Path) -> None:
    """A saved artifact, once loaded, preserves state_to_regime and reproduces predictions."""
    classifier = _fit_synthetic_classifier()
    artifact_path = tmp_path / "macro_hmm.joblib"
    classifier.save(artifact_path, start_date="2015-01-01")

    loaded = RegimeClassifier.load(artifact_path)

    assert loaded.is_fitted
    assert loaded.state_to_regime == classifier.state_to_regime
    assert loaded.n_train_observations == classifier.n_train_observations

    window = pd.DataFrame([_EXPANSION_POINT] * 5)
    orig_regime, orig_confidence = classifier.predict(window)
    loaded_regime, loaded_confidence = loaded.predict(window)
    assert loaded_regime == orig_regime
    assert loaded_confidence == pytest.approx(orig_confidence)


def test_load_missing_file_returns_unfitted_and_logs_error(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A missing artifact path logs an ERROR and returns an unfitted classifier."""
    missing_path = tmp_path / "does_not_exist.joblib"

    with caplog.at_level(logging.ERROR, logger="argus.macro"):
        classifier = RegimeClassifier.load(missing_path)

    assert not classifier.is_fitted
    assert any("failed to load artifact" in r.message for r in caplog.records)

    # predict() takes the documented rule-based path
    regime, confidence = classifier.predict({"vix": 35.0, "t10y2y": -0.6})
    assert regime == Regime.CONTRACTION.value
    assert confidence == 0.6


def test_load_corrupt_file_returns_unfitted_and_logs_error(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A corrupt pickle logs an ERROR and returns an unfitted classifier rather than raising."""
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"this is not a valid joblib pickle")

    with caplog.at_level(logging.ERROR, logger="argus.macro"):
        classifier = RegimeClassifier.load(corrupt_path)

    assert not classifier.is_fitted
    assert any("failed to load artifact" in r.message for r in caplog.records)


def test_load_version_mismatch_returns_unfitted_and_logs_error(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A hmmlearn/scikit-learn version mismatch in the artifact's metadata is treated as a failure."""
    classifier = _fit_synthetic_classifier()
    artifact_path = tmp_path / "macro_hmm.joblib"
    classifier.save(artifact_path)

    payload = joblib.load(artifact_path)
    payload["metadata"]["hmmlearn_version"] = "0.0.0-incompatible"
    joblib.dump(payload, artifact_path)

    with caplog.at_level(logging.ERROR, logger="argus.macro"):
        loaded = RegimeClassifier.load(artifact_path)

    assert not loaded.is_fitted
    assert any("version mismatch" in r.message for r in caplog.records)


def test_windowed_predict_classifies_a_stable_window_consistently() -> None:
    """A window of near-identical expansion-like observations classifies as EXPANSION."""
    classifier = _fit_synthetic_classifier()

    window = pd.DataFrame([_EXPANSION_POINT] * 10)
    regime, confidence = classifier.predict(window)

    assert regime == Regime.EXPANSION.value
    assert confidence > 0.5


def test_windowed_predict_resists_a_single_point_spike() -> None:
    """A noisy single observation flips the regime alone, but not inside a stable window.

    This is the property windowed inference exists to provide: informed by nine
    surrounding stable observations, the transition matrix resists flipping the
    session's regime call on one noisy print the way single-row decoding does.
    """
    classifier = _fit_synthetic_classifier()

    # Halfway between the expansion and contraction cluster means — an ambiguous
    # single print, not a full jump to another cluster's center.
    noisy_point = {
        col: _EXPANSION_POINT[col] + 0.5 * (_CONTRACTION_POINT[col] - _EXPANSION_POINT[col])
        for col in FEATURE_COLUMNS
    }

    # Decoded alone (no surrounding context), the ambiguous point does not read as EXPANSION
    spike_alone_regime, _ = classifier.predict(noisy_point)
    assert spike_alone_regime != Regime.EXPANSION.value

    # The same point as the final row of an otherwise-stable expansion window
    window = pd.DataFrame([_EXPANSION_POINT] * 9 + [noisy_point])
    windowed_regime, _ = classifier.predict(window)
    assert windowed_regime == Regime.EXPANSION.value


class _WindowAssemblyFailureMarketData:
    """A MarketDataProvider whose FRED calls always fail, forcing window assembly to fail."""

    def macro_bundle(self) -> dict:
        """Returns fixed macro indicator values used for the single-observation fallback."""
        return dict(_EXPANSION_POINT)

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns a fixed close-price series so the VIX-percentile calc still succeeds."""
        return pd.DataFrame({"close": [20.0, 20.0, 20.0, 20.0, 10.0]})

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Always raises, simulating a FRED outage during window assembly."""
        raise RuntimeError("FRED unavailable")


def test_analyze_falls_back_to_single_observation_when_window_assembly_fails(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A window-assembly failure degrades to single-observation inference, not a crash."""
    agent = MacroStatisticalAgent(
        market_data=_WindowAssemblyFailureMarketData(),
        model_path=tmp_path / "no_artifact_here.joblib",
    )
    agent.classifier = _fit_synthetic_classifier()

    with caplog.at_level(logging.WARNING, logger="argus.macro"):
        ctx = agent.analyze()

    assert ctx is not None
    assert any("feature window assembly failed" in r.message for r in caplog.records)

    expected_regime, expected_confidence = agent.classifier.predict(
        agent.market_data.macro_bundle()
    )
    assert ctx.macro_regime.value == expected_regime
    assert ctx.regime_confidence == pytest.approx(expected_confidence)
