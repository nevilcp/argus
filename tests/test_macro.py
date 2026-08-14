"""
Tests for the Macro-Economic Agent (argus/agents/macro.py).
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from argus.agents.macro import (
    DISCRIMINATION_SCENARIOS,
    FEATURE_COLUMNS,
    MacroStatisticalAgent,
    RegimeClassifier,
)
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
_EXPANSION_POINT = {
    "d_fed_funds_6m": -0.5,
    "d_unemp_12m": -0.5,
    "cpi_yoy": 2.0,
    "t10y2y": 1.5,
    "vix_pctile": 15.0,
}
_CONTRACTION_POINT = {
    "d_fed_funds_6m": -1.5,
    "d_unemp_12m": 2.0,
    "cpi_yoy": 1.0,
    "t10y2y": -1.0,
    "vix_pctile": 90.0,
}
_TRANSITIONAL_POINT = {
    "d_fed_funds_6m": 0.25,
    "d_unemp_12m": 0.25,
    "cpi_yoy": 2.5,
    "t10y2y": 0.2,
    "vix_pctile": 50.0,
}


_N_PER_BLOCK = 60


def _synthetic_history(seed: int = 3) -> pd.DataFrame:
    """Builds synthetic, well-separated 3-regime feature history, unfitted.

    Blocks (not shuffled) so a fitted HMM also learns a sensible transition
    structure, rather than i.i.d. noise around three means.

    Args:
        seed: RNG seed for the per-point Gaussian noise.

    Returns:
        DataFrame with FEATURE_COLUMNS, four blocks of _N_PER_BLOCK rows each
        (expansion, contraction, transitional, expansion), daily-indexed from
        2015-01-01.
    """
    rng = np.random.default_rng(seed)

    def block(point: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: point[col] + rng.normal(0, 0.05 if col != "vix_pctile" else 2.0, _N_PER_BLOCK)
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
    return history


def _fit_synthetic_classifier(seed: int = 3) -> RegimeClassifier:
    """Fits a RegimeClassifier on synthetic, well-separated 3-regime data.

    fit() itself tries multiple random restarts (see RegimeClassifier.fit), so
    no particular constructor seed needs to be hand-picked for this data to
    separate cleanly.

    Args:
        seed: RNG seed for the per-point Gaussian noise.

    Returns:
        A fitted RegimeClassifier whose state_to_regime covers all three regimes.
    """
    history = _synthetic_history(seed)

    # Recession flag over the contraction block only, so _map_states' NBER-enrichment
    # labelling (see RegimeClassifier._label_contraction_state) has something to key
    # CONTRACTION on, the same way it would from a real USREC series.
    recession_series = pd.Series(0.0, index=history.index)
    recession_series.iloc[_N_PER_BLOCK : 2 * _N_PER_BLOCK] = 1.0

    classifier = RegimeClassifier(n_components=3)
    classifier.fit(history, recession_series=recession_series)

    # save()/load() round-trip so the returned classifier carries load()'s stationary-
    # startprob_ fix (see RegimeClassifier.load) instead of a test-only hand patch —
    # hand-patching here would mask a regression in that fix from the rest of the suite.
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / "synthetic.joblib"
        classifier.save(artifact_path)
        classifier = RegimeClassifier.load(artifact_path)

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


def test_window_length_invariance() -> None:
    """The same constant window classifies identically at every length.

    Direct regression test for the parity bug that made the shipped artifact's
    regime a function of window-length parity rather than the macro data:
    tail(89) and tail(90) on the same real window returned different regimes.
    """
    classifier = _fit_synthetic_classifier()

    regimes = {classifier.predict(pd.DataFrame([_EXPANSION_POINT] * n))[0] for n in range(5, 15)}

    assert regimes == {Regime.EXPANSION.value}


def test_discrimination_crash_and_boom_windows_produce_different_labels() -> None:
    """A crash-like window and a boom-like window must not collapse to the same label.

    Regression test for the labelling bug where _map_states always assigned
    CONTRACTION to *some* state regardless of whether a genuinely contractionary
    state existed, which is what let a boom period read as CONTRACTION.
    """
    classifier = _fit_synthetic_classifier()

    boom_regime, _ = classifier.predict(pd.DataFrame([_EXPANSION_POINT] * 10))
    crash_regime, _ = classifier.predict(pd.DataFrame([_CONTRACTION_POINT] * 10))

    assert boom_regime == Regime.EXPANSION.value
    assert crash_regime == Regime.CONTRACTION.value


def test_contraction_reachable_from_a_single_observation() -> None:
    """CONTRACTION must be decodable from one observation with no window context.

    Regression test for the pre-fix bug: fitting one long training sequence
    leaves startprob_ a near one-hot pointing at whichever state the sequence
    happened to start in, making some states — including CONTRACTION, since
    training always starts on the expansion block — provably unreachable from a
    cold start. RegimeClassifier.load replaces startprob_ with the transition
    matrix's stationary distribution to fix this; _fit_synthetic_classifier
    routes through save()/load() rather than hand-patching startprob_, so this
    test exercises the real fix.
    """
    classifier = _fit_synthetic_classifier()

    regime, _ = classifier.predict(dict(_CONTRACTION_POINT))
    assert regime == Regime.CONTRACTION.value


def test_fit_raises_when_all_restarts_are_degenerate(monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() raises rather than silently accepting a degenerate model.

    Regression test for the shipped-artifact bug: two of its three states were
    the same distribution (|Δμ|max = 0.00097). Reproducing that exact EM
    collapse needs the same non-stationary daily forward-filled data that
    caused it; here GaussianHMM.fit is patched to always converge every state
    to identical means regardless of seed, so the test exercises fit()'s own
    degeneracy-rejection logic deterministically instead of hoping a
    synthetic dataset happens to degenerate under EM.
    """

    def _collapse_to_one_state(self, X, lengths=None):
        n_components = self.n_components
        n_features = X.shape[1]
        self.startprob_ = np.full(n_components, 1.0 / n_components)
        self.transmat_ = np.full((n_components, n_components), 1.0 / n_components)
        self.means_ = np.tile(X.mean(axis=0), (n_components, 1))
        cov = np.cov(X, rowvar=False) + np.eye(n_features) * 1e-3
        self.covars_ = np.tile(cov, (n_components, 1, 1))
        return self

    monkeypatch.setattr(GaussianHMM, "fit", _collapse_to_one_state)

    classifier = RegimeClassifier(n_components=3)
    with pytest.raises(ValueError, match="degenerate"):
        classifier.fit(_synthetic_history())


def test_validation_failures_flags_unfitted_classifier() -> None:
    """An unfitted classifier fails validation with a specific, non-empty reason."""
    classifier = RegimeClassifier()
    assert classifier.validation_failures() == ["classifier is not fitted"]


def test_validation_failures_flags_missing_contraction_enrichment() -> None:
    """Without a recession_series, fit() can't clear the CONTRACTION-enrichment gate."""
    classifier = RegimeClassifier(n_components=3)
    classifier.fit(_synthetic_history())

    failures = classifier.validation_failures()
    assert any("CONTRACTION enrichment" in f for f in failures)


def test_discrimination_scenarios_cover_all_feature_columns() -> None:
    """Every canned scenario supplies all FEATURE_COLUMNS.

    A scenario missing a key would have predict() silently substitute
    _FEATURE_DEFAULTS for it, weakening validation_failures()'s discrimination
    check without failing loudly.
    """
    for name, scenario in DISCRIMINATION_SCENARIOS.items():
        assert set(scenario.keys()) == set(FEATURE_COLUMNS), name
    assert len(DISCRIMINATION_SCENARIOS) >= 2


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
