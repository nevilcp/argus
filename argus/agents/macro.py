"""
argus/agents/macro.py

Statistical macroeconomic regime analysis agent.

Responsibilities:
  - Combine Federal Reserve (FRED) time series with the CBOE VIX using a Gaussian HMM
  - Classify hidden economic states (EXPANSION, CONTRACTION, TRANSITIONAL)
  - Expose regime multipliers used by specialist agents to dynamically weight convictions

Not responsible for:
  - Feature construction (see data/macro_features.py)
  - Raw data fetching beyond macro bundles (see data/fetchers.py)
  - Portfolio allocation (see agents/portfolio.py)
  - Risk constraint enforcement (see agents/risk.py)

Dependencies:
  - hmmlearn
  - sklearn
  - yfinance
  - FRED_API_KEY env var must be set (see .env.example)
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import hmmlearn
import joblib
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn
from hmmlearn.hmm import GaussianHMM
from scipy.spatial.distance import mahalanobis
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from argus.config import settings
from argus.data.macro_features import build_macro_feature_frame
from argus.params import MACRO
from argus.schemas.signals import MacroContext, Regime, SectorSignal, VixRegime, YieldCurve
from argus.seams import LiveMarketDataProvider, MarketDataProvider

logger = logging.getLogger("argus.macro")

# Column order the fit/predict paths always pass to the scaler and HMM; persisted
# in the artifact's metadata so a load-time mismatch is detectable rather than a
# silent feature misalignment. All five are stationary (diffs, a percentile rank,
# or a mean-reverting spread) — see data/macro_features.py for their derivation.
FEATURE_COLUMNS = ["d_fed_funds_6m", "d_unemp_12m", "cpi_yoy", "t10y2y", "vix_pctile"]

_FEATURE_DEFAULTS = {
    "d_fed_funds_6m": 0.0,
    "d_unemp_12m": 0.0,
    "cpi_yoy": 0.0,
    "t10y2y": 0.0,
    "vix_pctile": 50.0,
}

# Canned scenarios for scripts/train_macro_hmm.py's discrimination gate (and
# tested directly against it): a trained model that maps every one of these to
# the same label cannot be a function of the macro data. Values are on
# FEATURE_COLUMNS' scale, not raw indicator levels.
DISCRIMINATION_SCENARIOS: dict[str, dict[str, float]] = {
    "goldilocks_boom": {
        "d_fed_funds_6m": -0.1,
        "d_unemp_12m": -0.2,
        "cpi_yoy": 2.0,
        "t10y2y": 1.5,
        "vix_pctile": 10.0,
    },
    "gfc_crash": {
        "d_fed_funds_6m": -2.5,
        "d_unemp_12m": 3.5,
        "cpi_yoy": 0.5,
        "t10y2y": -0.3,
        "vix_pctile": 95.0,
    },
    "covid_shock": {
        "d_fed_funds_6m": -1.5,
        "d_unemp_12m": 8.0,
        "cpi_yoy": 0.3,
        "t10y2y": 0.5,
        "vix_pctile": 99.0,
    },
    "2023_curve_inversion": {
        "d_fed_funds_6m": 1.0,
        "d_unemp_12m": -0.3,
        "cpi_yoy": 4.0,
        "t10y2y": -1.0,
        "vix_pctile": 40.0,
    },
    "stagflation": {
        "d_fed_funds_6m": 0.5,
        "d_unemp_12m": 1.0,
        "cpi_yoy": 9.0,
        "t10y2y": -0.5,
        "vix_pctile": 70.0,
    },
}


def _dwell_times_months(hidden_states: np.ndarray, n_components: int) -> dict[int, float]:
    """Computes each state's mean run length (in observations) from a Viterbi path.

    Args:
        hidden_states: Decoded state sequence, ordered oldest to newest.
        n_components: Number of HMM states.

    Returns:
        Dict mapping state index to mean dwell time; 0.0 for a state with no runs.
    """
    runs: dict[int, list[int]] = {s: [] for s in range(n_components)}
    run_state = int(hidden_states[0])
    run_len = 1
    for state in hidden_states[1:]:
        state = int(state)
        if state == run_state:
            run_len += 1
        else:
            runs[run_state].append(run_len)
            run_state = state
            run_len = 1
    runs[run_state].append(run_len)
    return {s: (float(np.mean(lengths)) if lengths else 0.0) for s, lengths in runs.items()}


def _min_pairwise_mahalanobis(hmm: GaussianHMM, occupancy: np.ndarray) -> float:
    """Computes the minimum pairwise Mahalanobis distance between state means.

    Uses the occupancy-weighted pooled covariance across states, so a state's
    own (possibly ill-conditioned) covariance doesn't dominate the metric.

    Args:
        hmm: A fitted GaussianHMM with full covariances.
        occupancy: Fraction of decoded observations assigned to each state.

    Returns:
        The smallest pairwise distance, or 0.0 for a single-state model.
    """
    n = hmm.n_components
    if n < 2:
        return 0.0
    pooled_cov = np.average(hmm.covars_, axis=0, weights=occupancy)
    try:
        inv_cov = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(pooled_cov)

    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dist = mahalanobis(hmm.means_[i], hmm.means_[j], inv_cov)
            min_dist = min(min_dist, dist)
    return float(min_dist)


def _major_minor(version: str) -> str:
    """Truncates a dotted version string to its major.minor prefix."""
    return ".".join(version.split(".")[:2])


class RegimeClassifier:
    """Gaussian Hidden Markov Model mapping macro features to hidden economic states.

    Trained on a long FRED history (1990+), then applied to current macro observations
    to infer the latent regime. CONTRACTION is labelled by NBER (USREC) recession
    enrichment; EXPANSION and TRANSITIONAL are labelled heuristically from per-state
    feature means. See fit() and _map_states().
    """

    def __init__(self, n_components: int = 3) -> None:
        """Builds the underlying Gaussian HMM, untrained.

        Args:
            n_components: Number of latent regime states.
        """
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=300,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.state_to_regime: dict[int, str] = {}
        self.n_train_observations = 0
        # Diagnostic-only: per-state feature means from the last fit, keyed by
        # state index. Not persisted by save() — populated fresh by _map_states
        # so scripts/train_macro_hmm.py can print the human check on _map_states'
        # labeling immediately after fitting.
        self.state_means: dict[int, dict[str, float]] = {}
        # Populated by fit(): separation/occupancy/dwell/enrichment/recall evidence
        # for this fit, persisted into the artifact's metadata by save() and
        # checked by validation_failures().
        self.validation_metrics: dict = {}

    def fit(
        self, macro_history: pd.DataFrame, recession_series: Optional[pd.Series] = None
    ) -> None:
        """Fits the scaling transformer and HMM classifier on historic feature timelines.

        Tries MACRO.hmm_n_init random restarts (seeds 0..n_init-1), rejecting any
        restart whose states are not well separated or that leaves a state
        under-occupied, and keeps the highest-likelihood survivor. A single fixed
        seed is not enough: on this data, 3 of 5 seeds collapse two states onto
        the same distribution, and the shipped artifact's seed (42) was one of them.

        Args:
            macro_history: DataFrame containing at least FEATURE_COLUMNS. NaN rows
                are dropped before fitting.
            recession_series: Optional NBER USREC series (1.0 during a recession
                month, else 0.0), indexed to overlap macro_history. Used by
                _map_states to label the CONTRACTION state by recession
                enrichment; if None, no state is labelled CONTRACTION.

        Raises:
            ValueError: If every restart is rejected as degenerate or under-occupied.
        """
        df = macro_history.dropna().copy()
        if df.empty:
            logger.warning("RegimeClassifier.fit: Empty DataFrame after dropping NaNs.")
            return

        n_components = self.hmm.n_components
        features = df[FEATURE_COLUMNS].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        self._log_bic_diagnostic(scaled_features)

        best_hmm: Optional[GaussianHMM] = None
        best_score = -np.inf
        best_separation = 0.0
        best_occupancy = np.array([])
        rejections: list[str] = []

        for seed in range(MACRO.hmm_n_init):
            candidate = GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=300,
                random_state=seed,
            )
            with warnings.catch_warnings():
                # Sub-1e-4 monotonicity wobble near the optimum is benign, not a failure
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                candidate.fit(scaled_features)

            hidden_states = candidate.predict(scaled_features)
            occupancy = np.bincount(hidden_states, minlength=n_components) / len(hidden_states)
            if occupancy.min() < MACRO.hmm_min_state_occupancy:
                rejections.append(f"seed={seed}: min occupancy {occupancy.min():.3f}")
                continue

            separation = _min_pairwise_mahalanobis(candidate, occupancy)
            if separation < MACRO.hmm_min_state_separation:
                rejections.append(f"seed={seed}: min separation {separation:.3f}")
                continue

            score = candidate.score(scaled_features)
            if score > best_score:
                best_score = score
                best_hmm = candidate
                best_separation = separation
                best_occupancy = occupancy

        if best_hmm is None:
            raise ValueError(
                f"RegimeClassifier.fit: all {MACRO.hmm_n_init} restarts were degenerate "
                f"or under-occupied ({'; '.join(rejections)})"
            )

        self.hmm = best_hmm
        # Apply now, not just on the load() round-trip: validation_failures()'s
        # discrimination check calls predict() on this in-memory classifier before
        # it is ever saved, and a raw EM startprob_ (a one-hot artifact of wherever
        # the training sequence began) would dominate every single-observation
        # scenario regardless of its features.
        self.hmm.startprob_ = self._stationary_startprob(self.hmm.transmat_)
        self.scaler = scaler
        hidden_states = best_hmm.predict(scaled_features)

        self.validation_metrics = {
            "min_state_separation": best_separation,
            "state_occupancy": {int(s): float(o) for s, o in enumerate(best_occupancy)},
            "dwell_times_months": _dwell_times_months(hidden_states, n_components),
            "log_likelihood": float(best_score),
            "n_restarts_rejected": len(rejections),
        }

        self._map_states(df, recession_series)
        self.is_fitted = True
        self.n_train_observations = len(df)

    @staticmethod
    def _stationary_startprob(transmat: np.ndarray) -> np.ndarray:
        """Computes a Markov chain's stationary distribution from its transition matrix.

        Args:
            transmat: The HMM's n x n transition matrix.

        Returns:
            A probability vector over states (the left eigenvector for eigenvalue 1).
        """
        eigenvalues, eigenvectors = scipy.linalg.eig(transmat.T)
        stationary = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))].real
        return stationary / stationary.sum()

    def _log_bic_diagnostic(self, scaled_features: np.ndarray) -> None:
        """Logs BIC for k in {2..5} states as a diagnostic; does not affect the fitted model.

        n_components stays fixed at 3 (see the class docstring) — BIC decreases
        monotonically through k=5 on this data, but k>3 forces an arbitrary
        many-to-3 mapping onto the Regime enum for no validated gain. Logged so
        that choice stays visible instead of assumed.

        Args:
            scaled_features: Scaler-transformed training features.
        """
        n_obs, n_features = scaled_features.shape
        for k in range(2, 6):
            try:
                candidate = GaussianHMM(
                    n_components=k, covariance_type="full", n_iter=300, random_state=0
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    candidate.fit(scaled_features)
                log_likelihood = candidate.score(scaled_features)
                n_params = (
                    (k - 1) + k * (k - 1) + k * n_features + k * n_features * (n_features + 1) // 2
                )
                bic = -2 * log_likelihood + n_params * np.log(n_obs)
                logger.info("RegimeClassifier.fit: BIC diagnostic k=%d -> %.1f", k, bic)
            except Exception as exc:
                logger.debug("RegimeClassifier.fit: BIC diagnostic failed for k=%d: %s", k, exc)

    def _map_states(self, df: pd.DataFrame, recession_series: Optional[pd.Series] = None) -> None:
        """Maps hidden states to human-readable regimes.

        CONTRACTION = the state with the highest NBER (USREC) recession
            frequency, provided it clears MACRO.contraction_enrichment_min
            against the sample's base rate; otherwise no state is labelled
            CONTRACTION. This replaces a VIX-percentile heuristic that always
            assigned CONTRACTION to *some* state whether or not a
            contractionary state actually existed — which is how the ZIRP era
            got the label.
        EXPANSION = of the remaining states, the one with the lowest average
            d_unemp_12m (falling unemployment).
        TRANSITIONAL = all other states.

        Args:
            df: The same DataFrame used for fitting, used to label state sequences.
            recession_series: Optional NBER USREC series; see fit()'s docstring.
        """
        self.state_to_regime = {}

        features = df[FEATURE_COLUMNS].values
        scaled_features = self.scaler.transform(features)
        hidden_states = self.hmm.predict(scaled_features)

        df = df.copy()
        df["state"] = hidden_states
        full_means = df.groupby("state")[FEATURE_COLUMNS].mean()
        self.state_means = {int(state): row.to_dict() for state, row in full_means.iterrows()}

        contraction_state = self._label_contraction_state(df, recession_series)

        remaining = df[df["state"] != contraction_state] if contraction_state is not None else df
        expansion_state: Optional[int] = None
        if not remaining.empty:
            remaining_means = remaining.groupby("state")["d_unemp_12m"].mean()
            if not remaining_means.empty:
                expansion_state = int(remaining_means.idxmin())

        for state in sorted(df["state"].unique()):
            state = int(state)
            if state == contraction_state:
                self.state_to_regime[state] = Regime.CONTRACTION.value
            elif state == expansion_state:
                self.state_to_regime[state] = Regime.EXPANSION.value
            else:
                self.state_to_regime[state] = Regime.TRANSITIONAL.value

        logger.debug("HMM State Mapping: %s", self.state_to_regime)

    def _label_contraction_state(
        self, df_with_state: pd.DataFrame, recession_series: Optional[pd.Series]
    ) -> Optional[int]:
        """Picks the CONTRACTION state by NBER recession-frequency enrichment.

        Args:
            df_with_state: Training frame with a "state" column, from _map_states.
            recession_series: NBER USREC series (1.0/0.0), or None if unavailable.

        Returns:
            The state index to label CONTRACTION, or None if no candidate clears
            MACRO.contraction_enrichment_min. Either way, records enrichment and
            recall into validation_metrics for validation_failures() and the
            persisted artifact metadata.
        """
        if recession_series is None:
            self.validation_metrics["contraction_enrichment"] = None
            self.validation_metrics["contraction_recall"] = None
            return None

        aligned = recession_series.reindex(df_with_state.index).fillna(0.0)
        base_rate = float(aligned.mean())
        freq_by_state = aligned.groupby(df_with_state["state"]).mean()
        self.validation_metrics["recession_frequency_by_state"] = {
            int(s): float(f) for s, f in freq_by_state.items()
        }

        if base_rate <= 0 or freq_by_state.empty:
            self.validation_metrics["contraction_enrichment"] = None
            self.validation_metrics["contraction_recall"] = None
            return None

        candidate = int(freq_by_state.idxmax())
        enrichment = float(freq_by_state.loc[candidate] / base_rate)
        self.validation_metrics["contraction_enrichment"] = enrichment

        if enrichment < MACRO.contraction_enrichment_min:
            self.validation_metrics["contraction_recall"] = None
            return None

        recession_mask = aligned > 0
        recall = (
            float((df_with_state.loc[recession_mask, "state"] == candidate).mean())
            if recession_mask.any()
            else 0.0
        )
        self.validation_metrics["contraction_recall"] = recall
        return candidate

    def validation_failures(self) -> list[str]:
        """Checks the fitted model against the training-gate thresholds in MACRO.

        Meant for scripts/train_macro_hmm.py to hard-fail a training run before
        persisting a broken artifact. fit() already rejects degenerate/
        under-occupied restarts internally; this re-checks the winning fit plus
        the checks fit() doesn't perform on its own (dwell time, CONTRACTION
        enrichment, and a discrimination smoke test).

        Returns:
            Human-readable failure descriptions; empty if every check passes.
        """
        if not self.is_fitted:
            return ["classifier is not fitted"]

        failures: list[str] = []
        m = self.validation_metrics

        separation = m.get("min_state_separation", 0.0)
        if separation < MACRO.hmm_min_state_separation:
            failures.append(
                f"min state separation {separation:.3f} below floor "
                f"{MACRO.hmm_min_state_separation}"
            )

        occupancy = m.get("state_occupancy", {})
        low_occupancy = {s: o for s, o in occupancy.items() if o < MACRO.hmm_min_state_occupancy}
        if low_occupancy:
            failures.append(
                f"states below occupancy floor {MACRO.hmm_min_state_occupancy}: {low_occupancy}"
            )

        dwell = m.get("dwell_times_months", {})
        bad_dwell = {
            s: d
            for s, d in dwell.items()
            if not (MACRO.dwell_time_min_months <= d <= MACRO.dwell_time_max_months)
        }
        if bad_dwell:
            failures.append(
                f"states with dwell time outside [{MACRO.dwell_time_min_months}, "
                f"{MACRO.dwell_time_max_months}] months: {bad_dwell}"
            )

        enrichment = m.get("contraction_enrichment")
        if enrichment is None or enrichment < MACRO.contraction_enrichment_min:
            failures.append(
                f"CONTRACTION enrichment {enrichment} below floor "
                f"{MACRO.contraction_enrichment_min}"
            )

        labels = {
            self.predict(pd.DataFrame([scenario]))[0]
            for scenario in DISCRIMINATION_SCENARIOS.values()
        }
        if len(labels) < 2:
            failures.append(f"discrimination scenarios collapsed to a single label: {labels}")

        return failures

    def predict(self, current: dict | pd.DataFrame) -> Tuple[str, float]:
        """Classifies the current economic regime and returns posterior probability confidence.

        Falls back to static rule-based classification if the model has not been fitted.
        Accepts either a single observation (dict) or a multi-day feature window
        (DataFrame). A window lets the HMM's transition matrix inform the
        classification via the forward-backward algorithm instead of decoding an
        isolated point, which is what the posterior confidence is meant to reflect.

        Args:
            current: Either a dict with keys matching FEATURE_COLUMNS, or a DataFrame
                with those same columns, one row per period, ordered oldest to newest.

        Returns:
            Tuple of (regime_string, confidence), where confidence is the max posterior
            probability at the window's final timestep (or the single observation).

        Raises:
            ValueError: If the feature array contains a NaN or infinite value.
        """
        if not self.is_fitted:
            v = current.iloc[-1].to_dict() if isinstance(current, pd.DataFrame) else current
            return self._rule_based_fallback(v)

        if isinstance(current, pd.DataFrame):
            arr = current[FEATURE_COLUMNS].values
        else:
            arr = np.array([[current.get(col, _FEATURE_DEFAULTS[col]) for col in FEATURE_COLUMNS]])

        if not np.isfinite(arr).all():
            raise ValueError("RegimeClassifier.predict: feature array contains NaN/inf")

        scaled_arr = self.scaler.transform(arr)
        posteriors = self.hmm.predict_proba(scaled_arr)
        final_posterior = posteriors[-1]
        state = int(final_posterior.argmax())
        confidence = float(final_posterior.max())

        regime = self.state_to_regime.get(state, Regime.TRANSITIONAL.value)
        return regime, confidence

    def _rule_based_fallback(self, v: dict) -> Tuple[str, float]:
        """Applies static thresholds to classify the regime when the HMM is uncalibrated.

        Used during cold-start or when historical data fetch fails. Confidence is fixed
        at 0.6 (CONTRACTION/EXPANSION) or 0.5 (TRANSITIONAL) to signal lower certainty.

        Args:
            v: Dict with optional keys ``vix`` and ``t10y2y``.

        Returns:
            Tuple of (regime_string, confidence).
        """
        vix = v.get("vix")
        t10y2y = v.get("t10y2y")
        vix = vix if vix is not None else 20.0
        t10y2y = t10y2y if t10y2y is not None else 0.0

        if vix > 30.0 or t10y2y < -0.5:
            return Regime.CONTRACTION.value, 0.6
        elif vix < 18.0 and t10y2y > 0.3:
            return Regime.EXPANSION.value, 0.6
        else:
            return Regime.TRANSITIONAL.value, 0.5

    def save(self, path: str | Path, start_date: Optional[str] = None) -> None:
        """Persists the fitted hmm, scaler, and state_to_regime mapping to a joblib file.

        Args:
            path: Destination file path; parent directories are created if missing.
            start_date: ISO date string the training history began at, recorded in
                metadata for provenance only (fit() itself is agnostic to it).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hmm": self.hmm,
            "scaler": self.scaler,
            "state_to_regime": self.state_to_regime,
            "is_fitted": self.is_fitted,
            "metadata": {
                "hmmlearn_version": hmmlearn.__version__,
                "sklearn_version": sklearn.__version__,
                "feature_columns": list(FEATURE_COLUMNS),
                "n_train_observations": self.n_train_observations,
                "start_date": start_date,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "validation_metrics": self.validation_metrics,
            },
        }
        joblib.dump(payload, path)
        logger.info(
            "RegimeClassifier.save: wrote artifact to %s (%d observations)",
            path,
            self.n_train_observations,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClassifier":
        """Loads a persisted classifier artifact, never raising to the caller.

        Args:
            path: Path to a joblib artifact previously written by save().

        Returns:
            A fitted RegimeClassifier on success. On any failure (missing file,
            corrupt pickle, or a hmmlearn/scikit-learn/feature-column mismatch
            against the currently installed versions) logs at ERROR and returns
            an unfitted classifier, so predict() takes the rule-based path.
        """
        path = Path(path)
        try:
            payload = joblib.load(path)
            metadata = payload["metadata"]

            if metadata.get("feature_columns") != FEATURE_COLUMNS:
                raise ValueError(
                    f"feature column mismatch: artifact trained on "
                    f"{metadata.get('feature_columns')!r}, code expects {FEATURE_COLUMNS!r}"
                )
            # Major.minor only: a patch-level `pip install -U scikit-learn` shouldn't
            # silently drop production to the rule-based path
            artifact_hmmlearn = metadata.get("hmmlearn_version", "")
            if _major_minor(artifact_hmmlearn) != _major_minor(hmmlearn.__version__):
                raise ValueError(
                    f"hmmlearn version mismatch: artifact={artifact_hmmlearn!r}, "
                    f"installed={hmmlearn.__version__!r}"
                )
            artifact_sklearn = metadata.get("sklearn_version", "")
            if _major_minor(artifact_sklearn) != _major_minor(sklearn.__version__):
                raise ValueError(
                    f"scikit-learn version mismatch: artifact={artifact_sklearn!r}, "
                    f"installed={sklearn.__version__!r}"
                )

            classifier = cls()
            classifier.hmm = payload["hmm"]
            classifier.scaler = payload["scaler"]
            classifier.state_to_regime = payload["state_to_regime"]
            classifier.is_fitted = payload["is_fitted"]
            classifier.n_train_observations = metadata.get("n_train_observations", 0)
            classifier.validation_metrics = metadata.get("validation_metrics", {})

            if classifier.is_fitted:
                # fit() already applies this, but re-derive it here too so an
                # artifact saved before this fix still gets a reachable CONTRACTION
                # state from a cold start
                classifier.hmm.startprob_ = cls._stationary_startprob(classifier.hmm.transmat_)

            logger.info(
                "RegimeClassifier.load: loaded artifact from %s (trained %s, %d observations)",
                path,
                metadata.get("trained_at"),
                classifier.n_train_observations,
            )
            return classifier
        except Exception as exc:
            logger.error("RegimeClassifier.load: failed to load artifact from %s: %s", path, exc)
            return cls()


class MacroStatisticalAgent:
    """Orchestrates macroeconomic analysis pipelines across external data sources.

    Loads a persisted HMM classifier artifact on construction — so every
    construction site (API, collector, replay) gets a fitted model with no
    per-caller changes — and serves cached MacroContext objects with a 6-hour
    TTL to avoid redundant FRED fetches. If the artifact is missing or fails
    to load, the classifier degrades to rule-based classification (see
    RegimeClassifier.load).
    """

    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Loads the persisted classifier artifact for immediate use.

        Args:
            market_data: Provider for FRED/macro fetches; defaults to live fetches.
            model_path: Path to a RegimeClassifier artifact; defaults to
                settings.ARGUS_HMM_MODEL_PATH.
        """
        self.classifier = RegimeClassifier.load(model_path or settings.ARGUS_HMM_MODEL_PATH)
        self._cache: Tuple[MacroContext, datetime] | None = None
        self._cache_ttl_hours = 6
        self.market_data = market_data or LiveMarketDataProvider()

    def fit_on_history(self, start_date: str = "2010-01-01") -> None:
        """Fetches complete macroeconomic histories to fit the latent HMM classifier.

        Offline training utility, not exercised by the live `analyze()` path
        graph.py invokes, but now drawn through the same MarketDataProvider seam
        and the same build_macro_feature_frame() analyze() uses — so fit and
        predict can no longer silently drift apart.

        Args:
            start_date: ISO date string defining the beginning of the training window.
                Defaults to 2010-01-01 to include the post-GFC recovery cycle.
        """
        logger.debug("Fetching macro history from %s for HMM fitting...", start_date)
        try:
            df = build_macro_feature_frame(self.market_data, start=start_date)

            recession_series = None
            try:
                usrec = self.market_data.fred_series("USREC", start=start_date)
                recession_series = usrec.resample("ME").last()
            except Exception as exc:
                logger.warning(
                    "fit_on_history: failed to fetch NBER USREC series (%s); "
                    "CONTRACTION will not be labelled on this fit.",
                    exc,
                )

            self.classifier.fit(df, recession_series=recession_series)
            logger.debug("HMM fitted on %d observations from %s.", len(df), start_date)
        except Exception as e:
            logger.error("Failed to fit HMM on history: %s", e)

    def analyze(self) -> Optional[MacroContext]:
        """Compiles real-time economic indicators into a unified MacroContext.

        Returns a cached result if one exists within the 6-hour TTL. Otherwise
        fetches live FRED and VIX data, derives all secondary indicators, and
        builds a fresh MacroContext with agent multipliers.

        Returns:
            A fully populated MacroContext, or None if the core FRED data bundle
            (vix, fed_funds, t10y2y) is entirely unavailable. Returning None
            signals the orchestrator to exit gracefully rather than proceeding
            with a MacroContext built from fabricated defaults.
        """
        if self._cache:
            ctx, cached_at = self._cache
            if (datetime.now() - cached_at).total_seconds() < self._cache_ttl_hours * 3600:
                logger.debug("MacroStatisticalAgent.analyze: Cache hit.")
                return ctx

        current = self.market_data.macro_bundle()

        vix = current.get("vix")
        fed_funds_raw = current.get("fed_funds")
        t10y2y_raw = current.get("t10y2y")

        # All three primary fields being None indicates a complete data feed failure.
        # Return None rather than building a MacroContext from fabricated zero-defaults.
        if vix is None and fed_funds_raw is None and t10y2y_raw is None:
            logger.error(
                "analyze: FRED bundle returned no usable data (vix, fed_funds, t10y2y all None). "
                "Returning None to allow graceful exit."
            )
            return None

        if vix is None:
            logger.warning("analyze: vix is None from FRED bundle; proceeding with regime classification only.")
            vix = 20.0

        # window_final carries the monthly feature frame's last row so the regime-driving
        # fields below reflect the same data the classifier actually scored, instead of a
        # second, independently-fetched macro_bundle() that can disagree with it
        window_final: pd.Series | None = None
        try:
            history_start = (
                datetime.now() - timedelta(days=365 * MACRO.inference_history_years)
            ).strftime("%Y-%m-%d")
            frame = build_macro_feature_frame(self.market_data, start=history_start)
            window = frame.tail(MACRO.inference_window_months)
            if window.empty:
                raise ValueError("feature window empty after build_macro_feature_frame")
            regime_str, confidence = self.classifier.predict(window)
            window_final = window.iloc[-1]
        except Exception as exc:
            logger.warning(
                "analyze: feature window assembly failed (%s); "
                "falling back to single-observation inference.",
                exc,
            )
            regime_str, confidence = self.classifier.predict(current)
        regime = Regime(regime_str)
        model_healthy = self.classifier.is_fitted

        vix_percentile = 50.0
        try:
            vix_hist = self.market_data.ohlcv_daily("^VIX", period="2y")
            closes = vix_hist["close"].dropna()
            if not closes.empty:
                vix_percentile = float((closes < vix).mean() * 100.0)
        except Exception as exc:
            logger.warning("Failed to fetch VIX history for percentile: %s", exc)

        if vix_percentile < 30:
            vix_regime = VixRegime.LOW
        elif vix_percentile < 60:
            vix_regime = VixRegime.MEDIUM
        elif vix_percentile < 80:
            vix_regime = VixRegime.HIGH
        else:
            vix_regime = VixRegime.EXTREME

        if window_final is not None:
            fed_funds = float(window_final["fed_funds"])
        else:
            fed_funds = fed_funds_raw if fed_funds_raw is not None else 0.0
            if fed_funds_raw is None:
                logger.warning("analyze: fed_funds is None from FRED bundle; defaulting to 0.0 for trend calc.")

        # Determine 6-month trailing interest rate trend against historical baseline
        interest_rate_trend = "STABLE"
        try:
            ff_hist = self.market_data.fred_series("FEDFUNDS")
            if len(ff_hist) > 6:
                ff_6m_ago = ff_hist.iloc[-7]
                if fed_funds > ff_6m_ago + 0.25:
                    interest_rate_trend = "RISING"
                elif fed_funds < ff_6m_ago - 0.25:
                    interest_rate_trend = "FALLING"
        except Exception:
            pass

        if window_final is not None:
            t10y2y = float(window_final["t10y2y"])
        else:
            t10y2y = t10y2y_raw if t10y2y_raw is not None else 0.0
            if t10y2y_raw is None:
                logger.warning("analyze: t10y2y is None from FRED bundle; defaulting to 0.0 for yield curve.")

        if t10y2y < -0.1:
            yield_curve = YieldCurve.INVERTED
        elif abs(t10y2y) < 0.15:
            yield_curve = YieldCurve.FLAT
        else:
            yield_curve = YieldCurve.NORMAL

        if window_final is not None:
            cpi_yoy = float(window_final["cpi_yoy"])
        else:
            cpi_yoy_raw = current.get("cpi_yoy")
            cpi_yoy = cpi_yoy_raw if cpi_yoy_raw is not None else 0.0

        # Determine 3-month trailing inflation trajectory
        inflation_traj = "STABLE"
        try:
            cpi_hist = self.market_data.fred_series("CPIAUCSL")
            cpi_yoy_hist = cpi_hist.pct_change(12) * 100.0
            cpi_yoy_hist = cpi_yoy_hist.dropna()
            if len(cpi_yoy_hist) > 3:
                cpi_3m_ago = cpi_yoy_hist.iloc[-4]
                if cpi_yoy > cpi_3m_ago + 0.1:
                    inflation_traj = "RISING"
                elif cpi_yoy < cpi_3m_ago - 0.1:
                    inflation_traj = "FALLING"
        except Exception:
            pass

        if regime == Regime.EXPANSION and fed_funds < 4.5:
            sector_signal = SectorSignal.GROWTH_FAVORED
        elif regime == Regime.CONTRACTION or yield_curve == YieldCurve.INVERTED:
            sector_signal = SectorSignal.DEFENSIVE
        else:
            sector_signal = SectorSignal.VALUE_FAVORED

        # Scale agent multipliers by volatility regime; high VIX amplifies technical and sentiment signals
        fund_mult = 1.3 if (regime == Regime.EXPANSION and vix_percentile < 40) else 0.9
        tech_mult = 1.2 if vix_percentile > 60 else 1.0
        sent_mult = 1.15 if vix_percentile > 50 else 0.9

        multipliers = {
            "fundamental": fund_mult,
            "technical": tech_mult,
            "sentiment": sent_mult,
        }

        if window_final is not None:
            unemployment = float(window_final["unemployment"])
        else:
            unemployment = (
                current.get("unemployment", 0.0) if current.get("unemployment") is not None else 0.0
            )

        ctx = MacroContext(
            fed_funds=fed_funds,
            cpi_yoy=cpi_yoy,
            unemployment=unemployment,
            t10y2y=t10y2y,
            consumer_sentiment=current.get("consumer_sentiment", 50.0)
            if current.get("consumer_sentiment") is not None
            else 50.0,
            vix_level=vix,
            macro_regime=regime,
            regime_confidence=confidence,
            model_healthy=model_healthy,
            interest_rate_trend=interest_rate_trend,  # type: ignore
            yield_curve_shape=yield_curve,
            vix_regime=vix_regime,
            vix_percentile=vix_percentile,
            inflation_trajectory=inflation_traj,  # type: ignore
            sector_rotation_signal=sector_signal,
            agent_multipliers=multipliers,
            api_calls_used=0,
            timestamp=datetime.now(),
        )

        self._cache = (ctx, datetime.now())
        logger.debug("analyze: new MacroContext generated (Regime: %s)", regime.value)
        return ctx
