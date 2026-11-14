"""Gaussian HMM training, validation, persistence, and regime inference."""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

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

from argus.agents.macro_context import rule_based
from argus.params import MACRO
from argus.schemas.signals import Regime

logger = logging.getLogger(__name__)

# Scaler/HMM column order; persisted in metadata to catch mismatches. See data/macro_features.py.
FEATURE_COLUMNS = ["d_fed_funds_6m", "d_unemp_12m", "cpi_yoy", "t10y2y", "vix_pctile"]

# Discrimination-gate scenarios (scripts/train_macro_hmm.py); values on FEATURE_COLUMNS' scale.
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


def _new_hmm(n_components: int, random_state: int | None = None) -> GaussianHMM:
    """Build an untrained GaussianHMM with the project's fixed hyperparameters."""
    return GaussianHMM(
        n_components=n_components,
        covariance_type=MACRO.hmm_covariance_type,
        n_iter=MACRO.hmm_n_iter,
        random_state=random_state,
    )


def _dwell_times_months(hidden_states: np.ndarray, n_components: int) -> dict[int, float]:
    """Compute each state's mean run length (in observations) from a Viterbi path."""
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
    """Compute the minimum pairwise Mahalanobis distance between state means."""
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
    """Truncate a dotted version string to its major.minor prefix."""
    return ".".join(version.split(".")[:2])


def _check_artifact_compatibility(metadata: dict, hmm: GaussianHMM) -> None:
    """Verify a persisted artifact was trained against the installed code and libraries."""
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            f"feature column mismatch: artifact trained on "
            f"{metadata.get('feature_columns')!r}, code expects {FEATURE_COLUMNS!r}"
        )
    # Major.minor only: a patch-level scikit-learn upgrade shouldn't force the rule-based path.
    artifact_hmmlearn = metadata.get("hmmlearn_version", "")
    if _major_minor(artifact_hmmlearn) != _major_minor(hmmlearn.__version__):
        raise ValueError(
            f"hmmlearn version mismatch: artifact={artifact_hmmlearn!r}, installed={hmmlearn.__version__!r}"
        )
    artifact_sklearn = metadata.get("sklearn_version", "")
    if _major_minor(artifact_sklearn) != _major_minor(sklearn.__version__):
        raise ValueError(
            f"scikit-learn version mismatch: artifact={artifact_sklearn!r}, installed={sklearn.__version__!r}"
        )
    if (hmm.covariance_type, hmm.n_iter) != (MACRO.hmm_covariance_type, MACRO.hmm_n_iter):
        raise ValueError(
            f"HMM hyperparameter mismatch: artifact covariance_type={hmm.covariance_type!r}, "
            f"n_iter={hmm.n_iter}; configured covariance_type={MACRO.hmm_covariance_type!r}, "
            f"n_iter={MACRO.hmm_n_iter}"
        )


class RegimeClassifier:
    """Gaussian Hidden Markov Model mapping macro features to hidden economic states."""

    def __init__(self, n_components: int = 3) -> None:
        self.hmm = _new_hmm(n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.state_to_regime: dict[int, str] = {}
        self.n_train_observations = 0
        # Diagnostic state means are refreshed by _map_states and excluded from persisted artifacts.
        self.state_means: dict[int, dict[str, float]] = {}
        # Persist fit validation evidence so the publication gate can inspect the winning model.
        self.validation_metrics: dict = {}

    def fit(self, macro_history: pd.DataFrame, recession_series: pd.Series | None = None) -> bool:
        """Fit the scaling transformer and HMM classifier on historic feature timelines."""
        df = macro_history.dropna().copy()
        if df.empty:
            logger.warning("Empty DataFrame after dropping NaNs.")
            return False

        n_components = self.hmm.n_components
        features = df[FEATURE_COLUMNS].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        self._log_bic_diagnostic(scaled_features)

        best_hmm: GaussianHMM | None = None
        best_score = -np.inf
        best_separation = 0.0
        best_occupancy = np.array([])
        rejections: list[str] = []

        for seed in range(MACRO.hmm_n_init):
            candidate = _new_hmm(n_components, random_state=seed)
            with warnings.catch_warnings():
                # Sub-1e-4 monotonicity wobble near the optimum is benign, not a failure.
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
        # Set the stationary prior before validation calls predict on the unsaved model.
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
        return True

    @staticmethod
    def _stationary_startprob(transmat: np.ndarray) -> np.ndarray:
        """Compute a Markov chain's stationary distribution from its transition matrix."""
        eigenvalues, eigenvectors = scipy.linalg.eig(transmat.T)
        stationary = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))].real
        return stationary / stationary.sum()

    def _log_bic_diagnostic(self, scaled_features: np.ndarray) -> None:
        """Log BIC for two through five states without changing the fitted three-state model."""
        n_obs, n_features = scaled_features.shape
        for k in range(2, 6):
            try:
                candidate = _new_hmm(k, random_state=0)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    candidate.fit(scaled_features)
                log_likelihood = candidate.score(scaled_features)
                n_params = (k - 1) + k * (k - 1) + k * n_features + k * n_features * (n_features + 1) // 2
                bic = -2 * log_likelihood + n_params * np.log(n_obs)
                logger.info("BIC diagnostic k=%d -> %.1f", k, bic)
            except Exception as exc:
                logger.debug("BIC diagnostic failed for k=%d: %s", k, exc)

    def _map_states(self, df: pd.DataFrame, recession_series: pd.Series | None = None) -> None:
        """Map hidden states to human-readable regimes."""
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
        expansion_state: int | None = None
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

    def _label_contraction_state(self, df_with_state: pd.DataFrame, recession_series: pd.Series | None) -> int | None:
        """Pick the CONTRACTION state by NBER recession-frequency enrichment."""
        if recession_series is None:
            self.validation_metrics["contraction_enrichment"] = None
            self.validation_metrics["contraction_recall"] = None
            return None

        aligned = recession_series.reindex(df_with_state.index).fillna(0.0)
        base_rate = float(aligned.mean())
        freq_by_state = aligned.groupby(df_with_state["state"]).mean()
        self.validation_metrics["recession_frequency_by_state"] = {int(s): float(f) for s, f in freq_by_state.items()}

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
            float((df_with_state.loc[recession_mask, "state"] == candidate).mean()) if recession_mask.any() else 0.0
        )
        self.validation_metrics["contraction_recall"] = recall
        return candidate

    def validation_failures(self) -> list[str]:
        """Check the fitted model against the training-gate thresholds in MACRO."""
        if not self.is_fitted:
            return ["classifier is not fitted"]

        failures: list[str] = []
        m = self.validation_metrics

        separation = m.get("min_state_separation", 0.0)
        if separation < MACRO.hmm_min_state_separation:
            failures.append(f"min state separation {separation:.3f} below floor {MACRO.hmm_min_state_separation}")

        occupancy = m.get("state_occupancy", {})
        low_occupancy = {s: o for s, o in occupancy.items() if o < MACRO.hmm_min_state_occupancy}
        if low_occupancy:
            failures.append(f"states below occupancy floor {MACRO.hmm_min_state_occupancy}: {low_occupancy}")

        dwell = m.get("dwell_times_months", {})
        bad_dwell = {
            s: d for s, d in dwell.items() if not (MACRO.dwell_time_min_months <= d <= MACRO.dwell_time_max_months)
        }
        if bad_dwell:
            failures.append(
                f"states with dwell time outside [{MACRO.dwell_time_min_months}, "
                f"{MACRO.dwell_time_max_months}] months: {bad_dwell}"
            )

        enrichment = m.get("contraction_enrichment")
        if enrichment is None or enrichment < MACRO.contraction_enrichment_min:
            failures.append(f"CONTRACTION enrichment {enrichment} below floor {MACRO.contraction_enrichment_min}")

        labels = {self.predict(pd.DataFrame([scenario]))[0] for scenario in DISCRIMINATION_SCENARIOS.values()}
        if len(labels) < 2:
            failures.append(f"discrimination scenarios collapsed to a single label: {labels}")

        return failures

    def predict(self, current: dict | pd.DataFrame) -> tuple[str, float]:
        """Classify the final feature row using window posteriors, or raw-level rules when unfitted."""
        if not self.is_fitted:
            if isinstance(current, pd.DataFrame):
                raise ValueError(
                    "RegimeClassifier.predict: classifier is not fitted; "
                    "a feature window cannot be classified by the rule-based path"
                )
            return rule_based(current)

        if isinstance(current, pd.DataFrame):
            arr = current[FEATURE_COLUMNS].values
        else:
            # Reject missing features instead of scoring an invented neutral observation.
            missing = [col for col in FEATURE_COLUMNS if current.get(col) is None]
            if missing:
                raise ValueError(f"RegimeClassifier.predict: missing or None feature(s): {', '.join(missing)}")
            arr = np.array([[current[col] for col in FEATURE_COLUMNS]])

        if not np.isfinite(arr).all():
            raise ValueError("RegimeClassifier.predict: feature array contains NaN/inf")

        scaled_arr = self.scaler.transform(arr)
        posteriors = self.hmm.predict_proba(scaled_arr)
        final_posterior = posteriors[-1]
        state = int(final_posterior.argmax())
        confidence = float(final_posterior.max())

        regime = self.state_to_regime.get(state, Regime.TRANSITIONAL.value)
        return regime, confidence

    def save(self, path: str | Path, start_date: str | None = None) -> None:
        """Persist the fitted hmm, scaler, and state_to_regime mapping to a joblib file."""
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
                "trained_at": datetime.now(UTC).isoformat(),
                "validation_metrics": self.validation_metrics,
            },
        }
        joblib.dump(payload, path)
        logger.info(
            "wrote artifact to %s (%d observations)",
            path,
            self.n_train_observations,
        )

    @classmethod
    def load(cls, path: str | Path) -> RegimeClassifier:
        """Load a persisted classifier artifact, never raising to the caller."""
        path = Path(path)
        try:
            payload = joblib.load(path)
            metadata = payload["metadata"]
            _check_artifact_compatibility(metadata, payload["hmm"])

            classifier = cls()
            classifier.hmm = payload["hmm"]
            classifier.scaler = payload["scaler"]
            classifier.state_to_regime = payload["state_to_regime"]
            classifier.is_fitted = payload["is_fitted"]
            classifier.n_train_observations = metadata.get("n_train_observations", 0)
            classifier.validation_metrics = metadata.get("validation_metrics", {})

            if classifier.is_fitted:
                # Reapply the stationary prior so older artifacts can reach every regime at cold start.
                classifier.hmm.startprob_ = cls._stationary_startprob(classifier.hmm.transmat_)

            logger.info(
                "loaded artifact from %s (trained %s, %d observations)",
                path,
                metadata.get("trained_at"),
                classifier.n_train_observations,
            )
            return classifier
        except Exception as exc:
            logger.error("failed to load artifact from %s: %s", path, exc)
            return cls()
