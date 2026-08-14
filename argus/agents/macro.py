"""
argus/agents/macro.py

Statistical macroeconomic regime analysis agent.

Responsibilities:
  - Combine Federal Reserve (FRED) time series with the CBOE VIX using a Gaussian HMM
  - Classify hidden economic states (EXPANSION, CONTRACTION, TRANSITIONAL)
  - Expose regime multipliers used by specialist agents to dynamically weight convictions

Not responsible for:
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import hmmlearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from argus.config import settings
from argus.data.fetchers import fetch_fred_series
from argus.params import MACRO
from argus.schemas.signals import MacroContext, Regime, SectorSignal, VixRegime, YieldCurve
from argus.seams import LiveMarketDataProvider, MarketDataProvider

logger = logging.getLogger("argus.macro")

# Column order the fit/predict paths always pass to the scaler and HMM; persisted
# in the artifact's metadata so a load-time mismatch is detectable rather than a
# silent feature misalignment.
FEATURE_COLUMNS = ["fed_funds", "cpi_yoy", "t10y2y", "unemployment", "vix"]

_FEATURE_DEFAULTS = {"fed_funds": 0.0, "cpi_yoy": 0.0, "t10y2y": 0.0, "unemployment": 0.0, "vix": 20.0}


class RegimeClassifier:
    """Gaussian Hidden Markov Model mapping macro features to hidden economic states.

    Trained on a long FRED history (2010+), then applied to current macro observations
    to infer the latent regime. State-to-regime mapping is determined heuristically from
    per-state feature means after fitting.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42) -> None:
        """Builds the underlying Gaussian HMM, untrained.

        Args:
            n_components: Number of latent regime states.
            random_state: Seed for the HMM's random initialization.
        """
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=300,
            random_state=random_state,
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

    def fit(self, macro_history: pd.DataFrame) -> None:
        """Fits the scaling transformer and HMM classifier on historic feature timelines.

        Args:
            macro_history: DataFrame with columns [fed_funds, cpi_yoy, t10y2y, unemployment, vix].
                NaN rows are dropped before fitting.
        """
        df = macro_history.dropna().copy()
        if df.empty:
            logger.warning("RegimeClassifier.fit: Empty DataFrame after dropping NaNs.")
            return

        features = df[FEATURE_COLUMNS].values
        scaled_features = self.scaler.fit_transform(features)

        self.hmm.fit(scaled_features)
        self._map_states(df)
        self.is_fitted = True
        self.n_train_observations = len(df)

    def _map_states(self, df: pd.DataFrame) -> None:
        """Maps hidden states to human-readable regimes based on per-state feature means.

        CONTRACTION = state with the highest average VIX.
        EXPANSION = from remaining states, the one with the lowest fed_funds rate
            and a positive yield curve (t10y2y > 0).
        TRANSITIONAL = all other states.

        Args:
            df: The same DataFrame used for fitting, used to label state sequences.
        """
        features = df[FEATURE_COLUMNS].values
        scaled_features = self.scaler.transform(features)
        hidden_states = self.hmm.predict(scaled_features)

        df["state"] = hidden_states
        full_means = df.groupby("state")[FEATURE_COLUMNS].mean()
        self.state_means = {int(state): row.to_dict() for state, row in full_means.iterrows()}

        means = df.groupby("state")[["vix", "fed_funds", "t10y2y"]].mean()

        contraction_state = int(means["vix"].idxmax())

        remaining = means.drop(index=contraction_state, errors="ignore")
        expansion_state = -1

        if not remaining.empty:
            pos_curve = remaining[remaining["t10y2y"] > 0]
            if not pos_curve.empty:
                expansion_state = int(pos_curve["fed_funds"].idxmin())
            else:
                expansion_state = int(remaining["fed_funds"].idxmin())

        for state in means.index:
            state = int(state)
            if state == contraction_state:
                self.state_to_regime[state] = Regime.CONTRACTION.value
            elif state == expansion_state:
                self.state_to_regime[state] = Regime.EXPANSION.value
            else:
                self.state_to_regime[state] = Regime.TRANSITIONAL.value

        logger.debug("HMM State Mapping: %s", self.state_to_regime)

    def predict(self, current: dict | pd.DataFrame) -> Tuple[str, float]:
        """Classifies the current economic regime and returns posterior probability confidence.

        Falls back to static rule-based classification if the model has not been fitted.
        Accepts either a single observation (dict) or a multi-day feature window
        (DataFrame). A window lets the HMM's transition matrix inform the
        classification via the forward-backward algorithm instead of decoding an
        isolated point, which is what the posterior confidence is meant to reflect.

        Args:
            current: Either a dict with keys fed_funds, cpi_yoy, t10y2y, unemployment,
                vix, or a DataFrame with those same columns, one row per day, ordered
                oldest to newest.

        Returns:
            Tuple of (regime_string, confidence), where confidence is the max posterior
            probability at the window's final timestep (or the single observation).
        """
        if not self.is_fitted:
            v = current.iloc[-1].to_dict() if isinstance(current, pd.DataFrame) else current
            return self._rule_based_fallback(v)

        if isinstance(current, pd.DataFrame):
            arr = current[FEATURE_COLUMNS].values
        else:
            arr = np.array([[current.get(col, _FEATURE_DEFAULTS[col]) for col in FEATURE_COLUMNS]])

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
            if metadata.get("hmmlearn_version") != hmmlearn.__version__:
                raise ValueError(
                    f"hmmlearn version mismatch: artifact={metadata.get('hmmlearn_version')!r}, "
                    f"installed={hmmlearn.__version__!r}"
                )
            if metadata.get("sklearn_version") != sklearn.__version__:
                raise ValueError(
                    f"scikit-learn version mismatch: artifact={metadata.get('sklearn_version')!r}, "
                    f"installed={sklearn.__version__!r}"
                )

            classifier = cls()
            classifier.hmm = payload["hmm"]
            classifier.scaler = payload["scaler"]
            classifier.state_to_regime = payload["state_to_regime"]
            classifier.is_fitted = payload["is_fitted"]
            classifier.n_train_observations = metadata.get("n_train_observations", 0)
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

        Not part of the injectable seam: this is an offline training utility,
        not exercised by the live `analyze()` path graph.py invokes, and its
        direct `yf.download` call for VIX history has no equivalent in
        MarketDataProvider.

        Args:
            start_date: ISO date string defining the beginning of the training window.
                Defaults to 2010-01-01 to include the post-GFC recovery cycle.
        """
        logger.debug("Fetching macro history from %s for HMM fitting...", start_date)
        try:
            fed_funds = fetch_fred_series("FEDFUNDS", start=start_date)
            cpi = fetch_fred_series("CPIAUCSL", start=start_date)
            t10y2y = fetch_fred_series("T10Y2Y", start=start_date)
            unrate = fetch_fred_series("UNRATE", start=start_date)

            cpi_yoy = cpi.pct_change(12) * 100.0

            vix_df = yf.download("^VIX", start=start_date, progress=False, threads=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)

            # yfinance column naming varies between minor versions
            vix_col = "Close" if "Close" in vix_df.columns else "close"
            vix = vix_df[vix_col]

            df = pd.DataFrame(
                {
                    "fed_funds": fed_funds,
                    "cpi_yoy": cpi_yoy,
                    "t10y2y": t10y2y,
                    "unemployment": unrate,
                    "vix": vix,
                }
            )

            df = df.resample("D").ffill().dropna()
            self.classifier.fit(df)
            logger.debug("HMM fitted on %d observations from %s.", len(df), start_date)
        except Exception as e:
            logger.error("Failed to fit HMM on history: %s", e)

    def _assemble_feature_window(self, window_days: int) -> pd.DataFrame:
        """Builds a daily feature window for windowed HMM inference.

        Draws on the same fred_series/ohlcv_daily calls analyze() already makes
        elsewhere for trend calculations, so this costs no extra network round-trips
        live (fetch_fred_series has a 6-hour in-process cache).

        Args:
            window_days: Number of most-recent calendar days to keep after
                forward-filling monthly/weekly FRED series to daily resolution.

        Returns:
            DataFrame with FEATURE_COLUMNS, oldest to newest, at most window_days rows.

        Raises:
            Exception: Any fetch failure, or an empty result after resampling and
                dropping NaNs — propagated so analyze() can fall back to
                single-observation inference.
        """
        fed_funds = self.market_data.fred_series("FEDFUNDS")
        cpi = self.market_data.fred_series("CPIAUCSL")
        t10y2y = self.market_data.fred_series("T10Y2Y")
        unrate = self.market_data.fred_series("UNRATE")
        cpi_yoy = cpi.pct_change(12) * 100.0

        vix_hist = self.market_data.ohlcv_daily("^VIX", period="2y")
        vix = vix_hist["close"]

        df = pd.DataFrame(
            {
                "fed_funds": fed_funds,
                "cpi_yoy": cpi_yoy,
                "t10y2y": t10y2y,
                "unemployment": unrate,
                "vix": vix,
            }
        )
        df = df.resample("D").ffill().dropna()
        if df.empty:
            raise ValueError("feature window empty after resample/ffill/dropna")
        return df.tail(window_days)

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

        try:
            window = self._assemble_feature_window(MACRO.inference_window_days)
            regime_str, confidence = self.classifier.predict(window)
        except Exception as exc:
            logger.warning(
                "analyze: feature window assembly failed (%s); "
                "falling back to single-observation inference.",
                exc,
            )
            regime_str, confidence = self.classifier.predict(current)
        regime = Regime(regime_str)

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

        t10y2y = t10y2y_raw if t10y2y_raw is not None else 0.0
        if t10y2y_raw is None:
            logger.warning("analyze: t10y2y is None from FRED bundle; defaulting to 0.0 for yield curve.")

        if t10y2y < -0.1:
            yield_curve = YieldCurve.INVERTED
        elif abs(t10y2y) < 0.15:
            yield_curve = YieldCurve.FLAT
        else:
            yield_curve = YieldCurve.NORMAL

        cpi_yoy = current.get("cpi_yoy")
        if cpi_yoy is None:
            cpi_yoy = 0.0

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

        ctx = MacroContext(
            fed_funds=fed_funds,
            cpi_yoy=cpi_yoy,
            unemployment=current.get("unemployment", 0.0)
            if current.get("unemployment") is not None
            else 0.0,
            t10y2y=t10y2y,
            consumer_sentiment=current.get("consumer_sentiment", 50.0)
            if current.get("consumer_sentiment") is not None
            else 50.0,
            vix_level=vix,
            macro_regime=regime,
            regime_confidence=confidence,
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
