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
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from argus.data.fetchers import fetch_fred_series
from argus.schemas.signals import MacroContext, Regime, SectorSignal, VixRegime, YieldCurve
from argus.seams import LiveMarketDataProvider, MarketDataProvider

logger = logging.getLogger("argus.macro")


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

        features = df[["fed_funds", "cpi_yoy", "t10y2y", "unemployment", "vix"]].values
        scaled_features = self.scaler.fit_transform(features)

        self.hmm.fit(scaled_features)
        self._map_states(df)
        self.is_fitted = True

    def _map_states(self, df: pd.DataFrame) -> None:
        """Maps hidden states to human-readable regimes based on per-state feature means.

        CONTRACTION = state with the highest average VIX.
        EXPANSION = from remaining states, the one with the lowest fed_funds rate
            and a positive yield curve (t10y2y > 0).
        TRANSITIONAL = all other states.

        Args:
            df: The same DataFrame used for fitting, used to label state sequences.
        """
        features = df[["fed_funds", "cpi_yoy", "t10y2y", "unemployment", "vix"]].values
        scaled_features = self.scaler.transform(features)
        hidden_states = self.hmm.predict(scaled_features)

        df["state"] = hidden_states
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

    def predict(self, current_values: dict) -> Tuple[str, float]:
        """Classifies the current economic regime and returns posterior probability confidence.

        Falls back to static rule-based classification if the model has not been fitted.

        Args:
            current_values: Dict with keys fed_funds, cpi_yoy, t10y2y, unemployment, vix.

        Returns:
            Tuple of (regime_string, confidence) where confidence is the max posterior probability.
        """
        if not self.is_fitted:
            return self._rule_based_fallback(current_values)

        arr = np.array(
            [
                [
                    current_values.get("fed_funds", 0.0),
                    current_values.get("cpi_yoy", 0.0),
                    current_values.get("t10y2y", 0.0),
                    current_values.get("unemployment", 0.0),
                    current_values.get("vix", 20.0),
                ]
            ]
        )

        scaled_arr = self.scaler.transform(arr)
        state = int(self.hmm.predict(scaled_arr)[0])
        posteriors = self.hmm.predict_proba(scaled_arr)[0]
        confidence = float(posteriors.max())

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


class MacroStatisticalAgent:
    """Orchestrates macroeconomic analysis pipelines across external data sources.

    Loads and fits the underlying HMM classifier on startup, then serves
    cached MacroContext objects with a 6-hour TTL to avoid redundant FRED fetches.
    """

    def __init__(self, market_data: Optional[MarketDataProvider] = None) -> None:
        """Builds an untrained classifier; call fit_on_history before first use.

        Args:
            market_data: Provider for FRED/macro fetches; defaults to live fetches.
        """
        self.classifier = RegimeClassifier()
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
