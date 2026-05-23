"""
argus/agents/macro.py
=====================
Macro-Economic Agent — Purely statistical regime classification via Gaussian HMM.

Combines key FRED time series (fed funds, CPI, yield curve, unemployment) with
VIX to estimate the hidden macroeconomic state (EXPANSION, CONTRACTION, TRANSITIONAL).
Outputs a ``MacroContext`` schema that downstream agents use to scale conviction.

Zero LLM calls. Runs once per day or when the cache expires (6 hours).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from argus.data.fetchers import fetch_fred_series, fetch_macro_bundle, fetch_ohlcv_daily
from argus.schemas.signals import MacroContext, Regime, SectorSignal, VixRegime, YieldCurve

logger = logging.getLogger("argus.macro")


# ──────────────────────────────────────────────────────────────────────────────
# HMM Regime Classifier
# ──────────────────────────────────────────────────────────────────────────────


class RegimeClassifier:
    """
    Gaussian Hidden Markov Model for macroeconomic regime classification.

    Trains on 5 features: fed_funds, cpi_yoy, t10y2y, unemployment, vix.
    Maps the 3 hidden states to human-readable regimes based on their
    learned mean characteristics.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42) -> None:
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
        """
        Fit the HMM on historical macro data and map hidden states.

        Parameters
        ----------
        macro_history:
            DataFrame containing columns: ``fed_funds``, ``cpi_yoy``,
            ``t10y2y``, ``unemployment``, ``vix``.
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
        """Assign meaningful names to the learned hidden states based on means."""
        features = df[["fed_funds", "cpi_yoy", "t10y2y", "unemployment", "vix"]].values
        scaled_features = self.scaler.transform(features)
        hidden_states = self.hmm.predict(scaled_features)

        df["state"] = hidden_states
        means = df.groupby("state")[["vix", "fed_funds", "t10y2y"]].mean()

        # Highest mean VIX → CONTRACTION
        contraction_state = int(means["vix"].idxmax())

        remaining = means.drop(index=contraction_state, errors="ignore")
        expansion_state = -1

        if not remaining.empty:
            # Lowest fed funds and positive yield curve → EXPANSION
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
        """
        Predict the current regime and confidence.

        Falls back to simple heuristics if the model has not been fitted.
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
        """Heuristic fallback for when HMM history isn't loaded yet."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent Class
# ──────────────────────────────────────────────────────────────────────────────


class MacroStatisticalAgent:
    """
    Evaluates current macroeconomic conditions using FRED and yfinance data.
    """

    def __init__(self) -> None:
        self.classifier = RegimeClassifier()
        self._cache: Tuple[MacroContext, datetime] | None = None
        self._cache_ttl_hours = 6

    def fit_on_history(self, start_date: str = "2010-01-01") -> None:
        """Fetch historical data and fit the HMM classifier."""
        logger.info("Fetching macro history from %s for HMM fitting...", start_date)
        try:
            fed_funds = fetch_fred_series("FEDFUNDS", start=start_date)
            cpi = fetch_fred_series("CPIAUCSL", start=start_date)
            t10y2y = fetch_fred_series("T10Y2Y", start=start_date)
            unrate = fetch_fred_series("UNRATE", start=start_date)

            cpi_yoy = cpi.pct_change(12) * 100.0

            # VIX from yfinance
            vix_df = yf.download("^VIX", start=start_date, progress=False, threads=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            
            # ensure case insensitivity or handle correctly
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
            logger.info("HMM fitted on %d observations from %s.", len(df), start_date)
        except Exception as e:
            logger.error("Failed to fit HMM on history: %s", e)

    def analyze(self) -> MacroContext:
        """
        Produce a MacroContext summarizing the current macroeconomic regime.
        """
        # ── 1. Check Cache ──
        if self._cache:
            ctx, cached_at = self._cache
            if (datetime.now() - cached_at).seconds < self._cache_ttl_hours * 3600:
                logger.debug("MacroStatisticalAgent.analyze: Cache hit.")
                return ctx

        # ── 2. Fetch current data ──
        current = fetch_macro_bundle()
        regime_str, confidence = self.classifier.predict(current)
        regime = Regime(regime_str)

        vix = current.get("vix")
        if vix is None:
            vix = 20.0

        # ── 3. VIX Percentile and Regime ──
        vix_percentile = 50.0
        try:
            vix_hist = fetch_ohlcv_daily("^VIX", period="2y")
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

        # ── 4. Interest Rate Trend ──
        fed_funds = current.get("fed_funds")
        if fed_funds is None:
            fed_funds = 0.0
        
        interest_rate_trend = "STABLE"
        try:
            ff_hist = fetch_fred_series("FEDFUNDS")
            if len(ff_hist) > 6:
                ff_6m_ago = ff_hist.iloc[-7]
                if fed_funds > ff_6m_ago + 0.25:
                    interest_rate_trend = "RISING"
                elif fed_funds < ff_6m_ago - 0.25:
                    interest_rate_trend = "FALLING"
        except Exception:
            pass

        # ── 5. Yield Curve Shape ──
        t10y2y = current.get("t10y2y")
        if t10y2y is None:
            t10y2y = 0.0
            
        if t10y2y < -0.1:
            yield_curve = YieldCurve.INVERTED
        elif abs(t10y2y) < 0.15:
            yield_curve = YieldCurve.FLAT
        else:
            yield_curve = YieldCurve.NORMAL

        # ── 6. Inflation Trajectory ──
        cpi_yoy = current.get("cpi_yoy")
        if cpi_yoy is None:
            cpi_yoy = 0.0
            
        inflation_traj = "STABLE"
        try:
            cpi_hist = fetch_fred_series("CPIAUCSL")
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

        # ── 7. Sector Rotation Signal ──
        if regime == Regime.EXPANSION and fed_funds < 4.5:
            sector_signal = SectorSignal.GROWTH_FAVORED
        elif regime == Regime.CONTRACTION or yield_curve == YieldCurve.INVERTED:
            sector_signal = SectorSignal.DEFENSIVE
        else:
            sector_signal = SectorSignal.VALUE_FAVORED

        # ── 8. Agent Conviction Multipliers ──
        fund_mult = 1.3 if (regime == Regime.EXPANSION and vix_percentile < 40) else 0.9
        tech_mult = 1.2 if vix_percentile > 60 else 1.0
        sent_mult = 1.15 if vix_percentile > 50 else 0.9

        multipliers = {
            "fundamental": fund_mult,
            "technical": tech_mult,
            "sentiment": sent_mult,
        }

        # ── 9. Build and Cache Context ──
        ctx = MacroContext(
            macro_regime=regime,
            interest_rate_trend=interest_rate_trend,  # type: ignore
            yield_curve_shape=yield_curve,
            vix_level=vix,
            vix_regime=vix_regime,
            vix_percentile=vix_percentile,
            inflation_trajectory=inflation_traj,      # type: ignore
            sector_rotation_signal=sector_signal,
            agent_multipliers=multipliers,
            regime_confidence=confidence,
            api_calls_used=0,
            timestamp=datetime.now(),
        )

        self._cache = (ctx, datetime.now())
        logger.info("analyze: new MacroContext generated (Regime: %s)", regime.value)
        return ctx
