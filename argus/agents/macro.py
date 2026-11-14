"""
argus/agents/macro.py

Statistical macroeconomic regime analysis agent.

Responsibilities:
  - Fetch the FRED macro bundle and the VIX history, classify the regime with
    the persisted HMM (or the rule-based fallback), and assemble a MacroContext
  - Cache that context for a few hours so repeated sessions do not refetch
  - Expose regime multipliers consumed once, by orchestration/aggregator.py, as a
    per-agent vote-weight scalar during signal aggregation — never by the
    specialist agents' own analysis, which runs independently of this agent

Not responsible for:
  - The HMM itself: fitting, state labelling, persistence (see agents/macro_hmm.py)
  - The label and multiplier derivations (see agents/macro_context.py)
  - Feature construction (see data/macro_features.py)
  - Raw data fetching beyond macro bundles (see data/fetchers.py)
  - Portfolio allocation (see agents/portfolio.py)
  - Risk constraint enforcement (see agents/risk.py)

Dependencies:
  - FRED_API_KEY env var must be set (see .env.example)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from argus.agents.macro_context import (
    agent_multipliers,
    rule_based,
    sector_rotation,
    trend_from_lag,
    vix_regime,
    yield_curve_shape,
)
from argus.agents.macro_hmm import RegimeClassifier
from argus.config import settings
from argus.data.macro_features import (
    CPI_SERIES,
    FED_FUNDS_SERIES,
    RECESSION_SERIES,
    build_macro_feature_frame,
    cpi_yoy,
)
from argus.infra.clock import now_local
from argus.params import MACRO
from argus.ports import MarketDataProvider
from argus.schemas.signals import MacroContext, Regime, Trend
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger(__name__)


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
        market_data: MarketDataProvider | None = None,
        model_path: str | None = None,
    ) -> None:
        """Loads the persisted classifier artifact for immediate use.

        Args:
            market_data: Provider for FRED/macro fetches; defaults to live fetches.
            model_path: Path to a RegimeClassifier artifact; defaults to
                settings.ARGUS_HMM_MODEL_PATH.
        """
        self.classifier = RegimeClassifier.load(model_path or settings.ARGUS_HMM_MODEL_PATH)
        self._cache: tuple[MacroContext, datetime] | None = None
        self._cache_ttl_hours = 6
        self.market_data = market_data or LiveMarketDataProvider()

    def fit_on_history(self, start_date: str = MACRO.hmm_fit_start) -> bool:
        """Fetches complete macroeconomic histories to fit the latent HMM classifier.

        Offline training utility, not exercised by the live `analyze()` path
        graph.py invokes, but drawn through the same MarketDataProvider seam
        and the same build_macro_feature_frame() analyze() uses — so fit and
        predict cannot silently drift apart. A successful refit drops the cached
        MacroContext, which was classified by the previous model.

        Args:
            start_date: ISO date string defining the beginning of the training window.
                Defaults to MACRO.hmm_fit_start, which includes the post-GFC recovery cycle.

        Returns:
            True if the classifier was fitted; False if fetching or fitting failed,
            leaving the classifier as it was.
        """
        logger.debug("Fetching macro history from %s for HMM fitting...", start_date)
        try:
            df = build_macro_feature_frame(self.market_data, start=start_date)

            recession_series = None
            try:
                usrec = self.market_data.fred_series(RECESSION_SERIES, start=start_date)
                recession_series = usrec.resample("ME").last()
            except Exception as exc:
                logger.warning(
                    "failed to fetch NBER USREC series (%s); "
                    "CONTRACTION will not be labelled on this fit.",
                    exc,
                )

            if not self.classifier.fit(df, recession_series=recession_series):
                return False
            logger.debug("HMM fitted on %d observations from %s.", len(df), start_date)
        except Exception as e:
            logger.error("Failed to fit HMM on history: %s", e)
            return False
        self._cache = None
        return True

    def _resolve_field(
        self,
        window_final: pd.Series | None,
        field_name: str,
        raw_value: float | None,
        *,
        warn_message: str | None = None,
    ) -> float:
        """Reads one macro field from the feature window, falling back to the FRED bundle.

        Args:
            window_final: Final row of the monthly feature frame, or None if the
                window could not be assembled.
            field_name: Column to read from window_final.
            raw_value: The same field as it came back from macro_bundle(), used
                only when window_final is unavailable.
            warn_message: Logged at WARNING when both sources are unavailable and
                the field falls back to 0.0. Omit for fields that degrade silently.

        Returns:
            The field's value, or 0.0 if neither source supplied one.
        """
        if window_final is not None:
            return float(window_final[field_name])
        if raw_value is not None:
            return raw_value
        if warn_message is not None:
            logger.warning(warn_message)
        return 0.0

    def _trend_vs_lag(
        self,
        series_fetch_fn: Callable[[], pd.Series],
        current_value: float,
        lag: int,
        threshold: float,
        *,
        label: str,
    ) -> Trend:
        """Buckets a value's move against the same series `lag` periods back.

        Args:
            series_fetch_fn: Returns the already-transformed FRED series to compare
                against, oldest observation first. Any transform the comparison
                needs (e.g. a year-over-year percent change) belongs in here.
            current_value: The present-day value, on the returned series' scale.
            lag: How many periods back to compare against.
            threshold: Move size, in the series' units, that separates a direction
                from STABLE.
            label: Short name of the trend, used only in the failure log line.

        Returns:
            A Trend: RISING, FALLING, or STABLE — the last also being the fallback
            when the series cannot be fetched or is shorter than `lag`.
        """
        try:
            return trend_from_lag(series_fetch_fn(), current_value, lag, threshold)
        except Exception as exc:
            logger.warning("Failed to compute %s trend; defaulting to STABLE: %s", label, exc)
            return Trend.STABLE

    def _fresh_cache(self) -> MacroContext | None:
        """Returns the cached MacroContext if it is still within its TTL."""
        if self._cache:
            ctx, cached_at = self._cache
            if (now_local() - cached_at).total_seconds() < self._cache_ttl_hours * 3600:
                logger.debug("Cache hit.")
                return ctx
        return None

    def _classify_regime(self, current: dict) -> tuple[str, float, pd.Series | None, bool]:
        """Classifies the regime from the HMM's feature window, else from raw levels.

        Args:
            current: The FRED macro bundle, whose raw vix/t10y2y the rule-based
                path reads (a feature-window row carries neither).

        Returns:
            ``(regime, confidence, window_final, model_healthy)``. ``window_final``
            is the classifier's actual scored row, so later fields read from it
            instead of a disagreeing second fetch; it is None on the rule-based
            paths. ``model_healthy`` is False whenever the HMM did not produce the
            regime, whether unfitted or because the window failed to assemble.
        """
        if not self.classifier.is_fitted:
            regime, confidence = rule_based(current)
            return regime, confidence, None, False
        try:
            history_start = (
                now_local() - timedelta(days=365 * MACRO.inference_history_years)
            ).strftime("%Y-%m-%d")
            frame = build_macro_feature_frame(self.market_data, start=history_start)
            window = frame.tail(MACRO.inference_window_months)
            if window.empty:
                raise ValueError("feature window empty after build_macro_feature_frame")
            regime, confidence = self.classifier.predict(window)
            return regime, confidence, window.iloc[-1], True
        except Exception as exc:
            logger.warning(
                "feature window assembly failed (%s); "
                "falling back to single-observation inference.",
                exc,
            )
            regime, confidence = rule_based(current)
            return regime, confidence, None, False

    def _vix_percentile(self, vix: float | None) -> float:
        """Ranks today's VIX against two years of daily closes.

        Args:
            vix: The VIX level.

        Returns:
            The percentage of closes below `vix`, or the neutral 50.0 when the level
            is unavailable or the history could not be fetched.
        """
        if vix is None:
            logger.warning("VIX unavailable; skipping percentile computation, using neutral 50.0")
            return 50.0
        try:
            closes = self.market_data.ohlcv_daily("^VIX", period="2y")["close"].dropna()
            if not closes.empty:
                return float((closes < vix).mean() * 100.0)
        except Exception as exc:
            logger.warning("Failed to fetch VIX history for percentile: %s", exc)
        return 50.0

    def analyze(self) -> MacroContext | None:
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
        cached = self._fresh_cache()
        if cached is not None:
            return cached

        current = self.market_data.macro_bundle()
        vix = current.get("vix")
        fed_funds_raw = current.get("fed_funds")
        t10y2y_raw = current.get("t10y2y")
        if vix is None and fed_funds_raw is None and t10y2y_raw is None:
            logger.error(
                "FRED bundle returned no usable data (vix, fed_funds, t10y2y all None). "
                "Returning None to allow graceful exit."
            )
            return None

        if vix is None:
            logger.warning(
                "vix is None from FRED bundle; proceeding with regime classification only."
            )

        regime_str, confidence, window_final, model_healthy = self._classify_regime(current)
        regime = Regime(regime_str)
        vix_percentile = self._vix_percentile(vix)

        fed_funds = self._resolve_field(
            window_final,
            "fed_funds",
            fed_funds_raw,
            warn_message=(
                "analyze: fed_funds is None from FRED bundle; defaulting to 0.0 for trend calc."
            ),
        )
        interest_rate_trend = self._trend_vs_lag(
            lambda: self.market_data.fred_series(FED_FUNDS_SERIES),
            fed_funds,
            lag=6,
            threshold=0.25,
            label="interest rate",
        )
        t10y2y = float(window_final["t10y2y"]) if window_final is not None else t10y2y_raw
        if t10y2y is None:
            logger.warning("Yield curve unavailable: no t10y2y measurement")
        curve = yield_curve_shape(t10y2y)
        cpi = self._resolve_field(window_final, "cpi_yoy", current.get("cpi_yoy"))
        inflation_trajectory = self._trend_vs_lag(
            lambda: cpi_yoy(self.market_data.fred_series(CPI_SERIES)).dropna(),
            cpi,
            lag=3,
            threshold=0.1,
            label="inflation",
        )
        consumer_sentiment = current.get("consumer_sentiment")

        ctx = MacroContext(
            fed_funds=fed_funds,
            cpi_yoy=cpi,
            unemployment=self._resolve_field(
                window_final, "unemployment", current.get("unemployment")
            ),
            t10y2y=t10y2y,
            consumer_sentiment=50.0 if consumer_sentiment is None else consumer_sentiment,
            vix_level=vix,
            macro_regime=regime,
            regime_confidence=confidence,
            model_healthy=model_healthy,
            interest_rate_trend=interest_rate_trend,
            yield_curve_shape=curve,
            vix_regime=vix_regime(vix),
            vix_percentile=vix_percentile,
            inflation_trajectory=inflation_trajectory,
            sector_rotation_signal=sector_rotation(regime, fed_funds, curve),
            agent_multipliers=agent_multipliers(regime, vix_percentile),
            api_calls_used=0,
            timestamp=now_local(),
        )

        self._cache = (ctx, now_local())
        logger.debug("new MacroContext generated (Regime: %s)", regime.value)
        return ctx
