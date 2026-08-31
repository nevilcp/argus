"""
argus/data/macro_features.py

Shared point-in-time feature construction for the macro regime HMM's fit and
predict paths.

Responsibilities:
  - Fetch FEDFUNDS, UNRATE, CPIAUCSL, T10Y2Y, and ^VIX through the
    MarketDataProvider seam, so training and inference build features through
    one code path instead of two independently-drifting ones
  - Shift each monthly FRED series by its publication lag, so no row reflects
    information not yet public as of that row's own timestamp
  - Derive the five stationary features RegimeClassifier fits and predicts on

Not responsible for:
  - HMM fitting, state labelling, or the artifact validation gate (see
    agents/macro.py)
  - Publication-lag and window-length values themselves (see params.py's
    MacroParams)
"""

from __future__ import annotations

import pandas as pd

from argus.params import MACRO
from argus.seams import MarketDataProvider


def _lagged_fred_series(
    provider: MarketDataProvider, series_id: str, start: str, lag_days: int
) -> pd.Series:
    """Fetches a monthly FRED series and shifts it forward by its publication lag.

    FRED indexes monthly series by *reference* date, not publication date; shifting
    by the release lag makes each row reflect only what a real-time observer
    actually knew as of that date.

    Args:
        provider: Source for FRED series.
        series_id: FRED series identifier.
        start: ISO date string for the beginning of the fetched history.
        lag_days: Days between the series' reference date and its release.

    Returns:
        A copy of the series, re-indexed by publication date.
    """
    series = provider.fred_series(series_id, start=start).copy()
    series.index = series.index + pd.Timedelta(days=lag_days)
    return series


def build_macro_feature_frame(provider: MarketDataProvider, start: str) -> pd.DataFrame:
    """Builds the shared monthly macro feature frame for fitting and inference.

    Args:
        provider: Source for FRED series and VIX OHLCV history.
        start: ISO date string for the beginning of the fetched FRED history.

    Returns:
        Month-end DataFrame indexed by date, carrying both the raw levels
        (fed_funds, unemployment, t10y2y) and the five stationary features
        RegimeClassifier is fit/predicted on (d_fed_funds_6m, d_unemp_12m,
        cpi_yoy, t10y2y, vix_pctile — see agents.macro.FEATURE_COLUMNS).
        Rows with any NaN after a bounded 2-month forward-fill are dropped.
    """
    fed_funds = _lagged_fred_series(
        provider, "FEDFUNDS", start, MACRO.fed_funds_publication_lag_days
    )
    unemployment = _lagged_fred_series(
        provider, "UNRATE", start, MACRO.unemployment_publication_lag_days
    )
    cpi = _lagged_fred_series(provider, "CPIAUCSL", start, MACRO.cpi_publication_lag_days)
    t10y2y = provider.fred_series("T10Y2Y", start=start)
    # "max", not a caller-supplied period, so training and inference always draw identical
    # VIX history — the 5-year percentile window needs ~1260 trading days of warmup
    vix = provider.ohlcv_daily("^VIX", period="max")["close"]

    frame = pd.DataFrame(
        {
            "fed_funds": fed_funds,
            "unemployment": unemployment,
            "t10y2y": t10y2y,
            "cpi_yoy": cpi.pct_change(12) * 100.0,
            "d_fed_funds_6m": fed_funds.diff(6),
            "d_unemp_12m": unemployment.diff(12),
            "vix_pctile": vix.rolling(1260, min_periods=252).rank(pct=True) * 100.0,
        }
    )
    frame = frame.resample("ME").last()
    return frame.ffill(limit=2).dropna()
