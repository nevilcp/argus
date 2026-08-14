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
    fed_funds = provider.fred_series("FEDFUNDS", start=start).copy()
    unemployment = provider.fred_series("UNRATE", start=start).copy()
    cpi = provider.fred_series("CPIAUCSL", start=start).copy()
    t10y2y = provider.fred_series("T10Y2Y", start=start)
    # "max", not a caller-supplied period, so training and inference always draw
    # identical VIX history for the same provider — the 5-year percentile rolling
    # window needs ~1260 trading days of warmup regardless of caller
    vix = provider.ohlcv_daily("^VIX", period="max")["close"]

    # FRED indexes monthly series by *reference* date, not publication date;
    # shifting by the release lag makes each row reflect only what a real-time
    # observer actually knew as of that date
    fed_funds.index = fed_funds.index + pd.Timedelta(days=MACRO.fed_funds_publication_lag_days)
    unemployment.index = unemployment.index + pd.Timedelta(
        days=MACRO.unemployment_publication_lag_days
    )
    cpi.index = cpi.index + pd.Timedelta(days=MACRO.cpi_publication_lag_days)

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
