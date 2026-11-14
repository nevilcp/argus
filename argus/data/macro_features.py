"""Shared point-in-time macro features for HMM training and inference."""

from __future__ import annotations

import pandas as pd

from argus.params import MACRO
from argus.ports import MarketDataProvider

FED_FUNDS_SERIES = "FEDFUNDS"
UNEMPLOYMENT_SERIES = "UNRATE"
CPI_SERIES = "CPIAUCSL"
T10Y2Y_SERIES = "T10Y2Y"
T10YIE_SERIES = "T10YIE"
CONSUMER_SENTIMENT_SERIES = "UMCSENT"
RECESSION_SERIES = "USREC"


def cpi_yoy(cpi: pd.Series) -> pd.Series:
    """Convert a monthly CPI index into its year-over-year percent change."""
    return cpi.pct_change(12) * 100.0


def _lagged_fred_series(provider: MarketDataProvider, series_id: str, start: str, lag_days: int) -> pd.Series:
    """Fetch a monthly FRED series and shifts it forward by its publication lag."""
    series = provider.fred_series(series_id, start=start).copy()
    series.index = series.index + pd.Timedelta(days=lag_days)
    return series


def build_macro_feature_frame(provider: MarketDataProvider, start: str) -> pd.DataFrame:
    """Build the shared monthly macro feature frame for fitting and inference."""
    fed_funds = _lagged_fred_series(provider, FED_FUNDS_SERIES, start, MACRO.fed_funds_publication_lag_days)
    unemployment = _lagged_fred_series(provider, UNEMPLOYMENT_SERIES, start, MACRO.unemployment_publication_lag_days)
    cpi = _lagged_fred_series(provider, CPI_SERIES, start, MACRO.cpi_publication_lag_days)
    t10y2y = provider.fred_series(T10Y2Y_SERIES, start=start)
    # Use identical full VIX history in fit and inference to warm the five-year percentile window.
    vix = provider.ohlcv_daily("^VIX", period="max")["close"]

    frame = pd.DataFrame(
        {
            "fed_funds": fed_funds,
            "unemployment": unemployment,
            "t10y2y": t10y2y,
            "cpi_yoy": cpi_yoy(cpi),
            "d_fed_funds_6m": fed_funds.diff(6),
            "d_unemp_12m": unemployment.diff(12),
            "vix_pctile": vix.rolling(
                MACRO.vix_percentile_window_days, min_periods=MACRO.vix_percentile_min_periods
            ).rank(pct=True)
            * 100.0,
        }
    )
    frame = frame.resample("ME").last()
    return frame.ffill(limit=2).dropna()
