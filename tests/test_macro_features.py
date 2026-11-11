import numpy as np
import pandas as pd
import pytest

from argus.data.macro_features import build_macro_feature_frame
from tests.helpers.market_data import FakeMarketData

_MONTHS = pd.date_range("2019-01-01", "2022-12-01", freq="MS")


def _fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    """Returns a controlled monthly series for the given FRED series ID.

    All series are otherwise flat/smooth so cpi_yoy, d_fed_funds_6m, d_unemp_12m
    and vix_pctile all come out non-NaN across the test window without needing
    to hand-construct realistic macro trajectories.
    """
    if series_id == "FEDFUNDS":
        values = pd.Series(1.0, index=_MONTHS)
        # The one non-flat print: this is the value the point-in-time test
        # checks is absent from the frame until its real publication month.
        values.loc["2021-06-01"] = 99.0
        return values
    if series_id == "UNRATE":
        return pd.Series(5.0, index=_MONTHS)
    if series_id == "CPIAUCSL":
        return pd.Series(100.0 + np.arange(len(_MONTHS)) * 0.2, index=_MONTHS)
    if series_id == "T10Y2Y":
        return pd.Series(1.0, index=_MONTHS)
    raise KeyError(series_id)


def _ohlcv_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Returns a fixed-seed daily close series, long enough for the 5-year
    VIX-percentile rolling window to warm up within the test range."""
    dates = pd.bdate_range("2019-01-01", "2022-12-31")
    closes = 20.0 + np.random.default_rng(0).normal(0, 3.0, len(dates))
    return pd.DataFrame({"close": closes}, index=dates)


def _provider() -> FakeMarketData:
    """A provider with a controlled FEDFUNDS release, for testing the publication-lag shift."""
    return FakeMarketData(fred_series=_fred_series, ohlcv_daily=_ohlcv_daily)


def test_publication_lag_delays_a_release_to_its_actual_publication_month() -> None:
    """A FEDFUNDS print doesn't appear in the frame until after its real release date.

    FRED indexes FEDFUNDS by reference date, not publication date, so a frame
    built without the lag would show the 2021-06-01 print in the June row. The
    real publication lag (32 days) pushes it to the July row instead.
    """
    provider = _provider()
    frame = build_macro_feature_frame(provider, start="2019-01-01")

    assert frame.loc["2021-06-30", "fed_funds"] != pytest.approx(99.0)
    assert frame.loc["2021-07-31", "fed_funds"] == pytest.approx(99.0)


def test_build_macro_feature_frame_is_deterministic_across_call_sites() -> None:
    """Two calls against the same provider and start date produce identical frames.

    RegimeClassifier's fit path (fit_on_history) and predict path (analyze)
    both delegate here (see argus/agents/macro.py and macro_hmm.py), so train/serve skew is
    only ruled out if this is a pure function of (provider, start) with no
    hidden state.
    """
    provider = _provider()

    train_frame = build_macro_feature_frame(provider, start="2019-01-01")
    serve_frame = build_macro_feature_frame(provider, start="2019-01-01")

    pd.testing.assert_frame_equal(train_frame, serve_frame)
