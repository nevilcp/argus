import numpy as np
import pandas as pd
import pytest

from argus.data.macro_features import build_macro_feature_frame
from tests.helpers.market_data import FakeMarketData

_MONTHS = pd.date_range("2019-01-01", "2022-12-01", freq="MS")


def _fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    """Return a controlled monthly series for the given FRED series ID."""
    if series_id == "FEDFUNDS":
        values = pd.Series(1.0, index=_MONTHS)
        # Delay this distinct unemployment print until its publication month in the point-in-time feature test.
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
    """Return reproducible daily closes with enough history to warm the VIX percentile window."""
    dates = pd.bdate_range("2019-01-01", "2022-12-31")
    closes = 20.0 + np.random.default_rng(0).normal(0, 3.0, len(dates))
    return pd.DataFrame({"close": closes}, index=dates)


def _provider() -> FakeMarketData:
    """Build a provider with a controlled Fed funds release date."""
    return FakeMarketData(fred_series=_fred_series, ohlcv_daily=_ohlcv_daily)


def test_publication_lag_delays_a_release_to_its_actual_publication_month() -> None:
    provider = _provider()
    frame = build_macro_feature_frame(provider, start="2019-01-01")

    assert frame.loc["2021-06-30", "fed_funds"] != pytest.approx(99.0)
    assert frame.loc["2021-07-31", "fed_funds"] == pytest.approx(99.0)


def test_build_macro_feature_frame_is_deterministic_across_call_sites() -> None:
    provider = _provider()

    train_frame = build_macro_feature_frame(provider, start="2019-01-01")
    serve_frame = build_macro_feature_frame(provider, start="2019-01-01")

    pd.testing.assert_frame_equal(train_frame, serve_frame)
