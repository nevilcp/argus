"""Tests for argus/data/indicators.py."""

import importlib
import math
import sys

import pandas as pd
import pandas_ta as ta
import pytest
from hypothesis import given
from hypothesis import strategies as st

import argus.data.indicators as indicators_module
from argus.data.indicators import (
    MACD_WARMUP_BARS,
    _finite_or_none,
    _resample_ohlcv,
    _return_since,
    _volume_ratio,
    compress_candles,
    required_raw_bars,
)
from tests.helpers.candles import ET, ohlcv_frame


def test_compress_candles():
    # Cover the full one-day momentum lookback with reproducible monotonic closes.
    result = compress_candles(ohlcv_frame(400), 1)

    assert "rsi_14" in result
    assert "macd_histogram" in result
    assert "bb_percent_b" in result
    assert "atr_pct" in result
    assert "adx_14" in result
    assert "vwap_distance" in result
    assert "volume_ratio" in result
    assert "momentum_30m" in result
    assert "momentum_1d" in result
    assert "close" in result
    assert "timestamp" in result

    assert result["close"] == 399.0
    assert result["rsi_14"] == 100.0
    assert result["macd_histogram"] is not None


def test_resample_ohlcv_aggregates_correctly():
    df = ohlcv_frame(10, volume=100)

    resampled = _resample_ohlcv(df, interval_minutes=1, target_minutes=5)

    assert len(resampled) == 2
    assert resampled["open"].iloc[0] == 0
    assert resampled["close"].iloc[0] == 4
    assert resampled["high"].iloc[0] == 5
    assert resampled["low"].iloc[0] == 0
    assert resampled["volume"].iloc[0] == 500


def test_resample_ohlcv_is_a_noop_when_target_not_coarser():
    df = ohlcv_frame(5, freq="5min")

    result = _resample_ohlcv(df, interval_minutes=5, target_minutes=5)

    pd.testing.assert_frame_equal(result, df)


def test_momentum_windows_scale_with_interval():
    result_1m = compress_candles(ohlcv_frame(400), 1)
    assert result_1m["momentum_30m"] == pytest.approx((399 / 369) - 1.0, abs=1e-6)
    assert result_1m["momentum_1d"] is None

    result_5m = compress_candles(ohlcv_frame(100, freq="5min"), 5)
    assert result_5m["momentum_30m"] == pytest.approx((99 / 93) - 1.0, abs=1e-6)
    assert result_5m["momentum_1d"] is None


def test_finite_or_none():
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(None) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("-inf")) is None


def test_volume_ratio_computes_latest_over_rolling_mean():
    volume_s = pd.Series([100.0] * 19 + [200.0])
    assert _volume_ratio(volume_s) == pytest.approx(200.0 / 105.0)


def test_volume_ratio_returns_none_for_an_exactly_zero_mean():
    assert _volume_ratio(pd.Series([0.0] * 20)) is None


def test_volume_ratio_returns_none_for_a_near_zero_mean_instead_of_an_outlier():
    volume_s = pd.Series([1e-9] * 20)
    assert _volume_ratio(volume_s) is None


def test_macd_warmup_bars_matches_pandas_ta_first_valid_histogram():
    close = pd.Series(range(1, 101), dtype=float)
    histogram = ta.macd(close, fast=12, slow=26, signal=9)["MACDh_12_26_9"]
    assert histogram.first_valid_index() + 1 == MACD_WARMUP_BARS == 34


def test_required_raw_bars_matches_the_issue_table():
    assert required_raw_bars(1) == 391
    assert required_raw_bars(5) == 79
    assert required_raw_bars(15) == 34


def test_compress_candles_readiness_cliff_at_macd_warmup_floor():
    assert compress_candles(ohlcv_frame(33, freq="5min"), 5) is None

    result_34 = compress_candles(ohlcv_frame(34, freq="5min"), 5)
    assert result_34 is not None
    assert result_34["macd_histogram"] is not None


def test_compress_candles_never_emits_nan():
    result = compress_candles(ohlcv_frame(400), 1)
    assert result is not None
    for value in result.values():
        if isinstance(value, float):
            assert not math.isnan(value)


def test_return_since_momentum_1d_none_without_a_prior_session():
    close_s = ohlcv_frame(50, tz=ET)["close"]
    assert _return_since(close_s, pd.Timedelta(days=1), same_session=False) is None


def test_return_since_momentum_30m_does_not_reach_into_the_prior_session():
    day1 = ohlcv_frame(390, start="2024-01-02 09:30", tz=ET)
    day2 = ohlcv_frame(5, start="2024-01-03 09:30", tz=ET)
    combined_close = pd.concat([day1["close"], day2["close"]])
    assert _return_since(combined_close, pd.Timedelta(minutes=30), same_session=True) is None


@given(drop=st.integers(min_value=0, max_value=10))
def test_momentum_1d_is_stable_under_dropping_up_to_10_oldest_bars(drop: int):
    day1 = ohlcv_frame(390, start="2024-01-02 09:30", tz=ET)
    day2 = ohlcv_frame(50, start="2024-01-03 09:30", tz=ET)
    close_s = pd.concat([day1["close"], day2["close"]])

    baseline = _return_since(close_s, pd.Timedelta(days=1), same_session=False)
    trimmed = _return_since(close_s.iloc[drop:], pd.Timedelta(days=1), same_session=False)

    assert baseline is not None
    assert trimmed == baseline


def test_pandas_ta_import_failure_is_a_boot_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas_ta", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(indicators_module)
    finally:
        monkeypatch.undo()
        importlib.reload(indicators_module)
