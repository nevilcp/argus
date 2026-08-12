"""
Tests for the MFT Data Pipeline.
"""

import logging

import pandas as pd
import pytest

from argus.data.cache import OHLCVBuffer
from argus.data.pipeline import (
    _FETCH_INTERVAL,
    _ESTIMATED_FETCH_SECONDS_PER_TICKER,
    MFTDataPipeline,
    _bars_per_day,
    _derive_buffer_size,
    _parse_interval_minutes,
    _resample_ohlcv,
)


def test_register_tickers():
    """Registering tickers already present in the pipeline is a no-op for those tickers."""
    pipeline = MFTDataPipeline([])
    pipeline.register_tickers(["AAPL", "MSFT"])
    assert "AAPL" in pipeline.tickers
    assert "MSFT" in pipeline.tickers
    assert len(pipeline.tickers) == 2

    pipeline.register_tickers(["AAPL", "TSLA"])
    assert len(pipeline.tickers) == 3
    assert "TSLA" in pipeline.tickers


def test_compress_candles():
    """Compressing candles returns the full set of pandas-ta derived indicator keys."""
    pipeline = MFTDataPipeline([])

    # 400 minutes covers the 390-bar (1 trading day) momentum lookback at 1m resolution
    dates = pd.date_range("2024-01-01 09:30", periods=400, freq="1min")

    # Monotonic close series so RSI has a known, deterministic value to assert on
    df = pd.DataFrame({
        "open": range(400),
        "high": range(1, 401),
        "low": range(400),
        "close": range(400),
        "volume": [1000] * 400,
    }, index=dates)

    result = pipeline._compress_candles(df)

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
    # A monotonically rising close series saturates RSI at 100
    assert result["rsi_14"] == 100.0


def test_ohlcv_buffer_insertion():
    """Inserting a candle at an existing timestamp overwrites it rather than appending."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20)

    # 15 candles clears the buffer's 14-candle minimum before it returns data
    for i in range(15):
        buffer.insert_candle("AAPL", {"timestamp": f"2024-01-01T10:{i:02d}:00", "close": 150.0 + i})

    df = buffer.get_candles("AAPL")
    assert df is not None
    assert len(df) == 15
    assert df["close"].iloc[-1] == 164.0

    buffer.insert_candle("AAPL", {"timestamp": "2024-01-01T10:14:00", "close": 165.0})
    df2 = buffer.get_candles("AAPL")
    assert len(df2) == 15
    assert df2["close"].iloc[-1] == 165.0


def test_parse_interval_minutes():
    """yfinance interval strings convert to their minute equivalents."""
    assert _parse_interval_minutes("1m") == 1
    assert _parse_interval_minutes("5m") == 5
    assert _parse_interval_minutes("30m") == 30
    assert _parse_interval_minutes("1h") == 60


def test_parse_interval_minutes_rejects_unsupported_shape():
    """An interval string outside the '<n>m' / '<n>h' shape raises rather than silently misparsing."""
    with pytest.raises(ValueError):
        _parse_interval_minutes("daily")


def test_derive_buffer_size_matches_fetch_period():
    """Buffer size holds a full fetch period's candles at the given interval."""
    assert _bars_per_day(1) == 390
    assert _bars_per_day(5) == 78
    assert _derive_buffer_size(1) == 780
    assert _derive_buffer_size(5) == 156


def test_pipeline_derives_buffer_size_from_interval():
    """MFTDataPipeline sizes its buffer from the candle interval when CANDLE_BUFFER_SIZE isn't overridden."""
    pipeline_1m = MFTDataPipeline([], interval="1m")
    assert pipeline_1m.buffer._buffer_size == 780

    pipeline_5m = MFTDataPipeline([], interval="5m")
    assert pipeline_5m.buffer._buffer_size == 156


def test_resample_ohlcv_aggregates_correctly():
    """Resampling combines OHLCV bars with the standard first/max/min/last/sum rules."""
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="1min")
    df = pd.DataFrame({
        "open": range(10),
        "high": [x + 1 for x in range(10)],
        "low": range(10),
        "close": range(10),
        "volume": [100] * 10,
    }, index=dates)

    resampled = _resample_ohlcv(df, interval_minutes=1, target_minutes=5)

    assert len(resampled) == 2
    assert resampled["open"].iloc[0] == 0
    assert resampled["close"].iloc[0] == 4
    assert resampled["high"].iloc[0] == 5
    assert resampled["low"].iloc[0] == 0
    assert resampled["volume"].iloc[0] == 500


def test_resample_ohlcv_is_a_noop_when_target_not_coarser():
    """Resampling to a resolution no coarser than the input returns the input unchanged."""
    dates = pd.date_range("2024-01-01", periods=5, freq="5min")
    df = pd.DataFrame({
        "open": range(5), "high": range(5), "low": range(5), "close": range(5), "volume": [1] * 5,
    }, index=dates)

    result = _resample_ohlcv(df, interval_minutes=5, target_minutes=5)

    pd.testing.assert_frame_equal(result, df)


def test_momentum_windows_scale_with_interval():
    """momentum_30m/momentum_1d convert minutes to bars using the pipeline's own interval, at 1m and 5m."""

    def expected_momentum(closes: list[int], bars: int, n: int) -> float:
        idx = max(-bars, -n)
        return (closes[-1] / closes[idx]) - 1.0

    dates_1m = pd.date_range("2024-01-01 09:30", periods=400, freq="1min")
    closes_1m = list(range(400))
    df_1m = pd.DataFrame({
        "open": closes_1m, "high": [c + 1 for c in closes_1m], "low": closes_1m,
        "close": closes_1m, "volume": [1000] * 400,
    }, index=dates_1m)
    pipeline_1m = MFTDataPipeline([], interval="1m")
    result_1m = pipeline_1m._compress_candles(df_1m)
    assert result_1m["momentum_30m"] == pytest.approx(expected_momentum(closes_1m, 30, 400), abs=1e-6)
    assert result_1m["momentum_1d"] == pytest.approx(expected_momentum(closes_1m, 390, 400), abs=1e-6)

    dates_5m = pd.date_range("2024-01-01 09:30", periods=100, freq="5min")
    closes_5m = list(range(100))
    df_5m = pd.DataFrame({
        "open": closes_5m, "high": [c + 1 for c in closes_5m], "low": closes_5m,
        "close": closes_5m, "volume": [1000] * 100,
    }, index=dates_5m)
    pipeline_5m = MFTDataPipeline([], interval="5m")
    result_5m = pipeline_5m._compress_candles(df_5m)
    assert result_5m["momentum_30m"] == pytest.approx(expected_momentum(closes_5m, 6, 100), abs=1e-6)
    assert result_5m["momentum_1d"] == pytest.approx(expected_momentum(closes_5m, 78, 100), abs=1e-6)


def test_inter_request_sleep_derives_spacing_from_universe_size():
    """Inter-ticker spacing splits the fetch cycle's slack evenly across the universe."""
    pipeline = MFTDataPipeline([], interval="1m")
    n = 20
    expected = (_FETCH_INTERVAL - n * _ESTIMATED_FETCH_SECONDS_PER_TICKER) / n
    assert pipeline._inter_request_sleep(n) == pytest.approx(expected)


def test_inter_request_sleep_returns_zero_with_no_gaps_to_space():
    """A sweep of zero or one tickers has no gaps between fetches to pace."""
    pipeline = MFTDataPipeline([], interval="1m")
    assert pipeline._inter_request_sleep(0) == 0.0
    assert pipeline._inter_request_sleep(1) == 0.0


def test_inter_request_sleep_floors_at_zero_and_warns_when_universe_outgrows_cadence(caplog):
    """A universe too large to fit even an unpaced sweep logs a warning and sweeps back-to-back."""
    pipeline = MFTDataPipeline([], interval="1m")
    huge = int(_FETCH_INTERVAL // _ESTIMATED_FETCH_SECONDS_PER_TICKER) + 10

    with caplog.at_level(logging.WARNING, logger="argus.pipeline"):
        result = pipeline._inter_request_sleep(huge)

    assert result == 0.0
    assert any("won't fit" in record.message for record in caplog.records)
