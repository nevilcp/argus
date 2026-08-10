"""
Tests for the MFT Data Pipeline.
"""

from datetime import datetime

import pandas as pd
import pytest

from argus.data.pipeline import MFTDataPipeline
from argus.data.cache import OHLCVBuffer

def test_register_tickers():
    pipeline = MFTDataPipeline([])
    pipeline.register_tickers(["AAPL", "MSFT"])
    assert "AAPL" in pipeline.tickers
    assert "MSFT" in pipeline.tickers
    assert len(pipeline.tickers) == 2

    # Test duplicates are ignored
    pipeline.register_tickers(["AAPL", "TSLA"])
    assert len(pipeline.tickers) == 3
    assert "TSLA" in pipeline.tickers

def test_compress_candles():
    pipeline = MFTDataPipeline([])

    # Generate a dummy dataframe with enough rows for indicators (e.g. MACD needs 26)
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    
    # A simple upward trend
    df = pd.DataFrame({
        "open": range(100),
        "high": range(1, 101),
        "low": range(100),
        "close": range(100),
        "volume": [1000] * 100,
    }, index=dates)

    # Some basic checks to ensure pandas-ta runs without throwing errors
    # and returns the correct keys
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

    assert result["close"] == 99.0
    # RSI for a straight line up is usually 100
    assert result["rsi_14"] == 100.0

def test_ohlcv_buffer_insertion():
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20)
    
    # Need at least 14 candles to bypass the return None check
    for i in range(15):
        buffer.insert_candle("AAPL", {"timestamp": f"2024-01-01T10:{i:02d}:00", "close": 150.0 + i})
    
    df = buffer.get_candles("AAPL")
    assert df is not None
    assert len(df) == 15
    assert df["close"].iloc[-1] == 164.0

    # Overwrite the same timestamp
    buffer.insert_candle("AAPL", {"timestamp": "2024-01-01T10:14:00", "close": 165.0})
    df2 = buffer.get_candles("AAPL")
    assert len(df2) == 15
    assert df2["close"].iloc[-1] == 165.0
