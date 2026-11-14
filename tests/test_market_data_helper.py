"""Tests for tests/helpers/market_data.py's FakeMarketData."""

import pandas as pd
import pytest

from tests.helpers.market_data import FakeMarketData


def test_unconfigured_method_fails_loudly():
    with pytest.raises(NotImplementedError, match="vix"):
        FakeMarketData(news=[]).vix()


def test_unknown_keyword_is_rejected():
    with pytest.raises(TypeError, match="nws"):
        FakeMarketData(nws=[])


def test_response_kinds_value_callable_and_exception():
    market_data = FakeMarketData(
        vix=21.5,
        fundamentals=lambda ticker: {"ticker": ticker},
        news=RuntimeError("outage"),
    )

    assert market_data.vix() == 21.5
    assert market_data.fundamentals("AAPL") == {"ticker": "AAPL"}
    with pytest.raises(RuntimeError, match="outage"):
        market_data.news("AAPL", "Apple")


def test_results_are_copied_so_callers_cannot_leak_mutations():
    market_data = FakeMarketData(macro_bundle={"vix": 15.0})

    market_data.macro_bundle()["vix"] = 99.0

    assert market_data.macro_bundle() == {"vix": 15.0}


def test_calls_are_counted_per_method():
    market_data = FakeMarketData(ticker_info={"sector": "Technology"})

    market_data.ticker_info("AAPL")
    market_data.ticker_info("MSFT")

    assert market_data.calls["ticker_info"] == 2


def test_from_closes_serves_a_close_frame_per_ticker():
    closes = {"AAA": pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))}

    frame = FakeMarketData.from_closes(closes).ohlcv_daily("AAA")

    assert list(frame["close"]) == [1.0, 2.0]
