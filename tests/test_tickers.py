"""Tests is_valid_ticker, the shared validator behind ticker input handling."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from argus.data.tickers import is_valid_ticker

_REPRO_REJECTIONS = [
    "",
    "A" * 500,
    "'; DROP TABLE ohlcv; --",
    "../../etc/passwd",
    "AAPLLL",
    ".B",
    "AAPL.",
    "AAPL\n",
    "BRK.B\n",
]


@given(st.sampled_from(_REPRO_REJECTIONS))
def test_rejects_the_repro_set(ticker):
    assert not is_valid_ticker(ticker)


_ACCEPTED = [
    "AAPL",
    "MSFT",
    "BRK.B",
    "BRK-B",
    "BF-B",
    "F",
    "T",
    "V",
    "UNH",
    "GOOGL",
]

_INDICES_AND_CRYPTO_PAIRS = ["^GSPC", "^DJI", "BTC-USD", "ETH-USD"]


@given(st.sampled_from(_ACCEPTED))
def test_accepts_the_class_share_set(ticker):
    assert is_valid_ticker(ticker)


@given(st.sampled_from(_INDICES_AND_CRYPTO_PAIRS))
def test_rejects_indices_and_crypto_pairs(ticker):
    assert not is_valid_ticker(ticker)


@given(st.text(alphabet=" '\"/\x00", min_size=1, max_size=10))
def test_rejects_any_string_with_whitespace_quote_slash_or_nul(s):
    assert not is_valid_ticker(s)
