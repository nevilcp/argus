"""
tests/test_tickers.py

Tests for argus/data/tickers.py's is_valid_ticker, the shared validator
behind AnalysisRequest and MFTDataPipeline.register_tickers (PR5a, API-1).
"""

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
]


@given(st.sampled_from(_REPRO_REJECTIONS))
def test_rejects_the_repro_set(ticker):
    """Every string from the audit's hostile-ticker repro set is rejected."""
    assert not is_valid_ticker(ticker)


_ACCEPTED = [
    "AAPL", "MSFT", "BRK.B", "BRK-B", "BF-B", "F", "T", "V", "UNH", "GOOGL",
]


@given(st.sampled_from(_ACCEPTED))
def test_accepts_the_class_share_set(ticker):
    """Plain tickers and dotted/hyphenated share-class symbols are accepted."""
    assert is_valid_ticker(ticker)


@given(st.sampled_from(["^GSPC", "^DJI", "BTC-USD", "ETH-USD"]))
def test_rejects_indices_and_crypto_pairs(ticker):
    """Indices and crypto pairs are deliberately excluded: no GICS sector or financials."""
    assert not is_valid_ticker(ticker)


@given(st.text(alphabet=" '\"/\x00", min_size=1, max_size=10))
def test_rejects_any_string_with_whitespace_quote_slash_or_nul(s):
    """Any string built only from whitespace, quote, slash, or NUL characters is rejected."""
    assert not is_valid_ticker(s)
