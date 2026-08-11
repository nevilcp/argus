"""
Tests for the Fundamental Agent and its pure-Python helpers.
"""

from datetime import datetime, timedelta


from argus.agents.fundamental import anonymize_ticker, build_compact_prompt, FundamentalCache
from argus.schemas.signals import FundamentalSignal, Signal

def test_anonymize_ticker():
    """Anonymized IDs are deterministic per (ticker, seed) and vary if either input changes."""
    id1 = anonymize_ticker("AAPL", 20240101)
    id2 = anonymize_ticker("AAPL", 20240101)
    assert id1 == id2
    assert id1.startswith("COMP_")

    id3 = anonymize_ticker("AAPL", 20240102)
    assert id1 != id3

    id4 = anonymize_ticker("MSFT", 20240101)
    assert id1 != id4

def test_build_compact_prompt():
    """The compact prompt embeds fundamentals and substitutes the anon ID when provided."""
    pit_data = {
        "as_of_date": "2024-01-01",
        "fundamentals": {
            "sector": "Technology",
            "pe_ttm": 25.5,
            "revenue_growth_yoy": 0.15,
            "custom_metric": 100
        }
    }

    prompt_real = build_compact_prompt("AAPL", pit_data)
    assert 'ticker="AAPL"' in prompt_real
    assert 'as_of="2024-01-01"' in prompt_real
    assert 'sector="Technology"' in prompt_real
    assert 'P/E Ratio: 25.5' in prompt_real
    assert 'custom_metric: 100' in prompt_real
    # Value comes from _SECTOR_PE_MEDIANS, not the input data
    assert 'industry median ~32.0x' in prompt_real

    prompt_anon = build_compact_prompt("AAPL", pit_data, anon_id="COMP_XYZ")
    assert 'ticker="COMP_XYZ"' in prompt_anon
    assert 'sector="Technology"' in prompt_anon
    assert "AAPL" not in prompt_anon

def test_fundamental_cache():
    """A cached signal is served until it exceeds the 7-day TTL, then evicted."""
    cache = FundamentalCache()
    assert cache.is_stale("AAPL")

    signal = FundamentalSignal(
        ticker="AAPL",
        sector="Technology",
        industry="Consumer Electronics",
        data_as_of_date=datetime.now().date(),
        signal=Signal.BULLISH,
        conviction=0.9,
        moat_score=8,
        reasoning="Test",
        api_calls_used=0,
        timestamp=datetime.now(),
    )

    cache.set("AAPL", signal)
    assert not cache.is_stale("AAPL")
    assert cache.get("AAPL") == signal

    # Backdate the entry past the 7-day TTL rather than waiting for it to expire
    cache._cache["AAPL"] = (signal, datetime.now() - timedelta(days=8))
    assert cache.is_stale("AAPL")
    assert cache.get("AAPL") is None
