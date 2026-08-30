"""
Tests for the Fundamental Agent and its pure-Python helpers.
"""

import json
from datetime import datetime


import argus.agents.fundamental as fundamental_module
from argus.agents.fundamental import (
    FundamentalAgent,
    _session_seed_to_date,
    _use_backtest_seed,
    anonymize_ticker,
    build_compact_prompt,
)
from argus.seams import FixtureLLMClient


class _StubMarketData:
    """Returns a fixed fundamentals payload regardless of ticker."""

    def __init__(self, fundamentals: dict) -> None:
        self._fundamentals = fundamentals

    def fundamentals(self, ticker: str) -> dict:
        return dict(self._fundamentals)


class _RecordingLLMClient:
    """Stub LLMClient returning a fixed response while recording the prompt it received."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_user_prompt: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        return self._response_text

    def remaining_capacity(self) -> int:
        return 1_000_000

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

def test_use_backtest_seed_treats_zero_as_a_valid_seed():
    """session_seed=0 is a legal Optional[int] and must not be treated as falsy."""
    assert _use_backtest_seed(True, 0) is True
    assert _use_backtest_seed(True, None) is False
    assert _use_backtest_seed(False, 20240101) is False

def test_session_seed_to_date_parses_yyyymmdd_stamp():
    """session_seed is an integer date stamp, e.g. 20240115 -> 2024-01-15."""
    assert _session_seed_to_date(20240115) == datetime(2024, 1, 15).date()

def test_analyze_overwrites_llm_echoed_ratios_with_measured_data():
    """Every measured ratio in the persisted signal comes from the fetched payload, never the LLM's echo."""
    measured = {
        "pe_ttm": 31.5,
        "revenue_growth_yoy": 0.02,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "fcf_yield": 0.02,
        "debt_to_equity": 0.8,
        "current_ratio": 1.1,
        "roe": 0.4,
        "roic": 0.2,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3_000_000_000_000,
        "p_fcf": 45.0,
    }
    market_data = _StubMarketData(measured)

    # A wrong numeric echo: e.g. as if the model misread 31.5x/2% growth as 12x/45% growth.
    llm_response = json.dumps(
        {
            "signal": "BULLISH",
            "conviction": 0.7,
            "pe_ttm": 12.0,
            "revenue_growth_yoy": 0.45,
            "operating_margin": 0.99,
            "fcf_yield": 0.99,
            "debt_to_equity": 0.01,
            "roic": 0.99,
            "moat_score": 8,
            "reasoning": "Test",
        }
    )
    llm = FixtureLLMClient({"only": llm_response}, key_fn=lambda _p: "only")

    agent = FundamentalAgent(llm_client=llm, market_data=market_data)
    signal = agent.analyze("AAPL")

    assert signal is not None
    for key, value in measured.items():
        assert getattr(signal, key) == value

def test_analyze_anonymizes_and_derives_as_of_date_from_session_seed():
    """A seeded backtest call anonymizes the ticker and stamps data_as_of_date from session_seed, not today."""
    market_data = _StubMarketData({"sector": "Technology", "industry": "Consumer Electronics"})
    llm = _RecordingLLMClient(
        json.dumps({"signal": "NEUTRAL", "conviction": 0.5, "moat_score": 5, "reasoning": "r"})
    )
    agent = FundamentalAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL", backtest_mode=True, session_seed=20240115)

    assert signal is not None
    assert "AAPL" not in llm.last_user_prompt
    assert signal.data_as_of_date == _session_seed_to_date(20240115)

def test_analyze_retries_with_the_prior_validation_error_in_the_prompt():
    """A malformed first response's error is carried into the retry prompt, not silently re-sent."""
    market_data = _StubMarketData({"sector": "Technology", "industry": "Consumer Electronics"})
    prompts: list[str] = []

    class _FlakyThenValidLLMClient:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            prompts.append(user_prompt)
            if len(prompts) == 1:
                return "not valid json"
            return json.dumps(
                {"signal": "NEUTRAL", "conviction": 0.5, "moat_score": 5, "reasoning": "r"}
            )

        def remaining_capacity(self) -> int:
            return 1_000_000

    agent = FundamentalAgent(llm_client=_FlakyThenValidLLMClient(), market_data=market_data)
    signal = agent.analyze("AAPL")

    assert signal is not None
    assert len(prompts) == 2
    assert "Your previous response was invalid" in prompts[1]


def test_analyze_degrades_the_ticker_on_governor_exhaustion_without_retrying():
    """A RateLimitExceeded from the LLM client fails that ticker immediately, once, into errors."""
    market_data = _StubMarketData({"sector": "Technology", "industry": "Consumer Electronics"})

    class _RateLimitedLLMClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            raise fundamental_module.RateLimitExceeded("gpt-oss-120b quota exhausted")

        def remaining_capacity(self) -> int:
            return 1_000_000

    llm = _RateLimitedLLMClient()
    agent = FundamentalAgent(llm_client=llm, market_data=market_data)

    errors: list[str] = []
    signal = agent.analyze("AAPL", errors=errors)

    assert signal is None
    assert llm.calls == 1
    assert len(errors) == 1
    assert "rate limited" in errors[0]


def test_analyze_degrades_the_ticker_when_measured_data_fails_signal_validation():
    """A schema violation in the merged-in measured data (not the verdict) must not crash the batch."""
    market_data = _StubMarketData(
        {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "debt_to_equity": -1.5,  # FundamentalSignal requires ge=0.0
        }
    )
    llm = _RecordingLLMClient(
        json.dumps({"signal": "NEUTRAL", "conviction": 0.5, "moat_score": 5, "reasoning": "r"})
    )
    agent = FundamentalAgent(llm_client=llm, market_data=market_data)

    errors: list[str] = []
    signal = agent.analyze("AAPL", errors=errors)

    assert signal is None
    assert len(errors) == 1
    assert "failed validation" in errors[0]


def test_analyze_does_not_skip_every_ticker_at_a_low_configured_rpm(monkeypatch):
    """A low ARGUS_GROQ_RPM must not make the pre-flight capacity reserve exceed
    the budget itself (GOV-13) — a client reporting that full budget as remaining
    capacity should still admit the call.
    """
    monkeypatch.setattr(fundamental_module.settings, "ARGUS_GROQ_RPM", 10)

    market_data = _StubMarketData({"sector": "Technology", "industry": "Consumer Electronics"})
    llm = FixtureLLMClient(
        {"only": json.dumps({"signal": "NEUTRAL", "conviction": 0.5, "moat_score": 5, "reasoning": "r"})},
        key_fn=lambda _p: "only",
        capacity=10,
    )
    agent = FundamentalAgent(llm_client=llm, market_data=market_data)

    errors: list[str] = []
    signal = agent.analyze("AAPL", errors=errors)

    assert signal is not None
    assert errors == []


def test_analyze_skips_the_ticker_and_makes_no_calls_when_capacity_is_low():
    """A low-capacity client skips the ticker before any market-data fetch or LLM call.

    The capacity check must stay ahead of the fetch so a skipped ticker still
    costs no network call — asserted here by a market-data stub that raises
    if it is ever invoked.
    """

    class _ExplodingMarketData:
        def fundamentals(self, ticker: str) -> dict:
            raise AssertionError("fundamentals() must not be called when capacity is low")

    llm = FixtureLLMClient({}, capacity=0)
    agent = FundamentalAgent(llm_client=llm, market_data=_ExplodingMarketData())

    errors: list[str] = []
    signal = agent.analyze("AAPL", errors=errors)

    assert signal is None
    assert len(errors) == 1
    assert "capacity too low" in errors[0]
