"""
Tests exercising argus/seams.py's fixture-backed implementations against the
real agent classes — the point of the injection seam is that
FundamentalAgent/SentimentAgent produce a real, schema-valid signal from
fixture data with zero network or LLM calls. If these fixtures ever go
stale or the seam breaks, these tests fail; nothing here is decorative.
"""

import json
from pathlib import Path
from unittest import mock

import groq
import httpx
import pytest

from argus.agents.fundamental import FundamentalAgent
from argus.agents.sentiment import SentimentAgent
from argus.schemas.signals import FundamentalSignal, SentimentSignal
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider, GroqLLMClient, _groq_retry_delay

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _single_response_llm(fixture_file: str, ticker: str) -> FixtureLLMClient:
    """Builds a FixtureLLMClient that replays one recorded response for every call.

    Args:
        fixture_file: JSON file under fixtures/llm_responses/ keyed by ticker.
        ticker: Ticker whose recorded response to replay.

    Returns:
        A FixtureLLMClient returning that ticker's response for any prompt.
    """
    with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
        responses = json.load(f)
    # One ticker per instance, so every call in a single analyze() gets the same response
    return FixtureLLMClient({"only": responses[ticker]}, key_fn=lambda _prompt: "only")


def test_fundamental_agent_with_fixtures_produces_valid_signal():
    """FundamentalAgent produces a schema-valid signal from fixture data alone."""
    market_data = FixtureMarketDataProvider()
    llm_client = _single_response_llm("fundamental.json", "AAPL")
    agent = FundamentalAgent(llm_client=llm_client, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert isinstance(signal, FundamentalSignal)
    assert signal.ticker == "AAPL"
    assert signal.signal.value == "BEARISH"
    assert signal.conviction == 0.75
    assert signal.moat_score == 8
    # Fetched (not LLM-produced) fields came through the fixture market data provider
    assert signal.sector == "Technology"
    assert signal.marketCap is not None


def test_sentiment_agent_with_fixtures_produces_valid_signal():
    """SentimentAgent produces a schema-valid signal from fixture data alone."""
    market_data = FixtureMarketDataProvider()
    llm_client = _single_response_llm("sentiment.json", "AAPL")
    agent = SentimentAgent(llm_client=llm_client, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert isinstance(signal, SentimentSignal)
    assert signal.ticker == "AAPL"
    assert signal.signal.value == "NEUTRAL"
    assert signal.conviction == 0.4
    assert signal.sentiment_decay_risk == "LOW"


def test_fixture_market_data_provider_ohlcv_daily_matches_fixture():
    """The fixture provider serves real OHLCV data, not an empty stand-in."""
    market_data = FixtureMarketDataProvider()
    df = market_data.ohlcv_daily("AAPL")
    assert not df.empty
    assert "close" in df.columns


def test_fixture_market_data_provider_missing_ticker_raises():
    """A ticker with no fixture data raises KeyError rather than failing silently."""
    market_data = FixtureMarketDataProvider()
    try:
        market_data.fundamentals("NOT_A_REAL_TICKER")
        assert False, "expected KeyError for a ticker with no fixture"
    except KeyError:
        pass


def _synthetic_rate_limit_error(retry_after: str | None) -> groq.RateLimitError:
    """Builds a groq.RateLimitError with synthetic headers — no real httpx.Client involved.

    Args:
        retry_after: Value for the Retry-After header, or None to omit it.

    Returns:
        A RateLimitError carrying an in-memory httpx.Response with those headers.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=request)
    return groq.RateLimitError("rate limited", response=response, body=None)


def test_groq_retry_delay_honors_retry_after():
    """A RateLimitError's Retry-After header is honored verbatim, not re-jittered."""
    exc = _synthetic_rate_limit_error(retry_after="12.5")
    assert _groq_retry_delay(exc, attempt=0) == 12.5


def test_groq_retry_delay_falls_back_to_jittered_backoff_without_header():
    """Without a Retry-After header, delay is jittered exponential back-off capped at 30s."""
    exc = _synthetic_rate_limit_error(retry_after=None)
    for attempt in range(6):
        delay = _groq_retry_delay(exc, attempt)
        assert 0.0 <= delay <= 30.0


def _fake_success_response(prompt_tokens: int = 10, completion_tokens: int = 5):
    """Minimal stand-in for a ChatGroq.invoke() return value.

    Args:
        prompt_tokens: Value reported under response_metadata["token_usage"].
        completion_tokens: Value reported under response_metadata["token_usage"].

    Returns:
        A Mock exposing the ``content``/``response_metadata`` attributes complete() reads.
    """
    return mock.Mock(
        content="synthetic response",
        response_metadata={
            "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        },
    )


def _client_with_mocked_llm() -> GroqLLMClient:
    """Builds a GroqLLMClient with a real (network-free) httpx.Client but a mocked ChatGroq.

    Construction alone never touches the network — only ``._llm.invoke()`` would —
    so swapping in a Mock for that one call is enough to test complete()'s retry
    and governance logic without a live Groq key.

    Returns:
        A GroqLLMClient for a registered model, ready to have ``._llm.invoke``
        given a side_effect.
    """
    client = GroqLLMClient(
        model="llama-3.1-8b-instant", temperature=0.1, max_tokens=50, api_key="test-key"
    )
    client._llm = mock.Mock()
    return client


def test_groq_llm_client_retries_retryable_error_then_succeeds():
    """A transient APIConnectionError is retried and the eventual success is returned."""
    client = _client_with_mocked_llm()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client._llm.invoke.side_effect = [
        groq.APIConnectionError(request=request),
        _fake_success_response(),
    ]

    with mock.patch("argus.seams.time.sleep"):
        result = client.complete("system", "user")

    assert result == "synthetic response"
    assert client._llm.invoke.call_count == 2


def test_groq_llm_client_terminal_error_raises_without_retry():
    """AuthenticationError propagates on the first attempt — retrying a bad key wastes nothing but time."""
    client = _client_with_mocked_llm()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request)
    client._llm.invoke.side_effect = groq.AuthenticationError(
        "bad key", response=response, body=None
    )

    with mock.patch("argus.seams.time.sleep") as mock_sleep:
        with pytest.raises(groq.AuthenticationError):
            client.complete("system", "user")

    assert client._llm.invoke.call_count == 1
    assert mock_sleep.call_count == 0


def test_groq_llm_client_exhausts_retries_and_raises():
    """A persistently retryable error is retried up to the attempt cap, then propagates."""
    client = _client_with_mocked_llm()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client._llm.invoke.side_effect = groq.APIConnectionError(request=request)

    with mock.patch("argus.seams.time.sleep"):
        with pytest.raises(groq.APIConnectionError):
            client.complete("system", "user")

    from argus.seams import _MAX_COMPLETE_ATTEMPTS

    assert client._llm.invoke.call_count == _MAX_COMPLETE_ATTEMPTS


def test_groq_llm_client_releases_reservation_on_terminal_error():
    """A terminal error releases the pre-flight reservation instead of leaking it (GOV-1)."""
    client = _client_with_mocked_llm()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request)
    client._llm.invoke.side_effect = groq.AuthenticationError(
        "bad key", response=response, body=None
    )

    with mock.patch("argus.seams.time.sleep"), mock.patch(
        "argus.seams.governor.release_reservation"
    ) as mock_release:
        with pytest.raises(groq.AuthenticationError):
            client.complete("system", "user")

    mock_release.assert_called_once()
    assert mock_release.call_args.args[0] == client._model


def test_groq_llm_client_releases_reservation_after_exhausting_retries():
    """Retries-exhausted also releases the reservation instead of leaking it (GOV-1)."""
    client = _client_with_mocked_llm()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client._llm.invoke.side_effect = groq.APIConnectionError(request=request)

    with mock.patch("argus.seams.time.sleep"), mock.patch(
        "argus.seams.governor.release_reservation"
    ) as mock_release:
        with pytest.raises(groq.APIConnectionError):
            client.complete("system", "user")

    mock_release.assert_called_once()


def test_groq_llm_client_success_does_not_release_reservation():
    """A successful call records usage but never releases the reservation it actually spent."""
    client = _client_with_mocked_llm()
    client._llm.invoke.return_value = _fake_success_response()

    with mock.patch("argus.seams.governor.release_reservation") as mock_release, mock.patch(
        "argus.seams.governor.record_usage"
    ) as mock_record:
        client.complete("system", "user")

    mock_release.assert_not_called()
    mock_record.assert_called_once()
