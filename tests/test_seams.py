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
from argus.seams import (
    FixtureLLMClient,
    FixtureMarketDataProvider,
    GroqLLMClient,
    RetryableTransportError,
    _groq_retry_delay,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


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
    with pytest.raises(KeyError):
        market_data.fundamentals("NOT_A_REAL_TICKER")


def _synthetic_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    """Builds an in-memory httpx.Response for the chat-completions endpoint — no network."""
    request = httpx.Request("POST", CHAT_COMPLETIONS_URL)
    return httpx.Response(status_code, headers=headers or {}, request=request)


def _synthetic_rate_limit_error(retry_after: str | None) -> groq.RateLimitError:
    """Builds a groq.RateLimitError with synthetic headers — no real httpx.Client involved.

    Args:
        retry_after: Value for the Retry-After header, or None to omit it.

    Returns:
        A RateLimitError carrying an in-memory httpx.Response with those headers.
    """
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = _synthetic_response(429, headers)
    return groq.RateLimitError("rate limited", response=response, body=None)


def _synthetic_connection_error() -> groq.APIConnectionError:
    """Builds the transient, retryable transport failure the adapter must translate."""
    return groq.APIConnectionError(request=httpx.Request("POST", CHAT_COMPLETIONS_URL))


def _synthetic_auth_error() -> groq.AuthenticationError:
    """Builds the terminal 401 the adapter must surface without retrying."""
    return groq.AuthenticationError("bad key", response=_synthetic_response(401), body=None)


def test_groq_retry_delay_honors_retry_after():
    """A RateLimitError's Retry-After header is honored verbatim."""
    exc = _synthetic_rate_limit_error(retry_after="12.5")
    assert _groq_retry_delay(exc) == 12.5


def test_groq_retry_delay_returns_none_without_header():
    """Without a Retry-After header, the adapter gives no hint — the caller decides its own back-off."""
    exc = _synthetic_rate_limit_error(retry_after=None)
    assert _groq_retry_delay(exc) is None


def test_groq_retry_delay_returns_none_for_non_rate_limit_errors():
    """A retryable error other than RateLimitError carries no provider retry hint."""
    assert _groq_retry_delay(_synthetic_connection_error()) is None


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


def _groq_client() -> GroqLLMClient:
    """Builds a GroqLLMClient for a registered model.

    Construction alone never touches the network — only ``._llm.invoke()`` would —
    so no live Groq key is needed.
    """
    return GroqLLMClient(
        model="openai/gpt-oss-20b", temperature=0.1, max_tokens=50, api_key="test-key"
    )


def _client_with_mocked_llm() -> GroqLLMClient:
    """Builds a GroqLLMClient whose ChatGroq is a Mock.

    Mocking that one call is enough to test complete()'s retry and governance
    logic offline.

    Returns:
        A GroqLLMClient ready to have ``._llm.invoke`` given a side_effect.
    """
    client = _groq_client()
    client._llm = mock.Mock()
    return client


def test_groq_llm_client_sets_reasoning_effort_low():
    """The registered gpt-oss models are reasoning models; low effort caps their token spend."""
    assert _groq_client()._llm.reasoning_effort == "low"


def test_groq_llm_client_retryable_error_raises_retryable_transport_error():
    """A transient APIConnectionError becomes RetryableTransportError after a single invocation."""
    client = _client_with_mocked_llm()
    exc = _synthetic_connection_error()
    client._llm.invoke.side_effect = exc

    with pytest.raises(RetryableTransportError) as exc_info:
        client.complete("system", "user")

    assert exc_info.value.cause is exc
    assert client._llm.invoke.call_count == 1


def test_groq_llm_client_retryable_error_carries_retry_after_hint():
    """A RateLimitError's Retry-After header rides along on the RetryableTransportError."""
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_rate_limit_error(retry_after="12.5")

    with pytest.raises(RetryableTransportError) as exc_info:
        client.complete("system", "user")

    assert exc_info.value.retry_after == 12.5


def test_groq_llm_client_terminal_error_raises_without_retry():
    """AuthenticationError propagates on the first attempt — retrying a bad key wastes nothing but time."""
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_auth_error()

    with pytest.raises(groq.AuthenticationError):
        client.complete("system", "user")

    assert client._llm.invoke.call_count == 1


def test_groq_llm_client_releases_reservation_on_terminal_error():
    """A terminal error releases the pre-flight reservation instead of leaking it (GOV-1)."""
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_auth_error()

    with mock.patch("argus.seams.governor.release_reservation") as mock_release:
        with pytest.raises(groq.AuthenticationError):
            client.complete("system", "user")

    mock_release.assert_called_once()
    assert mock_release.call_args.args[0] == client._model


def test_groq_llm_client_releases_reservation_on_retryable_error():
    """A retryable error also releases the reservation instead of leaking it (GOV-1)."""
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_connection_error()

    with mock.patch("argus.seams.governor.release_reservation") as mock_release:
        with pytest.raises(RetryableTransportError):
            client.complete("system", "user")

    mock_release.assert_called_once()
    assert mock_release.call_args.args[0] == client._model


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
