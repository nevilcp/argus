"""Exercises argus/seams.py's fixture-backed implementations against real agents."""

import json
import logging
from unittest import mock

import groq
import httpx
import pytest

from argus.agents.fundamental import FundamentalAgent
from argus.agents.sentiment import SentimentAgent
from argus.ports import RetryableTransportError
from argus.schemas.signals import FundamentalSignal, SentimentSignal
from argus.seams import (
    FixtureLLMClient,
    FixtureMarketDataProvider,
    GroqLLMClient,
    LiveMarketDataProvider,
    _groq_retry_delay,
    default_groq_client,
)
from tests.helpers.paths import FIXTURES_DIR

CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def _single_response_llm(fixture_file: str, ticker: str) -> FixtureLLMClient:
    """Build a FixtureLLMClient that replays one recorded response for every call."""
    with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
        responses = json.load(f)
    return FixtureLLMClient({"only": responses[ticker]}, key_fn=lambda _prompt: "only")


def test_fundamental_agent_with_fixtures_produces_valid_signal():
    market_data = FixtureMarketDataProvider()
    llm_client = _single_response_llm("fundamental.json", "AAPL")
    agent = FundamentalAgent(llm_client=llm_client, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert isinstance(signal, FundamentalSignal)
    assert signal.ticker == "AAPL"
    assert signal.signal.value == "BEARISH"
    assert signal.conviction == 0.75
    assert signal.moat_score == 8
    assert signal.sector == "Technology"
    assert signal.marketCap is not None


def test_sentiment_agent_with_fixtures_produces_valid_signal():
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
    market_data = FixtureMarketDataProvider()
    df = market_data.ohlcv_daily("AAPL")
    assert not df.empty
    assert "close" in df.columns


def test_fixture_market_data_provider_missing_ticker_raises():
    market_data = FixtureMarketDataProvider()
    with pytest.raises(KeyError):
        market_data.fundamentals("NOT_A_REAL_TICKER")


def _synthetic_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    """Build an in-memory httpx.Response for the chat-completions endpoint - no network."""
    request = httpx.Request("POST", CHAT_COMPLETIONS_URL)
    return httpx.Response(status_code, headers=headers or {}, request=request)


def _synthetic_rate_limit_error(retry_after: str | None) -> groq.RateLimitError:
    """Build a groq.RateLimitError with synthetic headers - no real httpx.Client involved."""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = _synthetic_response(429, headers)
    return groq.RateLimitError("rate limited", response=response, body=None)


def _synthetic_connection_error() -> groq.APIConnectionError:
    """Build the transient, retryable transport failure the adapter must translate."""
    return groq.APIConnectionError(request=httpx.Request("POST", CHAT_COMPLETIONS_URL))


def _synthetic_auth_error() -> groq.AuthenticationError:
    """Build the terminal 401 the adapter must surface without retrying."""
    return groq.AuthenticationError("bad key", response=_synthetic_response(401), body=None)


def test_groq_retry_delay_honors_retry_after():
    exc = _synthetic_rate_limit_error(retry_after="12.5")
    assert _groq_retry_delay(exc) == 12.5


def test_groq_retry_delay_returns_none_without_header():
    exc = _synthetic_rate_limit_error(retry_after=None)
    assert _groq_retry_delay(exc) is None


def test_groq_retry_delay_returns_none_for_non_rate_limit_errors():
    assert _groq_retry_delay(_synthetic_connection_error()) is None


def _fake_success_response(prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a minimal successful model response."""
    return mock.Mock(
        content="synthetic response",
        response_metadata={"token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    )


def _groq_client() -> GroqLLMClient:
    """Build a GroqLLMClient for a registered model."""
    return GroqLLMClient(model="openai/gpt-oss-20b", temperature=0.1, max_tokens=50, api_key="test-key")


def _client_with_mocked_llm() -> GroqLLMClient:
    """Build a GroqLLMClient whose ChatGroq is a Mock."""
    client = _groq_client()
    client._llm = mock.Mock()
    return client


def test_groq_llm_client_sets_reasoning_effort_low():
    assert _groq_client()._llm.reasoning_effort == "low"


def test_groq_llm_client_retryable_error_raises_retryable_transport_error():
    client = _client_with_mocked_llm()
    exc = _synthetic_connection_error()
    client._llm.invoke.side_effect = exc

    with pytest.raises(RetryableTransportError) as exc_info:
        client.complete("system", "user")

    assert exc_info.value.cause is exc
    assert client._llm.invoke.call_count == 1


def test_groq_llm_client_retryable_error_carries_retry_after_hint():
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_rate_limit_error(retry_after="12.5")

    with pytest.raises(RetryableTransportError) as exc_info:
        client.complete("system", "user")

    assert exc_info.value.retry_after == 12.5


def test_groq_llm_client_terminal_error_raises_without_retry():
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_auth_error()

    with pytest.raises(groq.AuthenticationError):
        client.complete("system", "user")

    assert client._llm.invoke.call_count == 1


def test_groq_llm_client_releases_reservation_on_terminal_error():
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_auth_error()

    with (
        mock.patch("argus.seams.governor.release_reservation") as mock_release,
        pytest.raises(groq.AuthenticationError),
    ):
        client.complete("system", "user")

    mock_release.assert_called_once()
    assert mock_release.call_args.args[0] == client._model


def test_groq_llm_client_releases_reservation_on_retryable_error():
    client = _client_with_mocked_llm()
    client._llm.invoke.side_effect = _synthetic_connection_error()

    with (
        mock.patch("argus.seams.governor.release_reservation") as mock_release,
        pytest.raises(RetryableTransportError),
    ):
        client.complete("system", "user")

    mock_release.assert_called_once()
    assert mock_release.call_args.args[0] == client._model


def test_groq_llm_client_success_does_not_release_reservation():
    client = _client_with_mocked_llm()
    client._llm.invoke.return_value = _fake_success_response()

    with (
        mock.patch("argus.seams.governor.release_reservation") as mock_release,
        mock.patch("argus.seams.governor.record_usage") as mock_record,
    ):
        client.complete("system", "user")

    mock_release.assert_not_called()
    mock_record.assert_called_once()


def test_groq_llm_client_warns_when_reply_is_truncated(caplog):
    client = _client_with_mocked_llm()
    response = _fake_success_response()
    response.response_metadata["finish_reason"] = "length"
    client._llm.invoke.return_value = response

    with caplog.at_level("WARNING", logger="argus.seams"):
        client.complete("system", "user")

    assert "finish_reason=length" in caplog.text


def test_groq_llm_client_does_not_warn_on_normal_stop(caplog):
    client = _client_with_mocked_llm()
    response = _fake_success_response()
    response.response_metadata["finish_reason"] = "stop"
    client._llm.invoke.return_value = response

    with caplog.at_level("WARNING", logger="argus.seams"):
        client.complete("system", "user")

    assert "finish_reason=length" not in caplog.text


def test_default_groq_client_uses_the_configured_key(monkeypatch):
    monkeypatch.setattr("argus.seams.settings.GROQ_API_KEY", "configured-key")

    client = default_groq_client("openai/gpt-oss-20b", 0.1, 50)

    assert isinstance(client, GroqLLMClient)
    assert client._model == "openai/gpt-oss-20b"
    assert client._max_tokens == 50
    assert client._llm.groq_api_key.get_secret_value() == "configured-key"


def test_default_groq_client_warns_but_builds_without_a_key(monkeypatch, caplog):
    monkeypatch.setattr("argus.seams.settings.GROQ_API_KEY", "")

    with caplog.at_level(logging.WARNING, logger="argus.seams"):
        client = default_groq_client("openai/gpt-oss-20b", 0.1, 50)

    assert isinstance(client, GroqLLMClient)
    assert "GROQ_API_KEY is not set" in caplog.text


def test_market_data_providers_report_whether_they_are_live():
    assert LiveMarketDataProvider.is_live is True
    assert FixtureMarketDataProvider.is_live is False
