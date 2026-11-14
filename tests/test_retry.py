"""Tests for argus/data/retry.py."""

from unittest import mock

import pytest
import requests
import yfinance as yf
import yfinance.exceptions as yf_exceptions
from newsapi.newsapi_exception import NewsAPIException

from argus.data import retry
from tests.helpers.http_errors import crumb_error, http_error


def test_is_retryable_yfinance_rate_limit():
    assert retry.is_retryable(yf_exceptions.YFRateLimitError()) is True


def test_is_retryable_newsapi_rate_limited_code():
    exc = NewsAPIException({"status": "error", "code": "rateLimited", "message": "slow down"})
    assert retry.is_retryable(exc) is True


def test_is_retryable_newsapi_non_rate_limit_code():
    exc = NewsAPIException({"status": "error", "code": "apiKeyInvalid", "message": "bad key"})
    assert retry.is_retryable(exc) is False


def test_is_retryable_timeout_and_connection_errors():
    assert retry.is_retryable(requests.exceptions.Timeout()) is True
    assert retry.is_retryable(requests.exceptions.ConnectionError()) is True


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, False), (403, False), (404, False), (429, True), (423, True), (500, True), (503, True)],
)
def test_is_retryable_http_status_codes(status_code, expected):
    assert retry.is_retryable(http_error(status_code)) is expected


def test_is_retryable_defaults_true_for_unclassified_exceptions():
    assert retry.is_retryable(ValueError("unexpected")) is True


def test_is_crumb_handshake_failure_detects_yahoo_invalid_crumb():
    assert retry._is_crumb_handshake_failure(crumb_error()) is True


def test_is_crumb_handshake_failure_false_for_credential_401():
    assert retry._is_crumb_handshake_failure(http_error(401)) is False


def test_is_crumb_handshake_failure_false_for_non_401_status():
    response = mock.Mock(status_code=500, headers={})
    exc = requests.exceptions.HTTPError("Invalid Crumb", response=response)
    assert retry._is_crumb_handshake_failure(exc) is False


def test_reset_yfinance_session_clears_in_memory_and_persisted_cookie():
    data_client = yf.data.YfData()
    data_client._cookie = "stale-cookie"
    data_client._crumb = "stale-crumb"
    yf.cache.get_cookie_cache().store("curlCffi", {"stale": "cookie-jar"})

    retry._reset_yfinance_session()

    assert data_client._cookie is None
    assert data_client._crumb is None
    assert yf.cache.get_cookie_cache().lookup("curlCffi") is None


def test_retry_after_seconds_reads_header():
    assert retry.retry_after_seconds(http_error(429, {"Retry-After": "5.5"})) == 5.5


def test_retry_after_seconds_missing_header_returns_none():
    assert retry.retry_after_seconds(http_error(429)) is None


def test_retry_after_seconds_no_response_returns_none():
    assert retry.retry_after_seconds(ValueError("boom")) is None


def test_retry_after_seconds_malformed_value_returns_none():
    assert retry.retry_after_seconds(http_error(429, {"Retry-After": "soon"})) is None


def test_with_retry_does_not_retry_terminal_errors():
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        raise http_error(401)

    with pytest.raises(retry.DataFetchError):
        flaky()
    assert calls["n"] == 1


def test_with_retry_retries_retryable_errors_until_exhausted(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        raise requests.exceptions.Timeout("timed out")

    with pytest.raises(retry.DataFetchError):
        flaky()
    assert calls["n"] == retry._RETRY_ATTEMPTS


def test_with_retry_honors_retry_after_header(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise http_error(429, {"Retry-After": "9"})
        return "ok"

    assert flaky() == "ok"
    assert sleeps == [9.0]


def test_with_retry_succeeds_without_sleeping_on_first_attempt(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", mock.Mock(side_effect=AssertionError("should not sleep")))

    @retry.with_retry
    def ok():
        return "value"

    assert ok() == "value"


def test_with_retry_resets_yfinance_session_and_retries_crumb_handshake_failures(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    reset_calls = mock.Mock()
    monkeypatch.setattr(retry, "_reset_yfinance_session", reset_calls)
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise crumb_error()
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2
    reset_calls.assert_called_once()


def test_with_retry_raises_named_error_after_exhausting_crumb_handshake_retries(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    monkeypatch.setattr(retry, "_reset_yfinance_session", mock.Mock())
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        raise crumb_error()

    with pytest.raises(retry.YahooCrumbHandshakeError):
        flaky()
    assert calls["n"] == retry._RETRY_ATTEMPTS
