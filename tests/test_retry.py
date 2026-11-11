"""Tests for argus/data/retry.py.

Covers retry classification, the Retry-After header, the yfinance crumb reset,
and the DataFetchError the retry decorator raises.
"""

from unittest import mock

import pytest
import requests
import yfinance as yf
import yfinance.exceptions as yf_exceptions
from newsapi.newsapi_exception import NewsAPIException

from argus.data import retry
from tests.helpers.http_errors import crumb_error, http_error

# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------


def test_is_retryable_yfinance_rate_limit():
    """A yfinance rate-limit error is retryable."""
    assert retry.is_retryable(yf_exceptions.YFRateLimitError()) is True


def test_is_retryable_newsapi_rate_limited_code():
    """A NewsAPIException whose body code is rateLimited is retryable."""
    exc = NewsAPIException({"status": "error", "code": "rateLimited", "message": "slow down"})
    assert retry.is_retryable(exc) is True


def test_is_retryable_newsapi_non_rate_limit_code():
    """A NewsAPIException for any other cause (e.g. a bad key) is terminal."""
    exc = NewsAPIException({"status": "error", "code": "apiKeyInvalid", "message": "bad key"})
    assert retry.is_retryable(exc) is False


def test_is_retryable_timeout_and_connection_errors():
    """Network timeouts and connection failures are retryable."""
    assert retry.is_retryable(requests.exceptions.Timeout()) is True
    assert retry.is_retryable(requests.exceptions.ConnectionError()) is True


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, False), (403, False), (404, False), (429, True), (423, True), (500, True), (503, True)],
)
def test_is_retryable_http_status_codes(status_code, expected):
    """A retry would fail identically on 401/403/404; 429/423/5xx are worth retrying."""
    assert retry.is_retryable(http_error(status_code)) is expected


def test_is_retryable_defaults_true_for_unclassified_exceptions():
    """An exception outside the known taxonomy defaults to retryable, matching prior behavior."""
    assert retry.is_retryable(ValueError("unexpected")) is True


# ---------------------------------------------------------------------------
# _is_crumb_handshake_failure
# ---------------------------------------------------------------------------


def test_is_crumb_handshake_failure_detects_yahoo_invalid_crumb():
    """Yahoo's actual 401 body is recognized as a crumb handshake failure."""
    assert retry._is_crumb_handshake_failure(crumb_error()) is True


def test_is_crumb_handshake_failure_false_for_credential_401():
    """A plain 401 with no crumb mention (e.g. a bad FRED/NewsAPI key) is not a crumb failure."""
    assert retry._is_crumb_handshake_failure(http_error(401)) is False


def test_is_crumb_handshake_failure_false_for_non_401_status():
    """The "crumb" text alone, on a different status code, does not qualify."""
    response = mock.Mock(status_code=500, headers={})
    exc = requests.exceptions.HTTPError("Invalid Crumb", response=response)
    assert retry._is_crumb_handshake_failure(exc) is False


# ---------------------------------------------------------------------------
# _reset_yfinance_session
# ---------------------------------------------------------------------------


def test_reset_yfinance_session_clears_in_memory_and_persisted_cookie():
    """Reset clears both the singleton's in-memory crumb/cookie and the on-disk cookie cache.

    Regression test: clearing only _cookie/_crumb isn't enough — yfinance
    reloads a persisted cookie from disk before ever attempting a new
    network handshake (see _reset_yfinance_session's docstring), so a stale
    cookie there would survive an in-memory-only reset and get reused.
    """
    data_client = yf.data.YfData()
    data_client._cookie = "stale-cookie"
    data_client._crumb = "stale-crumb"
    yf.cache.get_cookie_cache().store("curlCffi", {"stale": "cookie-jar"})

    retry._reset_yfinance_session()

    assert data_client._cookie is None
    assert data_client._crumb is None
    assert yf.cache.get_cookie_cache().lookup("curlCffi") is None


# ---------------------------------------------------------------------------
# retry_after_seconds
# ---------------------------------------------------------------------------


def test_retry_after_seconds_reads_header():
    """A Retry-After header is parsed as seconds."""
    assert retry.retry_after_seconds(http_error(429, {"Retry-After": "5.5"})) == 5.5


def test_retry_after_seconds_missing_header_returns_none():
    """No Retry-After header means no override."""
    assert retry.retry_after_seconds(http_error(429)) is None


def test_retry_after_seconds_no_response_returns_none():
    """An exception with no .response attribute at all is handled, not raised."""
    assert retry.retry_after_seconds(ValueError("boom")) is None


def test_retry_after_seconds_malformed_value_returns_none():
    """A non-numeric Retry-After value is treated as absent rather than raising."""
    assert retry.retry_after_seconds(http_error(429, {"Retry-After": "soon"})) is None


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------


def test_with_retry_does_not_retry_terminal_errors():
    """A 401 fails on the first attempt instead of burning the full retry budget."""
    calls = {"n": 0}

    @retry.with_retry
    def flaky():
        calls["n"] += 1
        raise http_error(401)

    with pytest.raises(retry.DataFetchError):
        flaky()
    assert calls["n"] == 1


def test_with_retry_retries_retryable_errors_until_exhausted(monkeypatch):
    """A persistently retryable error is retried exactly _RETRY_ATTEMPTS times."""
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
    """The Retry-After header value is used verbatim as the sleep duration."""
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
    """A call that succeeds immediately never sleeps."""
    monkeypatch.setattr(
        retry.time, "sleep", mock.Mock(side_effect=AssertionError("should not sleep"))
    )

    @retry.with_retry
    def ok():
        return "value"

    assert ok() == "value"


def test_with_retry_resets_yfinance_session_and_retries_crumb_handshake_failures(monkeypatch):
    """A crumb handshake failure resets yfinance's session and gets retried, unlike a plain 401."""
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
    """Persistent crumb failures raise YahooCrumbHandshakeError, not a bare upstream 401."""
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
