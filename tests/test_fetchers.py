"""Tests for argus/data/fetchers.py.

Covers retry classification, NewsAPI budget enforcement, and the "None
means unavailable, not neutral" contract for NewsAPI.
"""

from datetime import timedelta
from unittest import mock

import pandas as pd
import pytest
import requests
import yfinance.exceptions as yf_exceptions
from newsapi.newsapi_exception import NewsAPIException

from argus.data import fetchers
from argus.data.cache import DailyBarCache


def _http_error(status_code: int, headers: dict | None = None) -> requests.exceptions.HTTPError:
    """Builds an HTTPError whose .response mimics a real requests response."""
    response = mock.Mock(status_code=status_code, headers=headers or {})
    return requests.exceptions.HTTPError(response=response)


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


def test_is_retryable_yfinance_rate_limit():
    """A yfinance rate-limit error is retryable."""
    assert fetchers._is_retryable(yf_exceptions.YFRateLimitError()) is True


def test_is_retryable_newsapi_rate_limited_code():
    """A NewsAPIException whose body code is rateLimited is retryable."""
    exc = NewsAPIException({"status": "error", "code": "rateLimited", "message": "slow down"})
    assert fetchers._is_retryable(exc) is True


def test_is_retryable_newsapi_non_rate_limit_code():
    """A NewsAPIException for any other cause (e.g. a bad key) is terminal."""
    exc = NewsAPIException({"status": "error", "code": "apiKeyInvalid", "message": "bad key"})
    assert fetchers._is_retryable(exc) is False


def test_is_retryable_timeout_and_connection_errors():
    """Network timeouts and connection failures are retryable."""
    assert fetchers._is_retryable(requests.exceptions.Timeout()) is True
    assert fetchers._is_retryable(requests.exceptions.ConnectionError()) is True


@pytest.mark.parametrize(
    "status_code,expected",
    [(401, False), (403, False), (404, False), (429, True), (423, True), (500, True), (503, True)],
)
def test_is_retryable_http_status_codes(status_code, expected):
    """A retry would fail identically on 401/403/404; 429/423/5xx are worth retrying."""
    assert fetchers._is_retryable(_http_error(status_code)) is expected


def test_is_retryable_defaults_true_for_unclassified_exceptions():
    """An exception outside the known taxonomy defaults to retryable, matching prior behavior."""
    assert fetchers._is_retryable(ValueError("unexpected")) is True


# ---------------------------------------------------------------------------
# _retry_after_seconds
# ---------------------------------------------------------------------------


def test_retry_after_seconds_reads_header():
    """A Retry-After header is parsed as seconds."""
    assert fetchers._retry_after_seconds(_http_error(429, {"Retry-After": "5.5"})) == 5.5


def test_retry_after_seconds_missing_header_returns_none():
    """No Retry-After header means no override."""
    assert fetchers._retry_after_seconds(_http_error(429)) is None


def test_retry_after_seconds_no_response_returns_none():
    """An exception with no .response attribute at all is handled, not raised."""
    assert fetchers._retry_after_seconds(ValueError("boom")) is None


def test_retry_after_seconds_malformed_value_returns_none():
    """A non-numeric Retry-After value is treated as absent rather than raising."""
    assert fetchers._retry_after_seconds(_http_error(429, {"Retry-After": "soon"})) is None


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------


def test_with_retry_does_not_retry_terminal_errors():
    """A 401 fails on the first attempt instead of burning the full retry budget."""
    calls = {"n": 0}

    @fetchers._with_retry
    def flaky():
        calls["n"] += 1
        raise _http_error(401)

    with pytest.raises(fetchers.DataFetchError):
        flaky()
    assert calls["n"] == 1


def test_with_retry_retries_retryable_errors_until_exhausted(monkeypatch):
    """A persistently retryable error is retried exactly _RETRY_ATTEMPTS times."""
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    @fetchers._with_retry
    def flaky():
        calls["n"] += 1
        raise requests.exceptions.Timeout("timed out")

    with pytest.raises(fetchers.DataFetchError):
        flaky()
    assert calls["n"] == fetchers._RETRY_ATTEMPTS


def test_with_retry_honors_retry_after_header(monkeypatch):
    """The Retry-After header value is used verbatim as the sleep duration."""
    sleeps = []
    monkeypatch.setattr(fetchers.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    @fetchers._with_retry
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(429, {"Retry-After": "9"})
        return "ok"

    assert flaky() == "ok"
    assert sleeps == [9.0]


def test_with_retry_succeeds_without_sleeping_on_first_attempt(monkeypatch):
    """A call that succeeds immediately never sleeps."""
    monkeypatch.setattr(
        fetchers.time, "sleep", mock.Mock(side_effect=AssertionError("should not sleep"))
    )

    @fetchers._with_retry
    def ok():
        return "value"

    assert ok() == "value"


# ---------------------------------------------------------------------------
# NewsAPI budget + fetch_news
# ---------------------------------------------------------------------------


def test_newsapi_budget_exhausts_after_daily_limit():
    """try_reserve returns False once the daily limit is spent."""
    budget = fetchers._NewsApiBudget(daily_limit=2)
    assert budget.try_reserve() is True
    assert budget.try_reserve() is True
    assert budget.try_reserve() is False


def test_newsapi_budget_resets_on_new_day():
    """Rolling over to a new UTC date resets the counter."""
    budget = fetchers._NewsApiBudget(daily_limit=1)
    assert budget.try_reserve() is True
    assert budget.try_reserve() is False

    budget._date = budget._date - timedelta(days=1)
    assert budget.try_reserve() is True


def test_newsapi_budget_survives_a_fresh_process():
    """A fresh _NewsApiBudget resumes today's count instead of restarting at zero.

    This is the Actions collector's cold-start case: a second process
    sharing the same ARGUS_DATA_DIR must not silently reopen the daily
    budget.
    """
    first = fetchers._NewsApiBudget(daily_limit=2)
    assert first.try_reserve() is True

    second = fetchers._NewsApiBudget(daily_limit=2)
    assert second.try_reserve() is True
    assert second.try_reserve() is False


def test_newsapi_budget_ignores_a_stale_persisted_date():
    """A persisted count from a prior UTC day is not carried into a new process's day."""
    stale = fetchers._NewsApiBudget(daily_limit=2)
    assert stale.try_reserve() is True
    stale._date = stale._date - timedelta(days=1)
    stale._save_to_disk()

    fresh = fetchers._NewsApiBudget(daily_limit=2)
    assert fresh.try_reserve() is True
    assert fresh.try_reserve() is True
    assert fresh.try_reserve() is False


def test_fetch_news_returns_none_without_api_key(monkeypatch):
    """No NEWSAPI_KEY means no request is even attempted."""
    monkeypatch.setattr(fetchers.settings, "newsapi_key", "")
    assert fetchers.fetch_news("AAPL", "Apple Inc") is None


def test_fetch_news_returns_none_when_budget_exhausted(monkeypatch):
    """An exhausted daily budget returns None distinguishable from an empty result."""
    monkeypatch.setattr(fetchers.settings, "newsapi_key", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", fetchers._NewsApiBudget(daily_limit=0))

    with mock.patch("newsapi.NewsApiClient") as mock_client_cls:
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None
    mock_client_cls.assert_not_called()


def test_fetch_news_returns_articles_at_max_page_size(monkeypatch):
    """A successful fetch requests NewsAPI's maximum page size and filters titleless articles."""
    monkeypatch.setattr(fetchers.settings, "newsapi_key", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", fetchers._NewsApiBudget())

    mock_client = mock.Mock()
    mock_client.get_everything.return_value = {
        "articles": [
            {"title": "Real headline", "description": "d", "publishedAt": "t", "source": {"name": "s"}},
            {"title": "", "description": "no title, dropped"},
        ]
    }

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result == [{"title": "Real headline", "description": "d", "published_at": "t", "source": "s"}]
    assert mock_client.get_everything.call_args.kwargs["page_size"] == 100


def test_fetch_news_returns_none_after_retries_exhausted(monkeypatch):
    """A retryable failure that never clears returns None, not an empty list read as success."""
    monkeypatch.setattr(fetchers.settings, "newsapi_key", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", fetchers._NewsApiBudget())
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)

    mock_client = mock.Mock()
    mock_client.get_everything.side_effect = _http_error(500)

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None


# ---------------------------------------------------------------------------
# fetch_multiple_daily caching
# ---------------------------------------------------------------------------


def _stub_daily_frame() -> pd.DataFrame:
    """Minimal single-row daily OHLCV frame for cache round-trip tests."""
    return pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100.0]},
        index=pd.to_datetime(["2026-08-12"]),
    )


def test_fetch_multiple_daily_serves_second_call_from_cache(monkeypatch):
    """A ticker already refreshed today is not refetched on a second call the same day."""
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", DailyBarCache(db_path=":memory:"))

    calls = {"n": 0}

    def fake_fetch(ticker, period):
        calls["n"] += 1
        return _stub_daily_frame()

    monkeypatch.setattr(fetchers, "fetch_ohlcv_daily", fake_fetch)

    first = fetchers.fetch_multiple_daily(["AAPL", "MSFT"], period="1y")
    assert calls["n"] == 2
    assert set(first.keys()) == {"AAPL", "MSFT"}

    second = fetchers.fetch_multiple_daily(["AAPL", "MSFT"], period="1y")
    assert calls["n"] == 2, "second same-day call must be served entirely from cache"
    assert set(second.keys()) == {"AAPL", "MSFT"}


def test_daily_bar_cache_lives_under_argus_data_dir(monkeypatch, tmp_path):
    """The lazily-built module singleton lands under settings.ARGUS_DATA_DIR, not the cwd.

    The old cwd-relative default landed outside docker-compose's mounted
    volumes and was destroyed on every container restart.
    """
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", None)

    cache = fetchers._daily_bar_cache()

    assert cache._db_path == str(tmp_path / "daily_bars_cache.db")


def test_daily_bar_cache_is_constructed_once_and_reused(monkeypatch):
    """A second call to `_daily_bar_cache()` returns the same instance rather than rebuilding it."""
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", None)

    first = fetchers._daily_bar_cache()
    second = fetchers._daily_bar_cache()

    assert first is second
