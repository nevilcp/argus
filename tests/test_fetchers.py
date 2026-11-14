"""Tests for argus/data/fetchers.py."""

import json
from datetime import timedelta
from unittest import mock

import pandas as pd
import pytest

from argus.data import fetchers, newsapi_budget, retry
from argus.data.cache import DailyBarCache, TTLCache
from argus.data.newsapi_budget import NewsApiBudget
from tests.helpers.candles import daily_bars
from tests.helpers.http_errors import http_error


def _raise_os_error(*_args: object, **_kwargs: object) -> None:
    raise OSError("simulated disk error")


def test_fetch_news_returns_none_when_reservation_hits_a_disk_error(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", NewsApiBudget())
    monkeypatch.setattr(newsapi_budget, "open", _raise_os_error, raising=False)

    with mock.patch("newsapi.NewsApiClient") as mock_client_cls:
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None
    mock_client_cls.return_value.get_everything.assert_not_called()


def test_fetch_news_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "")
    assert fetchers.fetch_news("AAPL", "Apple Inc") is None


def test_fetch_news_returns_none_when_budget_exhausted(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", NewsApiBudget(daily_limit=0))

    with mock.patch("newsapi.NewsApiClient") as mock_client_cls:
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None
    mock_client_cls.return_value.get_everything.assert_not_called()


def test_fetch_news_returns_articles_at_max_page_size(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", NewsApiBudget())

    mock_client = mock.Mock()
    mock_client.get_everything.return_value = {
        "articles": [
            {
                "title": "Real headline",
                "description": "d",
                "publishedAt": "t",
                "source": {"name": "s"},
            },
            {"title": "", "description": "no title, dropped"},
        ]
    }

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result == [{"title": "Real headline", "description": "d", "published_at": "t", "source": "s"}]
    assert mock_client.get_everything.call_args.kwargs["page_size"] == 100


def test_fetch_news_reserves_one_budget_slot_per_attempt(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    budget = NewsApiBudget(daily_limit=100)
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", budget)
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)

    mock_client = mock.Mock()
    mock_client.get_everything.side_effect = [http_error(500), {"articles": []}]

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result == []
    assert mock_client.get_everything.call_count == 2
    assert json.loads(budget._persist_path().read_text())["requests_today"] == 2


def test_fetch_news_stops_retrying_once_the_budget_runs_out(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", NewsApiBudget(daily_limit=1))
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)

    mock_client = mock.Mock()
    mock_client.get_everything.side_effect = http_error(500)

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None
    assert mock_client.get_everything.call_count == 1


def test_fetch_news_returns_none_after_retries_exhausted(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_NEWSAPI_BUDGET", NewsApiBudget())
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)

    mock_client = mock.Mock()
    mock_client.get_everything.side_effect = http_error(500)

    with mock.patch("newsapi.NewsApiClient", return_value=mock_client):
        result = fetchers.fetch_news("AAPL", "Apple Inc")

    assert result is None


def test_fetch_multiple_daily_serves_second_call_from_cache(monkeypatch):
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", DailyBarCache(db_path=":memory:"))

    calls = {"n": 0}

    def fake_fetch(ticker, period):
        calls["n"] += 1
        return daily_bars(["2026-08-12"])

    monkeypatch.setattr(fetchers, "fetch_ohlcv_daily", fake_fetch)

    first = fetchers.fetch_multiple_daily(["AAPL", "MSFT"], period="1y")
    assert calls["n"] == 2
    assert set(first.keys()) == {"AAPL", "MSFT"}

    second = fetchers.fetch_multiple_daily(["AAPL", "MSFT"], period="1y")
    assert calls["n"] == 2, "second same-day call must be served entirely from cache"
    assert set(second.keys()) == {"AAPL", "MSFT"}

    fetchers.fetch_multiple_daily(["MSFT"])
    assert fetchers._DAILY_BAR_CACHE.get("AAPL") is None
    assert fetchers._DAILY_BAR_CACHE.get("MSFT") is not None


def test_daily_bar_cache_lives_under_argus_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", None)

    cache = fetchers._daily_bar_cache()

    assert cache._db_path == str(tmp_path / "daily_bars_cache.db")


def test_daily_bar_cache_is_constructed_once_and_reused(monkeypatch):
    monkeypatch.setattr(fetchers, "_DAILY_BAR_CACHE", None)

    first = fetchers._daily_bar_cache()
    second = fetchers._daily_bar_cache()

    assert first is second


class _FakeFred:
    """Stub fredapi.Fred that counts calls instead of hitting the network."""

    calls = 0

    def __init__(self, api_key):
        pass

    def get_series(self, series_id, observation_start):
        _FakeFred.calls += 1
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))


def test_fetch_fred_series_cache_evicts_the_oldest_entry_at_capacity(monkeypatch):
    monkeypatch.setattr(fetchers.settings, "FRED_API_KEY", "test-key")
    monkeypatch.setattr(fetchers, "_FRED_CACHE", TTLCache(ttl=timedelta(hours=6), max_entries=2))
    monkeypatch.setattr("fredapi.Fred", _FakeFred)
    _FakeFred.calls = 0

    fetchers.fetch_fred_series("A")
    fetchers.fetch_fred_series("B")
    fetchers.fetch_fred_series("C")
    assert _FakeFred.calls == 3

    fetchers.fetch_fred_series("A")
    assert _FakeFred.calls == 4, "A was evicted to keep the cache at its cap, so this must refetch"

    fetchers.fetch_fred_series("C")
    assert _FakeFred.calls == 4, "C is still cached"


def _raw_download(columns=("Close", "High", "Low", "Open", "Volume"), rows=3) -> pd.DataFrame:
    """Build a frame shaped like yf.download's single-ticker output."""
    index = pd.date_range("2024-01-02", periods=rows, freq="D")
    return pd.DataFrame({c: [float(i + 1) for i in range(rows)] for c in columns}, index=index)


def test_daily_and_intraday_fetchers_share_one_download_path(monkeypatch):
    download = mock.Mock(return_value=_raw_download())
    monkeypatch.setattr(fetchers.yf, "download", download)

    daily = fetchers.fetch_ohlcv_daily("AAPL", period="1y")
    intraday = fetchers.fetch_ohlcv_intraday("AAPL", interval="5m", period="2d")

    assert list(daily.columns) == ["open", "high", "low", "close", "volume"]
    assert list(intraday.columns) == list(daily.columns)
    (daily_call, intraday_call) = download.call_args_list
    assert (daily_call.kwargs["interval"], daily_call.kwargs["period"]) == ("1d", "1y")
    assert (intraday_call.kwargs["interval"], intraday_call.kwargs["period"]) == ("5m", "2d")


def test_download_raises_on_an_empty_frame(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fetchers.yf, "download", mock.Mock(return_value=pd.DataFrame()))

    with pytest.raises(retry.DataFetchError, match="interval='5m'"):
        fetchers.fetch_ohlcv_intraday("AAPL", interval="5m", period="1d")


def test_normalize_ohlcv_raises_when_the_close_column_is_missing():
    with pytest.raises(retry.DataFetchError, match="no close column"):
        fetchers._normalize_ohlcv(_raw_download(columns=("Open", "High")))


def test_fetch_vix_returns_the_latest_close(monkeypatch):
    monkeypatch.setattr(fetchers.yf, "download", mock.Mock(return_value=_raw_download(rows=4)))

    assert fetchers.fetch_vix() == 4.0


def test_fetch_vix_raises_instead_of_guessing_a_column_when_close_is_missing(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    download = mock.Mock(return_value=_raw_download(columns=("Open", "High")))
    monkeypatch.setattr(fetchers.yf, "download", download)

    with pytest.raises(retry.DataFetchError):
        fetchers.fetch_vix()
    assert download.call_args.args[0] == "^VIX"


def test_fetch_fundamentals_is_built_from_fetch_ticker_info(monkeypatch):
    info = {
        "freeCashflow": 50.0,
        "marketCap": 1000.0,
        "debtToEquity": 79.5,
        "trailingPE": 22.0,
        "sector": "Technology",
    }
    monkeypatch.setattr(fetchers, "fetch_ticker_info", mock.Mock(return_value=info))

    result = fetchers.fetch_fundamentals("AAPL")

    fetchers.fetch_ticker_info.assert_called_once_with("AAPL")
    assert result["fcf_yield"] == pytest.approx(0.05)
    assert result["p_fcf"] == pytest.approx(20.0)
    assert result["debt_to_equity"] == pytest.approx(0.795)
    assert result["pe_ttm"] == 22.0
    assert result["sector"] == "Technology"
    assert result["roe"] is None


def test_fetch_fundamentals_retries_the_info_call_once_per_attempt(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    ticker_cls = mock.Mock()
    type(ticker_cls.return_value).info = mock.PropertyMock(side_effect=http_error(500))
    monkeypatch.setattr(fetchers.yf, "Ticker", ticker_cls)

    with pytest.raises(retry.DataFetchError):
        fetchers.fetch_fundamentals("AAPL")

    assert ticker_cls.call_count == retry._RETRY_ATTEMPTS
