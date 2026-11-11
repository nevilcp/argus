"""
tests/helpers/market_data.py

One configurable MarketDataProvider double for the tests.

Responsibilities:
  - Answer only the provider methods a test configures, so a test names
    exactly the data its agent reads
  - Raise NotImplementedError for any other method, so an agent reaching for
    data the test did not provide fails loudly instead of getting a default
  - Count calls per method

Not responsible for:
  - Doubles that must inherit real provider behaviour (those subclass
    FixtureMarketDataProvider or LiveMarketDataProvider directly)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

_METHODS = frozenset(
    {
        "ohlcv_daily",
        "ohlcv_intraday",
        "multiple_daily",
        "fundamentals",
        "fred_series",
        "macro_bundle",
        "news",
        "vix",
        "ticker_info",
        "ticker_calendar",
    }
)


class FakeMarketData:
    """A MarketDataProvider whose methods return what the test configured.

    Each keyword names a provider method and gives its response, one of:
      - an exception instance, which the method raises;
      - a callable, which is called with the method's arguments;
      - anything else, which the method returns. Dicts, frames and series are
        copied per call, so a caller mutating a result cannot leak into the
        next call.

    Attributes:
        calls: How many times each method was called, keyed by method name.
    """

    is_live = False

    def __init__(self, **responses: Any) -> None:
        """Stores the per-method responses.

        Raises:
            TypeError: If a keyword is not a MarketDataProvider method.
        """
        unknown = set(responses) - _METHODS
        if unknown:
            raise TypeError(f"not MarketDataProvider methods: {sorted(unknown)}")
        self._responses = responses
        self.calls: Counter[str] = Counter()

    @classmethod
    def from_closes(cls, closes: dict[str, pd.Series], **responses: Any) -> FakeMarketData:
        """Builds a provider whose `ohlcv_daily` serves a close-only frame per ticker.

        Args:
            closes: Daily close series keyed by ticker. An unknown ticker raises KeyError.
            **responses: Any further per-method responses.
        """
        return cls(
            ohlcv_daily=lambda ticker, period="2y": pd.DataFrame({"close": closes[ticker]}),
            **responses,
        )

    def _respond(self, method: str, *args: Any) -> Any:
        self.calls[method] += 1
        if method not in self._responses:
            raise NotImplementedError(f"FakeMarketData.{method} was not configured")
        response = self._responses[method]
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(*args)
        if isinstance(response, (pd.DataFrame, pd.Series, dict)):
            return response.copy()
        return response

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns the configured daily OHLCV frame."""
        return self._respond("ohlcv_daily", ticker, period)

    def ohlcv_intraday(self, ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Returns the configured intraday OHLCV frame."""
        return self._respond("ohlcv_intraday", ticker, interval, period)

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Returns the configured per-ticker daily frames."""
        return self._respond("multiple_daily", tickers, period)

    def fundamentals(self, ticker: str) -> dict:
        """Returns the configured fundamentals payload."""
        return self._respond("fundamentals", ticker)

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Returns the configured FRED series."""
        return self._respond("fred_series", series_id, start)

    def macro_bundle(self) -> dict:
        """Returns the configured macro indicator bundle."""
        return self._respond("macro_bundle")

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict] | None:
        """Returns the configured news articles (or None for an unavailable provider)."""
        return self._respond("news", ticker, company_name, days_back)

    def vix(self) -> float:
        """Returns the configured VIX level."""
        return self._respond("vix")

    def ticker_info(self, ticker: str) -> dict:
        """Returns the configured raw ticker info."""
        return self._respond("ticker_info", ticker)

    def ticker_calendar(self, ticker: str) -> Any:
        """Returns the configured raw ticker calendar."""
        return self._respond("ticker_calendar", ticker)


def ten_day_series(
    start: datetime, start_price: float = 100.0, ticker: str = "TEST"
) -> FakeMarketData:
    """Builds a provider serving a 10-day daily close series rising 1.0 per day from start_price."""
    dates = pd.date_range(start=start, periods=10, freq="D")
    closes = pd.Series([start_price + i for i in range(10)], index=dates)
    return FakeMarketData.from_closes({ticker: closes})
