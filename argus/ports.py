"""Market-data and model transport protocols and retryable transport errors."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from argus.params import MACRO


class MarketDataProvider(Protocol):
    """Market and economic data operations injected into agents."""

    is_live: bool

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Return daily OHLCV history for a ticker."""
        ...

    def ohlcv_intraday(self, ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Return intraday OHLCV candles at `interval` over `period` for a ticker."""
        ...

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Return daily OHLCV history for multiple tickers, keyed by ticker."""
        ...

    def fundamentals(self, ticker: str) -> dict:
        """Return the fundamentals ratio payload for a ticker."""
        ...

    def fred_series(self, series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
        """Return a FRED economic time series."""
        ...

    def macro_bundle(self) -> dict:
        """Return the current macro indicator bundle."""
        ...

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict] | None:
        """Return recent news articles for a ticker, or None if unavailable."""
        ...

    def vix(self) -> float:
        """Return the current CBOE VIX level."""
        ...

    def ticker_info(self, ticker: str) -> dict:
        """Return yfinance's raw `.info` dict for a ticker (e.g. GICS sector)."""
        ...

    def ticker_calendar(self, ticker: str) -> Any:
        """Return yfinance's raw `.calendar` object for a ticker (e.g. earnings date)."""
        ...


class LLMClient(Protocol):
    """Single model invocation with capacity reporting and explicit retryable failures."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system + user prompt to the LLM and return the raw response text."""
        ...

    def remaining_capacity(self) -> int:
        """Return how many more calls this client can make right now without waiting."""
        ...


class RetryableTransportError(Exception):
    """Transient model transport failure carrying an optional provider retry delay."""

    def __init__(self, cause: Exception, retry_after: float | None = None) -> None:
        self.cause = cause
        self.retry_after = retry_after
        super().__init__(str(cause))
