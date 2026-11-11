"""
argus/ports.py

The two boundaries agent code depends on, as Protocols, plus the one error
type that crosses the LLM boundary.

  - `MarketDataProvider`: everything an agent needs from market/macro data.
  - `LLMClient`: one structured-text call plus the back-pressure a caller
    needs to schedule it.
  - `RetryableTransportError`: how an `LLMClient` says "this attempt failed
    transiently; retry it".

The adapters that implement these (live and fixture-backed) live in
argus/seams.py. Callers import the Protocols from here, so they depend on
the contract and not on a particular implementation.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from argus.params import MACRO


class MarketDataProvider(Protocol):
    """Everything an agent needs from external market/macro data sources.

    Method names and signatures mirror the argus/data/fetchers.py functions
    they wrap; see that module's docstrings for the semantics of each field.

    Attributes:
        is_live: True when calls reach the network. Callers use it to pace
            requests and to decide whether a process-wide cache is safe to
            share, since a fixture-backed provider is cheap and per-session.
    """

    is_live: bool

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns daily OHLCV history for a ticker."""
        ...

    def ohlcv_intraday(self, ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Returns intraday OHLCV candles at ``interval`` over ``period`` for a ticker."""
        ...

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Returns daily OHLCV history for multiple tickers, keyed by ticker."""
        ...

    def fundamentals(self, ticker: str) -> dict:
        """Returns the fundamentals ratio payload for a ticker."""
        ...

    def fred_series(self, series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
        """Returns a FRED economic time series."""
        ...

    def macro_bundle(self) -> dict:
        """Returns the current macro indicator bundle."""
        ...

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict] | None:
        """Returns recent news articles for a ticker, or None if unavailable."""
        ...

    def vix(self) -> float:
        """Returns the current CBOE VIX level."""
        ...

    def ticker_info(self, ticker: str) -> dict:
        """Returns yfinance's raw ``.info`` dict for a ticker (e.g. GICS sector)."""
        ...

    def ticker_calendar(self, ticker: str) -> Any:
        """Returns yfinance's raw ``.calendar`` object for a ticker (e.g. earnings date)."""
        ...


class LLMClient(Protocol):
    """A structured-text call plus the back-pressure a caller needs to schedule it.

    Callers keep owning response parsing (JSON extraction, markdown-fence
    stripping, schema validation) — this seam only replaces "how do I get an
    LLM to answer this prompt," not "what does the answer mean." Back-pressure
    is part of the same contract: an agent deciding whether a call is worth
    making asks its own port, not a global.

    An implementation makes exactly one invocation per call — it does not
    loop. A transient failure worth a caller-side retry (see
    argus/structured_output.py's decode()) must be raised as
    RetryableTransportError; anything else propagates as-is and is treated
    as terminal.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a system + user prompt to the LLM and returns the raw response text."""
        ...

    def remaining_capacity(self) -> int:
        """Returns how many more calls this client can make right now without waiting."""
        ...


class RetryableTransportError(Exception):
    """An LLMClient.complete() attempt failed transiently and is worth retrying.

    Raised by an LLMClient implementation, not by callers. A caller with its
    own attempt budget (argus/structured_output.py's decode()) catches this
    to retry the call, without needing to know which provider-specific
    exceptions are retryable — that classification stays with the adapter
    that raises this.

    Attributes:
        cause: The underlying transport exception.
        retry_after: Seconds the provider explicitly said to wait before
            retrying (e.g. a 429's Retry-After header), or None if the
            provider gave no hint and the caller should decide its own
            back-off.
    """

    def __init__(self, cause: Exception, retry_after: float | None = None) -> None:
        """Wraps the underlying transport exception with an optional retry hint.

        Args:
            cause: The underlying transport exception.
            retry_after: Seconds the provider explicitly said to wait before
                retrying, or None if the provider gave no hint.
        """
        self.cause = cause
        self.retry_after = retry_after
        super().__init__(str(cause))
