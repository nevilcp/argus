"""
argus/seams.py

Dependency-injection seam between agent logic and two external boundaries:
network market-data fetches (argus/data/fetchers.py) and Groq LLM calls
(inline `ChatGroq` construction previously duplicated in fundamental.py,
sentiment.py, and portfolio.py).

Each boundary gets one Protocol and two implementations:
  - `MarketDataProvider` / `LiveMarketDataProvider` (delegates to
    argus/data/fetchers.py, unchanged behavior) / `FixtureMarketDataProvider`
    (serves pre-captured JSON from tests/fixtures/market_data/).
  - `LLMClient` / `GroqLLMClient` (wraps ChatGroq construction + invocation)
    / `FixtureLLMClient` (serves pre-captured response text from
    tests/fixtures/llm_responses/).

Agents accept these via constructor injection with a real default, so
production wiring (graph.py's `build_graph()`) is unchanged unless a
caller explicitly passes a fixture-backed instance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from argus.data import fetchers

logger = logging.getLogger("argus.seams")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class MarketDataProvider(Protocol):
    """Everything an agent needs from external market/macro data sources.

    Method names and signatures mirror the argus/data/fetchers.py functions
    they wrap; see that module's docstrings for the semantics of each field.
    """

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns daily OHLCV history for a ticker."""
        ...

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Returns daily OHLCV history for multiple tickers, keyed by ticker."""
        ...

    def fundamentals(self, ticker: str) -> dict:
        """Returns the fundamentals ratio payload for a ticker."""
        ...

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Returns a FRED economic time series."""
        ...

    def macro_bundle(self) -> dict:
        """Returns the current macro indicator bundle."""
        ...

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict]:
        """Returns recent news articles for a ticker."""
        ...

    def social_sentiment(self, ticker: str) -> dict:
        """Returns social media sentiment metrics for a ticker."""
        ...

    def vix(self) -> float:
        """Returns the current CBOE VIX level."""
        ...


class LiveMarketDataProvider:
    """Thin delegating wrapper around argus/data/fetchers.py. No behavior change."""

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Delegates to fetchers.fetch_ohlcv_daily."""
        return fetchers.fetch_ohlcv_daily(ticker, period=period)

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Delegates to fetchers.fetch_multiple_daily."""
        return fetchers.fetch_multiple_daily(tickers, period=period)

    def fundamentals(self, ticker: str) -> dict:
        """Delegates to fetchers.fetch_fundamentals."""
        return fetchers.fetch_fundamentals(ticker)

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Delegates to fetchers.fetch_fred_series."""
        return fetchers.fetch_fred_series(series_id, start=start)

    def macro_bundle(self) -> dict:
        """Delegates to fetchers.fetch_macro_bundle."""
        return fetchers.fetch_macro_bundle()

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict]:
        """Delegates to fetchers.fetch_news."""
        return fetchers.fetch_news(ticker, company_name, days_back=days_back)

    def social_sentiment(self, ticker: str) -> dict:
        """Delegates to fetchers.fetch_social_sentiment."""
        return fetchers.fetch_social_sentiment(ticker)

    def vix(self) -> float:
        """Delegates to fetchers.fetch_vix."""
        return fetchers.fetch_vix()


class FixtureMarketDataProvider:
    """Serves pre-captured data from tests/fixtures/market_data/ instead of the network.

    Fixtures are plain JSON keyed by ticker (see scripts/capture_fixtures.py).
    A ticker missing from a fixture file raises KeyError rather than silently
    returning empty/zero data — a test exercising a ticker with no fixture
    should fail loudly, not pass on fabricated data.
    """

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR / "market_data") -> None:
        """Initializes the provider against a fixtures directory.

        Args:
            fixtures_dir: Directory containing the per-domain JSON fixture files.
        """
        self._dir = fixtures_dir
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> Any:
        if name not in self._cache:
            path = self._dir / f"{name}.json"
            with open(path) as f:
                self._cache[name] = json.load(f)
        return self._cache[name]

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns cached OHLCV history from the price_history fixture."""
        record = self._load("price_history")[ticker]
        return pd.DataFrame(
            {"close": record["prices"]},
            index=pd.to_datetime(record["dates"]),
        )

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Returns cached OHLCV history for each ticker present in the price_history fixture."""
        return {t: self.ohlcv_daily(t, period=period) for t in tickers if t in self._load("price_history")}

    def fundamentals(self, ticker: str) -> dict:
        """Returns the cached fundamentals payload from the fundamentals fixture."""
        return dict(self._load("fundamentals")[ticker])

    def fred_series(self, series_id: str, start: str = "2018-01-01") -> pd.Series:
        """Returns a cached FRED series from the fred_series fixture."""
        record = self._load("fred_series")[series_id]
        return pd.Series(record["values"], index=pd.to_datetime(record["dates"]))

    def macro_bundle(self) -> dict:
        """Returns the cached macro indicator bundle from the macro_bundle fixture."""
        return dict(self._load("macro_bundle"))

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict]:
        """Returns cached news articles from the news fixture, or an empty list if none."""
        return list(self._load("news").get(ticker, []))

    def social_sentiment(self, ticker: str) -> dict:
        """Returns cached social sentiment metrics from the social_sentiment fixture."""
        return dict(self._load("social_sentiment").get(ticker, {}))

    def vix(self) -> float:
        """Returns the cached VIX level from the macro_bundle fixture."""
        return float(self._load("macro_bundle")["vix"])


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """A single structured-text call: system + user prompt in, raw response text out.

    Callers keep owning response parsing (JSON extraction, markdown-fence
    stripping, schema validation) — this seam only replaces "how do I get an
    LLM to answer this prompt," not "what does the answer mean."
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a system + user prompt to the LLM and returns the raw response text."""
        ...


class GroqLLMClient:
    """Wraps a ChatGroq model+params combination. One instance per (model, temperature, max_tokens)."""

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str) -> None:
        """Constructs the underlying ChatGroq client for this model+params combination.

        Args:
            model: Groq model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum completion length in tokens.
            api_key: Groq API key.
        """
        self._llm = ChatGroq(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Invokes the Groq model and returns its response as plain text.

        Returns:
            The response text, flattened from a content-block list if needed.
        """
        response = self._llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(
                item.get("text", "") for item in raw if isinstance(item, dict) and item.get("type") == "text"
            )
        elif not isinstance(raw, str):
            raw = str(raw)
        return raw


class FixtureLLMClient:
    """Returns pre-captured response text instead of calling Groq.

    `responses` maps an opaque key (agent's choosing — e.g. ticker) to the
    exact raw response text the real model previously produced, captured by
    scripts/capture_fixtures.py from a real langsmith trace. `complete` is
    keyed by `key_fn(user_prompt)` so a fixture-backed agent test doesn't
    need to guess call order.
    """

    def __init__(self, responses: dict[str, str], key_fn: Optional[Any] = None) -> None:
        """Stores the response map and the key function used to look up responses.

        Args:
            responses: Mapping of key → pre-captured raw response text.
            key_fn: Function deriving a lookup key from the user prompt. Defaults
                to using the user prompt itself as the key.
        """
        self._responses = responses
        self._key_fn = key_fn or (lambda user_prompt: user_prompt)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the pre-captured response text keyed by ``key_fn(user_prompt)``.

        Raises:
            KeyError: If no response was recorded for the derived key.
        """
        key = self._key_fn(user_prompt)
        if key not in self._responses:
            raise KeyError(f"FixtureLLMClient has no recorded response for key {key!r}")
        return self._responses[key]

    @classmethod
    def from_fixture_file(cls, path: Path, key_fn: Optional[Any] = None) -> "FixtureLLMClient":
        """Builds a FixtureLLMClient from a JSON file of key → response text.

        Args:
            path: Path to the JSON fixture file.
            key_fn: Function deriving a lookup key from the user prompt.

        Returns:
            A FixtureLLMClient backed by the loaded responses.
        """
        with open(path) as f:
            responses = json.load(f)
        return cls(responses, key_fn=key_fn)
