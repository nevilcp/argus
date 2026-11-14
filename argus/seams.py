"""Live and fixture implementations of market-data and model transport ports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import groq
import httpx
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from argus.config import settings
from argus.data import fetchers
from argus.data.retry import retry_after_seconds
from argus.governor import estimate_tokens, governor
from argus.params import MACRO
from argus.ports import RetryableTransportError

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


class LiveMarketDataProvider:
    """Thin delegating wrapper around argus/data/fetchers.py. No behavior change."""

    is_live = True

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        return fetchers.fetch_ohlcv_daily(ticker, period=period)

    def ohlcv_intraday(self, ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        return fetchers.fetch_ohlcv_intraday(ticker, interval=interval, period=period)

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        return fetchers.fetch_multiple_daily(tickers, period=period)

    def fundamentals(self, ticker: str) -> dict:
        return fetchers.fetch_fundamentals(ticker)

    def fred_series(self, series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
        return fetchers.fetch_fred_series(series_id, start=start)

    def macro_bundle(self) -> dict:
        return fetchers.fetch_macro_bundle()

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict] | None:
        return fetchers.fetch_news(ticker, company_name, days_back=days_back)

    def vix(self) -> float:
        return fetchers.fetch_vix()

    def ticker_info(self, ticker: str) -> dict:
        return fetchers.fetch_ticker_info(ticker)

    def ticker_calendar(self, ticker: str) -> Any:
        return fetchers.fetch_ticker_calendar(ticker)


class FixtureMarketDataProvider:
    """Captured market data provider that raises for missing ticker fixtures."""

    is_live = False

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR / "market_data") -> None:
        self._dir = fixtures_dir
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> Any:
        if name not in self._cache:
            path = self._dir / f"{name}.json"
            with open(path) as f:
                self._cache[name] = json.load(f)
        return self._cache[name]

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Return cached OHLCV history from the price_history fixture."""
        record = self._load("price_history")[ticker]
        return pd.DataFrame(
            {"close": record["prices"]},
            index=pd.to_datetime(record["dates"]),
        )

    def ohlcv_intraday(self, ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Reject intraday reads because the captured fixtures contain no intraday candles."""
        raise NotImplementedError("FixtureMarketDataProvider has no intraday candles")

    def multiple_daily(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Return cached OHLCV history for each ticker present in the price_history fixture."""
        return {t: self.ohlcv_daily(t, period=period) for t in tickers if t in self._load("price_history")}

    def fundamentals(self, ticker: str) -> dict:
        """Return the cached fundamentals payload from the fundamentals fixture."""
        return dict(self._load("fundamentals")[ticker])

    def fred_series(self, series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
        """Return a cached FRED series from the fred_series fixture."""
        record = self._load("fred_series")[series_id]
        return pd.Series(record["values"], index=pd.to_datetime(record["dates"]))

    def macro_bundle(self) -> dict:
        """Return the cached macro indicator bundle from the macro_bundle fixture."""
        return dict(self._load("macro_bundle"))

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> list[dict]:
        """Return cached news articles from the news fixture, or an empty list if none."""
        return list(self._load("news").get(ticker, []))

    def vix(self) -> float:
        """Return the cached VIX level from the macro_bundle fixture."""
        return float(self._load("macro_bundle")["vix"])

    def ticker_info(self, ticker: str) -> dict:
        """Return the cached yfinance .info dict from the ticker_info fixture."""
        return dict(self._load("ticker_info")[ticker])

    def ticker_calendar(self, ticker: str) -> Any:
        """Return the cached yfinance .calendar dict from the ticker_calendar fixture."""
        return dict(self._load("ticker_calendar")[ticker])


# Classify transient provider errors here; terminal failures propagate without retries.
_RETRYABLE_GROQ_ERRORS = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)


def _groq_retry_delay(exc: Exception) -> float | None:
    """Read the provider's own retry hint off a retryable Groq error, if it gave one."""
    if isinstance(exc, groq.RateLimitError):
        return retry_after_seconds(exc)
    return None


def _response_text(content: Any) -> str:
    """Flatten an invoked model's `content` into plain response text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


class GroqLLMClient:
    """Governed single-call Groq transport with provider-specific failure classification."""

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str) -> None:
        self._model = model
        self._max_tokens = max_tokens
        http_client = httpx.Client(event_hooks={"response": [self._on_response]})
        self._llm = ChatGroq(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
            http_client=http_client,
            # Disable SDK retries so the shared decoder owns the attempt budget and the governor observes every call.
            max_retries=0,
            # Keep reasoning effort low because reasoning tokens share the completion budget.
            reasoning_effort="low",
        )

    def _on_response(self, response: httpx.Response) -> None:
        """Feed a Groq response's rate-limit headers to the governor."""
        governor.observe_headers(self._model, response.headers)

    def remaining_capacity(self) -> int:
        """Return the governor's remaining rolling-window request capacity for this model."""
        return governor.get_remaining_capacity(self._model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke the Groq model once and return its response as plain text."""
        estimated_tokens = estimate_tokens(system_prompt, user_prompt, self._max_tokens)
        governor.wait_if_needed(self._model, estimated_tokens)

        invoked = False
        try:
            response = self._llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            invoked = True
        except _RETRYABLE_GROQ_ERRORS as e:
            raise RetryableTransportError(e, retry_after=_groq_retry_delay(e)) from e
        finally:
            # Release the pre-flight reservation on failure before classifying it for the caller.
            if not invoked:
                governor.release_reservation(self._model, estimated_tokens)

        token_usage = response.response_metadata.get("token_usage") or {}
        governor.record_usage(
            self._model,
            estimated_tokens,
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
        )

        if response.response_metadata.get("finish_reason") == "length":
            logger.warning(
                "[%s] reply truncated at max_tokens=%d (finish_reason=length); "
                "reasoning tokens count against this budget",
                self._model,
                self._max_tokens,
            )

        return _response_text(response.content)


def default_groq_client(model: str, temperature: float, max_tokens: int) -> GroqLLMClient:
    """Build the Groq-backed client an LLM agent falls back to when none is injected."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        logger.warning("GROQ_API_KEY is not set - LLM calls will fail at invocation time.")
    return GroqLLMClient(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)


class FixtureLLMClient:
    """Captured model responses selected by prompt content without network access or governor use."""

    _UNLIMITED_CAPACITY = 1_000_000

    def __init__(
        self,
        responses: dict[str, str],
        key_fn: Any | None = None,
        capacity: int = _UNLIMITED_CAPACITY,
    ) -> None:
        self._responses = responses
        self._key_fn = key_fn or (lambda user_prompt: user_prompt)
        self._capacity = capacity

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the pre-captured response text keyed by `key_fn(user_prompt)`."""
        key = self._key_fn(user_prompt)
        if key not in self._responses:
            raise KeyError(f"FixtureLLMClient has no recorded response for key {key!r}")
        return self._responses[key]

    def remaining_capacity(self) -> int:
        """Return the fixed capacity value supplied at construction."""
        return self._capacity

    @classmethod
    def from_fixture_file(cls, path: Path, key_fn: Any | None = None) -> FixtureLLMClient:
        """Build a FixtureLLMClient from a JSON file of key -> response text."""
        with open(path) as f:
            responses = json.load(f)
        return cls(responses, key_fn=key_fn)
