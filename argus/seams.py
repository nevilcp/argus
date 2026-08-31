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
  - `LLMClient` / `GroqLLMClient` (wraps ChatGroq construction + invocation,
    and is the sole choke point for governor rate-limiting and header-driven
    limit observation; classifies which failures are worth retrying and
    raises RetryableTransportError for those, but the retry loop itself
    lives in argus/structured_output.py's decode() — see its own docstring;
    also answers `remaining_capacity()` by delegating to the governor, so a
    caller can back off before spending a call it would only lose)
    / `FixtureLLMClient` (serves pre-captured response text from
    tests/fixtures/llm_responses/, never touches the governor or the network,
    and answers `remaining_capacity()` from a value fixed at construction).

Agents accept these via constructor injection with a real default, so
production wiring (graph.py's `build_graph()`) is unchanged unless a
caller explicitly passes a fixture-backed instance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Protocol

import groq
import httpx
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from argus.data import fetchers
from argus.orchestration.governor import estimate_tokens, governor

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

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> Optional[list[dict]]:
        """Returns recent news articles for a ticker, or None if unavailable."""
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

    def news(self, ticker: str, company_name: str, days_back: int = 7) -> Optional[list[dict]]:
        """Delegates to fetchers.fetch_news."""
        return fetchers.fetch_news(ticker, company_name, days_back=days_back)

    def vix(self) -> float:
        """Delegates to fetchers.fetch_vix."""
        return fetchers.fetch_vix()


class FixtureMarketDataProvider:
    """Serves pre-captured data from tests/fixtures/market_data/ instead of the network.

    Fixtures are plain JSON keyed by ticker, hand-captured under
    tests/fixtures/market_data/. A ticker missing from a fixture file raises
    KeyError rather than silently returning empty/zero data — a test
    exercising a ticker with no fixture should fail loudly, not pass on
    fabricated data.
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

    def vix(self) -> float:
        """Returns the cached VIX level from the macro_bundle fixture."""
        return float(self._load("macro_bundle")["vix"])


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


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
    """Signals that an LLMClient.complete() attempt failed transiently and is worth retrying.

    Raised by an LLMClient implementation, not by callers. A caller with its
    own attempt budget (argus/structured_output.py's decode()) catches this
    to retry the call, without needing to know which provider-specific
    exceptions are retryable — that classification stays with the adapter
    that raises this.

    Args:
        cause: The underlying transport exception.
        retry_after: Seconds the provider explicitly said to wait before
            retrying (e.g. a 429's Retry-After header), or None if the
            provider gave no hint and the caller should decide its own
            back-off.
    """

    def __init__(self, cause: Exception, retry_after: Optional[float] = None) -> None:
        """Wraps the underlying transport exception with an optional retry hint."""
        self.cause = cause
        self.retry_after = retry_after
        super().__init__(str(cause))


# Retryable: transient conditions where a second attempt might succeed.
_RETRYABLE_GROQ_ERRORS = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)
# Terminal: a retry would fail identically (bad key, bad request, missing resource).
_TERMINAL_GROQ_ERRORS = (
    groq.AuthenticationError,
    groq.PermissionDeniedError,
    groq.BadRequestError,
    groq.NotFoundError,
)


def _groq_retry_delay(exc: Exception) -> Optional[float]:
    """Reads the provider's own retry hint off a retryable Groq error, if it gave one.

    Only a RateLimitError carries an explicit hint (its Retry-After header,
    honored verbatim since Groq is telling us exactly how long its own limit
    takes to clear). Every other retryable error has no such hint, so the
    caller decides its own back-off. Header extraction itself is shared with
    argus/data/fetchers.py's retry loop rather than reimplemented here.

    Args:
        exc: The exception the failed attempt raised.

    Returns:
        Seconds to wait before retrying per the provider's hint, or None.
    """
    if isinstance(exc, groq.RateLimitError):
        return fetchers._retry_after_seconds(exc)
    return None


class GroqLLMClient:
    """Wraps a ChatGroq model+params combination. One instance per (model, temperature, max_tokens).

    The sole choke point for governance: token estimation, the governor's pre-flight
    wait, and rate-limit-header observation all live here rather than duplicated
    across the three agents that construct this client, so none of them can bypass
    it by accident. It also classifies which Groq failures are worth retrying and
    supplies the provider's own retry hint, but performs no retry loop itself — see
    RetryableTransportError and argus/structured_output.py's decode(), which owns
    the retry loop this client's failures feed into.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str) -> None:
        """Constructs the underlying ChatGroq client for this model+params combination.

        Args:
            model: Groq model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum completion length in tokens.
            api_key: Groq API key.
        """
        self._model = model
        self._max_tokens = max_tokens
        http_client = httpx.Client(event_hooks={"response": [self._on_response]})
        self._llm = ChatGroq(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
            http_client=http_client,
            # The groq SDK's own retry (default 2) would silently double-retry
            # underneath this client's retry loop and hide 429s from the governor.
            max_retries=0,
            # Both registered gpt-oss models are reasoning models; "low" keeps
            # reasoning-token spend down since reasoning is billed against the
            # same completion budget as max_tokens (see fundamental.py/sentiment.py's
            # max_tokens comments). reasoning_format is deliberately left unset —
            # Groq does not support it on gpt-oss, and the reasoning text already
            # comes back in a separate `reasoning` field rather than inlined into
            # content, so the existing fence-strip parsers are unaffected either way.
            reasoning_effort="low",
        )

    def _on_response(self, response: httpx.Response) -> None:
        """Feeds a Groq response's rate-limit headers to the governor.

        Reads headers only — never the body (``response.read()``), which would
        break streaming responses by forcing them to buffer.

        Args:
            response: The raw httpx.Response for a Groq API call.
        """
        governor.observe_headers(self._model, response.headers)

    def remaining_capacity(self) -> int:
        """Returns the governor's remaining rolling-window request capacity for this model."""
        return governor.get_remaining_capacity(self._model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Invokes the Groq model once and returns its response as plain text.

        Waits for governor capacity, then makes a single invocation — a
        caller with its own attempt budget (argus/structured_output.py's
        decode()) is responsible for retrying. A retryable failure (per
        _RETRYABLE_GROQ_ERRORS) is wrapped in RetryableTransportError,
        carrying the provider's retry hint if it gave one; a terminal
        failure (bad key, bad request) propagates as-is. Either way the
        governor's pre-flight reservation is released first, since no
        tokens were ever actually spent (see
        RateLimitGovernor.release_reservation).

        Returns:
            The response text, flattened from a content-block list if needed.

        Raises:
            RetryableTransportError: If the call fails with a transient error.
            The underlying groq/langchain exception: For a terminal error.
        """
        estimated_tokens = estimate_tokens(system_prompt, user_prompt, self._max_tokens)
        governor.wait_if_needed(self._model, estimated_tokens)

        try:
            response = self._llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
        except _TERMINAL_GROQ_ERRORS:
            governor.release_reservation(self._model, estimated_tokens)
            raise
        except _RETRYABLE_GROQ_ERRORS as e:
            governor.release_reservation(self._model, estimated_tokens)
            raise RetryableTransportError(e, retry_after=_groq_retry_delay(e)) from e
        except Exception:
            governor.release_reservation(self._model, estimated_tokens)
            raise

        token_usage = response.response_metadata.get("token_usage") or {}
        governor.record_usage(
            self._model,
            estimated_tokens,
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
        )

        raw = response.content
        if isinstance(raw, list):
            raw = "".join(
                item.get("text", "")
                for item in raw
                if isinstance(item, dict) and item.get("type") == "text"
            )
        elif not isinstance(raw, str):
            raw = str(raw)
        return raw


class FixtureLLMClient:
    """Returns pre-captured response text instead of calling Groq.

    `responses` maps an opaque key (agent's choosing — e.g. ticker) to the
    exact raw response text the real model previously produced, hand-captured
    under tests/fixtures/llm_responses/. `complete` is keyed by
    `key_fn(user_prompt)` so a fixture-backed agent test doesn't need to
    guess call order.
    """

    _UNLIMITED_CAPACITY = 1_000_000

    def __init__(
        self,
        responses: dict[str, str],
        key_fn: Optional[Any] = None,
        capacity: int = _UNLIMITED_CAPACITY,
    ) -> None:
        """Stores the response map and the key function used to look up responses.

        Args:
            responses: Mapping of key → pre-captured raw response text.
            key_fn: Function deriving a lookup key from the user prompt. Defaults
                to using the user prompt itself as the key.
            capacity: Fixed value `remaining_capacity()` reports. Defaults to
                effectively unlimited, so a fixture-backed test only sees
                back-pressure when it deliberately asks for a low value.
        """
        self._responses = responses
        self._key_fn = key_fn or (lambda user_prompt: user_prompt)
        self._capacity = capacity

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the pre-captured response text keyed by ``key_fn(user_prompt)``.

        Raises:
            KeyError: If no response was recorded for the derived key.
        """
        key = self._key_fn(user_prompt)
        if key not in self._responses:
            raise KeyError(f"FixtureLLMClient has no recorded response for key {key!r}")
        return self._responses[key]

    def remaining_capacity(self) -> int:
        """Returns the fixed capacity value supplied at construction."""
        return self._capacity

    @classmethod
    def from_fixture_file(cls, path: Path, key_fn: Optional[Any] = None) -> FixtureLLMClient:
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
