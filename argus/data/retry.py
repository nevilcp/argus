"""
argus/data/retry.py

Jittered-backoff retries for the network fetchers, and the error type they raise.

Responsibilities:
  - Classify which failures a second attempt could plausibly fix
  - Retry those with full-jitter backoff, honoring a provider's Retry-After
  - Reset yfinance's session when Yahoo rejects its crumb, since retrying with
    the same crumb would just repeat the failure
  - Wrap everything else in DataFetchError so callers handle one exception type

Not responsible for:
  - What is fetched (see data/fetchers.py)
  - Retrying LLM calls (see argus/structured_output.py)

Dependencies:
  - yfinance, requests; newsapi-python is optional
"""

from __future__ import annotations

import functools
import logging
import math
import time
from collections.abc import Callable
from typing import Any

import yfinance as yf

from argus.infra.backoff import full_jitter_delay

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when a data fetch fails after all retry attempts."""


class YahooCrumbHandshakeError(DataFetchError):
    """Yahoo Finance rejected every attempt with an invalid/expired crumb.

    yfinance mints an anonymous session cookie + crumb per process and caches
    it for reuse; this is not a credential problem (there is no API key to be
    wrong), it is Yahoo's anti-bot layer invalidating that handshake — which
    then poisons every subsequent call in the process until a fresh crumb is
    minted.
    """


_RETRY_ATTEMPTS = 3
# Full-jitter backoff base; doubles the ceiling each attempt (base, 2*base, 4*base)
_RETRY_BASE_DELAY = 1.5


def _status_code_of(exc: Exception) -> int | None:
    """Extracts an HTTP status code from an exception's attached response, if any.

    Args:
        exc: An exception potentially carrying a ``requests``/``httpx``-style
            ``.response`` attribute.

    Returns:
        The response's status code, or None if the exception carries none.
    """
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def is_retryable(exc: Exception) -> bool:
    """Classifies whether a second attempt at the same call could plausibly succeed.

    An invalid API key or a missing resource fails identically on every retry, so
    retrying it only burns time; a rate limit or a transient 5xx often clears within
    seconds. Unrecognized exceptions default to retryable, so anything not
    explicitly known to be terminal gets the full retry budget.

    Args:
        exc: The exception a fetch attempt raised.

    Returns:
        True if retrying is worthwhile; False for 401/403/404 and their
        provider-specific equivalents.
    """
    import yfinance.exceptions as yf_exceptions

    if isinstance(exc, yf_exceptions.YFRateLimitError):
        return True

    try:
        from newsapi.newsapi_exception import NewsAPIException

        if isinstance(exc, NewsAPIException):
            return exc.get_code() == "rateLimited"
    except ImportError:
        pass

    import requests

    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True

    status_code = _status_code_of(exc)
    if status_code is None:
        return True
    if status_code in (401, 403, 404):
        return False
    return status_code in (429, 423) or status_code >= 500


def _is_crumb_handshake_failure(exc: Exception) -> bool:
    """Distinguishes Yahoo's "Invalid Crumb" 401 from a genuine credential 401.

    FRED and NewsAPI 401s mean a bad/missing API key and retrying is pointless
    (see is_retryable). A yfinance 401 is different: yfinance has no key at
    all, so a 401 there means Yahoo rejected the anonymous crumb it minted —
    a session handshake a fresh mint can often fix, not a permanent failure.

    Args:
        exc: The exception a fetch attempt raised.

    Returns:
        True if this looks like Yahoo's crumb rejection specifically.
    """
    return _status_code_of(exc) == 401 and "crumb" in str(exc).lower()


def _reset_yfinance_session() -> None:
    """Forces yfinance's singleton client to mint a fresh cookie and crumb next call.

    yfinance caches the crumb it mints for the life of the process; once Yahoo
    invalidates it, every subsequent call reuses the same broken crumb and
    fails identically. Clearing only the in-memory _cookie/_crumb isn't
    enough: yfinance also persists the cookie to an on-disk cache keyed
    'curlCffi' (see its data.py's _load_cookie_curlCffi), and reloads it
    before ever attempting a new network handshake — so a stale, Yahoo-
    rejected cookie would otherwise survive the reset and get reused
    immediately. Clearing both is what lets a retry attempt an actual new
    handshake instead of repeating the failed one.
    """
    data_client = yf.data.YfData()
    data_client._cookie = None
    data_client._crumb = None
    yf.cache.get_cookie_cache().store("curlCffi", None)


def retry_after_seconds(exc: Exception) -> float | None:
    """Reads a Retry-After response header off an exception, if one is attached.

    Args:
        exc: The exception a fetch attempt raised.

    Returns:
        Seconds to wait before retrying, taken verbatim from the provider's own
        Retry-After header, or None if no such header is present.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except ValueError, TypeError:
        return None


def with_retry[F: Callable[..., Any]](fn: F) -> F:
    """Decorator implementing jittered back-off retries on retryable failures only.

    Wraps transient exceptions in DataFetchError after exhausting all attempts.
    Re-raises DataFetchError immediately without re-wrapping to avoid stacking.
    A terminal (non-retryable) failure raises DataFetchError on the first
    attempt rather than burning the full retry budget on a call that would
    fail identically every time. A Yahoo crumb/cookie handshake failure (see
    _is_crumb_handshake_failure) is retried with the yfinance session reset
    first, since retrying without a reset would just repeat the same failure;
    exhausting the budget there raises YahooCrumbHandshakeError instead.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function with retry semantics.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Calls fn, retrying retryable failures and re-raising as DataFetchError once exhausted."""
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except DataFetchError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_crumb_handshake_failure(exc):
                    if attempt >= _RETRY_ATTEMPTS:
                        raise YahooCrumbHandshakeError(
                            f"{fn.__name__}: Yahoo Finance rejected every crumb/cookie "
                            f"handshake attempt ({_RETRY_ATTEMPTS}); price data "
                            f"unavailable. Last error: {exc}"
                        ) from exc
                    _reset_yfinance_session()
                    logger.warning(
                        "%s attempt %d/%d failed with a crumb/cookie handshake error; "
                        "resetting the yfinance session and retrying: %s",
                        fn.__name__,
                        attempt,
                        _RETRY_ATTEMPTS,
                        exc,
                    )
                    time.sleep(full_jitter_delay(attempt - 1, _RETRY_BASE_DELAY, math.inf))
                    continue
                if not is_retryable(exc):
                    logger.warning(
                        "%s failed with a non-retryable error (%s: %s); not retrying",
                        fn.__name__,
                        type(exc).__name__,
                        exc,
                    )
                    raise DataFetchError(
                        f"{fn.__name__} failed with a non-retryable error: {exc}"
                    ) from exc

                retry_after = retry_after_seconds(exc)
                delay = full_jitter_delay(
                    attempt - 1, _RETRY_BASE_DELAY, math.inf, retry_after=retry_after
                )
                logger.warning(
                    "%s attempt %d/%d failed (%s: %s); retrying in %.1fs",
                    fn.__name__,
                    attempt,
                    _RETRY_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
        raise DataFetchError(
            f"{fn.__name__} failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    return wrapper  # type: ignore[return-value]
