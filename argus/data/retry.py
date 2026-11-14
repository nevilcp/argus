"""Classified data-fetch failures and bounded retries with jitter."""

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
    """Yahoo rejection of the cached anonymous cookie and crumb handshake."""


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.5


def _status_code_of(exc: Exception) -> int | None:
    """Extract an HTTP status code from an exception's attached response, if any."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def is_retryable(exc: Exception) -> bool:
    """Classify whether a second attempt at the same call could plausibly succeed."""
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
    """Distinguish Yahoo's "Invalid Crumb" 401 from a genuine credential 401."""
    return _status_code_of(exc) == 401 and "crumb" in str(exc).lower()


def _reset_yfinance_session() -> None:
    """Force yfinance's singleton client to mint a fresh cookie and crumb next call."""
    data_client = yf.data.YfData()
    data_client._cookie = None
    data_client._crumb = None
    yf.cache.get_cookie_cache().store("curlCffi", None)


def retry_after_seconds(exc: Exception) -> float | None:
    """Read a Retry-After response header off an exception, if one is attached."""
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
    """Retry transient fetch failures with jitter and reset rejected Yahoo sessions between attempts."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Call fn, retrying retryable failures and re-raising as DataFetchError once exhausted."""
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
                    raise DataFetchError(f"{fn.__name__} failed with a non-retryable error: {exc}") from exc

                retry_after = retry_after_seconds(exc)
                delay = full_jitter_delay(attempt - 1, _RETRY_BASE_DELAY, math.inf, retry_after=retry_after)
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
        raise DataFetchError(f"{fn.__name__} failed after {_RETRY_ATTEMPTS} attempts: {last_exc}") from last_exc

    return wrapper  # type: ignore[return-value]
