"""
argus/data/fetchers.py

Centralized data access layer for fetching pricing, fundamentals, and macro series.

Responsibilities:
  - Provide retried access to Yahoo Finance OHLCV and fundamentals, retrying
    only failures a second attempt could plausibly fix
  - Fetch and cache macroeconomic FRED series with a 6-hour TTL
  - Cache daily OHLCV bars per (ticker, trading day) so a sweep only pays for
    the first fetch of each trading day
  - Aggregate news headlines and social sentiment from free-tier APIs,
    enforcing NewsAPI's 100 request/day budget and returning None — never a
    fabricated empty result — when a source can't be reached

Not responsible for:
  - Data compression or indicator calculation (see data/pipeline.py)
  - Rolling intraday SQLite buffering (see data/cache.py's OHLCVBuffer)
  - Rate-limit governance for Groq LLM calls (see orchestration/governor.py)

Dependencies:
  - yfinance
  - fredapi (optional; requires FRED_API_KEY)
  - newsapi-python (optional; requires NEWSAPI_KEY)
  - pytrends (optional)
"""

from __future__ import annotations

import json
import logging
import random
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

import pandas as pd
import yfinance as yf

from argus.config import settings
from argus.data.cache import DailyBarCache

logger = logging.getLogger("argus.fetchers")

# yfinance ships its own retry/backoff but defaults it off; this is the one
# place that turns it on, per-process, for every yf.download/yf.Ticker call.
yf.config.network.retries = 2


class DataFetchError(Exception):
    """Raised when a data fetch fails after all retry attempts."""


F = TypeVar("F", bound=Callable[..., Any])

_RETRY_ATTEMPTS = 3
# Full-jitter backoff base; doubles the ceiling each attempt (base, 2*base, 4*base)
_RETRY_BASE_DELAY = 1.5


def _status_code_of(exc: Exception) -> Optional[int]:
    """Extracts an HTTP status code from an exception's attached response, if any.

    Args:
        exc: An exception potentially carrying a ``requests``/``httpx``-style
            ``.response`` attribute.

    Returns:
        The response's status code, or None if the exception carries none.
    """
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    """Classifies whether a second attempt at the same call could plausibly succeed.

    An invalid API key or a missing resource fails identically on every retry, so
    retrying it only burns time; a rate limit or a transient 5xx often clears within
    seconds. Unrecognized exceptions default to retryable, matching this module's
    prior blanket-retry behavior for anything not explicitly known to be terminal.

    Args:
        exc: The exception a fetch attempt raised.

    Returns:
        True if retrying is worthwhile; False for 401/403/404 and their
        provider-specific equivalents.
    """
    import yfinance.exceptions as yf_exceptions
    from pytrends.exceptions import TooManyRequestsError

    if isinstance(exc, (yf_exceptions.YFRateLimitError, TooManyRequestsError)):
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
    if status_code is not None:
        if status_code in (401, 403, 404):
            return False
        return status_code in (429, 423) or status_code >= 500

    return True


def _retry_after_seconds(exc: Exception) -> Optional[float]:
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
    except (ValueError, TypeError):
        return None


def _with_retry(fn: F) -> F:
    """Decorator implementing jittered back-off retries on retryable failures only.

    Wraps transient exceptions in DataFetchError after exhausting all attempts.
    Re-raises DataFetchError immediately without re-wrapping to avoid stacking.
    A terminal (non-retryable) failure raises DataFetchError on the first
    attempt rather than burning the full retry budget on a call that would
    fail identically every time.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function with retry semantics.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except DataFetchError:
                raise
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    logger.warning(
                        "%s failed with a non-retryable error (%s: %s); not retrying",
                        fn.__name__,
                        type(exc).__name__,
                        exc,
                    )
                    raise DataFetchError(
                        f"{fn.__name__} failed with a non-retryable error: {exc}"
                    ) from exc

                retry_after = _retry_after_seconds(exc)
                delay = (
                    retry_after
                    if retry_after is not None
                    else random.uniform(0.0, _RETRY_BASE_DELAY * (2 ** (attempt - 1)))
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


_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes DataFrame schemas to lower-case OHLCV columns.

    yfinance ≥ 0.2.40 returns MultiIndex columns; this function flattens them
    before column name normalization to ensure downstream consumers always
    receive a simple single-level column schema.

    Args:
        df: Raw DataFrame from yfinance.download().

    Returns:
        Filtered, sorted DataFrame with lowercase OHLCV columns and datetime index.

    Raises:
        DataFetchError: If no recognized OHLCV columns are present after normalization.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df.columns = [str(c).lower() for c in df.columns]

    available = [c for c in _OHLCV_COLUMNS if c in df.columns]
    if not available:
        raise DataFetchError(f"DataFrame has no OHLCV columns; found: {list(df.columns)}")
    df = df[available].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(subset=["close"], inplace=True)
    return df


@_with_retry
def fetch_ohlcv_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetches historical daily OHLCV candlestick data from Yahoo Finance.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        period: yfinance period string (e.g. '2y', '1y').

    Returns:
        Sorted DataFrame with lowercase OHLCV columns and a datetime index.

    Raises:
        DataFetchError: If the returned DataFrame is empty after all retries.
    """
    logger.debug("fetch_ohlcv_daily: ticker=%s period=%s", ticker, period)
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise DataFetchError(
            f"yfinance returned empty DataFrame for ticker={ticker!r} period={period!r}"
        )
    df = _normalise_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info("fetch_ohlcv_daily: %s → %d rows", ticker, len(df))
    return df


@_with_retry
def fetch_ohlcv_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetches intraday OHLCV candlestick intervals from Yahoo Finance.

    Args:
        ticker: Equity ticker symbol.
        interval: yfinance interval string (e.g. '5m', '15m', '1h').
        period: yfinance period string (e.g. '5d', '2d').

    Returns:
        Sorted DataFrame with lowercase OHLCV columns.

    Raises:
        DataFetchError: If the returned DataFrame is empty.
    """
    logger.debug(
        "fetch_ohlcv_intraday: ticker=%s interval=%s period=%s",
        ticker,
        interval,
        period,
    )
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise DataFetchError(
            f"yfinance returned empty intraday DataFrame for "
            f"ticker={ticker!r} interval={interval!r} period={period!r}"
        )
    df = _normalise_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info("fetch_ohlcv_intraday: %s [%s] → %d rows", ticker, interval, len(df))
    return df


_MULTI_DAILY_MAX_WORKERS = 5

# Daily bars change once per trading day; caching them turns every run after
# the first of the day into 0 requests for this node instead of one per ticker.
# Constructed lazily (not at import time) so it lands under whatever
# settings.ARGUS_DATA_DIR is current when first used, not whatever it was at
# import time — docker-compose mounts only data/ and chroma_db/, so the old
# cwd-relative default silently landed on the ephemeral container layer and
# was destroyed on every restart.
_DAILY_BAR_CACHE: Optional[DailyBarCache] = None


def _daily_bar_cache() -> DailyBarCache:
    """Returns the module-level DailyBarCache, constructing it under ARGUS_DATA_DIR on first use.

    Returns:
        The shared DailyBarCache instance.
    """
    global _DAILY_BAR_CACHE
    if _DAILY_BAR_CACHE is None:
        data_dir = Path(settings.ARGUS_DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        _DAILY_BAR_CACHE = DailyBarCache(db_path=str(data_dir / "daily_bars_cache.db"))
    return _DAILY_BAR_CACHE


def fetch_multiple_daily(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetches daily candlestick histories in parallel across a list of tickers.

    Tickers already refreshed today (UTC) are served from the on-disk
    DailyBarCache instead of hitting the network — daily bars only change once
    per trading day, so a same-day re-run costs 0 requests per cached ticker.

    Args:
        tickers: List of equity ticker symbols.
        period: yfinance period string applied to all tickers.

    Returns:
        Mapping of ticker → DataFrame. Failed tickers are omitted with a warning.
    """
    cache = _daily_bar_cache()
    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        cached = cache.get(ticker)
        if cached is not None:
            results[ticker] = cached
        else:
            to_fetch.append(ticker)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=_MULTI_DAILY_MAX_WORKERS) as pool:
            future_to_ticker = {
                pool.submit(fetch_ohlcv_daily, ticker, period): ticker for ticker in to_fetch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    results[ticker] = df
                    cache.put(ticker, df)
                except Exception as exc:
                    logger.warning(
                        "fetch_multiple_daily: failed for %s — %s: %s",
                        ticker,
                        type(exc).__name__,
                        exc,
                    )

    logger.info(
        "fetch_multiple_daily: %d/%d tickers fetched successfully (%d from cache)",
        len(results),
        len(tickers),
        len(tickers) - len(to_fetch),
    )
    return results


@_with_retry
def fetch_fundamentals(ticker: str) -> dict:
    """Retrieves fundamental balance sheet and income statements from Yahoo Finance.

    Note: ``returnOnAssets`` is used as a ROIC proxy because yfinance does not
    expose an invested-capital figure. This understates ROIC for capital-light
    companies but provides a consistent cross-ticker signal.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Dict of fundamental metrics with keys: pe_ttm, revenue_growth_yoy,
        operating_margin, net_margin, fcf_yield, debt_to_equity, current_ratio,
        roe, roic, sector, industry, marketCap, p_fcf, as_of_date.
    """
    logger.debug("fetch_fundamentals: ticker=%s", ticker)
    info: dict = yf.Ticker(ticker).info

    fcf = info.get("freeCashflow")
    mktcap = info.get("marketCap")
    fcf_yield = (fcf / mktcap) if fcf and mktcap else None
    p_fcf = (mktcap / fcf) if fcf and mktcap and fcf > 0 else None

    # yfinance returns debtToEquity as a percentage (e.g. 79.5 for 79.5%); normalize to decimal
    raw_de = info.get("debtToEquity")
    debt_to_equity = round(raw_de / 100.0, 4) if raw_de is not None else None

    result = {
        "pe_ttm": info.get("trailingPE"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "current_ratio": info.get("currentRatio"),
        "roe": info.get("returnOnEquity"),
        "roic": info.get("returnOnAssets"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": mktcap,
        "p_fcf": p_fcf,
        "as_of_date": date.today().isoformat(),  # noqa: DTZ011
    }
    logger.info(
        "fetch_fundamentals: %s → %d keys populated",
        ticker,
        sum(1 for v in result.values() if v is not None),
    )
    return result


@_with_retry
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches yfinance's raw ``.info`` dict for a ticker.

    Shared retried entry point for callers that only need a slice of `.info`
    (e.g. `sector`) — previously agents/risk.py called `yf.Ticker(ticker).info`
    directly, bypassing retry/back-off entirely.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Raw info dict as returned by yfinance.Ticker(ticker).info.
    """
    logger.debug("fetch_ticker_info: ticker=%s", ticker)
    return yf.Ticker(ticker).info


@_with_retry
def fetch_ticker_calendar(ticker: str) -> Any:
    """Fetches yfinance's raw ``.calendar`` object for a ticker.

    Shared retried entry point for earnings-date lookups — previously
    agents/sentiment.py called `yf.Ticker(ticker).calendar` directly,
    bypassing retry/back-off entirely.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Raw calendar object from yfinance.Ticker(ticker).calendar. Shape
        (DataFrame or dict) varies across yfinance versions; callers must
        handle both.
    """
    logger.debug("fetch_ticker_calendar: ticker=%s", ticker)
    return yf.Ticker(ticker).calendar


_FRED_CACHE: dict[str, tuple[datetime, pd.Series]] = {}
_FRED_CACHE_LOCK = Lock()
_FRED_CACHE_TTL = timedelta(hours=6)


@_with_retry
def fetch_fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    """Fetches a macroeconomic timeline from the Federal Reserve Economic Database (FRED).

    Responses are cached in-process for 6 hours to avoid redundant FRED API calls
    across multiple macro analysis cycles within a single trading session.

    Args:
        series_id: FRED series identifier (e.g. 'FEDFUNDS', 'T10Y2Y').
        start: ISO date string for the beginning of the requested history.

    Returns:
        Sorted, NaN-dropped pd.Series with a DatetimeIndex.

    Raises:
        DataFetchError: If FRED_API_KEY is not configured or the series is empty.
    """
    if not settings.fred_api_key:
        raise DataFetchError("FRED_API_KEY is not set — configure it in .env to use macro data.")

    cache_key = f"{series_id}::{start}"
    with _FRED_CACHE_LOCK:
        if cache_key in _FRED_CACHE:
            cached_at, series = _FRED_CACHE[cache_key]
            if datetime.now(timezone.utc).replace(tzinfo=None) - cached_at < _FRED_CACHE_TTL:
                logger.debug("fetch_fred_series: cache hit for %s", series_id)
                return series

    logger.debug("fetch_fred_series: fetching %s from FRED", series_id)
    from fredapi import Fred  # Lazy import; only loaded when FRED is actually used

    fred = Fred(api_key=settings.fred_api_key)
    raw: pd.Series = fred.get_series(series_id, observation_start=start)
    if raw.empty:
        raise DataFetchError(f"FRED returned empty series for {series_id!r}")

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna()

    with _FRED_CACHE_LOCK:
        _FRED_CACHE[cache_key] = (datetime.now(timezone.utc).replace(tzinfo=None), raw)

    logger.info("fetch_fred_series: %s → %d observations", series_id, len(raw))
    return raw


def fetch_macro_bundle() -> dict:
    """Gathers a consolidated set of macroeconomic indicators and interest rates.

    Returns:
        Dict with keys: fed_funds, unemployment, t10y2y, t10yie, consumer_sentiment,
        cpi_yoy, vix. Any key that fails to fetch is set to None.
    """
    bundle: dict[str, float | None] = {}

    _FRED_MAP = {
        "fed_funds": "FEDFUNDS",
        "unemployment": "UNRATE",
        "t10y2y": "T10Y2Y",
        "t10yie": "T10YIE",
        "consumer_sentiment": "UMCSENT",
    }

    for key, series_id in _FRED_MAP.items():
        try:
            s = fetch_fred_series(series_id)
            bundle[key] = float(s.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("fetch_macro_bundle: %s (%s) failed — %s", key, series_id, exc)
            bundle[key] = None

    # CPI YoY requires a 12-month pct_change transform on the raw CPI index
    try:
        cpi_raw = fetch_fred_series("CPIAUCSL")
        cpi_yoy = cpi_raw.pct_change(12) * 100
        bundle["cpi_yoy"] = float(cpi_yoy.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("fetch_macro_bundle: cpi_yoy (CPIAUCSL) failed — %s", exc)
        bundle["cpi_yoy"] = None

    try:
        bundle["vix"] = fetch_vix()
    except Exception as exc:
        logger.warning("fetch_macro_bundle: vix fetch failed — %s", exc)
        bundle["vix"] = None

    logger.info(
        "fetch_macro_bundle: populated %d/%d fields",
        sum(1 for v in bundle.values() if v is not None),
        len(bundle),
    )
    return bundle


_NEWSAPI_DAILY_LIMIT = 100
# NewsAPI's maximum page size; larger than the default 20, so each of the
# scarce daily requests returns as many articles as the tier allows.
_NEWSAPI_PAGE_SIZE = 100


class _NewsApiBudget:
    """Tracks NewsAPI's free-tier 100 request/day budget with UTC-day rollover.

    A 20-ticker sweep spends 20 of the 100 — a 5-run/day ceiling, tighter than
    anything Groq imposes — so this is checked before every request rather
    than discovered from a 426/429 response.

    Persisted under ARGUS_DATA_DIR (GOV-14): the Actions collector starts a
    fresh container on every tick, so an in-memory-only counter always read
    zero used and never actually enforced the daily ceiling in that
    deployment.
    """

    def __init__(self, daily_limit: int = _NEWSAPI_DAILY_LIMIT) -> None:
        """Initializes an empty budget tracker.

        Args:
            daily_limit: Maximum requests permitted per UTC day.
        """
        self._daily_limit = daily_limit
        self._lock = Lock()
        self._requests_today = 0
        self._date = datetime.now(timezone.utc).date()
        self._loaded_from_disk = False

    def _persist_path(self) -> Path:
        """Returns the counter's persistence path, read lazily so it honors
        whatever ARGUS_DATA_DIR is current when first used rather than at
        import time (matches _daily_bar_cache's convention above)."""
        return Path(settings.ARGUS_DATA_DIR) / "newsapi_budget.json"

    def _load_from_disk(self) -> None:
        """Restores today's persisted request count, if any, once per process."""
        try:
            raw = json.loads(self._persist_path().read_text())
            if date.fromisoformat(raw["date"]) == self._date:
                self._requests_today = raw["requests_today"]
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            pass

    def _save_to_disk(self) -> None:
        """Writes the current count to disk, tmp-file-then-replace for atomicity."""
        path = self._persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"date": self._date.isoformat(), "requests_today": self._requests_today})
        )
        tmp_path.replace(path)

    def try_reserve(self) -> bool:
        """Reserves one request against today's budget if any remains.

        Returns:
            True if a request was reserved, False if today's budget is spent.
        """
        with self._lock:
            if not self._loaded_from_disk:
                self._load_from_disk()
                self._loaded_from_disk = True
            today = datetime.now(timezone.utc).date()
            if today != self._date:
                self._requests_today = 0
                self._date = today
            if self._requests_today >= self._daily_limit:
                return False
            self._requests_today += 1
            self._save_to_disk()
            return True


_NEWSAPI_BUDGET = _NewsApiBudget()


@_with_retry
def _fetch_news_page(client: Any, query: str, from_date: str) -> list[dict]:
    """Fetches and normalizes one page of NewsAPI articles.

    Args:
        client: An initialized NewsApiClient.
        query: NewsAPI query string (e.g. "AAPL OR Apple Inc").
        from_date: ISO date string for the start of the search window.

    Returns:
        List of dicts with keys: title, description, published_at, source.
    """
    response = client.get_everything(
        q=query,
        language="en",
        from_param=from_date,
        sort_by="relevancy",
        page_size=_NEWSAPI_PAGE_SIZE,
    )
    articles = response.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "published_at": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")
    ]


def fetch_news(
    ticker: str,
    company_name: str,
    days_back: int = 7,
) -> Optional[list[dict]]:
    """Retrieves recent news articles matching a ticker from NewsAPI.

    NewsAPI's free Developer tier delays article availability by roughly 24
    hours and caps history at one month, so results are never same-day and a
    backtest window further back than that returns nothing.

    Args:
        ticker: Equity ticker symbol used in the search query.
        company_name: Human-readable company name appended to the query.
        days_back: Number of days of news history to retrieve (default 7).

    Returns:
        List of dicts with keys: title, description, published_at, source.
        None if NEWSAPI_KEY is not configured, the 100 request/day budget is
        exhausted, or the request failed after retries — distinguishable from
        an empty list, which means the request succeeded but found nothing.
    """
    if not settings.newsapi_key:
        logger.debug("fetch_news: NEWSAPI_KEY not set")
        return None

    if not _NEWSAPI_BUDGET.try_reserve():
        logger.warning(
            "fetch_news: daily budget of %d requests exhausted", _NEWSAPI_DAILY_LIMIT
        )
        return None

    try:
        from newsapi import NewsApiClient  # Lazy import; only loaded when NewsAPI is used

        client = NewsApiClient(api_key=settings.newsapi_key)
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{ticker} OR {company_name}"

        result = _fetch_news_page(client, query, from_date)
        logger.info("fetch_news: %s → %d articles", ticker, len(result))
        return result

    except DataFetchError as exc:
        logger.warning("fetch_news: %s failed — %s", ticker, exc)
        return None


def fetch_social_sentiment(ticker: str) -> dict:
    """Aggregates Google Trends signals into a normalized sentiment dict.

    Returns the same key set regardless of whether trends data is available,
    so downstream callers don't need to guard against missing keys — but
    ``social_data_available`` distinguishes a real observation from the
    neutral placeholder used when Google Trends couldn't be reached.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Dict with keys: mention_surge, volume_change_pct, top_posts,
        earnings_within_14d, social_data_available.
    """
    trends = _fetch_google_trends(ticker)

    if trends is None:
        return {
            "mention_surge": False,
            "volume_change_pct": 0.0,
            "top_posts": [],
            "earnings_within_14d": False,
            "social_data_available": False,
        }

    # Maps the 0–100 Google Trends score to a ±1 deviation from baseline (50 = neutral)
    volume_change_pct = (trends["trend_score"] - 50) / 50.0

    return {
        "mention_surge": trends["surge"],
        "volume_change_pct": round(volume_change_pct, 3),
        "top_posts": [],
        "earnings_within_14d": False,
        "social_data_available": True,
    }


@_with_retry
def _fetch_google_trends_raw(ticker: str) -> pd.DataFrame:
    """Fetches raw 7-day Google Trends interest-over-time data for a ticker.

    ``build_payload()`` internally calls pytrends' token endpoint before
    ``interest_over_time()`` hits the data endpoint, so each call here costs 2
    HTTP requests — a 20-ticker sweep is 40 Trends requests, not 20. Retries
    lean on pytrends' own built-in backoff (retries/backoff_factor) in
    addition to this module's outer retry loop.

    Args:
        ticker: Equity ticker symbol used as the Google Trends keyword.

    Returns:
        Raw interest-over-time DataFrame from pytrends.
    """
    from pytrends.request import TrendReq

    pt = TrendReq(hl="en-US", tz=360, timeout=(8, 20), retries=2, backoff_factor=0.5)
    pt.build_payload([ticker], timeframe="now 7-d")
    return pt.interest_over_time()


def _fetch_google_trends(ticker: str) -> Optional[dict]:
    """Fetches 7-day Google search interest for the ticker via pytrends.

    A score > 1.5× the 7-day average is flagged as a surge. Returns None — not
    a fabricated neutral score — when pytrends fails after retries or returns
    no data for this ticker: 50 is a real score some tickers legitimately
    report, so it must not double as "unknown."

    Args:
        ticker: Equity ticker symbol used as the Google Trends keyword.

    Returns:
        Dict with keys: trend_score (int 0–100), surge (bool). None on failure
        or when no data is available for this ticker.
    """
    try:
        df = _fetch_google_trends_raw(ticker)
    except DataFetchError as e:
        logger.warning("[GoogleTrends] Fetch failed for %s: %s", ticker, e)
        return None

    if df.empty or ticker not in df.columns:
        return None

    latest = int(df[ticker].iloc[-1])
    avg = df[ticker].mean()
    surge = avg > 0 and latest > avg * 1.5

    return {"trend_score": latest, "surge": bool(surge)}


@_with_retry
def fetch_vix() -> float:
    """Fetches the current CBOE Volatility Index (VIX) level.

    Returns:
        Most recent VIX close price as a float.

    Raises:
        DataFetchError: If yfinance returns an empty DataFrame for ^VIX.
    """
    logger.debug("fetch_vix: downloading ^VIX")
    raw = yf.download("^VIX", period="5d", interval="1d", progress=False, threads=False)
    if raw.empty:
        raise DataFetchError("yfinance returned empty DataFrame for ^VIX")

    # yfinance MultiIndex schema varies across minor versions; flatten for safety
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col = "Close" if "Close" in raw.columns else raw.columns[0]
    vix = float(raw[col].dropna().iloc[-1])
    logger.info("fetch_vix: current VIX = %.2f", vix)
    return vix
