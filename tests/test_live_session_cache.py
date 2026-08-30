"""
tests/test_live_session_cache.py

Tests for LiveSessionCache (issue #78): the freshness gate for MFT session
states, extracted from api/main.py so each clause can be verified by calling
a function with an explicit clock instead of standing up an HTTP client and
monkeypatching two module globals.

These exercise LiveSessionCache directly — no FastAPI app, no
MFTDataPipeline instance. publish() takes the tracked universe as a plain
argument instead of holding a pipeline reference. Publication age and bar
age are measured against one ET-aware clock (see the module docstring's note
on clock unification).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from argus.data.live_session_cache import (
    LiveSessionCache,
    max_bar_age_seconds,
    session_state_ttl_seconds,
)
from argus.data.pipeline import _FETCH_INTERVAL
from argus.params import SYSTEM

_ET = ZoneInfo("America/New_York")


def _publish(
    cache: LiveSessionCache,
    ticker: str,
    *,
    now: datetime,
    bar_age_seconds: float,
    write_age_seconds: float = 0.0,
    tracked_universe=None,
) -> None:
    """Publishes one ticker with independently controlled bar age and write age."""
    state = {"timestamp": (now - timedelta(seconds=bar_age_seconds)).isoformat()}
    published_at = now - timedelta(seconds=write_age_seconds)
    cache.publish(
        {ticker: state},
        tracked_universe if tracked_universe is not None else [ticker],
        now=published_at,
    )


def test_session_state_ttl_seconds_tolerates_one_missed_sweep():
    """The write-age TTL covers two fetch sweeps (MFT-14), not one decision cycle's worth."""
    assert session_state_ttl_seconds() == 2 * _FETCH_INTERVAL + SYSTEM.freshness_margin_seconds


def test_max_bar_age_seconds_scales_with_interval():
    """The bar-age budget grows with the candle interval, matching the issue's 480s-at-1m figure."""
    assert max_bar_age_seconds(1) == _FETCH_INTERVAL + 2 * 1 * 60 + SYSTEM.freshness_margin_seconds
    assert max_bar_age_seconds(1) == 480
    assert max_bar_age_seconds(5) > max_bar_age_seconds(1)


def test_admit_reports_a_never_published_ticker_as_absent():
    """A ticker with no cache entry at all is absent, not stalled or stale."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)

    result = cache.admit(["AAPL"], now)

    assert result.absent == ["AAPL"]
    assert result.stalled == []
    assert result.stale == []
    assert result.admitted == {}


def test_admit_reports_a_ticker_not_refreshed_recently_as_stalled():
    """A cache entry whose publication time exceeds the TTL is stalled, even with a fresh bar."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=5, write_age_seconds=10_000)

    result = cache.admit(["AAPL"], now)

    assert result.stalled == ["AAPL"]


def test_admit_reports_an_old_bar_as_stale():
    """A bar older than max_bar_age_seconds is stale, independent of how recently it was written."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=9 * 24 * 3600, write_age_seconds=5)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_fails_closed_on_a_malformed_bar_timestamp():
    """API-11: a non-ISO timestamp is treated as stale rather than raising."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": "not-a-timestamp"}}, ["AAPL"], now=now)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_fails_closed_on_a_naive_bar_timestamp():
    """API-11: a tz-less timestamp is treated as stale rather than raising a TypeError."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": datetime.now().isoformat()}}, ["AAPL"], now=now)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_passes_a_bar_one_full_fetch_interval_old():
    """MFT-14: a bar as old as one full _FETCH_INTERVAL still clears max_bar_age_seconds."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=_FETCH_INTERVAL, write_age_seconds=5)

    result = cache.admit(["AAPL"], now)

    assert "AAPL" in result.admitted
    assert result.stale == []
    assert result.stalled == []


def test_publish_evicts_tickers_no_longer_in_the_tracked_universe():
    """A ticker dropped from the tracked universe is dropped from the cache on the next publish."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "MSFT", now=now, bar_age_seconds=5, tracked_universe=["MSFT"])

    _publish(cache, "AAPL", now=now, bar_age_seconds=5)

    assert len(cache) == 1
    result = cache.admit(["MSFT", "AAPL"], now)
    assert result.absent == ["MSFT"]
    assert "AAPL" in result.admitted


def test_ages_reports_publication_and_bar_age_for_everything_held():
    """ages() reports numeric ages for every cached ticker, independent of whether it would pass admission."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=30, write_age_seconds=10)

    cache_age_seconds, bar_age_seconds = cache.ages(now)

    assert cache_age_seconds["AAPL"] == pytest.approx(10)
    assert bar_age_seconds["AAPL"] == pytest.approx(30)


def test_publish_defaults_to_the_same_clock_admit_measures_against():
    """A just-published entry with a fresh bar clears admit() using only each side's default clock."""
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"])

    result = cache.admit(["AAPL"], datetime.now(_ET))

    assert "AAPL" in result.admitted


def test_ages_measures_publication_age_correctly_across_a_dst_transition():
    """Two aware datetimes sharing one ZoneInfo instance must still yield real elapsed seconds.

    Plain `datetime` subtraction between operands with the same tzinfo
    object ignores tzinfo and compares wall-clock fields directly, which is
    wrong by one hour across a DST transition; ages()/admit() must use
    timestamp()-based arithmetic instead.
    """
    cache = LiveSessionCache(interval_minutes=1)
    published_at = datetime(2024, 3, 9, 12, 0, tzinfo=_ET)  # EST, before spring-forward
    now = datetime(2024, 3, 11, 12, 0, tzinfo=_ET)  # 2 wall-clock days later, after it
    cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"], now=published_at)

    cache_age_seconds, _ = cache.ages(now)

    # Real elapsed time is 47h (one DST hour was skipped), not the 48h a
    # same-tzinfo subtraction would report
    assert cache_age_seconds["AAPL"] == pytest.approx(47 * 3600)
