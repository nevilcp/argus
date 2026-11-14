"""Live feature admission with explicit publication and market clocks."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from argus.data.live_session_cache import (
    LiveSessionCache,
    max_bar_age_seconds,
    session_state_ttl_seconds,
)
from argus.params import SYSTEM

_ET = ZoneInfo("America/New_York")


def _publish(
    cache: LiveSessionCache,
    ticker: str,
    *,
    now: datetime,
    bar_age_seconds: float,
    write_age_seconds: float = 0.0,
    tracked_universe: list[str] | None = None,
) -> None:
    """Publish one ticker with independently controlled bar age and write age."""
    state = {"timestamp": (now - timedelta(seconds=bar_age_seconds)).isoformat()}
    universe = tracked_universe if tracked_universe is not None else [ticker]
    cache.publish({ticker: state}, universe, now=now - timedelta(seconds=write_age_seconds))


def test_session_state_ttl_seconds_tolerates_one_missed_sweep():
    assert session_state_ttl_seconds() == 2 * SYSTEM.mft_fetch_interval_seconds + SYSTEM.freshness_margin_seconds


def test_max_bar_age_seconds_scales_with_interval():
    assert max_bar_age_seconds(1) == SYSTEM.mft_fetch_interval_seconds + 2 * 1 * 60 + SYSTEM.freshness_margin_seconds
    assert max_bar_age_seconds(1) == 480
    assert max_bar_age_seconds(5) > max_bar_age_seconds(1)


def test_admit_reports_a_never_published_ticker_as_absent():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)

    result = cache.admit(["AAPL"], now)

    assert result.absent == ["AAPL"]
    assert result.stalled == []
    assert result.stale == []
    assert result.admitted == {}


def test_admit_reports_a_ticker_not_refreshed_recently_as_stalled():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=5, write_age_seconds=10_000)

    result = cache.admit(["AAPL"], now)

    assert result.stalled == ["AAPL"]


def test_admit_reports_an_old_bar_as_stale():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=9 * 24 * 3600, write_age_seconds=5)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_fails_closed_on_a_malformed_bar_timestamp():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": "not-a-timestamp"}}, ["AAPL"], now=now)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_fails_closed_on_a_naive_bar_timestamp():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": datetime.now().isoformat()}}, ["AAPL"], now=now)

    result = cache.admit(["AAPL"], now)

    assert result.stale == ["AAPL"]


def test_admit_passes_a_bar_one_full_fetch_interval_old():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(
        cache,
        "AAPL",
        now=now,
        bar_age_seconds=SYSTEM.mft_fetch_interval_seconds,
        write_age_seconds=5,
    )

    result = cache.admit(["AAPL"], now)

    assert "AAPL" in result.admitted
    assert result.stale == []
    assert result.stalled == []


def test_publish_evicts_tickers_no_longer_in_the_tracked_universe():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "MSFT", now=now, bar_age_seconds=5, tracked_universe=["MSFT"])

    _publish(cache, "AAPL", now=now, bar_age_seconds=5)

    assert len(cache) == 1
    result = cache.admit(["MSFT", "AAPL"], now)
    assert result.absent == ["MSFT"]
    assert "AAPL" in result.admitted


def test_ages_reports_publication_and_bar_age_for_everything_held():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    _publish(cache, "AAPL", now=now, bar_age_seconds=30, write_age_seconds=10)

    cache_age_seconds, bar_age_seconds = cache.ages(now)

    assert cache_age_seconds["AAPL"] == pytest.approx(10)
    assert bar_age_seconds["AAPL"] == pytest.approx(30)


def test_publish_defaults_to_the_same_clock_admit_measures_against():
    cache = LiveSessionCache(interval_minutes=1)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"])

    result = cache.admit(["AAPL"], datetime.now(_ET))

    assert "AAPL" in result.admitted


def test_ages_measures_publication_age_correctly_across_a_dst_transition():
    cache = LiveSessionCache(interval_minutes=1)
    published_at = datetime(2024, 3, 9, 12, 0, tzinfo=_ET)
    now = datetime(2024, 3, 11, 12, 0, tzinfo=_ET)
    cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"], now=published_at)

    cache_age_seconds, _ = cache.ages(now)

    # Measure 47 elapsed hours across spring-forward instead of 48 wall-clock hours.
    assert cache_age_seconds["AAPL"] == pytest.approx(47 * 3600)


def test_publication_evicts_oldest_at_capacity_and_filters_untracked_entries():
    cache = LiveSessionCache(1, max_entries=2)
    now = datetime.now(_ET)
    cache.publish({"AAPL": {}}, ["AAPL", "MSFT", "NVDA"], now=now - timedelta(seconds=1))
    cache.publish({"MSFT": {}, "NVDA": {}, "IGNORED": {}}, ["AAPL", "MSFT", "NVDA"], now=now)
    assert len(cache) == 2
    result = cache.admit(["AAPL", "IGNORED"], now)
    assert result.absent == ["AAPL", "IGNORED"]
