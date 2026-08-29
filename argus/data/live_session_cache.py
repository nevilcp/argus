"""
argus/data/live_session_cache.py

The live session cache: the named seam between the MFT pipeline that
produces per-ticker session states and the API gateway that allocates
against them. Owns publication, eviction, admission, and age reporting for
the freshness rule an /analyze request must clear before a cached session
state can be used (issue #78).

Responsibilities:
  - Publish a sweep's compressed session states, evicting entries for
    tickers no longer in the caller's tracked universe
  - Admit a requested set of tickers against an explicit clock, sorting
    rejections into absent / stalled / stale so a caller can report each
    distinctly
  - Report per-ticker publication and bar age for everything held, for a
    status endpoint that needs numbers rather than a pass/fail verdict
  - Own the two freshness thresholds (session_state_ttl_seconds,
    max_bar_age_seconds) beside the rule that applies them

Not responsible for:
  - Market hours — a property of the market, not of this cache's own data;
    stays a route concern (see api/main.py)
  - Fetching, buffering, or compressing candles (see argus/data/pipeline.py)
  - Re-validating session-state readiness at publish time — enforced once,
    by the pipeline's compress_all (see
    argus.schemas.signals.missing_session_state_keys)

Publication and admission both measure age against one timezone-aware
America/New_York clock. Publication used to stamp entries with naive local
time while admission measured bar age against ET — two clocks that happened
to agree only because this deployment already ran in ET, the same class of
latent bug GLOSSARY.md and this module's sibling docs already call out
elsewhere in the freshness path.

Dependencies:
  - argus.data.pipeline (_FETCH_INTERVAL, the sweep cadence both thresholds
    below are built from)
  - argus.params (SYSTEM.freshness_margin_seconds)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from argus.data.pipeline import _FETCH_INTERVAL
from argus.params import SYSTEM

_ET = ZoneInfo("America/New_York")


def session_state_ttl_seconds() -> int:
    """Derives how old a cache entry's publication time may be before admit() treats it as stalled.

    Returns:
        Budget in seconds: two fetch sweeps — tolerating one missed sweep
        before an entry counts as stalled — plus a fixed jitter margin.
    """
    return 2 * _FETCH_INTERVAL + SYSTEM.freshness_margin_seconds


def max_bar_age_seconds(interval_minutes: int) -> int:
    """Derives how old a session state's own bar timestamp may be before admit() treats it as stale.

    Distinct from session_state_ttl_seconds: that budget catches an entry
    that stopped being refreshed, this one catches a fresh publish that
    republishes an old bar (e.g. a restart replaying the persistent buffer).

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        Budget in seconds: one fetch sweep plus two candle intervals plus a
        fixed jitter margin.
    """
    return _FETCH_INTERVAL + 2 * interval_minutes * 60 + SYSTEM.freshness_margin_seconds


def _bar_age_seconds(state: dict, now_et: datetime) -> Optional[float]:
    """Computes a session state's bar age in seconds, failing closed on a bad timestamp (API-11).

    A missing, malformed, or naive ``timestamp`` must not raise out of
    admit() or ages() — it's treated as staleness instead.

    Args:
        state: A compressed technical feature dict.
        now_et: Current time, ET-aware, to measure age against.

    Returns:
        Age in seconds, or None if ``timestamp`` can't be parsed against `now_et`.
    """
    try:
        return (now_et - datetime.fromisoformat(state["timestamp"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class AdmissionResult:
    """Outcome of one LiveSessionCache.admit() call.

    Rejections are sorted into three lists rather than a single verdict:
    each means something operationally different (warming up vs. pipeline
    stalled vs. stale candles after a restart), and a caller building a
    response for each needs the grouping, not just a reason string.
    """

    admitted: dict[str, dict] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)
    stalled: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)


class LiveSessionCache:
    """Publication, eviction, admission, and age reporting for MFT session states (issue #78).

    Constructed once at startup and held in one module global by its
    caller. Holds no pipeline reference of its own — publish() takes the
    tracked universe as a plain argument, so eviction can be tested without
    a stand-in pipeline object.
    """

    def __init__(self, interval_minutes: int) -> None:
        """Creates an empty cache sized for the given candle resolution.

        Args:
            interval_minutes: Native candle resolution the tracked pipeline
                fetches at, used to size the max-bar-age threshold.
        """
        self._interval_minutes = interval_minutes
        self._states: dict[str, tuple[dict, datetime]] = {}

    def __len__(self) -> int:
        """Count of tickers currently held."""
        return len(self._states)

    def publish(
        self,
        session_states: dict[str, dict],
        tracked_universe: Iterable[str],
        now: Optional[datetime] = None,
    ) -> None:
        """Stores one sweep's compressed session states and evicts untracked tickers.

        Args:
            session_states: Mapping of ticker -> compressed technical feature
                dict, e.g. from MFTDataPipeline.compress_all().
            tracked_universe: The caller's full current tracked-ticker set.
                Every held ticker outside this set is dropped, not just ones
                absent from `session_states` — a repeated call with a
                shrinking universe is what actually evicts a dropped ticker.
            now: Publication timestamp to stamp every entry with. Defaults
                to the current ET-aware time — the same clock admit()/ages()
                measure against; tests pass an explicit value to control
                write age without reaching into private state.
        """
        published_at = now if now is not None else datetime.now(_ET)
        for ticker, state in session_states.items():
            self._states[ticker] = (state, published_at)

        tracked = set(tracked_universe)
        for ticker in list(self._states):
            if ticker not in tracked:
                del self._states[ticker]

    def admit(self, tickers: Iterable[str], now: datetime) -> AdmissionResult:
        """Determines which of the requested tickers may be allocated against right now.

        Args:
            tickers: Tickers to check.
            now: Current time, ET-aware — the one clock both publication age
                (against session_state_ttl_seconds) and bar age (against
                max_bar_age_seconds) are measured against.

        Returns:
            An AdmissionResult with every ticker sorted into exactly one of
            `admitted`, `absent`, `stalled`, or `stale`.
        """
        result = AdmissionResult()
        ttl = session_state_ttl_seconds()
        max_bar_age = max_bar_age_seconds(self._interval_minutes)

        for ticker in tickers:
            entry = self._states.get(ticker)
            if entry is None:
                result.absent.append(ticker)
                continue

            state, published_at = entry
            if (now - published_at).total_seconds() > ttl:
                result.stalled.append(ticker)
                continue

            age = _bar_age_seconds(state, now)
            if age is None or age > max_bar_age:
                result.stale.append(ticker)
                continue

            result.admitted[ticker] = state

        return result

    def ages(self, now: datetime) -> tuple[dict[str, float], dict[str, Optional[float]]]:
        """Reports per-ticker publication age and bar age for everything held.

        A different question from admit(): this answers "how old is X" for
        every cached ticker regardless of whether it would pass admission,
        for a status endpoint watching freshness degrade before it starts
        rejecting requests.

        Args:
            now: Current time, ET-aware — the same clock `publish` stamps
                entries with.

        Returns:
            Tuple of (cache_age_seconds, bar_age_seconds), each a mapping of
            ticker -> seconds; bar_age_seconds values are None where the
            entry's timestamp can't be parsed.
        """
        cache_age_seconds = {
            ticker: (now - published_at).total_seconds()
            for ticker, (_, published_at) in self._states.items()
        }
        bar_age_seconds = {
            ticker: _bar_age_seconds(state, now) for ticker, (state, _) in self._states.items()
        }
        return cache_age_seconds, bar_age_seconds
