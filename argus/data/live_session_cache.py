"""Live technical feature publication, eviction, and freshness admission."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from argus.infra.clock import now_market
from argus.params import SYSTEM


def session_state_ttl_seconds() -> int:
    """Derive how old a cache entry's publication time may be before admit() treats it as stalled."""
    return 2 * SYSTEM.mft_fetch_interval_seconds + SYSTEM.freshness_margin_seconds


def max_bar_age_seconds(interval_minutes: int) -> int:
    """Derive how old a session state's own bar timestamp may be before admit() treats it as stale."""
    return SYSTEM.mft_fetch_interval_seconds + 2 * interval_minutes * 60 + SYSTEM.freshness_margin_seconds


def _bar_age_seconds(state: dict, now_et: datetime) -> float | None:
    """Return elapsed bar age, treating missing, invalid, or naive timestamps as stale."""
    try:
        return (now_et - datetime.fromisoformat(state["timestamp"])).total_seconds()
    except KeyError, TypeError, ValueError:
        return None


@dataclass
class AdmissionResult:
    """Outcome of one LiveSessionCache.admit() call."""

    admitted: dict[str, dict] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)
    stalled: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)


class LiveSessionCache:
    """Publication, eviction, admission, and age reporting for MFT session states."""

    def __init__(self, interval_minutes: int, *, max_entries: int = SYSTEM.max_tracked_tickers) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._interval_minutes = interval_minutes
        self._states: dict[str, tuple[dict, datetime]] = {}

    def __len__(self) -> int:
        """Count of tickers currently held."""
        return len(self._states)

    def publish(
        self,
        session_states: dict[str, dict],
        tracked_universe: Iterable[str],
        now: datetime | None = None,
    ) -> None:
        """Store one sweep's compressed session states and evicts untracked tickers."""
        published_at = now if now is not None else now_market()
        tracked = set(tracked_universe)
        self._states = {t: entry for t, entry in self._states.items() if t in tracked}
        for ticker, state in session_states.items():
            if ticker not in tracked:
                continue
            self._states[ticker] = (state, published_at)
        while len(self._states) > self._max_entries:
            oldest = min(self._states, key=lambda t: self._states[t][1].timestamp())
            del self._states[oldest]

    def admit(self, tickers: Iterable[str], now: datetime) -> AdmissionResult:
        """Determine which of the requested tickers may be allocated against right now."""
        result = AdmissionResult()
        ttl = session_state_ttl_seconds()
        max_bar_age = max_bar_age_seconds(self._interval_minutes)

        for ticker in tickers:
            entry = self._states.get(ticker)
            if entry is None:
                result.absent.append(ticker)
                continue

            state, published_at = entry
            # Use epoch seconds because same-ZoneInfo subtraction measures wall time across DST changes.
            if (now.timestamp() - published_at.timestamp()) > ttl:
                result.stalled.append(ticker)
                continue

            age = _bar_age_seconds(state, now)
            if age is None or age > max_bar_age:
                result.stale.append(ticker)
                continue

            result.admitted[ticker] = state

        return result

    def ages(self, now: datetime) -> tuple[dict[str, float], dict[str, float | None]]:
        """Report per-ticker publication age and bar age for everything held."""
        cache_age_seconds = {
            ticker: now.timestamp() - published_at.timestamp() for ticker, (_, published_at) in self._states.items()
        }
        bar_age_seconds = {ticker: _bar_age_seconds(state, now) for ticker, (state, _) in self._states.items()}
        return cache_age_seconds, bar_age_seconds
