"""Builders and patch targets shared by the kill-switch test modules."""

from __future__ import annotations

from datetime import datetime
from unittest import mock

from argus.risk.kill_switch import KillSwitch

FETCH_VIX = "argus.data.fetchers.fetch_vix"


def make_ks(risk_tolerance: str = "MODERATE", inception: float = 100_000.0) -> KillSwitch:
    """Builds a KillSwitch with inception state set directly, bypassing start()'s thread.

    Args:
        risk_tolerance: Risk tier string.
        inception: Seed value for inception/current/high-water-mark.

    Returns:
        A KillSwitch ready for _check() to be driven directly.
    """
    ks = KillSwitch(risk_tolerance)
    ks._portfolio_inception_value = inception
    ks._current_portfolio_value = inception
    ks._high_water_mark = inception
    return ks


def expire_vix_cache(ks: KillSwitch) -> None:
    """Forces the next _fetch_vix_cached() call to hit fetch_vix rather than the cache.

    The VIX reading cache (TTL KILL_SWITCH.vix_cache_ttl_seconds) means back-to-back
    _check() calls in a test would otherwise reuse the first mocked reading
    instead of observing the next one — call this between _check() calls
    that mock a different fetch_vix outcome.
    """
    ks._vix_cache_value = None
    ks._vix_cache_fetched_at = None


def check_at_vix(ks: KillSwitch, vix: float) -> None:
    """Drives one _check() against a mocked VIX reading."""
    with mock.patch(FETCH_VIX, return_value=vix):
        ks._check()


def seed_persisted_halt() -> KillSwitch:
    """Writes a halt dump into the cwd, as a prior halted process would have left behind.

    Returns:
        The seed KillSwitch that wrote it, for comparing against the restored one.
    """
    seed = make_ks("MODERATE", inception=100_000.0)
    seed._current_portfolio_value = 80_000.0
    seed._halt_reason = "Drawdown 20.0% >= 12% limit (MODERATE)"
    seed._halt_time = datetime.now()
    seed._persist_halt_event(seed._halt_reason, seed._halt_time, 0.20, 15.0)
    return seed
