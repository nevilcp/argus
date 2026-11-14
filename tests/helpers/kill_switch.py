"""Builders and patch targets shared by the kill-switch test modules."""

from __future__ import annotations

from datetime import datetime
from unittest import mock

from argus.risk.kill_switch import KillSwitch

FETCH_VIX = "argus.data.fetchers.fetch_vix"


def make_ks(risk_tolerance: str = "MODERATE", inception: float = 100_000.0) -> KillSwitch:
    """Build a KillSwitch with inception state set directly, bypassing start()'s thread."""
    ks = KillSwitch(risk_tolerance)
    ks._portfolio_inception_value = inception
    ks._current_portfolio_value = inception
    ks._high_water_mark = inception
    return ks


def expire_vix_cache(ks: KillSwitch) -> None:
    """Force the next _fetch_vix_cached() call to hit fetch_vix rather than the cache."""
    ks._vix_cache_value = None
    ks._vix_cache_fetched_at = None


def check_at_vix(ks: KillSwitch, vix: float) -> None:
    """Drive one _check() against a mocked VIX reading."""
    with mock.patch(FETCH_VIX, return_value=vix):
        ks._check()


def seed_persisted_halt() -> KillSwitch:
    """Write a halt dump into the cwd, as a prior halted process would have left behind."""
    seed = make_ks("MODERATE", inception=100_000.0)
    seed._current_portfolio_value = 80_000.0
    seed._halt_reason = "Drawdown 20.0% >= 12% limit (MODERATE)"
    seed._halt_time = datetime.now()
    seed._persist_halt_event(seed._halt_reason, seed._halt_time, 0.20, 15.0)
    return seed
