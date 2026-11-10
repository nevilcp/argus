"""
tests/test_kill_switch_vix.py

Tests for argus/risk/kill_switch.py's VIX blackout gate: readings, caching,
failure handling, and the VIX surfaced on status.

fetch_vix is mocked throughout; _check() is driven directly, so no test sleeps
on a real check interval.
"""

from __future__ import annotations

from unittest import mock

import pytest

from argus.risk.kill_switch import KillSwitch
from argus.seams import FixtureMarketDataProvider
from tests.helpers.kill_switch import FETCH_VIX, check_at_vix, expire_vix_cache, make_ks


def test_vix_blackout_sets_and_clears_on_genuine_readings():
    """The blackout sets on a high reading and clears on a later genuine low reading."""
    ks = make_ks("MODERATE")
    check_at_vix(ks, 40.0)
    assert not ks.new_positions_allowed

    expire_vix_cache(ks)
    check_at_vix(ks, 15.0)
    assert ks.new_positions_allowed


def test_vix_reading_is_cached_within_the_ttl():
    """A second _check() inside the TTL reuses the cached reading, not a new fetch."""
    ks = make_ks("MODERATE")
    with mock.patch(FETCH_VIX, return_value=15.0) as mock_vix:
        ks._check()
        ks._check()
    mock_vix.assert_called_once()


def test_vix_fetch_failure_does_not_clear_an_active_blackout():
    """A fetch failure must never clear a blackout that's already set."""
    ks = make_ks("MODERATE")
    check_at_vix(ks, 40.0)
    assert not ks.new_positions_allowed

    expire_vix_cache(ks)
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("network down")):
        ks._check()
    assert not ks.new_positions_allowed


def test_vix_fetch_failure_sets_blackout_after_n_consecutive_failures():
    """Repeated fetch failures fail closed instead of silently trusting a stale reading."""
    ks = make_ks("MODERATE")
    assert ks.new_positions_allowed

    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()
            assert ks.new_positions_allowed
        ks._check()

    assert not ks.new_positions_allowed


def test_vix_fetch_success_resets_failure_counter():
    """A single genuine reading between failures resets the consecutive-failure count."""
    ks = make_ks("MODERATE")
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()

    check_at_vix(ks, 15.0)
    assert ks._consecutive_vix_failures == 0

    expire_vix_cache(ks)  # otherwise these failures would be cache hits, not real fetches
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()
    assert ks.new_positions_allowed  # only 4 consecutive failures since the reset, not 5


def test_vix_gate_driven_entirely_from_fixture_data():
    """The VIX gate can be exercised off FixtureMarketDataProvider, with no network fetch."""
    ks = make_ks("MODERATE")
    ks.market_data = FixtureMarketDataProvider()

    with mock.patch(FETCH_VIX, side_effect=AssertionError("must not reach the live fetcher")):
        ks._check()

    assert ks.status.current_vix == pytest.approx(16.799999237060547)
    assert ks.new_positions_allowed  # fixture VIX is well below the 35.0 blackout threshold


# ---------------------------------------------------------------------------
# VIX surfaced on status
# ---------------------------------------------------------------------------


def test_status_reports_the_last_observed_vix():
    """KillSwitchStatus.current_vix reflects the last successful VIX reading."""
    ks = make_ks("MODERATE")
    check_at_vix(ks, 22.5)
    assert ks.status.current_vix == pytest.approx(22.5)
