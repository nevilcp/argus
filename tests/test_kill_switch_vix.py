"""VIX blackout readings, caching, fail-closed behavior, and status regressions."""

from __future__ import annotations

from unittest import mock

import pytest

from argus.params import KILL_SWITCH
from argus.seams import FixtureMarketDataProvider
from tests.helpers.kill_switch import FETCH_VIX, check_at_vix, expire_vix_cache, make_ks


def test_vix_blackout_sets_and_clears_on_genuine_readings():
    ks = make_ks("MODERATE")
    check_at_vix(ks, 40.0)
    assert not ks.new_positions_allowed

    expire_vix_cache(ks)
    check_at_vix(ks, 15.0)
    assert ks.new_positions_allowed


def test_vix_reading_is_cached_within_the_ttl():
    ks = make_ks("MODERATE")
    with mock.patch(FETCH_VIX, return_value=15.0) as mock_vix:
        ks._check()
        ks._check()
    mock_vix.assert_called_once()


def test_vix_fetch_failure_does_not_clear_an_active_blackout():
    ks = make_ks("MODERATE")
    check_at_vix(ks, 40.0)
    assert not ks.new_positions_allowed

    expire_vix_cache(ks)
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("network down")):
        ks._check()
    assert not ks.new_positions_allowed


def test_vix_fetch_failure_sets_blackout_after_n_consecutive_failures():
    ks = make_ks("MODERATE")
    assert ks.new_positions_allowed

    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KILL_SWITCH.vix_failure_halt_threshold - 1):
            ks._check()
            assert ks.new_positions_allowed
        ks._check()

    assert not ks.new_positions_allowed


def test_vix_fetch_success_resets_failure_counter():
    ks = make_ks("MODERATE")
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KILL_SWITCH.vix_failure_halt_threshold - 1):
            ks._check()

    check_at_vix(ks, 15.0)
    assert ks._consecutive_vix_failures == 0

    expire_vix_cache(ks)
    with mock.patch(FETCH_VIX, side_effect=RuntimeError("down")):
        for _ in range(KILL_SWITCH.vix_failure_halt_threshold - 1):
            ks._check()
    assert ks.new_positions_allowed


def test_vix_gate_driven_entirely_from_fixture_data():
    ks = make_ks("MODERATE")
    ks.market_data = FixtureMarketDataProvider()

    with mock.patch(FETCH_VIX, side_effect=AssertionError("must not reach the live fetcher")):
        ks._check()

    assert ks.status.current_vix == pytest.approx(16.799999237060547)
    assert ks.new_positions_allowed


def test_status_reports_the_last_observed_vix():
    ks = make_ks("MODERATE")
    check_at_vix(ks, 22.5)
    assert ks.status.current_vix == pytest.approx(22.5)


def test_status_reports_no_vix_before_the_first_measurement():
    assert make_ks("MODERATE").status.current_vix is None
