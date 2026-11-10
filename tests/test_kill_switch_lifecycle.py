"""
tests/test_kill_switch_lifecycle.py

Tests for argus/risk/kill_switch.py's monitor thread: start(), idempotent
start() and stop().

fetch_vix is mocked throughout; _check() is driven directly, so no test sleeps
on a real check interval.
"""

from __future__ import annotations

from unittest import mock

import pytest

from argus.risk.kill_switch import KillSwitch
from tests.helpers.kill_switch import FETCH_VIX


@mock.patch(FETCH_VIX, return_value=15.0)
def test_start_launches_a_running_daemon_thread(mock_vix):
    """start() launches a background thread that is alive immediately after."""
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    try:
        assert ks._thread is not None
        assert ks._thread.is_alive()
    finally:
        ks.stop(timeout=2.0)


@mock.patch(FETCH_VIX, return_value=15.0)
def test_start_is_idempotent_when_already_running(mock_vix):
    """A second start() call while the monitor thread is alive is a no-op."""
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    first_thread = ks._thread
    try:
        ks.start(200_000.0)
        assert ks._thread is first_thread
        assert ks._portfolio_inception_value == 100_000.0  # untouched by the second call
    finally:
        ks.stop(timeout=2.0)


@mock.patch(FETCH_VIX, return_value=15.0)
def test_stop_joins_the_monitor_thread(mock_vix):
    """stop() signals the loop to exit and waits for the thread to finish."""
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    ks.stop(timeout=2.0)
    assert not ks._thread.is_alive()


def test_risk_tolerance_must_be_a_string():
    """A non-string risk_tolerance raises a typed error, not an incidental one."""
    with pytest.raises(TypeError):
        KillSwitch(123)
