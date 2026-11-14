"""Tests for argus/risk/kill_switch.py's monitor thread: start(), idempotent start() and stop()."""

from __future__ import annotations

from unittest import mock

import pytest

from argus.risk.kill_switch import KillSwitch
from tests.helpers.kill_switch import FETCH_VIX


@mock.patch(FETCH_VIX, return_value=15.0)
def test_start_launches_a_running_daemon_thread(mock_vix):
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    try:
        assert ks._thread is not None
        assert ks._thread.is_alive()
    finally:
        ks.stop(timeout=2.0)


@mock.patch(FETCH_VIX, return_value=15.0)
def test_start_is_idempotent_when_already_running(mock_vix):
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    first_thread = ks._thread
    try:
        ks.start(200_000.0)
        assert ks._thread is first_thread
        assert ks._portfolio_inception_value == 100_000.0
    finally:
        ks.stop(timeout=2.0)


@mock.patch(FETCH_VIX, return_value=15.0)
def test_stop_joins_the_monitor_thread(mock_vix):
    ks = KillSwitch("MODERATE", check_interval_seconds=3600)
    ks.start(100_000.0)
    ks.stop(timeout=2.0)
    assert not ks._thread.is_alive()


def test_risk_tolerance_must_be_a_string():
    with pytest.raises(TypeError):
        KillSwitch(123)
