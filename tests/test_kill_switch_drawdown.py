"""Drawdown thresholds, peak tracking, and reset regressions."""

from __future__ import annotations

from unittest import mock

import pytest

from tests.helpers.kill_switch import FETCH_VIX, check_at_vix, make_ks


@pytest.mark.parametrize(
    ("tolerance", "threshold"),
    [("CONSERVATIVE", 0.08), ("MODERATE", 0.12), ("AGGRESSIVE", 0.18)],
)
@mock.patch(FETCH_VIX, return_value=15.0)
def test_drawdown_halts_at_tier_threshold(mock_vix, tolerance, threshold):
    ks = make_ks(tolerance, inception=100_000.0)
    ks.update_portfolio_value(100_000.0 * (1 - threshold))
    ks._check()
    assert ks.is_halted


@mock.patch(FETCH_VIX, return_value=15.0)
def test_drawdown_does_not_halt_below_threshold(mock_vix):
    ks = make_ks("MODERATE", inception=100_000.0)
    ks.update_portfolio_value(100_000.0 * (1 - 0.12) + 1.0)
    ks._check()
    assert not ks.is_halted


@mock.patch(FETCH_VIX, return_value=15.0)
def test_zero_portfolio_value_registers_as_real_drawdown(mock_vix):
    ks = make_ks("CONSERVATIVE", inception=100_000.0)
    ks.update_portfolio_value(0.0)
    ks._check()
    assert ks.is_halted
    assert ks.status.realized_drawdown == pytest.approx(1.0)


@mock.patch(FETCH_VIX, return_value=15.0)
def test_peak_to_trough_drawdown_not_inception_relative(mock_vix):
    ks = make_ks("AGGRESSIVE", inception=100_000.0)

    ks.update_portfolio_value(150_000.0)
    ks._check()
    assert not ks.is_halted

    ks.update_portfolio_value(105_000.0)
    ks._check()
    assert ks.is_halted
    assert ks.status.realized_drawdown == pytest.approx((150_000.0 - 105_000.0) / 150_000.0)


def test_reset_clears_halt_and_reseeds_high_water_mark():
    ks = make_ks("MODERATE", inception=100_000.0)
    ks.update_portfolio_value(80_000.0)
    check_at_vix(ks, 15.0)
    assert ks.is_halted

    ks.reset(120_000.0)

    assert not ks.is_halted
    assert ks.new_positions_allowed
    assert ks.status.realized_drawdown == 0.0
