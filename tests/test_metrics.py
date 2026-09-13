"""Tests for argus/backtesting/metrics.py."""

import math

import pandas as pd

from argus.backtesting.metrics import compute_all_metrics


def _returns(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_sortino_matches_hand_computed_downside_deviation():
    """Downside deviation is the RMS of (return - target) below zero over all observations."""
    returns = _returns([0.02, -0.01, 0.03, -0.02, 0.01])

    metrics = compute_all_metrics(returns, pd.Series(dtype=float), risk_free_rate=0.0)

    # mean_ret = 0.006; downside_deviation = sqrt((0.01**2 + 0.02**2) / 5) = 0.01
    # sortino = 0.006 / 0.01 * sqrt(252)
    expected = 0.006 / 0.01 * math.sqrt(252)
    assert metrics["sortino_ratio"] == round(expected, 4)


def test_sortino_is_none_with_no_negative_returns():
    """No return below the target means no downside deviation to divide by."""
    returns = _returns([0.01, 0.02, 0.03])

    metrics = compute_all_metrics(returns, pd.Series(dtype=float), risk_free_rate=0.0)

    assert metrics["sortino_ratio"] is None


def test_sortino_handles_all_zero_returns_without_nan_or_crash():
    """All-zero returns against a positive target degrade to a finite, non-NaN ratio."""
    returns = _returns([0.0, 0.0, 0.0, 0.0, 0.0])

    metrics = compute_all_metrics(returns, pd.Series(dtype=float), risk_free_rate=0.05)

    assert metrics["sortino_ratio"] is not None
    assert math.isfinite(metrics["sortino_ratio"])


def test_information_ratio_unchanged_by_the_sortino_fix():
    """The information ratio is annualized daily active-return-over-tracking-error; leave as is."""
    strategy = _returns([0.02, -0.01, 0.03, -0.02, 0.01])
    benchmark = _returns([0.01, 0.00, 0.02, -0.01, 0.00])

    metrics = compute_all_metrics(strategy, benchmark, risk_free_rate=0.0)

    active_return = strategy - benchmark
    expected = active_return.mean() / active_return.std() * math.sqrt(252)
    assert metrics["information_ratio"] == round(expected, 4)
