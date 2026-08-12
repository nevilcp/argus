"""
Tests for the RateLimitGovernor.
"""

from unittest.mock import patch

import pytest

from argus.orchestration.governor import (
    MODEL_LIMITS,
    RateLimitGovernor,
    UnregisteredModel,
)


@pytest.fixture
def governor():
    """A fresh RateLimitGovernor with no prior usage recorded.

    Returns:
        A new RateLimitGovernor instance.
    """
    gov = RateLimitGovernor()
    return gov


@patch("argus.orchestration.governor.time.sleep")
def test_governor_minute_limit(mock_sleep, governor):
    """Exceeding the per-minute request limit sleeps rather than raising."""
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]

    req_limit = limits["requests_per_minute"]

    for _ in range(req_limit):
        governor.wait_if_needed(model, 100)

    assert mock_sleep.call_count == 0

    governor.wait_if_needed(model, 100)
    assert mock_sleep.call_count == 1


def test_governor_unregistered_model_raises(governor):
    """A model absent from MODEL_LIMITS raises rather than being silently ungoverned."""
    with pytest.raises(UnregisteredModel):
        governor.wait_if_needed("some-typo-d-model-id", 100)


def test_governor_get_capacity_unregistered_model(governor):
    """Remaining capacity for an unregistered model is 0, not a silent pass-through."""
    assert governor.get_remaining_capacity("some-typo-d-model-id") == 0


def test_governor_get_capacity(governor):
    """Remaining per-minute capacity decrements by one for each request recorded."""
    model = list(MODEL_LIMITS.keys())[0]
    initial_capacity = governor.get_remaining_capacity(model)

    governor.wait_if_needed(model, 10)
    assert governor.get_remaining_capacity(model) == initial_capacity - 1


def test_governor_report(governor):
    """The usage report reflects today's usage alongside the published per-minute limits."""
    model = list(MODEL_LIMITS.keys())[0]
    governor.wait_if_needed(model, 100)

    report = governor.get_usage_report()
    assert model in report
    assert report[model]["requests_today"] == 1
    assert report[model]["tokens_today"] == 100
    assert report[model]["requests_per_minute_limit"] == MODEL_LIMITS[model]["requests_per_minute"]
    assert report[model]["tokens_per_minute_limit"] == MODEL_LIMITS[model]["tokens_per_minute"]
