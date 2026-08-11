"""
Tests for the RateLimitGovernor.
"""

from unittest.mock import patch

import pytest

from argus.orchestration.governor import (
    RateLimitExceeded,
    RateLimitGovernor,
    MODEL_LIMITS,
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

@patch("argus.orchestration.governor.time.sleep")
def test_governor_daily_limit(mock_sleep, governor):
    """Exceeding the daily request limit raises rather than sleeping."""
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]

    req_limit = limits["requests_per_day"]

    # Seed usage directly rather than making req_limit - 1 real calls
    usage = governor._get_usage(model)
    usage.requests_today = req_limit - 1

    governor.wait_if_needed(model, 10)
    assert mock_sleep.call_count == 0
    assert usage.requests_today == req_limit

    # Unlike the per-minute limit, the daily limit raises instead of blocking
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(model, 10)
    assert mock_sleep.call_count == 0
    assert usage.requests_today == req_limit

def test_governor_get_capacity(governor):
    """Remaining capacity decrements by one for each request recorded."""
    model = list(MODEL_LIMITS.keys())[0]
    initial_capacity = governor.get_remaining_capacity(model)
    
    governor.wait_if_needed(model, 10)
    assert governor.get_remaining_capacity(model) == initial_capacity - 1

def test_governor_report(governor):
    """The usage report reflects the requests and tokens recorded for a model."""
    model = list(MODEL_LIMITS.keys())[0]
    governor.wait_if_needed(model, 100)
    
    report = governor.get_usage_report()
    assert model in report
    assert report[model]["requests_today"] == 1
    assert report[model]["tokens_today"] == 100
