"""
Tests for the RateLimitGovernor.
"""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from argus.orchestration.governor import (
    RateLimitExceeded,
    RateLimitGovernor,
    ModelUsage,
    MODEL_LIMITS,
)

@pytest.fixture
def governor():
    # Clear any existing state
    gov = RateLimitGovernor()
    return gov

@patch("argus.orchestration.governor.time.sleep")
def test_governor_minute_limit(mock_sleep, governor):
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]
    
    req_limit = limits["requests_per_minute"]
    
    # Use up the minute limit
    for _ in range(req_limit):
        governor.wait_if_needed(model, 100)
    
    assert mock_sleep.call_count == 0
    
    # The next one should trigger a sleep
    governor.wait_if_needed(model, 100)
    assert mock_sleep.call_count == 1

@patch("argus.orchestration.governor.time.sleep")
def test_governor_daily_limit(mock_sleep, governor):
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]
    
    req_limit = limits["requests_per_day"]
    
    # Artificially set usage to just below daily limit
    usage = governor._get_usage(model)
    usage.requests_today = req_limit - 1
    
    governor.wait_if_needed(model, 10)
    assert mock_sleep.call_count == 0
    assert usage.requests_today == req_limit

    # The next one must raise instead of sleeping-and-proceeding
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(model, 10)
    assert mock_sleep.call_count == 0
    assert usage.requests_today == req_limit

def test_governor_get_capacity(governor):
    model = list(MODEL_LIMITS.keys())[0]
    initial_capacity = governor.get_remaining_capacity(model)
    
    governor.wait_if_needed(model, 10)
    assert governor.get_remaining_capacity(model) == initial_capacity - 1

def test_governor_report(governor):
    model = list(MODEL_LIMITS.keys())[0]
    governor.wait_if_needed(model, 100)
    
    report = governor.get_usage_report()
    assert model in report
    assert report[model]["requests_today"] == 1
    assert report[model]["tokens_today"] == 100
