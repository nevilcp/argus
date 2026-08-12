"""
Tests for the RateLimitGovernor.
"""

import threading
import time
from unittest.mock import patch

import pytest

from argus.orchestration.governor import (
    MODEL_LIMITS,
    RateLimitExceeded,
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
    """Exceeding the per-minute request limit sleeps, then admits the call after re-checking."""
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]

    req_limit = limits["requests_per_minute"]

    for _ in range(req_limit):
        governor.wait_if_needed(model, 100)

    assert mock_sleep.call_count == 0

    # The mocked sleep doesn't advance real time, so simulate the minute rolling
    # over — the re-check loop (the point of the lock-release fix) then finds
    # room on its second pass instead of spinning through every retry attempt.
    def _advance_minute(_seconds):
        governor._get_usage(model).current_minute = "1970-01-01T00:00"

    mock_sleep.side_effect = _advance_minute

    governor.wait_if_needed(model, 100)
    assert mock_sleep.call_count == 1


@patch("argus.orchestration.governor.time.sleep")
def test_governor_wait_gives_up_after_max_attempts(mock_sleep, governor):
    """A request too large to ever fit the per-minute budget raises rather than looping forever."""
    model = list(MODEL_LIMITS.keys())[0]
    limits = MODEL_LIMITS[model]

    # estimated_tokens alone exceeds the per-minute budget, so no amount of
    # waiting for the minute to reset ever makes this admissible.
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(model, limits["tokens_per_minute"] + 1)

    assert mock_sleep.call_count == governor._MAX_WAIT_ATTEMPTS
    usage = governor._get_usage(model)
    assert usage.requests_this_minute == 0
    assert usage.tokens_this_minute == 0


def test_governor_does_not_block_other_models_while_sleeping(governor):
    """A stall throttling one model must not block a concurrent call for a different model.

    Regression test for the lock being held across time.sleep(): with the lock
    held for the whole sleep, a call for `other_model` on another thread would
    have queued behind `throttled_model`'s wait instead of returning immediately.
    """
    models = list(MODEL_LIMITS.keys())
    assert len(models) >= 2, "test requires at least two registered models"
    throttled_model, other_model = models[0], models[1]

    usage = governor._get_usage(throttled_model)
    usage.requests_this_minute = MODEL_LIMITS[throttled_model]["requests_per_minute"]

    sleeping = threading.Event()
    # `time` is a single shared module: patching governor's time.sleep patches
    # this attribute globally, so a plain time.sleep(0.3) call in the side_effect
    # below would recurse into the mock. Capture the real function first.
    real_sleep = time.sleep

    def _blocking_sleep(_seconds):
        sleeping.set()
        real_sleep(0.3)

    def _run_throttled():
        with patch("argus.orchestration.governor.time.sleep", side_effect=_blocking_sleep):
            try:
                governor.wait_if_needed(throttled_model, 10)
            except RateLimitExceeded:
                pass

    t = threading.Thread(target=_run_throttled)
    t.start()
    assert sleeping.wait(timeout=2), "throttled thread never reached its sleep"

    start = time.perf_counter()
    governor.wait_if_needed(other_model, 10)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2, "call for an unrelated model was blocked by the throttled model's sleep"

    t.join(timeout=5)


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
