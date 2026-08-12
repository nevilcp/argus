"""
Tests for the RateLimitGovernor.
"""

import threading
import time
from unittest.mock import patch

import pytest

from argus.orchestration.governor import (
    BOOTSTRAP_LIMITS,
    REGISTERED_MODELS,
    RateLimitExceeded,
    RateLimitGovernor,
    UnregisteredModel,
    _parse_reset_duration,
    estimate_tokens,
)


@pytest.fixture
def governor():
    """A fresh RateLimitGovernor with no prior usage recorded.

    Returns:
        A new RateLimitGovernor instance.
    """
    gov = RateLimitGovernor()
    return gov


def _rate_limit_headers(
    limit_requests: int = 14_400,
    limit_tokens: int = 300_000,
    remaining_requests: int = 14_399,
    remaining_tokens: int = 299_500,
    reset_requests: str = "23h59m59s",
    reset_tokens: str = "7.66s",
) -> dict[str, str]:
    """Builds a synthetic set of Groq rate-limit response headers.

    Args:
        limit_requests: Value for x-ratelimit-limit-requests (a DAILY figure).
        limit_tokens: Value for x-ratelimit-limit-tokens (a PER-MINUTE figure).
        remaining_requests: Value for x-ratelimit-remaining-requests.
        remaining_tokens: Value for x-ratelimit-remaining-tokens.
        reset_requests: Go-style duration string for x-ratelimit-reset-requests.
        reset_tokens: Go-style duration string for x-ratelimit-reset-tokens.

    Returns:
        Dict of header name -> value, matching Groq's real header names.
    """
    return {
        "x-ratelimit-limit-requests": str(limit_requests),
        "x-ratelimit-limit-tokens": str(limit_tokens),
        "x-ratelimit-remaining-requests": str(remaining_requests),
        "x-ratelimit-remaining-tokens": str(remaining_tokens),
        "x-ratelimit-reset-requests": reset_requests,
        "x-ratelimit-reset-tokens": reset_tokens,
    }


@patch("argus.orchestration.governor.time.sleep")
def test_governor_minute_limit(mock_sleep, governor):
    """Exceeding the per-minute request limit sleeps, then admits the call after re-checking."""
    model = list(REGISTERED_MODELS)[0]
    limits = BOOTSTRAP_LIMITS[model]

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
    model = list(REGISTERED_MODELS)[0]
    limits = BOOTSTRAP_LIMITS[model]

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
    models = list(REGISTERED_MODELS)
    assert len(models) >= 2, "test requires at least two registered models"
    throttled_model, other_model = models[0], models[1]

    usage = governor._get_usage(throttled_model)
    usage.requests_this_minute = BOOTSTRAP_LIMITS[throttled_model]["requests_per_minute"]

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
    """A model absent from REGISTERED_MODELS raises rather than being silently ungoverned."""
    with pytest.raises(UnregisteredModel):
        governor.wait_if_needed("some-typo-d-model-id", 100)


def test_governor_get_capacity_unregistered_model(governor):
    """Remaining capacity for an unregistered model is 0, not a silent pass-through."""
    assert governor.get_remaining_capacity("some-typo-d-model-id") == 0


def test_governor_get_capacity(governor):
    """Remaining per-minute capacity decrements by one for each request recorded."""
    model = list(REGISTERED_MODELS)[0]
    initial_capacity = governor.get_remaining_capacity(model)

    governor.wait_if_needed(model, 10)
    assert governor.get_remaining_capacity(model) == initial_capacity - 1


def test_governor_report(governor):
    """The usage report reflects today's usage alongside the published per-minute limits."""
    model = list(REGISTERED_MODELS)[0]
    governor.wait_if_needed(model, 100)

    report = governor.get_usage_report()
    assert model in report
    assert report[model]["requests_today"] == 1
    assert report[model]["tokens_today"] == 100
    assert report[model]["requests_per_minute_limit"] == BOOTSTRAP_LIMITS[model]["requests_per_minute"]
    assert report[model]["tokens_per_minute_limit"] == BOOTSTRAP_LIMITS[model]["tokens_per_minute"]
    assert report[model]["limits_observed"] is False


@pytest.mark.parametrize(
    "value,expected_seconds",
    [
        ("2m59.56s", 2 * 60 + 59.56),
        ("7.66s", 7.66),
        ("88ms", 0.088),
        ("1h2m3s", 3600 + 2 * 60 + 3),
        ("0s", 0.0),
    ],
)
def test_parse_reset_duration(value, expected_seconds):
    """Go-style Groq reset durations parse into the equivalent number of seconds."""
    assert _parse_reset_duration(value) == pytest.approx(expected_seconds)


def test_parse_reset_duration_rejects_unparseable_input():
    """A string with no recognizable duration component raises rather than silently returning 0."""
    with pytest.raises(ValueError):
        _parse_reset_duration("not-a-duration")


def test_estimate_tokens_combines_prompts_and_max_tokens():
    """Word count across both prompts is scaled and added to the completion ceiling."""
    system_prompt = "one two three"
    user_prompt = "four five"
    max_tokens = 100

    estimated = estimate_tokens(system_prompt, user_prompt, max_tokens)
    assert estimated == int(5 * 1.3) + 100


def test_observe_headers_overrides_bootstrap(governor):
    """A real Groq response header sets limits_observed and overrides the bootstrap figures."""
    model = list(REGISTERED_MODELS)[0]
    assert governor._get_usage(model).limits_observed is False

    governor.observe_headers(model, _rate_limit_headers(limit_tokens=123_456, remaining_tokens=100_000))

    usage = governor._get_usage(model)
    assert usage.limits_observed is True
    assert usage.limit_tokens == 123_456
    assert usage.remaining_tokens == 100_000

    report = governor.get_usage_report()
    assert report[model]["limits_observed"] is True
    assert report[model]["tokens_per_minute_limit"] == 123_456


def test_observe_headers_unregistered_model_raises(governor):
    """Feeding headers for an unregistered model raises rather than fabricating a profile."""
    with pytest.raises(UnregisteredModel):
        governor.observe_headers("some-typo-d-model-id", _rate_limit_headers())


def test_observe_headers_skips_incomplete_headers(governor):
    """A response missing rate-limit headers leaves the bootstrap state untouched."""
    model = list(REGISTERED_MODELS)[0]
    governor.observe_headers(model, {"content-type": "application/json"})
    assert governor._get_usage(model).limits_observed is False


@patch("argus.orchestration.governor.time.sleep")
def test_observed_tpm_governs_wait_if_needed(mock_sleep, governor):
    """Once headers are observed, tokens-per-minute admission uses the header-reported remaining."""
    model = list(REGISTERED_MODELS)[0]
    governor.observe_headers(model, _rate_limit_headers(limit_tokens=1_000, remaining_tokens=1_000))

    # Bootstrap TPM would happily admit this, but the header-observed remaining won't.
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(model, 100_000)
    assert mock_sleep.call_count == governor._MAX_WAIT_ATTEMPTS


def test_observed_daily_exhaustion_raises_immediately(governor):
    """A provider-reported daily budget of zero fails fast rather than sleeping out an ~day-long reset."""
    model = list(REGISTERED_MODELS)[0]
    governor.observe_headers(model, _rate_limit_headers(remaining_requests=0, reset_requests="23h59m0s"))

    with patch("argus.orchestration.governor.time.sleep") as mock_sleep:
        with pytest.raises(RateLimitExceeded):
            governor.wait_if_needed(model, 10)
    assert mock_sleep.call_count == 0, "a day-long reset must not be slept out"


def test_record_usage_applies_delta_to_bootstrap_counters(governor):
    """record_usage corrects the local today/this-minute counters toward the actual token count."""
    model = list(REGISTERED_MODELS)[0]
    governor.wait_if_needed(model, 100)

    governor.record_usage(model, estimated_tokens=100, prompt_tokens=40, completion_tokens=10)

    usage = governor._get_usage(model)
    assert usage.tokens_this_minute == 50
    assert usage.tokens_today == 50


def test_record_usage_never_lets_counters_go_negative(governor):
    """An overestimate correcting downward floors at 0 rather than going negative."""
    model = list(REGISTERED_MODELS)[0]
    governor.wait_if_needed(model, 100)

    governor.record_usage(model, estimated_tokens=100, prompt_tokens=0, completion_tokens=0)

    usage = governor._get_usage(model)
    assert usage.tokens_this_minute == 0
    assert usage.tokens_today == 0


def test_record_usage_unregistered_model_raises(governor):
    """Recording usage for an unregistered model raises rather than fabricating a profile."""
    with pytest.raises(UnregisteredModel):
        governor.record_usage("some-typo-d-model-id", 100, 50, 10)


def test_record_usage_does_not_touch_header_derived_remaining(governor):
    """record_usage only corrects bootstrap counters — the header-observed remaining is untouched.

    observe_headers already refreshed the header-derived remaining with the
    provider's own post-call figure by the time record_usage runs (see
    GroqLLMClient.complete), so applying a second delta there would double-adjust.
    """
    model = list(REGISTERED_MODELS)[0]
    governor.observe_headers(model, _rate_limit_headers(remaining_tokens=5_000))

    governor.record_usage(model, estimated_tokens=1_000, prompt_tokens=1, completion_tokens=1)

    assert governor._get_usage(model).remaining_tokens == 5_000


