import contextlib
import threading
import time
from unittest.mock import patch

import pytest

from argus.governor import (
    BOOTSTRAP_LIMITS,
    REGISTERED_MODELS,
    RateLimitExceeded,
    RateLimitGovernor,
    UnregisteredModel,
    _parse_reset_duration,
    estimate_tokens,
)
from argus.params import GOVERNOR

# Use any registered model unless a test explicitly requires separate per-model budgets.
MODEL = next(iter(REGISTERED_MODELS))


def _tokens_this_window(usage) -> int:
    """Sum the rolling token window's entries."""
    return sum(tok for _, tok in usage.tokens_window)


@pytest.fixture
def governor():
    """Return a rate governor with empty usage counters."""
    return RateLimitGovernor()


def _rate_limit_headers(
    limit_requests: int = 14_400,
    limit_tokens: int = 300_000,
    remaining_requests: int = 14_399,
    remaining_tokens: int = 299_500,
    reset_requests: str = "23h59m59s",
    reset_tokens: str = "7.66s",
) -> dict[str, str]:
    """Build a synthetic set of Groq rate-limit response headers."""
    return {
        "x-ratelimit-limit-requests": str(limit_requests),
        "x-ratelimit-limit-tokens": str(limit_tokens),
        "x-ratelimit-remaining-requests": str(remaining_requests),
        "x-ratelimit-remaining-tokens": str(remaining_tokens),
        "x-ratelimit-reset-requests": reset_requests,
        "x-ratelimit-reset-tokens": reset_tokens,
    }


def test_governor_admits_after_window_rolls_over(monkeypatch, governor):
    req_limit = BOOTSTRAP_LIMITS[MODEL]["requests_per_minute"]

    fake_now = [1_000.0]
    monkeypatch.setattr("argus.governor.time.monotonic", lambda: fake_now[0])

    for _ in range(req_limit):
        governor.wait_if_needed(MODEL, 100)

    sleep_calls = []

    def _fake_sleep(seconds):
        sleep_calls.append(seconds)
        fake_now[0] += 61

    monkeypatch.setattr("argus.governor.time.sleep", _fake_sleep)

    governor.wait_if_needed(MODEL, 100)
    assert len(sleep_calls) == 1


@patch("argus.governor.time.sleep")
def test_governor_wait_gives_up_after_max_attempts(mock_sleep, governor):
    # Reject reservations larger than the entire minute budget because no window rollover can admit them.
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(MODEL, BOOTSTRAP_LIMITS[MODEL]["tokens_per_minute"] + 1)

    assert mock_sleep.call_count == GOVERNOR.max_wait_attempts
    usage = governor._get_usage(MODEL)
    assert len(usage.requests_window) == 0
    assert _tokens_this_window(usage) == 0


def test_governor_does_not_block_other_models_while_sleeping(governor):
    models = list(REGISTERED_MODELS)
    assert len(models) >= 2, "test requires at least two registered models"
    throttled_model, other_model = models[0], models[1]

    usage = governor._get_usage(throttled_model)
    now = time.monotonic()
    usage.requests_window.extend([now] * BOOTSTRAP_LIMITS[throttled_model]["requests_per_minute"])

    sleeping = threading.Event()
    # Capture real sleep before patching the shared time module so the mock side effect cannot recurse.
    real_sleep = time.sleep

    def _blocking_sleep(_seconds):
        sleeping.set()
        real_sleep(0.3)

    def _run_throttled():
        with (
            patch("argus.governor.time.sleep", side_effect=_blocking_sleep),
            contextlib.suppress(RateLimitExceeded),
        ):
            governor.wait_if_needed(throttled_model, 10)

    t = threading.Thread(target=_run_throttled)
    t.start()
    assert sleeping.wait(timeout=2), "throttled thread never reached its sleep"

    start = time.perf_counter()
    governor.wait_if_needed(other_model, 10)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2, "call for an unrelated model was blocked by the throttled model's sleep"

    t.join(timeout=5)


def test_governor_unregistered_model_raises(governor):
    with pytest.raises(UnregisteredModel):
        governor.wait_if_needed("some-typo-d-model-id", 100)


def test_governor_get_capacity_unregistered_model_raises(governor):
    with pytest.raises(UnregisteredModel):
        governor.get_remaining_capacity("some-typo-d-model-id")


def test_governor_get_capacity(governor):
    initial_capacity = governor.get_remaining_capacity(MODEL)

    governor.wait_if_needed(MODEL, 10)
    assert governor.get_remaining_capacity(MODEL) == initial_capacity - 1


def test_governor_report(governor):
    governor.wait_if_needed(MODEL, 100)

    report = governor.get_usage_report()
    assert MODEL in report
    assert report[MODEL]["requests_today"] == 1
    assert report[MODEL]["tokens_today"] == 100
    limits = BOOTSTRAP_LIMITS[MODEL]
    assert report[MODEL]["requests_per_minute_limit"] == limits["requests_per_minute"]
    assert report[MODEL]["tokens_per_minute_limit"] == limits["tokens_per_minute"]
    assert report[MODEL]["limits_observed"] is False


@pytest.mark.parametrize(
    ("value", "expected_seconds"),
    [
        ("2m59.56s", 2 * 60 + 59.56),
        ("7.66s", 7.66),
        ("88ms", 0.088),
        ("1h2m3s", 3600 + 2 * 60 + 3),
        ("0s", 0.0),
        ("60", 60.0),
    ],
)
def test_parse_reset_duration(value, expected_seconds):
    assert _parse_reset_duration(value) == pytest.approx(expected_seconds)


def test_parse_reset_duration_rejects_unparseable_input():
    with pytest.raises(ValueError, match="unparseable Groq reset duration"):
        _parse_reset_duration("not-a-duration")


def test_estimate_tokens_combines_prompts_and_max_tokens():
    system_prompt = "one two three"
    user_prompt = "four five"
    max_tokens = 100

    estimated = estimate_tokens(system_prompt, user_prompt, max_tokens)
    assert estimated == int(5 * GOVERNOR.word_to_token_ratio) + GOVERNOR.chat_template_overhead_tokens + 100


def test_observe_headers_overrides_bootstrap(governor):
    assert governor._get_usage(MODEL).limits_observed is False

    governor.observe_headers(MODEL, _rate_limit_headers(limit_tokens=123_456, remaining_tokens=100_000))

    usage = governor._get_usage(MODEL)
    assert usage.limits_observed is True
    assert usage.limit_tokens == 123_456
    assert usage.remaining_tokens == 100_000

    report = governor.get_usage_report()
    assert report[MODEL]["limits_observed"] is True
    assert report[MODEL]["tokens_per_minute_limit"] == 123_456


def test_observe_headers_unregistered_model_raises(governor):
    with pytest.raises(UnregisteredModel):
        governor.observe_headers("some-typo-d-model-id", _rate_limit_headers())


def test_observe_headers_skips_incomplete_headers(governor):
    governor.observe_headers(MODEL, {"content-type": "application/json"})
    assert governor._get_usage(MODEL).limits_observed is False


@patch("argus.governor.time.sleep")
def test_observed_tpm_governs_wait_if_needed(mock_sleep, governor):
    governor.observe_headers(MODEL, _rate_limit_headers(limit_tokens=1_000, remaining_tokens=1_000))

    # Bootstrap TPM would happily admit this, but the header-observed remaining won't.
    with pytest.raises(RateLimitExceeded):
        governor.wait_if_needed(MODEL, 100_000)
    assert mock_sleep.call_count == GOVERNOR.max_wait_attempts


def test_observed_daily_exhaustion_raises_immediately(governor):
    governor.observe_headers(MODEL, _rate_limit_headers(remaining_requests=0, reset_requests="23h59m0s"))

    with (
        patch("argus.governor.time.sleep") as mock_sleep,
        pytest.raises(RateLimitExceeded),
    ):
        governor.wait_if_needed(MODEL, 10)
    assert mock_sleep.call_count == 0, "a day-long reset must not be slept out"


def test_unobserved_daily_budget_is_permissive_by_design(governor):
    assert governor._get_usage(MODEL).limits_observed is False

    for _ in range(5):
        governor.wait_if_needed(MODEL, 10)

    assert governor._get_usage(MODEL).limits_observed is False


def test_record_usage_applies_delta_to_bootstrap_counters(governor):
    governor.wait_if_needed(MODEL, 100)

    governor.record_usage(MODEL, estimated_tokens=100, prompt_tokens=40, completion_tokens=10)

    usage = governor._get_usage(MODEL)
    assert _tokens_this_window(usage) == 50
    assert usage.tokens_today == 50


def test_record_usage_never_lets_counters_go_negative(governor):
    governor.wait_if_needed(MODEL, 100)

    governor.record_usage(MODEL, estimated_tokens=100, prompt_tokens=0, completion_tokens=0)

    usage = governor._get_usage(MODEL)
    assert _tokens_this_window(usage) == 0
    assert usage.tokens_today == 0


def test_record_usage_unregistered_model_raises(governor):
    with pytest.raises(UnregisteredModel):
        governor.record_usage("some-typo-d-model-id", 100, 50, 10)


def test_record_usage_does_not_touch_header_derived_remaining(governor):
    governor.observe_headers(MODEL, _rate_limit_headers(remaining_tokens=5_000))

    governor.record_usage(MODEL, estimated_tokens=1_000, prompt_tokens=1, completion_tokens=1)

    assert governor._get_usage(MODEL).remaining_tokens == 5_000


def test_release_reservation_restores_bootstrap_budget(governor):
    initial_capacity = governor.get_remaining_capacity(MODEL)

    governor.wait_if_needed(MODEL, 500)
    assert governor.get_remaining_capacity(MODEL) == initial_capacity - 1

    governor.release_reservation(MODEL, 500)

    usage = governor._get_usage(MODEL)
    assert governor.get_remaining_capacity(MODEL) == initial_capacity
    assert _tokens_this_window(usage) == 0
    assert usage.requests_today == 0
    assert usage.tokens_today == 0


def test_release_reservation_restores_header_observed_remaining(governor):
    governor.observe_headers(MODEL, _rate_limit_headers(limit_tokens=1_000, remaining_tokens=1_000))

    governor.wait_if_needed(MODEL, 400)
    usage = governor._get_usage(MODEL)
    assert usage.remaining_tokens == 600
    assert usage.remaining_requests == 14_398

    governor.release_reservation(MODEL, 400)
    assert usage.remaining_tokens == 1_000
    assert usage.remaining_requests == 14_399


def test_release_reservation_unregistered_model_raises(governor):
    with pytest.raises(UnregisteredModel):
        governor.release_reservation("some-typo-d-model-id", 100)


def test_observe_headers_updates_token_axis_without_request_axis(governor):
    headers = {
        "x-ratelimit-limit-tokens": "50000",
        "x-ratelimit-remaining-tokens": "40000",
        "x-ratelimit-reset-tokens": "10s",
    }
    governor.observe_headers(MODEL, headers)

    usage = governor._get_usage(MODEL)
    assert usage.limits_observed is True
    assert usage.limit_tokens == 50000
    assert usage.remaining_tokens == 40000
    assert usage.limit_requests is None
    assert usage.remaining_requests is None


def test_observe_headers_updates_request_axis_without_token_axis(governor):
    headers = {
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-remaining-requests": "14000",
        "x-ratelimit-reset-requests": "12h",
    }
    governor.observe_headers(MODEL, headers)

    usage = governor._get_usage(MODEL)
    assert usage.limits_observed is True
    assert usage.limit_requests == 14400
    assert usage.remaining_requests == 14000
    assert usage.limit_tokens is None
    assert usage.remaining_tokens is None


def test_observe_headers_malformed_reset_still_applies_limit_and_remaining(governor):
    governor.observe_headers(MODEL, _rate_limit_headers(reset_tokens="not-a-duration"))

    usage = governor._get_usage(MODEL)
    assert usage.limits_observed is True
    assert usage.limit_tokens == 300_000
    assert usage.remaining_tokens == 299_500
    assert usage.reset_tokens_at is None
