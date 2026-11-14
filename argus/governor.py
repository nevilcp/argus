"""Thread-safe Groq request and token quota admission and observation."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypedDict

from argus.config import settings
from argus.params import GOVERNOR

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a model's token or request quota is exhausted."""


class UnregisteredModel(ValueError):
    """Raised when a model with no known rate-limit profile is passed to the governor."""


# Register supported model names here; quota values come from bootstrap settings and observed headers.
REGISTERED_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
)

# Use conservative bootstrap RPM and TPM; enforce daily request quotas only after headers report them.
BOOTSTRAP_LIMITS: dict[str, dict[str, int]] = {
    model: {
        "tokens_per_minute": settings.ARGUS_GROQ_TPM,
        "requests_per_minute": settings.ARGUS_GROQ_RPM,
    }
    for model in REGISTERED_MODELS
}

# Observe request and token headers independently; requests are daily and tokens are per minute.
_REQUEST_AXIS_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
)
_TOKEN_AXIS_HEADERS = (
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
)

# Match ms before m so a reset such as 88ms cannot be parsed as 88 minutes.
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _require_registered(model: str) -> None:
    """Reject a model the governor holds no rate-limit profile for."""
    if model not in REGISTERED_MODELS:
        raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")


def _parse_reset_duration(value: str) -> float:
    """Parse a Go-style duration string into seconds."""
    total = 0.0
    matched = False
    for amount, unit in _DURATION_COMPONENT_RE.findall(value):
        total += float(amount) * _DURATION_UNIT_SECONDS[unit]
        matched = True
    if matched:
        return total
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"unparseable Groq reset duration: {value!r}") from e


def _observe_axis(
    model: str,
    headers: Mapping[str, str],
    axis: str,
    axis_headers: tuple[str, str, str],
    now: float,
) -> tuple[int, int, float | None] | None:
    """Read one rate-limit axis's (limit, remaining, reset) triple out of a Groq response."""
    limit_header, remaining_header, reset_header = axis_headers
    if not all(header in headers for header in axis_headers):
        return None

    try:
        limit = int(headers[limit_header])
        remaining = int(headers[remaining_header])
    except (ValueError, TypeError) as e:
        logger.warning("Malformed %s headers for %s: %s", axis, model, e)
        return None

    reset_at: float | None = None
    try:
        reset_at = now + _parse_reset_duration(headers[reset_header])
    except ValueError as e:
        logger.warning("Unparseable %s reset for %s: %s", axis, model, e)
    return limit, remaining, reset_at


def estimate_tokens(system_prompt: str, user_prompt: str, max_tokens: int) -> int:
    """Estimate a chat completion's total token consumption before making the call."""
    word_count = len(system_prompt.split()) + len(user_prompt.split())
    return int(word_count * GOVERNOR.word_to_token_ratio) + GOVERNOR.chat_template_overhead_tokens + max_tokens


@dataclass
class ModelUsage:
    """Rolling per-model usage and independently observed provider quotas."""

    requests_today: int = 0
    tokens_today: int = 0
    current_date: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))

    requests_window: deque[float] = field(default_factory=deque)
    tokens_window: deque[tuple[float, int]] = field(default_factory=deque)

    limit_requests: int | None = None
    limit_tokens: int | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_at: float | None = None
    reset_tokens_at: float | None = None
    limits_observed: bool = False


class ModelUsageReport(TypedDict):
    """One model's entry in `RateLimitGovernor.get_usage_report`."""

    requests_today: int
    tokens_today: int
    requests_per_minute_limit: int
    # Prefer observed token limits; fall back to bootstrap limits before observation.
    tokens_per_minute_limit: int | None
    limits_observed: bool
    remaining_requests_today: int | None
    remaining_tokens_this_minute: int | None


def _observed(usage: ModelUsage, remaining: int | None) -> int | None:
    """Return a provider-reported remaining figure, or None before any headers arrived."""
    return remaining if usage.limits_observed else None


class RateLimitGovernor:
    """Thread-safe Groq quota admission with bounded waits and provider header observation."""

    def __init__(self) -> None:
        self._usage: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()
        logger.info("RateLimitGovernor initialized with %d model profiles", len(REGISTERED_MODELS))

    def _get_usage(self, model: str) -> ModelUsage:
        """Retrieve or initializes the ModelUsage tracker for a given model."""
        if model not in self._usage:
            self._usage[model] = ModelUsage()
        return self._usage[model]

    def _reset_if_new_day(self, usage: ModelUsage) -> None:
        """Reset daily counters when the tracker's date diverges from the current UTC date."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if usage.current_date != today:
            usage.requests_today = 0
            usage.tokens_today = 0
            usage.current_date = today
            logger.debug("daily counters reset for new UTC day")

    def _prune_window(self, usage: ModelUsage) -> None:
        """Drop entries older than `GOVERNOR.window_seconds` from both rolling windows."""
        cutoff = time.monotonic() - GOVERNOR.window_seconds
        while usage.requests_window and usage.requests_window[0] < cutoff:
            usage.requests_window.popleft()
        while usage.tokens_window and usage.tokens_window[0][0] < cutoff:
            usage.tokens_window.popleft()

    def _refresh_observed_windows(self, usage: ModelUsage) -> None:
        """Assume a header-observed window has rolled over once its deadline passes."""
        now = time.monotonic()
        if usage.reset_tokens_at is not None and now >= usage.reset_tokens_at:
            usage.remaining_tokens = usage.limit_tokens
            usage.reset_tokens_at = None
        if usage.reset_requests_at is not None and now >= usage.reset_requests_at:
            usage.remaining_requests = usage.limit_requests
            usage.reset_requests_at = None

    def wait_if_needed(self, model: str, estimated_tokens: int = 500) -> None:
        """Block the calling thread until rate limits allow the next request."""
        _require_registered(model)

        for attempt in range(GOVERNOR.max_wait_attempts):
            with self._lock:
                usage = self._get_usage(model)
                self._reset_if_new_day(usage)
                self._prune_window(usage)
                self._refresh_observed_windows(usage)

                wait_seconds = self._admit_or_wait_seconds(model, usage, estimated_tokens)
                if wait_seconds is None:
                    return
                logger.warning(
                    "Window limit approached for %s (attempt %d/%d). Sleeping %.1fs.",
                    model,
                    attempt + 1,
                    GOVERNOR.max_wait_attempts,
                    wait_seconds,
                )

            time.sleep(wait_seconds)

        raise RateLimitExceeded(
            f"{model} still over its per-minute quota after {GOVERNOR.max_wait_attempts} "
            f"sleep(s); estimated_tokens={estimated_tokens} may itself exceed the "
            "model's tokens-per-minute budget"
        )

    def _admit_or_wait_seconds(self, model: str, usage: ModelUsage, estimated_tokens: int) -> float | None:
        """Admit or calculate the next wait while the usage lock is held."""
        bootstrap = BOOTSTRAP_LIMITS[model]
        remaining_tokens = _observed(usage, usage.remaining_tokens)
        remaining_requests = _observed(usage, usage.remaining_requests)

        rpm_ok = len(usage.requests_window) < bootstrap["requests_per_minute"]

        tokens_this_window = sum(tok for _, tok in usage.tokens_window)
        if remaining_tokens is not None:
            tpm_ok = remaining_tokens - estimated_tokens >= 0
        else:
            tpm_ok = tokens_this_window + estimated_tokens <= bootstrap["tokens_per_minute"]

        # Do not invent a daily quota before the first observed header; RPM and TPM still bound admission.
        rpd_ok = remaining_requests is None or remaining_requests > 0

        if rpm_ok and tpm_ok and rpd_ok:
            now = time.monotonic()
            usage.requests_today += 1
            usage.tokens_today += estimated_tokens
            usage.requests_window.append(now)
            usage.tokens_window.append((now, estimated_tokens))
            if remaining_tokens is not None:
                usage.remaining_tokens = max(0, remaining_tokens - estimated_tokens)
            if remaining_requests is not None:
                usage.remaining_requests = max(0, remaining_requests - 1)
            logger.debug(
                "%s - today: %d req, %d tok | this window: %d req, %d tok",
                model,
                usage.requests_today,
                usage.tokens_today,
                len(usage.requests_window),
                tokens_this_window + estimated_tokens,
            )
            return None

        if not rpd_ok:
            reset_in = max(0.0, (usage.reset_requests_at or 0.0) - time.monotonic())
            raise RateLimitExceeded(
                f"{model} has exhausted its Groq-reported daily request budget "
                f"({usage.limit_requests} req/day); resets in {reset_in:.0f}s"
            )

        oldest = usage.requests_window[0] if usage.requests_window else time.monotonic()
        return max(1.0, GOVERNOR.window_seconds - (time.monotonic() - oldest))

    def observe_headers(self, model: str, headers: Mapping[str, str]) -> None:
        """Overwrite this model's provider-reported limits from a Groq response."""
        _require_registered(model)

        now = time.monotonic()
        token_values = _observe_axis(model, headers, "token-axis", _TOKEN_AXIS_HEADERS, now)
        request_values = _observe_axis(model, headers, "request-axis", _REQUEST_AXIS_HEADERS, now)

        if token_values is None and request_values is None:
            return

        with self._lock:
            usage = self._get_usage(model)
            if token_values is not None:
                usage.limit_tokens, usage.remaining_tokens, usage.reset_tokens_at = token_values
            if request_values is not None:
                usage.limit_requests, usage.remaining_requests, usage.reset_requests_at = request_values
            usage.limits_observed = True
            logger.debug(
                "%s - observed limits: req=%s/%s tok=%s/%s",
                model,
                usage.remaining_requests,
                usage.limit_requests,
                usage.remaining_tokens,
                usage.limit_tokens,
            )

    def record_usage(self, model: str, estimated_tokens: int, prompt_tokens: int, completion_tokens: int) -> None:
        """Reconcile actual usage against local estimates without debiting observed headers again."""
        _require_registered(model)

        delta = (prompt_tokens + completion_tokens) - estimated_tokens
        with self._lock:
            usage = self._get_usage(model)
            self._reset_if_new_day(usage)
            self._prune_window(usage)
            usage.tokens_today = max(0, usage.tokens_today + delta)
            self._apply_window_delta(usage, delta)

    def release_reservation(self, model: str, estimated_tokens: int) -> None:
        """Return a pre-flight reservation that a failed call never actually spent."""
        _require_registered(model)

        with self._lock:
            usage = self._get_usage(model)
            self._reset_if_new_day(usage)
            self._prune_window(usage)

            usage.requests_today = max(0, usage.requests_today - 1)
            usage.tokens_today = max(0, usage.tokens_today - estimated_tokens)
            if usage.requests_window:
                usage.requests_window.pop()
            self._apply_window_delta(usage, -estimated_tokens)

            if usage.limits_observed:
                if usage.remaining_tokens is not None and usage.limit_tokens is not None:
                    usage.remaining_tokens = min(usage.limit_tokens, usage.remaining_tokens + estimated_tokens)
                if usage.remaining_requests is not None and usage.limit_requests is not None:
                    usage.remaining_requests = min(usage.limit_requests, usage.remaining_requests + 1)

    def _apply_window_delta(self, usage: ModelUsage, delta: int) -> None:
        """Append a clamped correction entry to the token window, flooring the sum at 0."""
        if delta == 0:
            return
        current_sum = sum(tok for _, tok in usage.tokens_window)
        applied = max(delta, -current_sum)
        if applied != 0:
            usage.tokens_window.append((time.monotonic(), applied))

    def get_remaining_capacity(self, model: str) -> int:
        """Return the remaining rolling-window request capacity for a given model."""
        _require_registered(model)
        with self._lock:
            usage = self._get_usage(model)
            self._prune_window(usage)
            return max(0, BOOTSTRAP_LIMITS[model]["requests_per_minute"] - len(usage.requests_window))

    def get_usage_report(self) -> dict[str, ModelUsageReport]:
        """Compile a per-model usage snapshot for health check endpoints."""
        report: dict[str, ModelUsageReport] = {}
        with self._lock:
            for model in REGISTERED_MODELS:
                usage = self._get_usage(model)
                self._reset_if_new_day(usage)
                self._prune_window(usage)
                self._refresh_observed_windows(usage)
                bootstrap = BOOTSTRAP_LIMITS[model]
                report[model] = ModelUsageReport(
                    requests_today=usage.requests_today,
                    tokens_today=usage.tokens_today,
                    requests_per_minute_limit=bootstrap["requests_per_minute"],
                    tokens_per_minute_limit=(
                        usage.limit_tokens if usage.limits_observed else bootstrap["tokens_per_minute"]
                    ),
                    limits_observed=usage.limits_observed,
                    remaining_requests_today=usage.remaining_requests,
                    remaining_tokens_this_minute=usage.remaining_tokens,
                )
        return report


governor = RateLimitGovernor()
