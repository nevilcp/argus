"""
argus/orchestration/governor.py

Centralized API rate-limit governance module across all LLM and data providers.

Responsibilities:
  - Enforce per-model tokens-per-minute and requests-per-minute quotas
  - Provide cooperative back-pressure via timed waits before each API call
  - Learn a model's real limits from Groq's own rate-limit response headers,
    falling back to a conservative bootstrap floor until the first header
    arrives
  - Expose usage telemetry for health checks and governor reports

Not responsible for:
  - Retry logic (handled per-agent with exponential back-off)
  - Authentication or credential management (see config.py)
  - Kill-switch circuit breakers (see risk/kill_switch.py)

Groq reports the account's real limits on every response, and those limits
depend on the API key's tier (Free vs Developer) — a fact this module cannot
know in advance. BOOTSTRAP_LIMITS below is the published Developer-plan
table (console.groq.com/docs/models, read 2026-08-12), used only until a
response header corrects it for the tier actually in use. It is a
conservative floor, not a claim about any particular account's tier.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional

logger = logging.getLogger("argus.governor")


class RateLimitExceeded(Exception):
    """Raised when a model's token or request quota is exhausted."""


class UnregisteredModel(ValueError):
    """Raised when a model with no known rate-limit profile is passed to the governor."""


# The two Llama IDs the system actually calls (see argus/seams.py, GroqLLMClient
# construction sites in fundamental.py/sentiment.py/portfolio.py). Membership only
# — the numbers live in BOOTSTRAP_LIMITS below since headers are the real source.
REGISTERED_MODELS: frozenset[str] = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    }
)

# Source: Groq's published Developer-plan model table (console.groq.com/docs/models),
# read 2026-08-12. Used as the pre-flight budget until observe_headers() corrects it
# for the account's actual tier. Groq publishes only TPM and RPM for these models —
# no daily column — so neither a requests-per-day nor a tokens-per-day figure is
# invented here; both stay unenforced until a response header reports one.
BOOTSTRAP_LIMITS: dict[str, dict[str, int]] = {
    "llama-3.3-70b-versatile": {
        "tokens_per_minute": 300_000,
        "requests_per_minute": 1_000,
    },
    "llama-3.1-8b-instant": {
        "tokens_per_minute": 250_000,
        "requests_per_minute": 1_000,
    },
}

# Groq's rate-limit response headers. x-ratelimit-limit-requests/-remaining-requests
# are a DAILY figure; x-ratelimit-limit-tokens/-remaining-tokens are a PER-MINUTE
# figure. Groq publishes no header for RPM or TPD at all, so those two axes stay
# BOOTSTRAP_LIMITS-enforced even once headers arrive for the other two.
_RATE_LIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

# Go-style duration components, as Groq's reset headers encode them
# ("2m59.56s", "7.66s", "88ms"). "ms" must be tried before "m" or "88ms" would
# parse as "88m" with a dangling "s".
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _parse_reset_duration(value: str) -> float:
    """Parses a Go-style duration string into seconds.

    Groq's x-ratelimit-reset-* headers use Go's time.Duration.String() format,
    not plain integers.

    Args:
        value: Duration string such as "2m59.56s", "7.66s", or "88ms".

    Returns:
        Total duration in seconds.

    Raises:
        ValueError: If ``value`` contains no recognizable duration component.
    """
    total = 0.0
    matched = False
    for amount, unit in _DURATION_COMPONENT_RE.findall(value):
        total += float(amount) * _DURATION_UNIT_SECONDS[unit]
        matched = True
    if not matched:
        raise ValueError(f"unparseable Groq reset duration: {value!r}")
    return total


def estimate_tokens(system_prompt: str, user_prompt: str, max_tokens: int) -> int:
    """Estimates a chat completion's total token consumption before making the call.

    Word count * 1.3 approximates prompt-side tokenization overhead; max_tokens
    stands in for the completion side since actual completion length is unknown
    pre-flight. Single estimator shared by every GroqLLMClient call site, replacing
    three divergent per-agent estimates that used to disagree on whether the
    system prompt or the configured max_tokens counted at all.

    Args:
        system_prompt: The system message text.
        user_prompt: The user message text.
        max_tokens: The completion's configured max_tokens ceiling.

    Returns:
        Estimated total token count to reserve against the governor's budget.
    """
    word_count = len(system_prompt.split()) + len(user_prompt.split())
    return int(word_count * 1.3) + max_tokens


@dataclass
class ModelUsage:
    """Per-model rolling usage counters, plus Groq's own header-reported limits once observed.

    ``requests_today``/``tokens_today``/``requests_this_minute``/``tokens_this_minute``
    are always tracked locally (informational, and the sole enforcement basis before
    the first header arrives). ``limit_requests``/``remaining_requests`` are the
    provider's own DAILY request accounting; ``limit_tokens``/``remaining_tokens`` are
    its PER-MINUTE token accounting. Both stay ``None`` — bootstrap enforcement only —
    until ``observe_headers`` sets ``limits_observed``.
    """

    requests_today: int = 0
    tokens_today: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    current_date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    current_minute: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    )

    limit_requests: Optional[int] = None
    limit_tokens: Optional[int] = None
    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    # time.monotonic() deadlines, immune to wall-clock adjustment
    reset_requests_at: Optional[float] = None
    reset_tokens_at: Optional[float] = None
    limits_observed: bool = False


class RateLimitGovernor:
    """Thread-safe governor enforcing Groq's real quotas, bootstrapped from a published floor.

    Designed for cooperative multi-threaded use: GroqLLMClient calls ``wait_if_needed``
    before each API invocation, causing a blocking sleep if adding the request would
    violate quotas, then ``observe_headers``/``record_usage`` after each response to
    keep the tracked state true to what the provider actually reports.
    """

    def __init__(self) -> None:
        """Initializes empty per-model usage tracking."""
        self._usage: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()
        logger.info("RateLimitGovernor initialized with %d model profiles", len(REGISTERED_MODELS))

    def _get_usage(self, model: str) -> ModelUsage:
        """Retrieves or initialises the ModelUsage tracker for a given model.

        Args:
            model: Provider model identifier string.

        Returns:
            Live ModelUsage instance for the model.
        """
        if model not in self._usage:
            self._usage[model] = ModelUsage()
        return self._usage[model]

    def _reset_if_new_day(self, usage: ModelUsage) -> None:
        """Resets daily counters when the tracker's date diverges from the current UTC date.

        Args:
            usage: Mutable ModelUsage instance to check and reset.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if usage.current_date != today:
            usage.requests_today = 0
            usage.tokens_today = 0
            usage.current_date = today
            logger.debug("RateLimitGovernor: daily counters reset for new UTC day")

    def _reset_if_new_minute(self, usage: ModelUsage) -> None:
        """Resets per-minute counters when the tracker's minute diverges from the current UTC minute.

        Args:
            usage: Mutable ModelUsage instance to check and reset.
        """
        now_minute = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        if usage.current_minute != now_minute:
            usage.requests_this_minute = 0
            usage.tokens_this_minute = 0
            usage.current_minute = now_minute

    def _refresh_observed_windows(self, usage: ModelUsage) -> None:
        """Assumes a header-observed window has rolled over once its deadline passes.

        No further response has arrived to confirm the reset, so this is an optimistic
        assumption bounded by the provider's own reported deadline — the same trust
        model the bootstrap counters' date/minute rollover already relies on.

        Args:
            usage: Mutable ModelUsage instance to check and reset.
        """
        now = time.monotonic()
        if usage.reset_tokens_at is not None and now >= usage.reset_tokens_at:
            usage.remaining_tokens = usage.limit_tokens
            usage.reset_tokens_at = None
        if usage.reset_requests_at is not None and now >= usage.reset_requests_at:
            usage.remaining_requests = usage.limit_requests
            usage.reset_requests_at = None

    _MAX_WAIT_ATTEMPTS = 3

    def wait_if_needed(self, model: str, estimated_tokens: int = 500) -> None:
        """Blocks the calling thread until rate limits allow the next request.

        Checks three independent axes: requests-per-minute (always bootstrap-enforced;
        Groq never reports an RPM header), tokens-per-minute (header-enforced once
        observed, bootstrap-enforced before that), and requests-per-day (header-enforced
        once observed, unenforced before that — matching Tier A's refusal to invent a
        daily figure nothing publishes). If a per-minute limit would be exceeded,
        releases the lock, sleeps out the remainder of the current minute, then
        re-checks. If the provider-reported daily budget is genuinely exhausted, raises
        immediately rather than sleeping — its reset window is on the order of a day,
        too long for a request thread to block on.

        Args:
            model: Provider model identifier string.
            estimated_tokens: Estimated token consumption of the upcoming request.

        Raises:
            UnregisteredModel: If ``model`` has no known rate-limit profile.
            RateLimitExceeded: If the daily request budget is exhausted, or the
                request still doesn't fit after ``_MAX_WAIT_ATTEMPTS`` sleeps —
                typically because estimated_tokens alone exceeds the tokens-per-minute
                budget.
        """
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")

        bootstrap = BOOTSTRAP_LIMITS[model]

        for attempt in range(self._MAX_WAIT_ATTEMPTS):
            with self._lock:
                usage = self._get_usage(model)
                self._reset_if_new_day(usage)
                self._reset_if_new_minute(usage)
                self._refresh_observed_windows(usage)

                rpm_ok = usage.requests_this_minute < bootstrap["requests_per_minute"]

                if usage.limits_observed and usage.remaining_tokens is not None:
                    tpm_ok = usage.remaining_tokens - estimated_tokens >= 0
                else:
                    tpm_ok = usage.tokens_this_minute + estimated_tokens <= bootstrap["tokens_per_minute"]

                if usage.limits_observed and usage.remaining_requests is not None:
                    rpd_ok = usage.remaining_requests > 0
                else:
                    rpd_ok = True

                if rpm_ok and tpm_ok and rpd_ok:
                    usage.requests_today += 1
                    usage.tokens_today += estimated_tokens
                    usage.requests_this_minute += 1
                    usage.tokens_this_minute += estimated_tokens
                    if usage.limits_observed and usage.remaining_tokens is not None:
                        usage.remaining_tokens = max(0, usage.remaining_tokens - estimated_tokens)
                    if usage.limits_observed and usage.remaining_requests is not None:
                        usage.remaining_requests = max(0, usage.remaining_requests - 1)
                    logger.debug(
                        "[Governor] %s — today: %d req, %d tok | this min: %d req, %d tok",
                        model,
                        usage.requests_today,
                        usage.tokens_today,
                        usage.requests_this_minute,
                        usage.tokens_this_minute,
                    )
                    return

                if not rpd_ok:
                    reset_in = max(0.0, (usage.reset_requests_at or 0.0) - time.monotonic())
                    raise RateLimitExceeded(
                        f"{model} has exhausted its Groq-reported daily request budget "
                        f"({usage.limit_requests} req/day); resets in {reset_in:.0f}s"
                    )

                wait_seconds = 60 - datetime.now(timezone.utc).second + 1
                logger.warning(
                    "[Governor] Minute limit approached for %s (attempt %d/%d). Sleeping %ds.",
                    model,
                    attempt + 1,
                    self._MAX_WAIT_ATTEMPTS,
                    wait_seconds,
                )

            time.sleep(wait_seconds)

        raise RateLimitExceeded(
            f"{model} still over its per-minute quota after {self._MAX_WAIT_ATTEMPTS} "
            f"sleep(s); estimated_tokens={estimated_tokens} may itself exceed the "
            "model's tokens-per-minute budget"
        )

    def observe_headers(self, model: str, headers: Mapping[str, str]) -> None:
        """Overwrites this model's provider-reported limits from a Groq response.

        Called after every Groq response, success and 429 alike (see
        GroqLLMClient._on_response in argus/seams.py). Provider numbers always win
        over local bookkeeping — a response missing any rate-limit header (e.g. a
        malformed or non-Groq stub in a test) is silently skipped rather than
        raising, since observing nothing is not an error.

        Args:
            model: Provider model identifier string.
            headers: Response headers from a Groq API call (case-insensitive mapping).

        Raises:
            UnregisteredModel: If ``model`` has no known rate-limit profile.
        """
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")

        if not all(h in headers for h in _RATE_LIMIT_HEADERS):
            return

        with self._lock:
            usage = self._get_usage(model)
            try:
                limit_requests = int(headers["x-ratelimit-limit-requests"])
                limit_tokens = int(headers["x-ratelimit-limit-tokens"])
                remaining_requests = int(headers["x-ratelimit-remaining-requests"])
                remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
                reset_requests_in = _parse_reset_duration(headers["x-ratelimit-reset-requests"])
                reset_tokens_in = _parse_reset_duration(headers["x-ratelimit-reset-tokens"])
            except (ValueError, TypeError) as e:
                logger.warning("[Governor] Malformed rate-limit headers for %s: %s", model, e)
                return

            now = time.monotonic()
            usage.limit_requests = limit_requests
            usage.limit_tokens = limit_tokens
            usage.remaining_requests = remaining_requests
            usage.remaining_tokens = remaining_tokens
            usage.reset_requests_at = now + reset_requests_in
            usage.reset_tokens_at = now + reset_tokens_in
            usage.limits_observed = True
            logger.debug(
                "[Governor] %s — observed limits: %d/%d req/day, %d/%d tok/min",
                model,
                remaining_requests,
                limit_requests,
                remaining_tokens,
                limit_tokens,
            )

    def record_usage(
        self, model: str, estimated_tokens: int, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Reconciles a completed call's actual token usage against its pre-flight estimate.

        Applies the delta (actual - estimated) to the local bootstrap counters only —
        never to the header-derived remaining_tokens, which observe_headers already
        refreshed with the provider's own post-call figure for this same response by
        the time this runs (GroqLLMClient calls observe_headers from its response hook
        before complete() returns). Groq also excludes cached tokens from rate-limit
        accounting, so this call's token_usage and the header-derived remaining can
        legitimately disagree — headers are authoritative for enforcement; this method
        only keeps the informational today/this-minute counters honest.

        Args:
            model: Provider model identifier string.
            estimated_tokens: The estimate originally passed to ``wait_if_needed``.
            prompt_tokens: Actual prompt tokens from response.response_metadata["token_usage"].
            completion_tokens: Actual completion tokens from the same source.

        Raises:
            UnregisteredModel: If ``model`` has no known rate-limit profile.
        """
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")

        delta = (prompt_tokens + completion_tokens) - estimated_tokens
        with self._lock:
            usage = self._get_usage(model)
            usage.tokens_today = max(0, usage.tokens_today + delta)
            usage.tokens_this_minute = max(0, usage.tokens_this_minute + delta)

    def get_remaining_capacity(self, model: str) -> int:
        """Returns the remaining per-minute request capacity for a given model.

        Args:
            model: Provider model identifier string.

        Returns:
            Remaining requests allowed in the current minute, or 0 if unregistered.
        """
        if model not in REGISTERED_MODELS:
            return 0
        with self._lock:
            usage = self._get_usage(model)
            self._reset_if_new_minute(usage)
            return max(
                0, BOOTSTRAP_LIMITS[model]["requests_per_minute"] - usage.requests_this_minute
            )

    def get_usage_report(self) -> dict:
        """Compiles a per-model usage snapshot for health check endpoints.

        Returns:
            Dict mapping model name → today's cumulative usage, the per-minute limit
            in force (header-observed tokens-per-minute once known, else the bootstrap
            floor), whether that limit is provider-observed or still a bootstrap guess,
            and the provider's own remaining counts where observed.
        """
        report: dict[str, dict] = {}
        with self._lock:
            for model in REGISTERED_MODELS:
                usage = self._get_usage(model)
                self._reset_if_new_day(usage)
                self._refresh_observed_windows(usage)
                bootstrap = BOOTSTRAP_LIMITS[model]
                report[model] = {
                    "requests_today": usage.requests_today,
                    "tokens_today": usage.tokens_today,
                    "requests_per_minute_limit": bootstrap["requests_per_minute"],
                    "tokens_per_minute_limit": (
                        usage.limit_tokens if usage.limits_observed else bootstrap["tokens_per_minute"]
                    ),
                    "limits_observed": usage.limits_observed,
                    "remaining_requests_today": usage.remaining_requests,
                    "remaining_tokens_this_minute": usage.remaining_tokens,
                }
        return report


# Module-level singleton shared by all agents within a process
governor = RateLimitGovernor()
