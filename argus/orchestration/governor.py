"""
argus/orchestration/governor.py

Centralized API rate-limit governance module across all LLM and data providers.

Responsibilities:
  - Enforce per-model tokens-per-minute and requests-per-minute quotas over a
    rolling 60s window
  - Provide cooperative back-pressure via timed waits before each API call
  - Learn a model's real limits from Groq's own rate-limit response headers,
    falling back to a conservative bootstrap floor until the first header
    arrives
  - Release a pre-flight reservation that a failed call never actually spent
  - Expose usage telemetry for health checks and governor reports

Not responsible for:
  - Retry logic (handled per-agent with exponential back-off)
  - Authentication or credential management (see config.py)
  - Kill-switch circuit breakers (see risk/kill_switch.py)

Groq reports the account's real limits on every response, and those limits
depend on the API key's tier — a fact this module cannot know in advance.
BOOTSTRAP_LIMITS below is seeded from settings.ARGUS_GROQ_RPM/TPM, a
conservative free-tier default, used only until a response header corrects it
for the tier actually in use. It is a floor, not a claim about any particular
account's tier.

This is one governor per process, not per account: it only sees calls made
by its own process. api/main.py enforces --workers 1 so the API never runs
two ignorant copies of it (see Dockerfile.api's CMD), but
scripts/reconcile_outcomes.py's Actions collector is a second, independent
process making calls against the same Groq key — a second governor this
module has no visibility into and cannot coordinate with (GOV-8).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Mapping, Optional

from argus.config import settings

logger = logging.getLogger("argus.governor")


class RateLimitExceeded(Exception):
    """Raised when a model's token or request quota is exhausted."""


class UnregisteredModel(ValueError):
    """Raised when a model with no known rate-limit profile is passed to the governor."""


# The two gpt-oss IDs the system actually calls (see argus/seams.py, GroqLLMClient
# construction sites in fundamental.py/sentiment.py/portfolio.py, sourced from
# argus/config.py's ARGUS_*_MODEL settings). Membership only — the numbers live
# in BOOTSTRAP_LIMITS below since headers are the real source.
REGISTERED_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
)

# Conservative free-tier bootstrap floor, applied uniformly to every registered
# model until observe_headers() corrects it per-model for the account's actual
# tier. Groq publishes no daily figure for either axis at this floor, so neither
# a requests-per-day nor a tokens-per-day number is invented here; both stay
# unenforced until a response header reports one.
BOOTSTRAP_LIMITS: dict[str, dict[str, int]] = {
    model: {
        "tokens_per_minute": settings.ARGUS_GROQ_TPM,
        "requests_per_minute": settings.ARGUS_GROQ_RPM,
    }
    for model in REGISTERED_MODELS
}

# Groq's rate-limit response headers, split by axis so one axis can be observed
# without the other (GOV-6) — a response is never required to carry both.
# x-ratelimit-*-requests is a DAILY figure; x-ratelimit-*-tokens is a PER-MINUTE
# figure. Groq publishes no header for RPM or TPD at all, so those two stay
# BOOTSTRAP_LIMITS-enforced even once headers arrive for the other two.
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

# Go-style duration components, as Groq's reset headers encode them
# ("2m59.56s", "7.66s", "88ms"). "ms" must be tried before "m" or "88ms" would
# parse as "88m" with a dangling "s".
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _parse_reset_duration(value: str) -> float:
    """Parses a Go-style duration string into seconds.

    Groq's x-ratelimit-reset-* headers use Go's time.Duration.String() format,
    not plain integers, but a bare number (no unit suffix) is accepted as
    seconds rather than rejected outright.

    Args:
        value: Duration string such as "2m59.56s", "7.66s", "88ms", or "60".

    Returns:
        Total duration in seconds.

    Raises:
        ValueError: If ``value`` contains no recognizable duration component
            and isn't a bare number either.
    """
    total = 0.0
    matched = False
    for amount, unit in _DURATION_COMPONENT_RE.findall(value):
        total += float(amount) * _DURATION_UNIT_SECONDS[unit]
        matched = True
    if matched:
        return total
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"unparseable Groq reset duration: {value!r}")


# Measured against representative reconstructed prompts (fundamental, sentiment,
# and portfolio system+user pairs built from tests/fixtures/market_data/): raw
# content averaged ~2.0 tokens/word (peak ~2.2 on the portfolio prompt's dense
# numeric signal table), far above the previous 1.3 guess, plus a constant
# ~15-token template overhead for the role wrapper and special tokens. gpt-oss
# uses a different tokenizer than the Llama-3.1 one this was originally measured
# against, but 2.2 is a deliberate over-estimate that still errs safe, so the
# value stays unchanged. Both figures are rounded up for headroom rather than
# fit exactly.
_WORD_TO_TOKEN_RATIO = 2.2
_CHAT_TEMPLATE_OVERHEAD_TOKENS = 20


def estimate_tokens(system_prompt: str, user_prompt: str, max_tokens: int) -> int:
    """Estimates a chat completion's total token consumption before making the call.

    Word count * _WORD_TO_TOKEN_RATIO plus a fixed chat-template overhead
    approximates prompt-side tokenization; max_tokens stands in for the
    completion side since actual completion length is unknown pre-flight.
    Single estimator shared by every GroqLLMClient call site, replacing three
    divergent per-agent estimates that used to disagree on whether the system
    prompt or the configured max_tokens counted at all.

    Args:
        system_prompt: The system message text.
        user_prompt: The user message text.
        max_tokens: The completion's configured max_tokens ceiling.

    Returns:
        Estimated total token count to reserve against the governor's budget.
    """
    word_count = len(system_prompt.split()) + len(user_prompt.split())
    return int(word_count * _WORD_TO_TOKEN_RATIO) + _CHAT_TEMPLATE_OVERHEAD_TOKENS + max_tokens


# Rolling admission window — replaces a fixed-minute bucket so a burst can never
# land 2x budget by straddling a bucket boundary (GOV-4).
_WINDOW_SECONDS = 60.0


@dataclass
class ModelUsage:
    """Per-model rolling usage counters, plus Groq's own header-reported limits once observed.

    ``requests_today``/``tokens_today`` are daily bootstrap counters, reset on
    UTC date rollover. ``requests_window``/``tokens_window`` hold the
    (monotonic timestamp[, tokens]) of every admitted call within the trailing
    ``_WINDOW_SECONDS``, pruned lazily on each access — the rolling
    replacement for a fixed-minute bucket. Both are informational and the sole
    enforcement basis before the first header arrives. ``limit_requests``/
    ``remaining_requests`` are the provider's own DAILY request accounting;
    ``limit_tokens``/``remaining_tokens`` are its PER-MINUTE token accounting.
    Both stay ``None`` — bootstrap enforcement only — until ``observe_headers``
    sets ``limits_observed``.
    """

    requests_today: int = 0
    tokens_today: int = 0
    current_date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    requests_window: Deque[float] = field(default_factory=deque)
    tokens_window: Deque[tuple[float, int]] = field(default_factory=deque)

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
    keep the tracked state true to what the provider actually reports, or
    ``release_reservation`` if the call never got a response at all.
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

    def _prune_window(self, usage: ModelUsage) -> None:
        """Drops entries older than ``_WINDOW_SECONDS`` from both rolling windows.

        Args:
            usage: Mutable ModelUsage instance to prune in place.
        """
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while usage.requests_window and usage.requests_window[0] < cutoff:
            usage.requests_window.popleft()
        while usage.tokens_window and usage.tokens_window[0][0] < cutoff:
            usage.tokens_window.popleft()

    def _refresh_observed_windows(self, usage: ModelUsage) -> None:
        """Assumes a header-observed window has rolled over once its deadline passes.

        No further response has arrived to confirm the reset, so this is an optimistic
        assumption bounded by the provider's own reported deadline — the same trust
        model the bootstrap counters' daily rollover already relies on.

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
        releases the lock, sleeps out the remainder of the rolling window's oldest
        entry, then re-checks. Total wait is capped at ``_MAX_WAIT_ATTEMPTS`` sleeps;
        a call still over budget after that raises rather than blocking indefinitely.
        If the provider-reported daily budget is genuinely exhausted, raises
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
                self._prune_window(usage)
                self._refresh_observed_windows(usage)

                rpm_ok = len(usage.requests_window) < bootstrap["requests_per_minute"]

                tokens_this_window = sum(tok for _, tok in usage.tokens_window)
                if usage.limits_observed and usage.remaining_tokens is not None:
                    tpm_ok = usage.remaining_tokens - estimated_tokens >= 0
                else:
                    tpm_ok = tokens_this_window + estimated_tokens <= bootstrap["tokens_per_minute"]

                if usage.limits_observed and usage.remaining_requests is not None:
                    rpd_ok = usage.remaining_requests > 0
                else:
                    rpd_ok = True

                if rpm_ok and tpm_ok and rpd_ok:
                    now = time.monotonic()
                    usage.requests_today += 1
                    usage.tokens_today += estimated_tokens
                    usage.requests_window.append(now)
                    usage.tokens_window.append((now, estimated_tokens))
                    if usage.limits_observed and usage.remaining_tokens is not None:
                        usage.remaining_tokens = max(0, usage.remaining_tokens - estimated_tokens)
                    if usage.limits_observed and usage.remaining_requests is not None:
                        usage.remaining_requests = max(0, usage.remaining_requests - 1)
                    logger.debug(
                        "[Governor] %s — today: %d req, %d tok | this window: %d req, %d tok",
                        model,
                        usage.requests_today,
                        usage.tokens_today,
                        len(usage.requests_window),
                        tokens_this_window + estimated_tokens,
                    )
                    return

                if not rpd_ok:
                    reset_in = max(0.0, (usage.reset_requests_at or 0.0) - time.monotonic())
                    raise RateLimitExceeded(
                        f"{model} has exhausted its Groq-reported daily request budget "
                        f"({usage.limit_requests} req/day); resets in {reset_in:.0f}s"
                    )

                oldest = usage.requests_window[0] if usage.requests_window else time.monotonic()
                wait_seconds = max(1.0, _WINDOW_SECONDS - (time.monotonic() - oldest))
                logger.warning(
                    "[Governor] Window limit approached for %s (attempt %d/%d). Sleeping %.1fs.",
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
        over local bookkeeping. The token and request axes are observed
        independently — a response carrying only one axis's headers still updates
        that axis rather than being discarded because the other axis is absent.
        A response missing rate-limit headers entirely is silently skipped rather
        than raising, since observing nothing is not an error.

        Args:
            model: Provider model identifier string.
            headers: Response headers from a Groq API call (case-insensitive mapping).

        Raises:
            UnregisteredModel: If ``model`` has no known rate-limit profile.
        """
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")

        has_token_axis = all(h in headers for h in _TOKEN_AXIS_HEADERS)
        has_request_axis = all(h in headers for h in _REQUEST_AXIS_HEADERS)
        if not has_token_axis and not has_request_axis:
            return

        now = time.monotonic()
        token_values: Optional[tuple[int, int, Optional[float]]] = None
        if has_token_axis:
            try:
                limit_tokens = int(headers["x-ratelimit-limit-tokens"])
                remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
            except (ValueError, TypeError) as e:
                logger.warning("[Governor] Malformed token-axis headers for %s: %s", model, e)
            else:
                reset_tokens_at = None
                try:
                    reset_tokens_at = now + _parse_reset_duration(headers["x-ratelimit-reset-tokens"])
                except ValueError as e:
                    logger.warning("[Governor] Unparseable token-axis reset for %s: %s", model, e)
                token_values = (limit_tokens, remaining_tokens, reset_tokens_at)

        request_values: Optional[tuple[int, int, Optional[float]]] = None
        if has_request_axis:
            try:
                limit_requests = int(headers["x-ratelimit-limit-requests"])
                remaining_requests = int(headers["x-ratelimit-remaining-requests"])
            except (ValueError, TypeError) as e:
                logger.warning("[Governor] Malformed request-axis headers for %s: %s", model, e)
            else:
                reset_requests_at = None
                try:
                    reset_requests_at = now + _parse_reset_duration(
                        headers["x-ratelimit-reset-requests"]
                    )
                except ValueError as e:
                    logger.warning("[Governor] Unparseable request-axis reset for %s: %s", model, e)
                request_values = (limit_requests, remaining_requests, reset_requests_at)

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
                "[Governor] %s — observed limits: req=%s/%s tok=%s/%s",
                model,
                usage.remaining_requests,
                usage.limit_requests,
                usage.remaining_tokens,
                usage.limit_tokens,
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
        only keeps the informational today/window counters honest.

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
            self._reset_if_new_day(usage)
            self._prune_window(usage)
            usage.tokens_today = max(0, usage.tokens_today + delta)
            self._apply_window_delta(usage, delta)

    def release_reservation(self, model: str, estimated_tokens: int) -> None:
        """Returns a pre-flight reservation that a failed call never actually spent.

        wait_if_needed debits a model's budget the moment it admits a call, before
        that call has actually reached the network. A terminal error or an
        exhausted retry loop (see GroqLLMClient.complete) means no tokens were
        ever spent and no request landed — without this, that reservation would
        sit in the local counters and the header-derived remaining figures for
        the rest of their windows, permanently shrinking real capacity for a
        call that never happened.

        Args:
            model: Provider model identifier string.
            estimated_tokens: The estimate originally passed to ``wait_if_needed``.

        Raises:
            UnregisteredModel: If ``model`` has no known rate-limit profile.
        """
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} has no registered rate-limit profile")

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
                    usage.remaining_tokens = min(
                        usage.limit_tokens, usage.remaining_tokens + estimated_tokens
                    )
                if usage.remaining_requests is not None and usage.limit_requests is not None:
                    usage.remaining_requests = min(usage.limit_requests, usage.remaining_requests + 1)

    def _apply_window_delta(self, usage: ModelUsage, delta: int) -> None:
        """Appends a clamped correction entry to the token window, flooring the sum at 0.

        Args:
            usage: Mutable ModelUsage instance whose tokens_window is corrected.
            delta: Signed token adjustment (actual - estimated for record_usage,
                or -estimated_tokens for a full release).
        """
        if delta == 0:
            return
        current_sum = sum(tok for _, tok in usage.tokens_window)
        applied = max(delta, -current_sum)
        if applied != 0:
            usage.tokens_window.append((time.monotonic(), applied))

    def get_remaining_capacity(self, model: str) -> int:
        """Returns the remaining rolling-window request capacity for a given model.

        Args:
            model: Provider model identifier string.

        Returns:
            Remaining requests allowed in the trailing window, or 0 if unregistered.
        """
        if model not in REGISTERED_MODELS:
            return 0
        with self._lock:
            usage = self._get_usage(model)
            self._prune_window(usage)
            return max(0, BOOTSTRAP_LIMITS[model]["requests_per_minute"] - len(usage.requests_window))

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
                self._prune_window(usage)
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
