"""
argus/orchestration/governor.py

Centralized API rate-limit governance module across all LLM and data providers.

Responsibilities:
  - Enforce per-model daily request and token quotas
  - Provide cooperative back-pressure via timed waits before each API call
  - Expose usage telemetry for health checks and governor reports

Not responsible for:
  - Retry logic (handled per-agent with exponential back-off)
  - Authentication or credential management (see config.py)
  - Kill-switch circuit breakers (see risk/kill_switch.py)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("argus.governor")


class RateLimitExceeded(Exception):
    """Raised when a model's daily request or token quota is exhausted."""

# Per-model token and request limits; tuned against each provider's free-tier daily caps
MODEL_LIMITS: dict[str, dict[str, int]] = {
    "llama-3.3-70b-versatile": {
        "requests_per_day": 1_000,
        "tokens_per_day": 100_000,
        "tokens_per_minute": 6_000,
        "requests_per_minute": 30,
    },
    "llama-3.1-8b-instant": {
        "requests_per_day": 14_400,
        "tokens_per_day": 500_000,
        "tokens_per_minute": 20_000,
        "requests_per_minute": 30,
    },
    "gemini-3.5-flash": {
        "requests_per_day": 1_500,
        "tokens_per_day": 1_000_000,
        "tokens_per_minute": 30_000,
        "requests_per_minute": 30,
    },
    "gemini-2.0-flash": {
        "requests_per_day": 1_500,
        "tokens_per_day": 1_000_000,
        "tokens_per_minute": 30_000,
        "requests_per_minute": 30,
    },
    "ProsusAI/finbert": {
        "requests_per_day": 10_000,
        "tokens_per_day": 1_000_000,
        "tokens_per_minute": 100_000,
        "requests_per_minute": 1_000,
    },
}


@dataclass
class ModelUsage:
    """Per-model rolling usage counters scoped to the current UTC day.

    Counters reset when ``current_date`` does not match the current UTC date,
    ensuring daily quotas refresh automatically at midnight UTC.
    """

    requests_today: int = 0
    tokens_today: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    current_date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    current_minute: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    )


class RateLimitGovernor:
    """Thread-safe governor enforcing daily and per-minute API quotas across all LLM providers.

    Designed for cooperative multi-threaded use: agents call ``wait_if_needed`` before
    each API invocation, causing a blocking sleep if adding the request would violate
    quotas. Counters reset automatically at their respective time boundaries.
    """

    def __init__(self) -> None:
        """Initializes empty per-model usage tracking."""
        self._usage: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()
        logger.info("RateLimitGovernor initialized with %d model profiles", len(MODEL_LIMITS))

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

    def wait_if_needed(self, model: str, estimated_tokens: int = 500) -> None:
        """Blocks the calling thread until rate limits allow the next API request.

        Checks both daily and per-minute quotas. If a per-minute limit would be
        exceeded, sleeps until the next UTC minute begins. If the daily limit is
        exhausted, raises ``RateLimitExceeded`` rather than proceeding.

        Args:
            model: Provider model identifier string.
            estimated_tokens: Estimated token consumption of the upcoming request.

        Raises:
            RateLimitExceeded: If the model's daily request or token quota is exhausted.
        """
        if model not in MODEL_LIMITS:
            return

        limits = MODEL_LIMITS[model]
        with self._lock:
            usage = self._get_usage(model)
            self._reset_if_new_day(usage)
            self._reset_if_new_minute(usage)

            daily_req_ok = usage.requests_today < limits["requests_per_day"]
            daily_tok_ok = usage.tokens_today + estimated_tokens <= limits["tokens_per_day"]
            minute_req_ok = usage.requests_this_minute < limits["requests_per_minute"]
            minute_tok_ok = (
                usage.tokens_this_minute + estimated_tokens <= limits["tokens_per_minute"]
            )

            if not (daily_req_ok and daily_tok_ok):
                logger.critical(
                    "[Governor] Daily limit reached for %s. req=%d/%d tok=%d/%d",
                    model,
                    usage.requests_today,
                    limits["requests_per_day"],
                    usage.tokens_today,
                    limits["tokens_per_day"],
                )
                raise RateLimitExceeded(
                    f"Daily quota exhausted for {model}: "
                    f"req={usage.requests_today}/{limits['requests_per_day']} "
                    f"tok={usage.tokens_today}/{limits['tokens_per_day']}"
                )

            if not (minute_req_ok and minute_tok_ok):
                seconds_left = 60 - datetime.now(timezone.utc).second
                logger.warning(
                    "[Governor] Minute limit approached for %s. Sleeping %ds.",
                    model,
                    seconds_left + 1,
                )
                time.sleep(seconds_left + 1)
                self._reset_if_new_minute(usage)

            usage.requests_today += 1
            usage.tokens_today += estimated_tokens
            usage.requests_this_minute += 1
            usage.tokens_this_minute += estimated_tokens

            logger.debug(
                "[Governor] %s — today: %d req, %d tok | this min: %d req, %d tok",
                model,
                usage.requests_today,
                usage.tokens_today,
                usage.requests_this_minute,
                usage.tokens_this_minute,
            )

    def get_remaining_capacity(self, model: str) -> int:
        """Returns the remaining daily request capacity for a given model.

        Args:
            model: Provider model identifier string.

        Returns:
            Remaining requests allowed today, or 0 if the model is unregistered.
        """
        if model not in MODEL_LIMITS:
            return 0
        with self._lock:
            usage = self._get_usage(model)
            self._reset_if_new_day(usage)
            return max(0, MODEL_LIMITS[model]["requests_per_day"] - usage.requests_today)

    def get_usage_report(self) -> dict:
        """Compiles a per-model usage snapshot for health check endpoints.

        Returns:
            Dict mapping model name → dict of today's request and token usage with limits.
        """
        report: dict[str, dict] = {}
        with self._lock:
            for model in MODEL_LIMITS:
                usage = self._get_usage(model)
                self._reset_if_new_day(usage)
                lim = MODEL_LIMITS[model]
                report[model] = {
                    "requests_today": usage.requests_today,
                    "requests_limit": lim["requests_per_day"],
                    "tokens_today": usage.tokens_today,
                    "tokens_limit": lim["tokens_per_day"],
                }
        return report


# Module-level singleton shared by all agents within a process
governor = RateLimitGovernor()
