"""
argus/orchestration/governor.py
===============================
Governor — Centralised rate limiter and quota tracker for LLM API calls.

Ensures that we do not breach Groq or Google free-tier rate limits.
Must be invoked before every single LLM call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

logger = logging.getLogger("argus.governor")


# ──────────────────────────────────────────────────────────────────────────────
# Rate Limit Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelLimits:
    rpm: int    # Requests per minute
    rpd: int    # Requests per day
    tpm: int    # Tokens per minute
    tpd: int    # Tokens per day (0 = unlimited)

MODEL_LIMITS: dict[str, ModelLimits] = {
    "gemini-3.1-flash-lite":   ModelLimits(rpm=15, rpd=500,   tpm=250_000, tpd=0),
    "llama-3.1-8b-instant":    ModelLimits(rpm=30, rpd=14_400, tpm=6_000,  tpd=500_000),
    "llama-3.3-70b-versatile": ModelLimits(rpm=30, rpd=1_000,  tpm=12_000, tpd=100_000),
    "llama-4-scout-17b":       ModelLimits(rpm=30, rpd=1_000,  tpm=30_000, tpd=500_000),
}


# ──────────────────────────────────────────────────────────────────────────────
# Exception
# ──────────────────────────────────────────────────────────────────────────────

class RateLimitExceeded(Exception):
    """Raised when the daily quota (RPD) for a model is exhausted."""
    def __init__(self, model: str, limit_type: str, current: int, limit: int) -> None:
        self.model = model
        self.limit_type = limit_type
        super().__init__(f"{model} {limit_type} limit: {current}/{limit}")


# ──────────────────────────────────────────────────────────────────────────────
# Governor Singleton
# ──────────────────────────────────────────────────────────────────────────────

class RateLimitGovernor:
    """
    Module-level singleton tracking LLM usage across all agents.
    Enforces RPM (via sleep) and RPD (via exceptions).
    """

    def __init__(self) -> None:
        self._call_log: dict[str, deque[float]] = defaultdict(deque)
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._daily_token_counts: dict[str, int] = defaultdict(int)
        self._last_reset_date: date = date.today()
        self._lock = threading.Lock()

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if the calendar day has changed."""
        if date.today() != self._last_reset_date:
            self._daily_counts.clear()
            self._daily_token_counts.clear()
            self._last_reset_date = date.today()

    def wait_if_needed(self, model: str, estimated_tokens: int = 500) -> None:
        """
        Block if the RPM limit is reached, or raise RateLimitExceeded
        if the daily limit is exhausted.

        Parameters
        ----------
        model:
            Model identifier matching a key in MODEL_LIMITS.
        estimated_tokens:
            Expected token usage for TPM warnings and daily accounting.
        """
        with self._lock:
            self._reset_daily_if_needed()

            if model not in MODEL_LIMITS:
                # Unrestricted model
                return

            limits = MODEL_LIMITS[model]

            # ── 1. Daily limit check ──
            if self._daily_counts[model] >= limits.rpd:
                raise RateLimitExceeded(model, "RPD", self._daily_counts[model], limits.rpd)

            # ── 2. RPM check (rolling 60s window) ──
            now = time.monotonic()
            log = self._call_log[model]
            
            # Prune calls older than 60 seconds
            while log and log[0] < now - 60.0:
                log.popleft()

            if len(log) >= limits.rpm:
                sleep_secs = 60.0 - (now - log[0]) + 0.2  # 200ms buffer
                logger.info("[Governor] %s RPM limit. Sleeping %.1fs...", model, sleep_secs)
                time.sleep(max(sleep_secs, 0.0))
                
                # Cleanup again after sleep
                now = time.monotonic()
                while log and log[0] < now - 60.0:
                    log.popleft()

            # ── 3. TPM check ──
            if limits.tpm > 0 and estimated_tokens > limits.tpm * 0.9:
                logger.warning(
                    "[Governor] %s single call uses %d tokens (%.0f%% of TPM limit)",
                    model, estimated_tokens, (estimated_tokens / limits.tpm) * 100
                )

            # ── 4. Log this call ──
            self._call_log[model].append(time.monotonic())
            self._daily_counts[model] += 1
            self._daily_token_counts[model] += estimated_tokens

    def record_actual_tokens(self, model: str, actual_tokens: int, estimated_tokens: int = 500) -> None:
        """
        Replaces the estimated token count with the actual from the API response.
        """
        with self._lock:
            self._reset_daily_if_needed()
            # Adjust the daily total by the difference
            self._daily_token_counts[model] += (actual_tokens - estimated_tokens)

    def get_usage_report(self) -> dict[str, Any]:
        """
        Generate a snapshot of today's LLM consumption across all models.
        """
        with self._lock:
            self._reset_daily_if_needed()
            report: dict[str, Any] = {}
            total_calls = 0
            
            for model, limits in MODEL_LIMITS.items():
                calls = self._daily_counts[model]
                tokens = self._daily_token_counts[model]
                total_calls += calls
                
                pct = (calls / limits.rpd) if limits.rpd > 0 else 0.0
                report[model] = {
                    "calls_today": calls,
                    "rpd_limit": limits.rpd,
                    "pct_used": f"{pct:.1%}",
                    "tokens_today": tokens,
                    "tpd_limit": limits.tpd,
                }
            
            report["total_api_calls_today"] = total_calls
            report["projected_daily_cost"] = 0.0  # We only use free tiers in V2
            return report

    def get_remaining_capacity(self, model: str) -> int:
        """
        Return the number of remaining daily calls for a model.
        Useful for deciding whether to fetch fresh or use cache.
        """
        with self._lock:
            self._reset_daily_if_needed()
            if model not in MODEL_LIMITS:
                return 999_999
            limits = MODEL_LIMITS[model]
            return max(0, limits.rpd - self._daily_counts[model])


# Global singleton instance
governor = RateLimitGovernor()
