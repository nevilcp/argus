"""Retry delay calculation shared by the data fetchers and the LLM decoder."""

from __future__ import annotations

import random


def capped_backoff(attempt: int, base: float, cap: float) -> float:
    """Return the deterministic exponential back-off ceiling for an attempt."""
    return min(cap, base * 2**attempt)


def full_jitter_delay(
    attempt: int,
    base: float,
    cap: float,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Return how long to wait before the next attempt."""
    if retry_after is not None:
        return retry_after
    return (rng or random).uniform(0.0, capped_backoff(attempt, base, cap))
