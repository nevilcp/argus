"""Retry delay calculation shared by the data fetchers and the LLM decoder."""

from __future__ import annotations

import random


def capped_backoff(attempt: int, base: float, cap: float) -> float:
    """Returns the deterministic exponential back-off ceiling for an attempt.

    Args:
        attempt: Zero-based attempt index that just failed.
        base: Seconds for the first retry.
        cap: Upper bound in seconds.
    """
    return min(cap, base * 2**attempt)


def full_jitter_delay(
    attempt: int,
    base: float,
    cap: float,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Returns how long to wait before the next attempt.

    A server-supplied `retry_after` wins; otherwise the delay is drawn
    uniformly from [0, capped_backoff], so parallel callers don't
    thundering-herd a shared outage.

    Args:
        attempt: Zero-based attempt index that just failed.
        base: Seconds for the first retry.
        cap: Upper bound in seconds.
        retry_after: Delay the server asked for, if any.
        rng: Random source; tests pass a seeded one. Defaults to the module RNG.

    Returns:
        Seconds to sleep.
    """
    if retry_after is not None:
        return retry_after
    return (rng or random).uniform(0.0, capped_backoff(attempt, base, cap))
