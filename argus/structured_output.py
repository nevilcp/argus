"""Shared model response parsing, schema validation, repair, and retry policy."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from argus.governor import RateLimitExceeded, UnregisteredModel
from argus.infra.backoff import capped_backoff, full_jitter_delay
from argus.infra.diagnostics import Diagnostics
from argus.params import STRUCTURED_OUTPUT
from argus.ports import LLMClient, RetryableTransportError

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Every decode attempt was exhausted; names which stage kept failing."""

    def __init__(self, stage: str, attempts: int, cause: Exception) -> None:
        self.stage = stage
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"{stage} failed after {attempts} attempt(s): {cause}")


def _strip_markdown_fence(raw: str) -> str:
    """Strip a markdown code fence from a raw LLM response, if present."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    raw = raw[3:]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.strip()
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _repair_prompt(user_prompt: str, last_error: str) -> str:
    """Append the prior attempt's failure to a prompt so the model can correct itself."""
    return (
        f"{user_prompt}\n\n"
        f"Your previous response was invalid: {last_error}\n"
        "Return a corrected response satisfying the schema above."
    )


@dataclass(frozen=True)
class _Failure:
    """One failed decode attempt: what failed, at which stage, and any server retry hint."""

    error: Exception
    stage: str
    retry_after: float | None = None


def _attempt_once[T: BaseModel](
    llm_client: LLMClient, system_prompt: str, prompt: str, schema: type[T]
) -> T | _Failure:
    """Make one send-parse-validate attempt, returning the instance or why it failed."""
    try:
        raw = llm_client.complete(system_prompt, prompt)
    except RetryableTransportError as e:
        return _Failure(e, "transport", e.retry_after)
    try:
        data = json.loads(_strip_markdown_fence(raw))
    except json.JSONDecodeError as e:
        return _Failure(e, "json_parse")
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        return _Failure(e, "schema_validation")


def _retry_delay(attempt: int, failure: _Failure) -> float:
    """Return how long to sleep before the next attempt."""
    base = STRUCTURED_OUTPUT.backoff_base_seconds
    cap = STRUCTURED_OUTPUT.backoff_max_seconds
    if failure.stage == "transport":
        return full_jitter_delay(attempt, base, cap, retry_after=failure.retry_after)
    return capped_backoff(attempt, base, cap)


def decode[T: BaseModel](
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    repair: bool,
) -> T:
    """Decode and validate model JSON with one bounded retry budget and optional persistent repair context."""
    last_content_error: Exception | None = None

    for attempt in range(STRUCTURED_OUTPUT.max_attempts):
        prompt = user_prompt
        if repair and last_content_error is not None:
            prompt = _repair_prompt(user_prompt, str(last_content_error))

        result = _attempt_once(llm_client, system_prompt, prompt, schema)
        if not isinstance(result, _Failure):
            return result
        if result.stage != "transport":
            last_content_error = result.error

        if attempt == STRUCTURED_OUTPUT.max_attempts - 1:
            raise StructuredOutputError(result.stage, attempt + 1, result.error)

        logger.warning("Attempt %d %s failed: %s", attempt + 1, result.stage, result.error)
        time.sleep(_retry_delay(attempt, result))

    raise AssertionError("unreachable: loop always returns or raises")


def decode_or_degrade[T: BaseModel](
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    repair: bool,
    diagnostics: Diagnostics,
    label: str,
    reraise_governor: bool = False,
    decode_level: Literal["warning", "error"] = "warning",
    api_error_text: str = "API error",
) -> T | None:
    """Turn decoder failures into stage-specific diagnostics and None."""
    try:
        return decode(llm_client, system_prompt, user_prompt, schema, repair=repair)
    except StructuredOutputError as e:
        diagnostics.record(f"{label}: {e}", decode_level)
    except (RateLimitExceeded, UnregisteredModel) as e:
        if reraise_governor:
            raise
        diagnostics.record(f"{label}: rate limited: {e}")
    except Exception as e:
        diagnostics.record(f"{label}: {api_error_text}: {e}", "error")
    return None
