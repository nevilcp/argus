"""
argus/structured_output.py

The expand half of the LLM seam refactor (issue #67): one structured-output
call replacing the three near-identical fence-strip/parse/validate/retry
blocks in agents/fundamental.py, agents/sentiment.py, and agents/portfolio.py.
Composed over the existing LLMClient transport port rather than widening it —
see argus/seams.py's LLMClient docstring. Used by agents/fundamental.py and
agents/sentiment.py; portfolio.py still carries its own copy pending #72.

Responsibilities:
  - Strip a markdown code fence from a raw LLM response, in any of its four
    observed shapes (unfenced, fenced, json-tagged, unterminated)
  - Parse the fence-stripped text as JSON and validate it against a
    caller-supplied Pydantic schema
  - Retry invalid-JSON and schema-violating responses up to a declared
    attempt cap, with exponential back-off between attempts
  - Optionally re-prompt with the prior failure appended (repair), so the
    model can correct itself
  - Raise a typed, stage-tagged error when every attempt is exhausted

Not responsible for:
  - Sending the request or governing rate limits (argus/seams.py's
    LLMClient / GroqLLMClient owns transport, and already retries transient
    network/API errors beneath this call)
  - Defining what a valid response looks like for any particular agent —
    callers supply the target schema
  - Deciding what to do when decoding ultimately fails — callers catch
    StructuredOutputError and degrade however fits their context
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from argus.params import STRUCTURED_OUTPUT
from argus.seams import LLMClient

logger = logging.getLogger("argus.structured_output")

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when every decode attempt is exhausted, naming which stage kept failing.

    Args:
        stage: "json_parse" if the response was never valid JSON, or
            "schema_validation" if it parsed but never matched the target schema.
        attempts: Number of attempts made before giving up.
        cause: The last attempt's underlying exception.
    """

    def __init__(self, stage: str, attempts: int, cause: Exception) -> None:
        """Builds the error message from stage, attempt count, and underlying cause."""
        self.stage = stage
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"{stage} failed after {attempts} attempt(s): {cause}")


def _strip_markdown_fence(raw: str) -> str:
    """Strips a markdown code fence from a raw LLM response, if present.

    Handles four shapes: unfenced text (returned unchanged), a closed fence
    (```...```), a json-tagged fence (```json...```), and an unterminated
    fence (a single opening marker with no closing one, which truncated
    generations have produced in practice). Only the leading and trailing
    fence markers are stripped, not every ``` occurrence in the text — a
    JSON string value that itself quotes a fenced snippet (e.g. a
    "reasoning" field citing code) must survive intact.

    Args:
        raw: The LLM's raw response text.

    Returns:
        The response text with any surrounding fence markers removed.
    """
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
    """Appends the prior attempt's failure to a prompt so the model can correct itself.

    Args:
        user_prompt: The original user prompt.
        last_error: ``str(exception)`` from the previous attempt's failure.

    Returns:
        The user prompt with a correction request appended.
    """
    return (
        f"{user_prompt}\n\n"
        f"Your previous response was invalid: {last_error}\n"
        "Return a corrected response satisfying the schema above."
    )


def decode(
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    repair: bool,
) -> T:
    """Sends a prompt and decodes the response into a validated instance of ``schema``.

    Owns everything between "send the prompt" and "hand back a usable
    object": strips a markdown fence, parses JSON, and validates against
    ``schema``. Invalid JSON and schema-violating JSON are each retried up
    to ``STRUCTURED_OUTPUT.max_attempts``, backing off exponentially between
    attempts. With ``repair=True``, a failed attempt's error is appended to
    the next attempt's prompt; with ``repair=False``, the same prompt is
    re-sent unchanged.

    ``llm_client.complete()`` may itself raise — a transport-level error
    (GroqLLMClient already retries transient ones beneath this call) or the
    governor's RateLimitExceeded/UnregisteredModel. None of those are caught
    here, so they propagate on the first occurrence: the governor has
    already exhausted its own bounded wait before raising either of its
    exceptions, and a second retry loop above it would only re-run into the
    same wall.

    Args:
        llm_client: Transport to send the prompt over.
        system_prompt: The system message text.
        user_prompt: The user message text.
        schema: Pydantic model class the decoded response must validate against.
        repair: Whether to append the prior failure to the next attempt's prompt.

    Returns:
        A validated instance of ``schema``.

    Raises:
        StructuredOutputError: If every attempt fails to produce valid JSON
            (stage="json_parse") or schema-valid data (stage="schema_validation").
    """
    last_error: Optional[Exception] = None
    stage = "json_parse"

    for attempt in range(STRUCTURED_OUTPUT.max_attempts):
        prompt = user_prompt
        if repair and last_error is not None:
            prompt = _repair_prompt(user_prompt, str(last_error))

        raw = llm_client.complete(system_prompt, prompt)
        stripped = _strip_markdown_fence(raw)

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            last_error = e
            stage = "json_parse"
        else:
            try:
                return schema.model_validate(data)
            except ValidationError as e:
                last_error = e
                stage = "schema_validation"

        if attempt == STRUCTURED_OUTPUT.max_attempts - 1:
            assert last_error is not None
            raise StructuredOutputError(stage, attempt + 1, last_error)

        delay = min(
            STRUCTURED_OUTPUT.backoff_max_seconds,
            STRUCTURED_OUTPUT.backoff_base_seconds * 2**attempt,
        )
        logger.warning(
            "[StructuredOutput] Attempt %d %s failed: %s", attempt + 1, stage, last_error
        )
        time.sleep(delay)

    raise AssertionError("unreachable: loop always returns or raises")
