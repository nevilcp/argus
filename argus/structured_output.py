"""
argus/structured_output.py

Decodes and validates an LLM's raw JSON response against a Pydantic schema.

Replaces the three near-identical fence-strip/parse/validate/retry blocks
that used to live in agents/fundamental.py, agents/sentiment.py, and
agents/portfolio.py, which now all call in here instead. Composed over the
existing LLMClient transport port rather than widening it — see
argus/ports.py's LLMClient docstring.

Responsibilities:
  - Strip a markdown code fence from a raw LLM response, in any of its four
    observed shapes (unfenced, fenced, json-tagged, unterminated)
  - Parse the fence-stripped text as JSON and validate it against a
    caller-supplied Pydantic schema
  - Retry invalid-JSON and schema-violating responses up to a declared
    attempt cap, with exponential back-off between attempts
  - Retry a transport-level RetryableTransportError from the same attempt
    budget, honoring the provider's own retry hint when the adapter
    supplies one
  - Optionally re-prompt with the prior failure appended (repair), so the
    model can correct itself
  - Raise a typed, stage-tagged error when every attempt is exhausted

Not responsible for:
  - Sending the request or governing rate limits (argus/seams.py's
    LLMClient owns transport). An adapter makes one invocation per call and
    performs no retry loop of its own — this module is the only place that
    loops
  - Classifying which transport failures are worth retrying versus terminal
    — that stays with the LLMClient implementation (e.g. GroqLLMClient)
  - Defining what a valid response looks like for any particular agent —
    callers supply the target schema
  - Deciding what a failed decode means for the caller — decode() raises
    StructuredOutputError, and decode_or_degrade() turns it into a
    diagnostics note and None for callers that degrade rather than abort
"""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

from pydantic import BaseModel, ValidationError

from argus.governor import RateLimitExceeded, UnregisteredModel
from argus.infra.backoff import capped_backoff, full_jitter_delay
from argus.infra.diagnostics import Diagnostics
from argus.params import STRUCTURED_OUTPUT
from argus.ports import LLMClient, RetryableTransportError

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Every decode attempt was exhausted; names which stage kept failing.

    Attributes:
        stage: "json_parse" if the response was never valid JSON,
            "schema_validation" if it parsed but never matched the target
            schema, or "transport" if the LLMClient call itself kept failing
            transiently.
        attempts: Number of attempts made before giving up.
        cause: The last attempt's underlying exception.
    """

    def __init__(self, stage: str, attempts: int, cause: Exception) -> None:
        """Builds the error message from stage, attempt count, and underlying cause.

        Args:
            stage: Which decode stage kept failing — "json_parse",
                "schema_validation", or "transport".
            attempts: Number of attempts made before giving up.
            cause: The last attempt's underlying exception.
        """
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


def decode[T: BaseModel](
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
    ``schema``. Invalid JSON, schema-violating JSON, and a retryable
    transport failure are each retried up to ``STRUCTURED_OUTPUT.max_attempts``,
    backing off exponentially between attempts (full-jitter for an unhinted
    transport retry, to avoid the parallel agents thundering-herding a
    shared provider outage; deterministic otherwise). With ``repair=True``,
    the most recent json_parse/schema_validation failure's error is kept
    appended to the prompt across attempts, including one interrupted by a
    transport retry in between — a transport hiccup carries no content
    feedback of its own, so it must not discard repair context that was
    about to be sent.

    ``llm_client.complete()`` may itself raise. A RetryableTransportError is
    retried from this same attempt budget — honoring its ``retry_after``
    hint when the adapter supplied one, else the same exponential back-off
    used for invalid-JSON/schema-violation retries. Anything else (a
    terminal transport error, or the governor's
    RateLimitExceeded/UnregisteredModel) propagates immediately: the
    governor has already exhausted its own bounded wait before raising
    either of its exceptions, and a terminal transport error would only
    fail identically on a second attempt.

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
            (stage="json_parse"), schema-valid data (stage="schema_validation"),
            or a successful transport call (stage="transport").
    """
    last_error: Exception | None = None
    last_content_error: Exception | None = None
    stage = "json_parse"
    retry_after: float | None = None

    for attempt in range(STRUCTURED_OUTPUT.max_attempts):
        prompt = user_prompt
        if repair and last_content_error is not None:
            prompt = _repair_prompt(user_prompt, str(last_content_error))

        try:
            raw = llm_client.complete(system_prompt, prompt)
        except RetryableTransportError as e:
            last_error = e
            stage = "transport"
            retry_after = e.retry_after
        else:
            retry_after = None
            stripped = _strip_markdown_fence(raw)
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as e:
                last_error = e
                last_content_error = e
                stage = "json_parse"
            else:
                try:
                    return schema.model_validate(data)
                except ValidationError as e:
                    last_error = e
                    last_content_error = e
                    stage = "schema_validation"

        if attempt == STRUCTURED_OUTPUT.max_attempts - 1:
            assert last_error is not None
            raise StructuredOutputError(stage, attempt + 1, last_error)

        base = STRUCTURED_OUTPUT.backoff_base_seconds
        cap = STRUCTURED_OUTPUT.backoff_max_seconds
        if stage == "transport":
            delay = full_jitter_delay(attempt, base, cap, retry_after=retry_after)
        else:
            delay = retry_after if retry_after is not None else capped_backoff(attempt, base, cap)
        logger.warning("Attempt %d %s failed: %s", attempt + 1, stage, last_error)
        time.sleep(delay)

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
    """Runs ``decode`` and turns every failure into a diagnostics note and ``None``.

    Replaces the except ladder each LLM agent used to carry. The three failure
    classes are noted differently so a reader can tell them apart:
    ``StructuredOutputError`` (``"<label>: <error>"``), a governor rejection
    (``"<label>: rate limited: <error>"``), and any other exception
    (``"<label>: <api_error_text>: <error>"``, always at error level).

    Args:
        llm_client: Transport to send the prompt over.
        system_prompt: The system message text.
        user_prompt: The user message text.
        schema: Pydantic model class the decoded response must validate against.
        repair: Passed through to ``decode``.
        diagnostics: Receives one note per failure.
        label: Prefix of every note, e.g. ``"fundamental_analysis[AAPL]"``.
        reraise_governor: When True, ``RateLimitExceeded`` and
            ``UnregisteredModel`` propagate instead of being noted. For a
            caller with no per-item fallback to degrade to.
        decode_level: Log level for a ``StructuredOutputError`` note.
        api_error_text: Wording between the label and the error for an
            unclassified exception.

    Returns:
        A validated instance of ``schema``, or None if the call failed.

    Raises:
        RateLimitExceeded: Only when ``reraise_governor`` is True.
        UnregisteredModel: Only when ``reraise_governor`` is True.
    """
    try:
        return decode(llm_client, system_prompt, user_prompt, schema, repair=repair)
    except StructuredOutputError as e:
        diagnostics.record(f"{label}: {e}", decode_level)
    except (RateLimitExceeded, UnregisteredModel) as e:
        if reraise_governor:
            raise
        # Governor already exhausted its bounded wait; degrade rather than retry into it.
        diagnostics.record(f"{label}: rate limited: {e}")
    except Exception as e:
        diagnostics.record(f"{label}: {api_error_text}: {e}", "error")
    return None
