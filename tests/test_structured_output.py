"""
Tests for argus/structured_output.py's decode() — fence-stripping, the
retry/repair loop, and typed-error reporting. No agent is migrated onto
this decoder yet (issue #69 is expand-only), so these tests exercise it
directly against a mocked LLMClient rather than through any agent.
"""

from unittest import mock

import pytest
from pydantic import BaseModel

from argus.orchestration.governor import RateLimitExceeded, UnregisteredModel
from argus.params import STRUCTURED_OUTPUT
from argus.seams import RetryableTransportError
from argus.structured_output import StructuredOutputError, _strip_markdown_fence, decode


class _Verdict(BaseModel):
    """Minimal schema for exercising decode() without depending on any agent's schema."""

    signal: str
    conviction: float


def _mock_llm(*raw_responses: str) -> mock.Mock:
    """Builds a Mock LLMClient whose complete() yields each response in order.

    Args:
        raw_responses: Raw response text to return on successive calls.

    Returns:
        A Mock exposing complete(system_prompt, user_prompt) -> str.
    """
    client = mock.Mock()
    client.complete.side_effect = list(raw_responses)
    return client


# ---------------------------------------------------------------------------
# Fence-stripping
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_unfenced_passthrough():
    """Text with no fence markers is returned unchanged (aside from whitespace)."""
    assert _strip_markdown_fence('{"signal": "BULLISH"}') == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_closed_fence():
    """A closed, untagged fence has its markers removed."""
    raw = '```\n{"signal": "BULLISH"}\n```'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_json_tagged_fence():
    """A closed, ```json-tagged fence has both the markers and the language tag removed."""
    raw = '```json\n{"signal": "BULLISH"}\n```'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_unterminated_fence():
    """A single opening ```json marker with no closing fence still strips cleanly."""
    raw = '```json\n{"signal": "BULLISH"}'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_preserves_embedded_backticks_in_string_values():
    """A JSON string value quoting its own fenced snippet must survive intact.

    A global split on every ``` occurrence would truncate the payload at the
    first embedded fence rather than only stripping the outer one.
    """
    raw = (
        '```json\n'
        '{"signal": "BULLISH", "conviction": 0.8, '
        '"reasoning": "See ```python\\nprint(1)\\n``` for reference"}\n'
        '```'
    )
    stripped = _strip_markdown_fence(raw)
    assert stripped == (
        '{"signal": "BULLISH", "conviction": 0.8, '
        '"reasoning": "See ```python\\nprint(1)\\n``` for reference"}'
    )
    import json

    assert json.loads(stripped)["reasoning"] == "See ```python\nprint(1)\n``` for reference"


# ---------------------------------------------------------------------------
# decode(): success paths
# ---------------------------------------------------------------------------


def test_decode_returns_validated_instance_on_first_attempt():
    """A well-formed, schema-valid response decodes on the first attempt."""
    llm = _mock_llm('{"signal": "BULLISH", "conviction": 0.8}')

    result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="BULLISH", conviction=0.8)
    assert llm.complete.call_count == 1


def test_decode_strips_fence_before_validating():
    """A fenced response is stripped before JSON parsing, not rejected as invalid."""
    llm = _mock_llm('```json\n{"signal": "BEARISH", "conviction": 0.5}\n```')

    result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="BEARISH", conviction=0.5)


def test_decode_retries_invalid_json_then_succeeds():
    """Invalid JSON on the first attempt is retried and a later valid response succeeds."""
    llm = _mock_llm("not json at all", '{"signal": "NEUTRAL", "conviction": 0.3}')

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


def test_decode_retries_schema_violation_then_succeeds():
    """Valid JSON that fails schema validation is retried and a later valid response succeeds."""
    llm = _mock_llm(
        '{"signal": "NEUTRAL"}',  # missing required "conviction"
        '{"signal": "NEUTRAL", "conviction": 0.3}',
    )

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# decode(): retryable transport failures
# ---------------------------------------------------------------------------


def test_decode_retries_retryable_transport_error_then_succeeds():
    """A RetryableTransportError from the transport is retried and the eventual success is returned."""
    llm = mock.Mock()
    llm.complete.side_effect = [
        RetryableTransportError(RuntimeError("connection reset")),
        '{"signal": "NEUTRAL", "conviction": 0.3}',
    ]

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


def test_decode_exhausts_attempts_on_retryable_transport_error_raises_transport_stage():
    """A persistent transport failure costs the declared attempt cap, then raises stage=transport."""
    llm = mock.Mock()
    llm.complete.side_effect = [
        RetryableTransportError(RuntimeError("connection reset"))
        for _ in range(STRUCTURED_OUTPUT.max_attempts)
    ]

    with mock.patch("argus.structured_output.time.sleep"):
        with pytest.raises(StructuredOutputError) as exc_info:
            decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "transport"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts
    assert llm.complete.call_count == STRUCTURED_OUTPUT.max_attempts


def test_decode_honors_transport_retry_after_hint():
    """A RetryableTransportError's retry_after hint is used as the sleep delay verbatim."""
    llm = mock.Mock()
    llm.complete.side_effect = [
        RetryableTransportError(RuntimeError("rate limited"), retry_after=7.5),
        '{"signal": "NEUTRAL", "conviction": 0.3}',
    ]

    with mock.patch("argus.structured_output.time.sleep") as mock_sleep:
        decode(llm, "system", "user", _Verdict, repair=False)

    mock_sleep.assert_called_once_with(7.5)


def test_decode_transport_retry_without_hint_uses_jittered_backoff():
    """Without a retry_after hint, a transport retry uses full-jitter back-off, not a fixed delay.

    The adapter's deleted internal loop used full jitter specifically to
    desynchronize retries across the parallel agents during a shared
    provider outage; decode() must preserve that for the unhinted case.
    """
    llm = mock.Mock()
    llm.complete.side_effect = [
        RetryableTransportError(RuntimeError("connection reset")),
        '{"signal": "NEUTRAL", "conviction": 0.3}',
    ]

    with mock.patch("argus.structured_output.time.sleep") as mock_sleep:
        decode(llm, "system", "user", _Verdict, repair=False)

    delay = mock_sleep.call_args.args[0]
    assert 0.0 <= delay <= STRUCTURED_OUTPUT.backoff_base_seconds


def test_decode_transport_retry_preserves_repair_context_from_earlier_content_failure():
    """A transport hiccup between two content failures does not discard the pending repair text."""
    llm = mock.Mock()
    llm.complete.side_effect = [
        '{"signal": "NEUTRAL"}',  # missing required "conviction" -> schema_validation failure
        RetryableTransportError(RuntimeError("connection reset")),  # transport hiccup
        '{"signal": "NEUTRAL"}',  # still missing "conviction" -> exhausts attempts
    ]

    with mock.patch("argus.structured_output.time.sleep"):
        with pytest.raises(StructuredOutputError):
            decode(llm, "system", "original user prompt", _Verdict, repair=True)

    assert llm.complete.call_count == 3
    third_call_user_prompt = llm.complete.call_args_list[2].args[1]
    assert "original user prompt" in third_call_user_prompt
    assert "Your previous response was invalid" in third_call_user_prompt


def test_decode_transport_retry_does_not_append_repair_text():
    """A transport retry resends the original prompt — repair only makes sense for content failures."""
    llm = mock.Mock()
    llm.complete.side_effect = [
        RetryableTransportError(RuntimeError("connection reset")),
        '{"signal": "NEUTRAL", "conviction": 0.3}',
    ]

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=True)

    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert second_call_user_prompt == "original user prompt"


# ---------------------------------------------------------------------------
# decode(): exhausted-attempts failure paths
# ---------------------------------------------------------------------------


def test_decode_exhausts_attempts_on_invalid_json_raises_json_parse_stage():
    """Persistently invalid JSON is retried up to the attempt cap, then raises stage=json_parse."""
    llm = _mock_llm(*(["not json"] * STRUCTURED_OUTPUT.max_attempts))

    with mock.patch("argus.structured_output.time.sleep"):
        with pytest.raises(StructuredOutputError) as exc_info:
            decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "json_parse"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts
    assert llm.complete.call_count == STRUCTURED_OUTPUT.max_attempts


def test_decode_exhausts_attempts_on_schema_violation_raises_schema_validation_stage():
    """Persistently schema-invalid JSON is retried up to the cap, then raises stage=schema_validation."""
    llm = _mock_llm(*(['{"signal": "NEUTRAL"}'] * STRUCTURED_OUTPUT.max_attempts))

    with mock.patch("argus.structured_output.time.sleep"):
        with pytest.raises(StructuredOutputError) as exc_info:
            decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "schema_validation"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts


# ---------------------------------------------------------------------------
# decode(): repair flag
# ---------------------------------------------------------------------------


def test_decode_repair_enabled_appends_prior_failure_to_next_prompt():
    """With repair=True, the retry's prompt carries the previous attempt's error."""
    llm = _mock_llm("not json", '{"signal": "NEUTRAL", "conviction": 0.3}')

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=True)

    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert "original user prompt" in second_call_user_prompt
    assert "Your previous response was invalid" in second_call_user_prompt


def test_decode_repair_disabled_resends_prompt_unchanged():
    """With repair=False, every attempt sends the exact same, unmodified prompt."""
    llm = _mock_llm("not json", '{"signal": "NEUTRAL", "conviction": 0.3}')

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=False)

    first_call_user_prompt = llm.complete.call_args_list[0].args[1]
    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert first_call_user_prompt == "original user prompt"
    assert second_call_user_prompt == "original user prompt"


# ---------------------------------------------------------------------------
# decode(): governor exceptions propagate unretried
# ---------------------------------------------------------------------------


def test_decode_propagates_rate_limit_exceeded_without_retry():
    """A RateLimitExceeded from the transport propagates immediately, with no retry attempt."""
    llm = mock.Mock()
    llm.complete.side_effect = RateLimitExceeded("daily budget exhausted")

    with pytest.raises(RateLimitExceeded):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert llm.complete.call_count == 1


def test_decode_propagates_unregistered_model_without_retry():
    """An UnregisteredModel from the transport propagates immediately, with no retry attempt."""
    llm = mock.Mock()
    llm.complete.side_effect = UnregisteredModel("no rate-limit profile")

    with pytest.raises(UnregisteredModel):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert llm.complete.call_count == 1


# ---------------------------------------------------------------------------
# decode() against real, fixture-captured responses
# ---------------------------------------------------------------------------


def test_decode_against_fundamental_fixture_response():
    """decode() validates a real, pre-captured LLM response, not just synthetic JSON."""
    from pathlib import Path

    from argus.seams import FixtureLLMClient

    class _FundamentalVerdict(BaseModel):
        signal: str
        conviction: float
        moat_score: int
        reasoning: str

    fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "llm_responses"
    llm = FixtureLLMClient.from_fixture_file(
        fixtures_dir / "fundamental.json", key_fn=lambda _prompt: "AAPL"
    )

    result = decode(llm, "system", "AAPL", _FundamentalVerdict, repair=False)

    assert result.signal == "BEARISH"
    assert result.conviction == 0.75
    assert result.moat_score == 8
