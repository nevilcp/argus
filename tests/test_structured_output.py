"""Tests argus/structured_output.py's decode() function."""

import json
from unittest import mock

import pytest
from pydantic import BaseModel

from argus.governor import RateLimitExceeded, UnregisteredModel
from argus.infra.diagnostics import Diagnostics
from argus.params import STRUCTURED_OUTPUT
from argus.ports import RetryableTransportError
from argus.seams import FixtureLLMClient
from argus.structured_output import (
    StructuredOutputError,
    _strip_markdown_fence,
    decode,
    decode_or_degrade,
)
from tests.helpers.paths import FIXTURES_DIR

VALID_JSON = '{"signal": "NEUTRAL", "conviction": 0.3}'
SCHEMA_INVALID_JSON = '{"signal": "NEUTRAL"}'


class _Verdict(BaseModel):
    """Minimal schema for exercising decode() without depending on any agent's schema."""

    signal: str
    conviction: float


def _mock_llm(*responses: str | BaseException) -> mock.Mock:
    """Build a Mock LLMClient whose complete() yields each response in order."""
    client = mock.Mock()
    client.complete.side_effect = list(responses)
    return client


def _transport_error(retry_after: float | None = None) -> RetryableTransportError:
    """Build the retryable transport failure the seam raises on a transient outage."""
    return RetryableTransportError(RuntimeError("connection reset"), retry_after=retry_after)


def test_strip_markdown_fence_unfenced_passthrough():
    assert _strip_markdown_fence('{"signal": "BULLISH"}') == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_closed_fence():
    raw = '```\n{"signal": "BULLISH"}\n```'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_json_tagged_fence():
    raw = '```json\n{"signal": "BULLISH"}\n```'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_unterminated_fence():
    raw = '```json\n{"signal": "BULLISH"}'
    assert _strip_markdown_fence(raw) == '{"signal": "BULLISH"}'


def test_strip_markdown_fence_preserves_embedded_backticks_in_string_values():
    raw = (
        "```json\n"
        '{"signal": "BULLISH", "conviction": 0.8, '
        '"reasoning": "See ```python\\nprint(1)\\n``` for reference"}\n'
        "```"
    )
    stripped = _strip_markdown_fence(raw)
    assert stripped == (
        '{"signal": "BULLISH", "conviction": 0.8, "reasoning": "See ```python\\nprint(1)\\n``` for reference"}'
    )
    assert json.loads(stripped)["reasoning"] == "See ```python\nprint(1)\n``` for reference"


def test_decode_returns_validated_instance_on_first_attempt():
    llm = _mock_llm('{"signal": "BULLISH", "conviction": 0.8}')

    result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="BULLISH", conviction=0.8)
    assert llm.complete.call_count == 1


def test_decode_strips_fence_before_validating():
    llm = _mock_llm('```json\n{"signal": "BEARISH", "conviction": 0.5}\n```')

    result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="BEARISH", conviction=0.5)


def test_decode_retries_invalid_json_then_succeeds():
    llm = _mock_llm("not json at all", VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


def test_decode_retries_schema_violation_then_succeeds():
    llm = _mock_llm(SCHEMA_INVALID_JSON, VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


def test_decode_retries_retryable_transport_error_then_succeeds():
    llm = _mock_llm(_transport_error(), VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode(llm, "system", "user", _Verdict, repair=False)

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert llm.complete.call_count == 2


def test_decode_exhausts_attempts_on_retryable_transport_error_raises_transport_stage():
    llm = _mock_llm(*(_transport_error() for _ in range(STRUCTURED_OUTPUT.max_attempts)))

    with (
        mock.patch("argus.structured_output.time.sleep"),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "transport"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts
    assert llm.complete.call_count == STRUCTURED_OUTPUT.max_attempts


def test_decode_honors_transport_retry_after_hint():
    llm = _mock_llm(_transport_error(retry_after=7.5), VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep") as mock_sleep:
        decode(llm, "system", "user", _Verdict, repair=False)

    mock_sleep.assert_called_once_with(7.5)


def test_decode_transport_retry_without_hint_uses_jittered_backoff():
    llm = _mock_llm(_transport_error(), VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep") as mock_sleep:
        decode(llm, "system", "user", _Verdict, repair=False)

    delay = mock_sleep.call_args.args[0]
    assert 0.0 <= delay <= STRUCTURED_OUTPUT.backoff_base_seconds


def test_decode_transport_retry_preserves_repair_context_from_earlier_content_failure():
    llm = _mock_llm(
        SCHEMA_INVALID_JSON,
        _transport_error(),
        SCHEMA_INVALID_JSON,
    )

    with mock.patch("argus.structured_output.time.sleep"), pytest.raises(StructuredOutputError):
        decode(llm, "system", "original user prompt", _Verdict, repair=True)

    assert llm.complete.call_count == 3
    third_call_user_prompt = llm.complete.call_args_list[2].args[1]
    assert "original user prompt" in third_call_user_prompt
    assert "Your previous response was invalid" in third_call_user_prompt


def test_decode_transport_retry_does_not_append_repair_text():
    llm = _mock_llm(_transport_error(), VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=True)

    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert second_call_user_prompt == "original user prompt"


def test_decode_exhausts_attempts_on_invalid_json_raises_json_parse_stage():
    llm = _mock_llm(*(["not json"] * STRUCTURED_OUTPUT.max_attempts))

    with (
        mock.patch("argus.structured_output.time.sleep"),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "json_parse"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts
    assert llm.complete.call_count == STRUCTURED_OUTPUT.max_attempts


def test_decode_exhausts_attempts_on_schema_violation_raises_schema_validation_stage():
    llm = _mock_llm(*([SCHEMA_INVALID_JSON] * STRUCTURED_OUTPUT.max_attempts))

    with (
        mock.patch("argus.structured_output.time.sleep"),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert exc_info.value.stage == "schema_validation"
    assert exc_info.value.attempts == STRUCTURED_OUTPUT.max_attempts


def test_decode_repair_enabled_appends_prior_failure_to_next_prompt():
    llm = _mock_llm("not json", VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=True)

    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert "original user prompt" in second_call_user_prompt
    assert "Your previous response was invalid" in second_call_user_prompt


def test_decode_repair_disabled_resends_prompt_unchanged():
    llm = _mock_llm("not json", VALID_JSON)

    with mock.patch("argus.structured_output.time.sleep"):
        decode(llm, "system", "original user prompt", _Verdict, repair=False)

    first_call_user_prompt = llm.complete.call_args_list[0].args[1]
    second_call_user_prompt = llm.complete.call_args_list[1].args[1]
    assert first_call_user_prompt == "original user prompt"
    assert second_call_user_prompt == "original user prompt"


def test_decode_propagates_rate_limit_exceeded_without_retry():
    llm = _mock_llm(RateLimitExceeded("daily budget exhausted"))

    with pytest.raises(RateLimitExceeded):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert llm.complete.call_count == 1


def test_decode_propagates_unregistered_model_without_retry():
    llm = _mock_llm(UnregisteredModel("no rate-limit profile"))

    with pytest.raises(UnregisteredModel):
        decode(llm, "system", "user", _Verdict, repair=False)

    assert llm.complete.call_count == 1


def test_decode_or_degrade_returns_the_decoded_object_and_records_nothing():
    diagnostics = Diagnostics()

    result = decode_or_degrade(
        _mock_llm(VALID_JSON),
        "system",
        "user",
        _Verdict,
        repair=False,
        diagnostics=diagnostics,
        label="x[T]",
    )

    assert result == _Verdict(signal="NEUTRAL", conviction=0.3)
    assert diagnostics.messages == []


def test_decode_or_degrade_notes_an_exhausted_decode():
    diagnostics = Diagnostics()

    with mock.patch("argus.structured_output.time.sleep"):
        result = decode_or_degrade(
            _mock_llm(*["not json"] * STRUCTURED_OUTPUT.max_attempts),
            "system",
            "user",
            _Verdict,
            repair=False,
            diagnostics=diagnostics,
            label="x[T]",
        )

    assert result is None
    assert len(diagnostics.messages) == 1
    assert diagnostics.messages[0].startswith("x[T]: json_parse failed after")


def test_decode_or_degrade_notes_a_governor_rejection():
    diagnostics = Diagnostics()
    llm = _mock_llm(RateLimitExceeded("budget gone"))

    result = decode_or_degrade(llm, "system", "user", _Verdict, repair=False, diagnostics=diagnostics, label="x[T]")

    assert result is None
    assert llm.complete.call_count == 1
    assert diagnostics.messages == ["x[T]: rate limited: budget gone"]


def test_decode_or_degrade_reraises_the_governor_when_asked():
    diagnostics = Diagnostics()

    with pytest.raises(UnregisteredModel):
        decode_or_degrade(
            _mock_llm(UnregisteredModel("no profile")),
            "system",
            "user",
            _Verdict,
            repair=False,
            diagnostics=diagnostics,
            label="x",
            reraise_governor=True,
        )

    assert diagnostics.messages == []


def test_decode_or_degrade_notes_an_unclassified_error_with_the_given_wording():
    diagnostics = Diagnostics()

    result = decode_or_degrade(
        _mock_llm(RuntimeError("boom")),
        "system",
        "user",
        _Verdict,
        repair=False,
        diagnostics=diagnostics,
        label="x",
        api_error_text="Custom API error",
    )

    assert result is None
    assert diagnostics.messages == ["x: Custom API error: boom"]


def test_decode_against_fundamental_fixture_response():

    class _FundamentalVerdict(BaseModel):
        signal: str
        conviction: float
        moat_score: int
        reasoning: str

    fixtures_dir = FIXTURES_DIR / "llm_responses"
    llm = FixtureLLMClient.from_fixture_file(fixtures_dir / "fundamental.json", key_fn=lambda _prompt: "AAPL")

    result = decode(llm, "system", "AAPL", _FundamentalVerdict, repair=False)

    assert result.signal == "BEARISH"
    assert result.conviction == 0.75
    assert result.moat_score == 8
