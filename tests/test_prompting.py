"""
Tests for argus/schemas/prompting.py's field_list() and schema_block() — the
two render functions that turn a Pydantic model's PromptText markers into the
literal text an LLM prompt shows. No agent is migrated onto this module yet
(issue #76 is expand-only), so these tests exercise it directly against
throwaway models built for each case.
"""

from typing import Annotated

import pytest
from pydantic import BaseModel

from argus.schemas.prompting import PromptText, field_list, schema_block


class _Flat(BaseModel):
    """A flat model with every field marked."""

    signal: Annotated[str, PromptText('"BULLISH|BEARISH|NEUTRAL"')]
    conviction: Annotated[float, PromptText("<float 0.0-1.0>")]


class _MissingMarker(BaseModel):
    """A flat model where one field carries no PromptText marker."""

    signal: Annotated[str, PromptText('"BULLISH|BEARISH|NEUTRAL"')]
    conviction: float


class _Nested(BaseModel):
    """A single nested-model field, unmarked as the contract requires."""

    ticker: Annotated[str, PromptText('""')]
    position: _Flat


class _NestedList(BaseModel):
    """A list-of-nested-model field, unmarked as the contract requires."""

    ticker: Annotated[str, PromptText('""')]
    positions: list[_Flat]


class _NestedWithMarker(BaseModel):
    """A nested-model field that incorrectly carries a PromptText marker."""

    position: Annotated[_Flat, PromptText("{}")]


class _NestedListWithMarker(BaseModel):
    """A list-of-nested-model field that incorrectly carries a PromptText marker."""

    positions: Annotated[list[_Flat], PromptText("[]")]


def test_schema_block_includes_every_field():
    """Every field's name and marker text appear in the rendered block."""
    result = schema_block(_Flat)
    assert result == '{"signal":"BULLISH|BEARISH|NEUTRAL","conviction":<float 0.0-1.0>}'


def test_schema_block_uses_marker_text_verbatim():
    """Marker text is copied through unmodified, not reformatted."""
    result = schema_block(_Flat)
    assert '"<float 0.0-1.0>"' not in result  # sanity: conviction's value has no outer quotes
    assert "<float 0.0-1.0>" in result


def test_field_list_includes_every_field_and_joins_with_comma_space():
    """field_list joins each field's marker text verbatim with ', '."""
    result = field_list(_Flat)
    assert result == '"BULLISH|BEARISH|NEUTRAL", <float 0.0-1.0>'


def test_schema_block_renders_nested_model_as_its_own_block():
    """A nested-model field renders as a JSON object, not a marker lookup."""
    result = schema_block(_Nested)
    assert result == (
        '{"ticker":"","position":{"signal":"BULLISH|BEARISH|NEUTRAL","conviction":<float 0.0-1.0>}}'
    )


def test_schema_block_wraps_list_of_nested_model_in_brackets():
    """A list-of-nested-model field renders its block wrapped in brackets."""
    result = schema_block(_NestedList)
    assert result == (
        '{"ticker":"","positions":[{"signal":"BULLISH|BEARISH|NEUTRAL","conviction":<float 0.0-1.0>}]}'
    )


def test_schema_block_raises_on_scalar_field_with_no_marker():
    """A scalar field with no PromptText marker fails loudly rather than rendering blank."""
    with pytest.raises(TypeError):
        schema_block(_MissingMarker)


def test_field_list_raises_on_scalar_field_with_no_marker():
    """field_list applies the same no-marker enforcement as schema_block."""
    with pytest.raises(TypeError):
        field_list(_MissingMarker)


def test_schema_block_raises_on_nested_model_field_carrying_a_marker():
    """A nested-model field must not carry a PromptText marker of its own."""
    with pytest.raises(TypeError):
        schema_block(_NestedWithMarker)


def test_schema_block_raises_on_nested_list_field_carrying_a_marker():
    """A list-of-nested-model field must not carry a PromptText marker of its own."""
    with pytest.raises(TypeError):
        schema_block(_NestedListWithMarker)
