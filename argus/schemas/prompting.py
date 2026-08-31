"""
argus/schemas/prompting.py

Renders each agent's declared LLM output schema directly from its Pydantic
verdict/proposal model, so the prompt text shown to the model and the
constraints its response is validated against cannot drift apart (issue
#76). Depends on pydantic and nothing else — deliberately kept out of
argus/structured_output.py, which reaches groq, httpx, langchain_groq, and
pandas through argus/seams.py, and out of argus/schemas/signals.py's own
import surface being widened with anything heavier, since that module is
imported by nearly everything in the system.

Responsibilities:
  - Define the PromptText marker attached to a field's Annotated metadata,
    carrying the exact text an LLM is shown for that field
  - Render a model's fields as a prose list (field_list) or a JSON schema
    block (schema_block), recursing into a nested model or a list of them
  - Raise when a field's declaration disagrees with its shape: a scalar
    field with no marker, or a nested-model field that carries one

Not responsible for:
  - Sending prompts or decoding responses (see argus/structured_output.py)
  - Defining validation constraints themselves (see argus/schemas/signals.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class PromptText:
    """Annotated marker carrying the exact text an LLM is shown for one field.

    Args:
        text: Verbatim text for this field — prose for field_list, a JSON
            value (already quoted if it's a string) for schema_block.
    """

    text: str


def _prompt_text(field: FieldInfo) -> PromptText | None:
    """Returns the PromptText marker in a field's metadata, if any."""
    for item in field.metadata:
        if isinstance(item, PromptText):
            return item
    return None


def _require_prompt_text(model: type[BaseModel], name: str, field: FieldInfo) -> str:
    """Returns a field's marker text, or raises naming the field that lacks one."""
    marker = _prompt_text(field)
    if marker is None:
        raise TypeError(f"{model.__name__}.{name} has no PromptText marker")
    return marker.text


def _nested_model(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """Identifies a field annotation as a direct or list-wrapped BaseModel.

    Args:
        annotation: A field's ``FieldInfo.annotation``.

    Returns:
        ``(model, is_list)`` where ``model`` is the nested BaseModel type,
        or ``(None, False)`` if the annotation is neither a BaseModel nor a
        list of one.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False
    if get_origin(annotation) is list:
        args = get_args(annotation)
        if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0], True
    return None, False


def field_list(model: type[BaseModel]) -> str:
    """Renders every field's PromptText marker as a comma-space-joined prose list.

    Each marker's text already names its own field, so no field name is
    added by this function.

    Args:
        model: Pydantic model whose every field carries a PromptText marker.

    Returns:
        The fields' marker text, verbatim, joined with ", ".

    Raises:
        TypeError: If any field lacks a PromptText marker, or is itself a
            nested model or a list of one.
    """
    parts = []
    for name, field in model.model_fields.items():
        nested, _ = _nested_model(field.annotation)
        if nested is not None:
            raise TypeError(f"{model.__name__}.{name}: field_list does not support nested models")
        parts.append(_require_prompt_text(model, name, field))
    return ", ".join(parts)


def schema_block(model: type[BaseModel]) -> str:
    """Renders a model as a JSON-object-shaped schema block for a prompt.

    A field annotated as a nested model, or a list of one, recurses into
    its own block instead of reading a marker — list fields are wrapped in
    brackets. Every other field must carry a PromptText marker giving the
    JSON value shown for it.

    Args:
        model: Pydantic model to render.

    Returns:
        A JSON-object-shaped string, e.g. ``{"ticker":"","allocation_pct":0.0}``,
        with no insignificant whitespace.

    Raises:
        TypeError: If a scalar field lacks a PromptText marker, or a
            nested-model field carries one.
    """
    entries = []
    for name, field in model.model_fields.items():
        nested, is_list = _nested_model(field.annotation)
        if nested is None:
            value = _require_prompt_text(model, name, field)
        else:
            if _prompt_text(field) is not None:
                raise TypeError(
                    f"{model.__name__}.{name} is a nested model "
                    "and must not carry a PromptText marker"
                )
            rendered = schema_block(nested)
            value = f"[{rendered}]" if is_list else rendered
        entries.append(f'"{name}":{value}')
    return "{" + ",".join(entries) + "}"
