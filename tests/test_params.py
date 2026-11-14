"""Guards on argus/params.py: every declared field must carry provenance metadata."""

import dataclasses

from argus import params
from argus.config import settings
from argus.params import (
    PORTFOLIO,
    RISK,
    SYSTEM,
    TECHNICAL_INDICATOR_WEIGHTS,
    Provenance,
    all_params,
)


def _label(row: dict) -> str:
    """Return the "group.field" identifier a failure message names the row by."""
    return f"{row['group']}.{row['field']}"


def test_every_param_has_provenance_metadata():
    rows = all_params()
    assert rows, "expected at least one declared parameter"
    missing = [_label(r) for r in rows if r["provenance"] is None]
    assert not missing, f"fields declared without a Provenance tag: {missing}"


def test_every_provenance_value_is_valid():
    invalid = [_label(r) for r in all_params() if not isinstance(r["provenance"], Provenance)]
    assert not invalid, f"fields with a non-Provenance tag: {invalid}"


def test_every_parameter_group_is_registered():
    group_classes = [
        obj
        for obj in vars(params).values()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__ == params.__name__
    ]
    registered = {type(group) for group in params._ALL_GROUPS.values()}
    unregistered = [cls.__name__ for cls in group_classes if cls not in registered]
    assert not unregistered, f"group classes with no module-level instance: {unregistered}"
    assert "TECHNICAL_INDICATOR_WEIGHTS" in params._ALL_GROUPS
    assert "Provenance" not in params._ALL_GROUPS
    not_frozen = [n for n, g in params._ALL_GROUPS.items() if not type(g).__dataclass_params__.frozen]
    assert not not_frozen, f"mutable parameter groups: {not_frozen}"


def test_derived_defaults_follow_their_source():
    assert PORTFOLIO.kelly_max_position == SYSTEM.max_single_position_pct
    assert RISK.returns_lookback_days == SYSTEM.trading_days_per_year
    assert dataclasses.asdict(TECHNICAL_INDICATOR_WEIGHTS) == settings.TECHNICAL_INDICATOR_WEIGHTS
