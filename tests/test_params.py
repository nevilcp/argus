"""Guards on argus/params.py: every declared field must carry provenance metadata."""

from argus.params import Provenance, all_params


def _label(row: dict) -> str:
    """Returns the "group.field" identifier a failure message names the row by."""
    return f"{row['group']}.{row['field']}"


def test_every_param_has_provenance_metadata():
    """Every declared parameter row carries a non-null provenance tag."""
    rows = all_params()
    assert rows, "expected at least one declared parameter"
    missing = [_label(r) for r in rows if r["provenance"] is None]
    assert not missing, f"fields declared without a Provenance tag: {missing}"


def test_every_provenance_value_is_valid():
    """Every provenance tag is a member of the Provenance enum, not an arbitrary value."""
    invalid = [_label(r) for r in all_params() if not isinstance(r["provenance"], Provenance)]
    assert not invalid, f"fields with a non-Provenance tag: {invalid}"
