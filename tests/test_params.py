"""Guards on argus/params.py: every declared field must carry provenance metadata."""

from argus.params import Provenance, all_params


def test_every_param_has_provenance_metadata():
    rows = all_params()
    assert rows, "expected at least one declared parameter"
    missing = [f"{r['group']}.{r['field']}" for r in rows if r["provenance"] is None]
    assert not missing, f"fields declared without a Provenance tag: {missing}"


def test_every_provenance_value_is_valid():
    rows = all_params()
    invalid = [
        f"{r['group']}.{r['field']}"
        for r in rows
        if not isinstance(r["provenance"], Provenance)
    ]
    assert not invalid, f"fields with a non-Provenance tag: {invalid}"
