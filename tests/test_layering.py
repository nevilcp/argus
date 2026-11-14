"""Enforces the import direction between ARGUS's packages."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Order modules from the lowest dependency layer; keys name top-level packages or argus modules.
LAYERS: dict[str, int] = {
    "params": 1,
    # Keep schemas below config because settings depend on shared request bounds.
    "schemas": 2,
    "config": 3,
    "infra": 3,
    "ports": 4,
    "governor": 4,
    "data": 5,
    "memory": 5,
    "outcomes": 5,
    "seams": 6,
    "structured_output": 6,
    "agents": 7,
    "risk": 8,
    "orchestration": 9,
    "backtesting": 10,
    "api": 11,
    "scripts": 11,
}

NON_CODE = {"models", "__pycache__"}


def _module_name(path: Path) -> str:
    """Return the dotted module name of a source file under ROOT."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module: str) -> str:
    """Return the LAYERS key a dotted module name belongs to."""
    parts = module.split(".")
    return parts[1] if parts[0] == "argus" and len(parts) > 1 else parts[0]


def imported_packages(source: str, module: str) -> set[str]:
    """Return the LAYERS keys that a source file imports from, at any depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = module.split(".")[: -node.level]
                base = ".".join([*anchor, base]) if base else ".".join(anchor)
            targets = [f"{base}.{a.name}" for a in node.names] if base == "argus" else [base]
        else:
            continue
        for target in targets:
            if target.split(".")[0] in {"argus", "api", "scripts"}:
                found.add(_package_of(target))
    return found


def _source_files() -> list[Path]:
    """Return every Python file under argus/, api/ and scripts/."""
    return sorted(
        path
        for top in ("argus", "api", "scripts")
        for path in (ROOT / top).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _violations() -> list[str]:
    """Return one line per import that points at a higher layer."""
    problems: list[str] = []
    for path in _source_files():
        module = _module_name(path)
        own = _package_of(module)
        if own not in LAYERS:
            continue
        for imported in sorted(imported_packages(path.read_text(encoding="utf-8"), module)):
            if imported == own or imported not in LAYERS:
                continue
            if LAYERS[imported] > LAYERS[own]:
                problems.append(f"{module} (layer {LAYERS[own]}) imports {imported} (layer {LAYERS[imported]})")
    return problems


def test_no_import_points_at_a_higher_layer():
    assert _violations() == []


def test_every_package_has_a_layer():
    top_level = {
        p.stem if p.is_file() else p.name
        for p in (ROOT / "argus").iterdir()
        if p.name not in NON_CODE and (p.is_dir() or p.suffix == ".py") and p.name != "__init__.py"
    }
    assert top_level - set(LAYERS) == set()


def test_layers_name_only_things_that_exist():
    existing = {p.stem if p.is_file() else p.name for p in (ROOT / "argus").iterdir()} | {
        "api",
        "scripts",
    }
    assert set(LAYERS) - existing == set()


@pytest.mark.parametrize(
    ("source", "module", "expected"),
    [
        ("import argus.agents.risk", "argus.data.x", {"agents"}),
        ("from argus.agents.risk import get_sector", "argus.data.x", {"agents"}),
        ("from argus.risk import paper_book", "argus.data.x", {"risk"}),
        ("from argus import governor", "argus.data.x", {"governor"}),
        ("from . import cache", "argus.data.pipeline", {"data"}),
        ("from ..params import SYSTEM", "argus.data.pipeline", {"params"}),
        ("import json\nimport pandas", "argus.data.x", set()),
    ],
)
def test_imports_resolve_by_full_dotted_path(source, module, expected):
    assert imported_packages(source, module) == expected


def test_imports_inside_functions_and_type_checking_blocks_are_found():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from argus.orchestration.state import ARGUSState\n"
        "def f():\n"
        "    from argus.risk import kill_switch\n"
    )
    assert imported_packages(source, "argus.data.x") == {"orchestration", "risk"}
