"""Enforces the import direction between ARGUS's packages.

Every import in argus/, api/ and scripts/ is found by walking the syntax tree,
so imports inside functions and under TYPE_CHECKING count as much as
module-level ones. A package may import from its own layer or any layer below
it, and from itself freely; anything else is an upward edge and fails here.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Lowest layer first. Keys are the first path component under argus/ (a module
# or a package), plus the two top-level application packages.
LAYERS: dict[str, int] = {
    "params": 1,
    # schemas is below config because config reads the shared request types and
    # bounds from schemas/requests.py; schemas itself imports only params
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

# Directories under argus/ that hold no Python and so take no layer.
NON_CODE = {"models", "__pycache__"}


def _module_name(path: Path) -> str:
    """Returns the dotted module name of a source file under ROOT."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module: str) -> str:
    """Returns the LAYERS key a dotted module name belongs to.

    Resolved by position, not by substring, so ``argus.agents.risk`` belongs to
    ``agents`` and is never mistaken for the ``risk`` package.
    """
    parts = module.split(".")
    return parts[1] if parts[0] == "argus" and len(parts) > 1 else parts[0]


def imported_packages(source: str, module: str) -> set[str]:
    """Returns the LAYERS keys that a source file imports from, at any depth.

    Args:
        source: Python source text.
        module: Dotted name of the module the source belongs to, used to
            resolve relative imports.

    Returns:
        The package of every absolute or relative import in the tree.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = module.split(".")[: -node.level]
                base = ".".join([*anchor, base]) if base else ".".join(anchor)
            # `from argus import governor` imports the module argus.governor
            targets = [f"{base}.{a.name}" for a in node.names] if base == "argus" else [base]
        else:
            continue
        for target in targets:
            if target.split(".")[0] in {"argus", "api", "scripts"}:
                found.add(_package_of(target))
    return found


def _source_files() -> list[Path]:
    """Returns every Python file under argus/, api/ and scripts/."""
    return sorted(
        path
        for top in ("argus", "api", "scripts")
        for path in (ROOT / top).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _violations() -> list[str]:
    """Returns one line per import that points at a higher layer."""
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
                problems.append(
                    f"{module} (layer {LAYERS[own]}) imports {imported} (layer {LAYERS[imported]})"
                )
    return problems


def test_no_import_points_at_a_higher_layer():
    """No module imports from a layer above its own."""
    assert _violations() == []


def test_every_package_has_a_layer():
    """A new top-level module or package must be placed in LAYERS, not silently exempt."""
    top_level = {
        p.stem if p.is_file() else p.name
        for p in (ROOT / "argus").iterdir()
        if p.name not in NON_CODE and (p.is_dir() or p.suffix == ".py") and p.name != "__init__.py"
    }
    assert top_level - set(LAYERS) == set()


def test_layers_name_only_things_that_exist():
    """A removed or renamed package is dropped from LAYERS rather than left behind."""
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
    """argus.agents.risk counts as agents, never as the risk package."""
    assert imported_packages(source, module) == expected


def test_imports_inside_functions_and_type_checking_blocks_are_found():
    """A deferred or type-only import is still an edge."""
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from argus.orchestration.state import ARGUSState\n"
        "def f():\n"
        "    from argus.risk import kill_switch\n"
    )
    assert imported_packages(source, "argus.data.x") == {"orchestration", "risk"}
