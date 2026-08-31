"""
ARGUS — top-level package.

Exposes the package version. Sub-packages (agents, data, orchestration,
memory, risk, backtesting, schemas) are imported directly, not re-exported
here, so importing `argus` stays cheap and free of import cycles.
"""

from importlib.metadata import version

# pyproject.toml is the single source of truth for the version — a literal
# here would drift from it (DEP-6)
__version__ = version("argus")
__author__ = "ARGUS Team"
