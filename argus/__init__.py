"""
ARGUS — top-level package.

Exposes the package version and provides a convenient import surface
for the core sub-packages (agents, data, orchestration, memory, risk,
backtesting, schemas).
"""

from importlib.metadata import version

# pyproject.toml is the single source of truth for the version — a literal
# here would drift from it (DEP-6)
__version__ = version("argus")
__author__ = "ARGUS Team"
