"""ARGUS package version."""

from importlib.metadata import version

# Read the version from package metadata so it cannot drift from pyproject.toml.
__version__ = version("argus")
__author__ = "ARGUS Team"
