"""Leaf utilities shared across layers: clock, atomic file writes, retry back-off.

Imports only `argus.config`, `argus.params` and the standard library, so any
other package may depend on it without creating a cycle.
"""
