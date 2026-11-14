"""Process-wide logging setup shared by the API and the CLI scripts."""

from __future__ import annotations

import logging

from argus.config import settings

# Configure package logger trees and __main__ so both module and direct script entry points emit logs.
LOG_ROOTS = ("argus", "api", "scripts", "__main__")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s"


def configure_logging() -> None:
    """Attach a stream handler to each logger tree so ARGUS_LOG_LEVEL takes effect."""
    formatter = logging.Formatter(_LOG_FORMAT)
    for name in LOG_ROOTS:
        tree = logging.getLogger(name)
        tree.setLevel(settings.ARGUS_LOG_LEVEL)
        tree.propagate = False
        if not tree.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            tree.addHandler(handler)
