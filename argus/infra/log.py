"""Process-wide logging setup shared by the API and the CLI scripts."""

from __future__ import annotations

import logging

from argus.config import settings

# Every logger is `getLogger(__name__)`, so the package trees are the roots to
# configure. "__main__" is there because `python -m scripts.x` and
# `python scripts/x.py` name the script's own logger "__main__", which is
# outside every other tree.
LOG_ROOTS = ("argus", "api", "scripts", "__main__")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s"


def configure_logging() -> None:
    """Attaches a stream handler to each logger tree so ARGUS_LOG_LEVEL takes effect.

    Under uvicorn the root logger has no handlers at level WARNING, so every
    `logger.info`/`debug` call is silently dropped and ARGUS_LOG_LEVEL is
    inert. `propagate = False` stops uvicorn's own logging config from
    resetting this. Idempotent, so re-running lifespan (e.g. across tests)
    doesn't stack duplicate handlers.
    """
    formatter = logging.Formatter(_LOG_FORMAT)
    for name in LOG_ROOTS:
        tree = logging.getLogger(name)
        tree.setLevel(settings.ARGUS_LOG_LEVEL)
        tree.propagate = False
        if not tree.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            tree.addHandler(handler)
