"""Startup checks and process exclusivity for the HTTP gateway."""

import fcntl
import logging
import sys
from typing import TextIO

from argus.config import settings, web_concurrency
from argus.governor import REGISTERED_MODELS, UnregisteredModel

logger = logging.getLogger(__name__)

# Keep the file open for the process lifetime; closing it releases the exclusivity lock.
_process_lock_file: TextIO | None = None


def assert_single_worker() -> None:
    """Fail fast on the argv/env spellings of "more than one worker" this can detect."""
    argv = sys.argv
    requested: list[tuple[str, str]] = []

    if "--workers" in argv:
        idx = argv.index("--workers")
        if idx + 1 < len(argv):
            requested.append((f"--workers {argv[idx + 1]}", argv[idx + 1]))
    requested.extend((arg, arg.removeprefix("--workers=")) for arg in argv if arg.startswith("--workers="))
    concurrency = web_concurrency()
    if concurrency is not None:
        requested.append((f"WEB_CONCURRENCY={concurrency}", concurrency))

    for spelling, count in requested:
        if count != "1":
            raise RuntimeError(
                "ARGUS must run with a single worker (in-process governor/kill-switch "
                f"singletons); got {spelling}. This is only a pre-flight check of argv/env — it "
                "can't see a gunicorn config file or uvicorn.run(workers=N), and "
                "acquire_process_lock's flock is the real guard."
            )


def acquire_process_lock() -> None:
    """Take an exclusive, non-blocking flock on `${ARGUS_DATA_DIR}/argus.lock`."""
    global _process_lock_file
    lock_path = settings.data_path("argus.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise RuntimeError(
            f"Another ARGUS process already holds {lock_path} — refusing to start a "
            f"second process against the same data directory."
        ) from exc
    _process_lock_file = lock_file


def release_process_lock() -> None:
    """Release the flock taken by `acquire_process_lock`; a no-op if none is held."""
    if _process_lock_file is not None:
        fcntl.flock(_process_lock_file, fcntl.LOCK_UN)
        _process_lock_file.close()


def warn_on_permissive_security_defaults() -> None:
    """Log a WARNING for each security-relevant setting still at its permissive default."""
    if settings.ARGUS_CORS_ORIGINS == ["*"]:
        logger.warning(
            '[Startup] ARGUS_CORS_ORIGINS is at its default ("*"): every origin is allowed. '
            "Set it explicitly for a deployment reachable outside a trusted network."
        )
    if not settings.ARGUS_API_KEY:
        logger.warning(
            "ARGUS_API_KEY is unset: /analyze and /kill-switch/reset accept requests "
            "with no X-API-Key check. Set it for a deployment reachable outside a trusted network."
        )


def assert_registered_models() -> None:
    """Fail fast if a configured agent model has no rate-limit profile."""
    for model in (
        settings.ARGUS_FUNDAMENTAL_MODEL,
        settings.ARGUS_SENTIMENT_MODEL,
        settings.ARGUS_PORTFOLIO_MODEL,
    ):
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} is configured but has no registered rate-limit profile")
