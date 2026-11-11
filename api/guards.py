"""
api/guards.py

Fail-fast checks the API runs once at startup, before it serves traffic.

Responsibilities:
  - assert_single_worker / acquire_process_lock: keep the in-process governor and
    kill-switch singletons to exactly one process per data directory
  - release_process_lock: drop that lock on shutdown
  - assert_registered_models: turn a typo'd model name into a boot failure
  - warn_on_permissive_security_defaults: say out loud when auth/CORS are wide open

Not responsible for:
  - Sequencing these checks or the rest of startup (see api/lifespan.py)
"""

import fcntl
import logging
import sys
from typing import TextIO

from argus.config import settings, web_concurrency
from argus.governor import REGISTERED_MODELS, UnregisteredModel

logger = logging.getLogger(__name__)

# Holds the open file object backing the process-exclusivity flock for as
# long as this process runs; garbage-collecting it would release the lock
_process_lock_file: TextIO | None = None


def assert_single_worker() -> None:
    """Fails fast on the argv/env spellings of "more than one worker" this can detect.

    The rate governor and kill switch are in-process singletons (see
    argus/governor.py). A second worker process would run its
    own ignorant copy of both, silently doubling real Groq usage against the
    same account and gating risk independently rather than jointly — the
    Actions collector (scripts/reconcile_outcomes.py) is already a second,
    independent governor against that same key, and this assertion exists so
    a deployment doesn't add a third by accident. See Dockerfile.api's CMD.

    This is a best-effort pre-flight only, not the real guard: it can't see
    gunicorn's `-w`, a gunicorn config file, or a programmatic
    `uvicorn.run(workers=N)`. `acquire_process_lock` is what actually
    catches those, and the case that costs real money — two containers on one
    data volume — which no argv/env spelling of "workers" describes at all.

    Raises:
        RuntimeError: If ``--workers``/``--workers=N`` in argv, or
            ``WEB_CONCURRENCY`` in the environment, requests a value other
            than "1".
    """
    argv = sys.argv
    requested: list[tuple[str, str]] = []

    if "--workers" in argv:
        idx = argv.index("--workers")
        if idx + 1 < len(argv):
            requested.append((f"--workers {argv[idx + 1]}", argv[idx + 1]))
    requested.extend(
        (arg, arg.removeprefix("--workers=")) for arg in argv if arg.startswith("--workers=")
    )
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
    """Takes an exclusive, non-blocking flock on ``${ARGUS_DATA_DIR}/argus.lock``.

    This is the real single-process guard, unlike `assert_single_worker`'s
    argv/env pre-flight: it catches every way two ARGUS processes could end up
    sharing one data directory — including two whole containers — not just
    the worker-count spellings that pre-flight can parse. Note this only runs
    during lifespan startup, so ``uvicorn --reload``'s parent process (which
    doesn't run lifespan) is not itself protected; the reloaded child is.
    Scoping the lock file to ``ARGUS_DATA_DIR`` also correctly leaves
    ``scripts/collect_session.py``'s GitHub Actions runner unblocked — it
    writes to its own runner-local directory, not this deployment's, and is
    already an accepted second governor/kill-switch process (see
    ``argus/orchestration/collector.py``'s module docstring).

    Raises:
        RuntimeError: If another process already holds the lock.
    """
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
    """Releases the flock taken by `acquire_process_lock`; a no-op if none is held."""
    if _process_lock_file is not None:
        fcntl.flock(_process_lock_file, fcntl.LOCK_UN)
        _process_lock_file.close()


def warn_on_permissive_security_defaults() -> None:
    """Logs a WARNING for each security-relevant setting still at its permissive default.

    ARGUS_CORS_ORIGINS defaulting to ``["*"]`` and a blank ARGUS_API_KEY are
    both fine for local development behind a trusted network boundary, but
    silent in a deployment actually exposed to the internet. Neither default
    changes here — only whether the deployment says so out loud.
    """
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
    """Fails fast if a configured agent model has no rate-limit profile.

    Converts what would otherwise be a first-call UnregisteredModel deep inside
    a request into a boot-time failure, so a typo'd or unsupported
    ARGUS_*_MODEL setting is caught before the process ever serves traffic.

    Raises:
        UnregisteredModel: If any of ARGUS_FUNDAMENTAL_MODEL, ARGUS_SENTIMENT_MODEL,
            or ARGUS_PORTFOLIO_MODEL is absent from REGISTERED_MODELS.
    """
    for model in (
        settings.ARGUS_FUNDAMENTAL_MODEL,
        settings.ARGUS_SENTIMENT_MODEL,
        settings.ARGUS_PORTFOLIO_MODEL,
    ):
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(
                f"{model!r} is configured but has no registered rate-limit profile"
            )
