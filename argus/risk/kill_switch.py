"""
argus/risk/kill_switch.py

Asynchronous kill switch daemon monitoring drawdown thresholds and volatility limits.

Responsibilities:
  - Monitor portfolio drawdown against configurable risk-tolerance thresholds
  - Enforce VIX blackout gates to block new positions during extreme volatility
  - Persist crash dump JSON on circuit breaker activation for manual review

Not responsible for:
  - Portfolio allocation (see agents/portfolio.py)
  - Signal aggregation (see orchestration/aggregator.py)
  - RateLimitGovernor (see orchestration/governor.py)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from argus.config import settings
from argus.params import KILL_SWITCH

logger = logging.getLogger("argus.kill_switch")


@dataclass
class KillSwitchStatus:
    """Snapshot of the active kill switch status and calculated indicators."""

    halted: bool
    new_positions_blocked: bool
    reason: Optional[str]
    triggered_at: Optional[datetime]
    realized_drawdown: float
    current_vix: float


class KillSwitch:
    """Monitors portfolio drawdown and market volatility limits to trigger circuit breakers.

    Runs a background daemon thread that evaluates portfolio value and VIX every
    ``check_interval_seconds``. Two independent gates exist:
      1. VIX blackout: blocks *new* positions when VIX ≥ VIX_BLACKOUT_THRESHOLD.
      2. Drawdown halt: freezes all activity when realized drawdown ≥ risk threshold.

    The drawdown threshold is fixed at construction from a single
    ``risk_tolerance`` (see api/main.py's ``settings.ARGUS_RISK_TOLERANCE``).
    ARGUS serves every /analyze request from the same process, so this gate
    is intentionally process-global rather than per-request: a request whose
    own ``risk_tolerance`` differs from the configured one is still governed
    by the configured threshold, not its own (KS-6).
    """

    DRAWDOWN_THRESHOLDS = {
        "CONSERVATIVE": KILL_SWITCH.conservative_drawdown_halt,
        "MODERATE": KILL_SWITCH.moderate_drawdown_halt,
        "AGGRESSIVE": KILL_SWITCH.aggressive_drawdown_halt,
    }

    # Consecutive VIX-fetch failures before failing closed (blocking new
    # positions) rather than continuing to trust a possibly-stale reading
    _VIX_FAILURE_HALT_THRESHOLD = 5

    # VIX is a daily-close figure; re-fetching it every check is redundant
    # network load for a value that can't have moved
    _VIX_CACHE_TTL_SECONDS = 900

    def __init__(self, user_risk_tolerance: str, check_interval_seconds: int = 900):
        """Initializes the kill switch, deferring monitoring until start() is called.

        Args:
            user_risk_tolerance: Risk tier string ('CONSERVATIVE', 'MODERATE',
                'AGGRESSIVE'); unrecognized values fall back to 'MODERATE'.
            check_interval_seconds: Seconds between monitor loop iterations.

        Raises:
            TypeError: If user_risk_tolerance isn't a string.
        """
        if not isinstance(user_risk_tolerance, str):
            raise TypeError(
                f"risk_tolerance must be a string, got {type(user_risk_tolerance).__name__}"
            )

        self.risk_tolerance = user_risk_tolerance.upper()
        if self.risk_tolerance not in self.DRAWDOWN_THRESHOLDS:
            logger.warning(f"Unknown risk tolerance {self.risk_tolerance}, defaulting to MODERATE.")
            self.risk_tolerance = "MODERATE"

        self.check_interval = check_interval_seconds
        # Read once from settings at construction, not at class-body import
        # time — the prior `getattr(settings, ..., 35.0)` default was
        # unreachable because settings always defines this field
        self.vix_blackout = settings.VIX_BLACKOUT_THRESHOLD

        self._halted = threading.Event()
        self._new_positions_blocked = threading.Event()
        self._stop_event = threading.Event()

        # Guards _halt_reason/_halt_time so a reader never observes one half
        # of a halt update torn from the other (KS-12)
        self._state_lock = threading.Lock()
        self._halt_reason: Optional[str] = None
        self._halt_time: Optional[datetime] = None

        self._portfolio_inception_value: Optional[float] = None
        self._current_portfolio_value: Optional[float] = None
        self._high_water_mark: float = 0.0

        self._last_vix: Optional[float] = None
        self._consecutive_vix_failures: int = 0
        self._vix_cache_value: Optional[float] = None
        self._vix_cache_fetched_at: Optional[float] = None

        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger("argus.kill_switch")

    def start(self, initial_portfolio_value: float) -> None:
        """Starts the background thread monitoring loop with the initial portfolio value.

        A no-op if the monitor thread is already running (idempotent).

        Args:
            initial_portfolio_value: Total portfolio value at session inception (USD).
        """
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning("start() called while already running; ignoring.")
            return

        self._stop_event.clear()
        self._portfolio_inception_value = initial_portfolio_value
        self._current_portfolio_value = initial_portfolio_value
        self._high_water_mark = initial_portfolio_value
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="KillSwitchMonitor"
        )
        self._thread.start()
        self._logger.info(
            f"Kill switch monitor started. Inception value: ${initial_portfolio_value:,.0f}"
        )

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signals the monitor loop to exit and waits for the thread to finish.

        Args:
            timeout: Max seconds to wait for the thread to join; None waits
                indefinitely.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._logger.info("Kill switch monitor stopped.")

    def update_portfolio_value(self, current_value: float) -> None:
        """Thread-safe update of the current portfolio valuation.

        Also advances the high-water mark if this value is a new peak, so
        drawdown is always measured peak-to-trough rather than against the
        (possibly stale) inception value.

        Args:
            current_value: Current mark-to-market portfolio value (USD).
        """
        self._current_portfolio_value = current_value
        self._high_water_mark = max(self._high_water_mark, current_value)

    def _monitor_loop(self) -> None:
        """Monitoring loop, waiting between checks until stop() is called."""
        while not self._stop_event.is_set():
            try:
                self._check()
            except Exception as e:
                self._logger.error(f"Kill switch monitor error: {e}")
            self._stop_event.wait(self.check_interval)

    def _fetch_vix_cached(self) -> float:
        """Fetches VIX, reusing a recent reading for _VIX_CACHE_TTL_SECONDS.

        Raises whatever the underlying fetch raises when the cache is empty
        or stale; a cache hit never raises.
        """
        now = time.monotonic()
        if (
            self._vix_cache_value is not None
            and self._vix_cache_fetched_at is not None
            and now - self._vix_cache_fetched_at < self._VIX_CACHE_TTL_SECONDS
        ):
            return self._vix_cache_value

        from argus.data.fetchers import fetch_vix

        value = fetch_vix()
        self._vix_cache_value = value
        self._vix_cache_fetched_at = now
        return value

    def _check(self) -> None:
        """Evaluates portfolio drawdown and VIX levels against safety limits.

        Drawdown is measured peak-to-trough against _high_water_mark, not
        against the inception value. A VIX fetch failure never clears an
        active blackout; after _VIX_FAILURE_HALT_THRESHOLD consecutive
        failures it sets one — a monitor that can't observe volatility fails
        closed rather than silently substituting a benign guess.
        """
        if self._portfolio_inception_value is None:
            return

        current = (
            self._current_portfolio_value
            if self._current_portfolio_value is not None
            else self._portfolio_inception_value
        )
        self._high_water_mark = max(self._high_water_mark, current)
        drawdown = (
            (self._high_water_mark - current) / self._high_water_mark
            if self._high_water_mark > 0
            else 0.0
        )
        threshold = self.DRAWDOWN_THRESHOLDS[self.risk_tolerance]

        try:
            vix = self._fetch_vix_cached()
        except Exception as exc:
            self._consecutive_vix_failures += 1
            self._logger.error(
                f"VIX fetch failed ({self._consecutive_vix_failures} consecutive): {exc}"
            )
            if (
                self._consecutive_vix_failures >= self._VIX_FAILURE_HALT_THRESHOLD
                and not self._new_positions_blocked.is_set()
            ):
                self._new_positions_blocked.set()
                self._logger.warning(
                    "VIX BLACKOUT (fail-closed): "
                    f"{self._consecutive_vix_failures} consecutive fetch failures."
                )
        else:
            self._consecutive_vix_failures = 0
            self._last_vix = vix

            if vix >= self.vix_blackout and not self._new_positions_blocked.is_set():
                self._new_positions_blocked.set()
                self._logger.warning(
                    f"VIX BLACKOUT: {vix:.1f} >= {self.vix_blackout}. New positions blocked."
                )
            elif vix < self.vix_blackout and self._new_positions_blocked.is_set():
                self._new_positions_blocked.clear()
                self._logger.info(f"VIX normalized to {vix:.1f}. New positions unblocked.")

        if drawdown >= threshold and not self._halted.is_set():
            reason = f"Drawdown {drawdown:.1%} >= {threshold:.0%} limit ({self.risk_tolerance})"
            halt_time = datetime.now()
            with self._state_lock:
                self._halt_reason = reason
                self._halt_time = halt_time
            self._halted.set()
            self._logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
            self._persist_halt_event(reason, halt_time, drawdown, self._last_vix or 0.0)

    def _persist_halt_event(
        self, reason: str, halt_time: datetime, drawdown: float, vix: float
    ) -> None:
        """Saves a JSON crash dump to disk upon triggering a circuit breaker halt.

        The dump includes a mandatory manual intervention instruction so operators
        cannot accidentally restart the system without reviewing the halt reason.
        Also prunes dumps beyond KILL_SWITCH.max_halt_dumps_retained (KS-13), so a
        long-lived deployment with many halt/reset cycles doesn't grow runs/
        unbounded.

        Args:
            reason: Human-readable halt reason, as set on ``self._halt_reason``.
            halt_time: Timestamp of the halt, as set on ``self._halt_time``.
            drawdown: Realized drawdown at the time of halt.
            vix: VIX level at the time of halt.
        """
        timestamp = halt_time.strftime("%Y%m%d_%H%M%S")
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        filename = runs_dir / f"argus_halt_{timestamp}.json"

        dump = {
            "halt_time": halt_time.isoformat(),
            "reason": reason,
            "risk_tolerance": self.risk_tolerance,
            "inception_value": self._portfolio_inception_value,
            "halt_value": self._current_portfolio_value,
            "realized_drawdown": drawdown,
            "vix_at_halt": vix,
            "instruction": "MANUAL INTERVENTION REQUIRED. Delete this file to allow system restart.",
        }

        try:
            with open(filename, "w") as f:
                json.dump(dump, f, indent=4)
        except Exception as e:
            self._logger.error(f"Failed to write halt event file: {e}")

        _prune_halt_dumps(str(runs_dir))

    @property
    def status(self) -> KillSwitchStatus:
        """Returns a point-in-time status snapshot without advancing any internal state."""
        if self._portfolio_inception_value is None:
            drawdown = 0.0
        else:
            current = (
                self._current_portfolio_value
                if self._current_portfolio_value is not None
                else self._portfolio_inception_value
            )
            hwm = self._high_water_mark if self._high_water_mark > 0 else self._portfolio_inception_value
            drawdown = (hwm - current) / hwm if hwm > 0 else 0.0

        with self._state_lock:
            reason = self._halt_reason
            triggered_at = self._halt_time

        return KillSwitchStatus(
            halted=self._halted.is_set(),
            new_positions_blocked=self._new_positions_blocked.is_set(),
            reason=reason,
            triggered_at=triggered_at,
            realized_drawdown=drawdown,
            current_vix=self._last_vix or 0.0,
        )

    def _restore_halt_from_file(self, path: Path) -> None:
        """Re-applies a previously persisted halt so a process restart can't silently clear it.

        Args:
            path: Path to a halt-event JSON file written by ``_persist_halt_event``.
        """
        try:
            with open(path) as f:
                dump = json.load(f)
        except Exception as e:
            self._logger.error(f"Failed to read halt event file {path}: {e}")
            return

        reason = dump.get("reason") or f"Restored from {path.name}"
        halt_time_raw = dump.get("halt_time")
        halt_time = datetime.fromisoformat(halt_time_raw) if halt_time_raw else datetime.now()
        with self._state_lock:
            self._halt_reason = reason
            self._halt_time = halt_time
        self._halted.set()
        self._logger.warning(
            f"Kill switch restored HALTED state from {path.name}: {reason}. "
            "Call POST /kill-switch/reset to resume (this also deletes the halt dump)."
        )

    @property
    def is_halted(self) -> bool:
        """Returns True if the drawdown circuit breaker has been triggered."""
        return self._halted.is_set()

    @property
    def new_positions_allowed(self) -> bool:
        """Returns True only when both the VIX blackout and drawdown halt gates are clear."""
        return not self._new_positions_blocked.is_set() and not self._halted.is_set()

    def reset(self, new_inception_value: float) -> None:
        """Resets the circuit breaker and re-initializes the inception value.

        Clears all halt flags and deletes any persisted halt dumps, so a
        process restart can't silently resurrect a halt the operator already
        reviewed and resolved through this call (KS-9).

        Args:
            new_inception_value: Replacement inception portfolio value (USD).
        """
        self._halted.clear()
        self._new_positions_blocked.clear()
        with self._state_lock:
            self._halt_reason = None
            self._halt_time = None
        self._portfolio_inception_value = new_inception_value
        self._current_portfolio_value = new_inception_value
        self._high_water_mark = new_inception_value
        self._consecutive_vix_failures = 0
        for path in _list_halt_dumps():
            try:
                path.unlink()
            except Exception as e:
                self._logger.error(f"Failed to delete halt dump {path}: {e}")
        self._logger.info(f"Kill switch reset. New inception value: ${new_inception_value:,.0f}")


_kill_switch: Optional[KillSwitch] = None


def _list_halt_dumps(runs_dir: str = "runs") -> list[Path]:
    """Lists halt-event dumps oldest-to-newest by their own ``halt_time`` field.

    Ranks by content rather than the filename's embedded timestamp (KS-13) —
    lexicographic filename sort only coincidentally matches chronological
    order. Falls back to file mtime for a dump that fails to parse.

    Args:
        runs_dir: Directory ``_persist_halt_event`` writes halt dumps to.

    Returns:
        Halt-dump paths sorted oldest to newest; empty if none exist.
    """
    directory = Path(runs_dir)
    if not directory.is_dir():
        return []

    def _halt_time(path: Path) -> datetime:
        try:
            with open(path) as f:
                raw = json.load(f).get("halt_time")
            if raw:
                return datetime.fromisoformat(raw)
        except Exception:
            pass
        return datetime.fromtimestamp(path.stat().st_mtime)

    return sorted(directory.glob("argus_halt_*.json"), key=_halt_time)


def _prune_halt_dumps(runs_dir: str = "runs") -> None:
    """Deletes all but the most recent KILL_SWITCH.max_halt_dumps_retained halt dumps.

    Args:
        runs_dir: Directory ``_persist_halt_event`` writes halt dumps to.
    """
    keep = KILL_SWITCH.max_halt_dumps_retained
    dumps = _list_halt_dumps(runs_dir)
    stale = dumps[:-keep] if keep > 0 else dumps
    for path in stale:
        try:
            path.unlink()
        except Exception as e:
            logger.error(f"Failed to prune halt dump {path}: {e}")


def _find_latest_halt_file(runs_dir: str = "runs") -> Optional[Path]:
    """Finds the most recently written halt-event dump, if any exist.

    Args:
        runs_dir: Directory ``_persist_halt_event`` writes halt dumps to.

    Returns:
        Path to the newest ``argus_halt_*.json`` file, or None if none exist.
    """
    dumps = _list_halt_dumps(runs_dir)
    return dumps[-1] if dumps else None


def initialize_kill_switch(
    risk_tolerance: str, portfolio_value: float, check_interval: int = 900
) -> KillSwitch:
    """Initializes and starts the module-level KillSwitch singleton.

    Safe to call multiple times; returns the existing instance if already initialized.
    An unattended process restarting after a halt should not silently resume trading,
    so a leftover ``runs/argus_halt_*.json`` from a prior run re-applies the halt
    immediately, per that file's own "manual intervention required" instruction.

    Args:
        risk_tolerance: Risk tier string ('CONSERVATIVE', 'MODERATE', 'AGGRESSIVE').
        portfolio_value: Initial portfolio value used as the drawdown base (USD).
        check_interval: Seconds between monitor loop iterations (default 900,
            matching the VIX cache TTL — a shorter interval would just re-hit
            the cache without observing anything new).

    Returns:
        The active KillSwitch singleton.
    """
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch(risk_tolerance, check_interval)
        halt_file = _find_latest_halt_file()
        if halt_file is not None:
            _kill_switch._restore_halt_from_file(halt_file)
        _kill_switch.start(portfolio_value)
    return _kill_switch


def get_kill_switch() -> Optional[KillSwitch]:
    """Retrieves the active KillSwitch singleton instance.

    Returns:
        The KillSwitch singleton, or None if not yet initialized.
    """
    return _kill_switch
