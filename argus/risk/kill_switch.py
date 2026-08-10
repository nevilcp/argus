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
    """

    DRAWDOWN_THRESHOLDS = {
        "CONSERVATIVE": 0.08,
        "MODERATE": 0.12,
        "AGGRESSIVE": 0.18,
    }
    VIX_BLACKOUT = getattr(settings, "VIX_BLACKOUT_THRESHOLD", 35.0)

    def __init__(self, user_risk_tolerance: str, check_interval_seconds: int = 60):
        self.risk_tolerance = user_risk_tolerance.upper()
        if self.risk_tolerance not in self.DRAWDOWN_THRESHOLDS:
            logger.warning(f"Unknown risk tolerance {self.risk_tolerance}, defaulting to MODERATE.")
            self.risk_tolerance = "MODERATE"

        self.check_interval = check_interval_seconds

        self._halted = threading.Event()
        self._new_positions_blocked = threading.Event()

        self._halt_reason: Optional[str] = None
        self._halt_time: Optional[datetime] = None

        self._portfolio_inception_value: Optional[float] = None
        self._current_portfolio_value: Optional[float] = None

        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger("argus.kill_switch")

    def start(self, initial_portfolio_value: float) -> None:
        """Starts the background thread monitoring loop with the initial portfolio value.

        Args:
            initial_portfolio_value: Total portfolio value at session inception (USD).
        """
        self._portfolio_inception_value = initial_portfolio_value
        self._current_portfolio_value = initial_portfolio_value
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="KillSwitchMonitor"
        )
        self._thread.start()
        self._logger.info(
            f"Kill switch monitor started. Inception value: ${initial_portfolio_value:,.0f}"
        )

    def update_portfolio_value(self, current_value: float) -> None:
        """Thread-safe update of the current portfolio valuation.

        Args:
            current_value: Current mark-to-market portfolio value (USD).
        """
        self._current_portfolio_value = current_value

    def _monitor_loop(self) -> None:
        """Infinite monitoring loop sleeping between checks."""
        while True:
            try:
                self._check()
            except Exception as e:
                self._logger.error(f"Kill switch monitor error: {e}")
            time.sleep(self.check_interval)

    def _check(self) -> None:
        """Evaluates portfolio drawdown and VIX levels against safety limits.

        Fetches the live VIX value; falls back to 20.0 if the API call fails to
        prevent a monitor crash from silently disabling the safety gate.
        """
        if self._portfolio_inception_value is None:
            return

        current = self._current_portfolio_value or self._portfolio_inception_value
        drawdown = (self._portfolio_inception_value - current) / self._portfolio_inception_value
        threshold = self.DRAWDOWN_THRESHOLDS[self.risk_tolerance]

        try:
            from argus.data.fetchers import fetch_vix

            vix = fetch_vix()
        except Exception:
            vix = 20.0

        if vix >= self.VIX_BLACKOUT and not self._new_positions_blocked.is_set():
            self._new_positions_blocked.set()
            self._logger.warning(
                f"VIX BLACKOUT: {vix:.1f} >= {self.VIX_BLACKOUT}. New positions blocked."
            )

        elif vix < self.VIX_BLACKOUT and self._new_positions_blocked.is_set():
            self._new_positions_blocked.clear()
            self._logger.info(f"VIX normalized to {vix:.1f}. New positions unblocked.")

        if drawdown >= threshold and not self._halted.is_set():
            self._halt_reason = (
                f"Drawdown {drawdown:.1%} >= {threshold:.0%} limit ({self.risk_tolerance})"
            )
            self._halt_time = datetime.now()
            self._halted.set()
            self._logger.critical(f"KILL SWITCH TRIGGERED: {self._halt_reason}")
            self._persist_halt_event(drawdown, vix)

    def _persist_halt_event(self, drawdown: float, vix: float) -> None:
        """Saves a JSON crash dump to disk upon triggering a circuit breaker halt.

        The dump includes a mandatory manual intervention instruction so operators
        cannot accidentally restart the system without reviewing the halt reason.

        Args:
            drawdown: Realized drawdown at the time of halt.
            vix: VIX level at the time of halt.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        filename = runs_dir / f"argus_halt_{timestamp}.json"

        dump = {
            "halt_time": datetime.now().isoformat(),
            "reason": self._halt_reason,
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

    @property
    def status(self) -> KillSwitchStatus:
        """Returns a point-in-time status snapshot without advancing any internal state."""
        if self._portfolio_inception_value is None:
            drawdown = 0.0
        else:
            current = self._current_portfolio_value or self._portfolio_inception_value
            drawdown = (self._portfolio_inception_value - current) / self._portfolio_inception_value

        return KillSwitchStatus(
            halted=self._halted.is_set(),
            new_positions_blocked=self._new_positions_blocked.is_set(),
            reason=self._halt_reason,
            triggered_at=self._halt_time,
            realized_drawdown=drawdown,
            current_vix=0.0,
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

        Clears all halt flags. The operator is responsible for ensuring that the
        halt event JSON file has been reviewed and deleted before calling this.

        Args:
            new_inception_value: Replacement inception portfolio value (USD).
        """
        self._halted.clear()
        self._new_positions_blocked.clear()
        self._halt_reason = None
        self._halt_time = None
        self._portfolio_inception_value = new_inception_value
        self._current_portfolio_value = new_inception_value
        self._logger.info(f"Kill switch reset. New inception value: ${new_inception_value:,.0f}")


_kill_switch: Optional[KillSwitch] = None


def initialize_kill_switch(
    risk_tolerance: str, portfolio_value: float, check_interval: int = 60
) -> KillSwitch:
    """Initializes and starts the module-level KillSwitch singleton.

    Safe to call multiple times; returns the existing instance if already initialized.

    Args:
        risk_tolerance: Risk tier string ('CONSERVATIVE', 'MODERATE', 'AGGRESSIVE').
        portfolio_value: Initial portfolio value used as the drawdown base (USD).
        check_interval: Seconds between monitor loop iterations (default 60).

    Returns:
        The active KillSwitch singleton.
    """
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch(risk_tolerance, check_interval)
        _kill_switch.start(portfolio_value)
    return _kill_switch


def get_kill_switch() -> Optional[KillSwitch]:
    """Retrieves the active KillSwitch singleton instance.

    Returns:
        The KillSwitch singleton, or None if not yet initialized.
    """
    return _kill_switch
