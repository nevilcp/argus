"""Synthetic paper equity from matured, non-overlapping decision runs."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from argus.config import settings
from argus.infra.clock import now_local
from argus.infra.files import atomic_write_text
from argus.outcomes import close_prices, compute_realized_return, needs_reconciliation
from argus.ports import MarketDataProvider
from argus.risk.kill_switch import drawdown_fraction
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)


def _weighted_run_return(
    run_decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
    prices_by_ticker: dict[str, pd.Series | None],
) -> float | None:
    """Average one run's matured decision outcomes, weighted by allocation_pct."""
    weighted_sum = 0.0
    weight_total = 0.0
    for decision in run_decisions:
        prices = close_prices(decision.ticker, market_data, prices_by_ticker)
        if prices is None:
            continue
        outcome = compute_realized_return(decision, market_data, horizon_days, prices=prices)
        if outcome is None:
            continue

        actual_return_pct, _, _ = outcome
        if decision.allocation is None:
            raise RuntimeError("needs_reconciliation already confirmed this")
        weight = decision.allocation.allocation_pct
        weighted_sum += actual_return_pct * weight
        weight_total += weight

    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def compute_run_returns(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
) -> list[tuple[datetime, float]]:
    """Group decisions into runs and compute each matured run's weighted return."""
    by_run: dict[datetime, list[ARGUSDecision]] = defaultdict(list)
    for decision in decisions:
        if needs_reconciliation(decision):
            by_run[decision.session_timestamp].append(decision)

    horizon = timedelta(days=horizon_days)
    prices_by_ticker: dict[str, pd.Series | None] = {}
    results: list[tuple[datetime, float]] = []
    last_kept_timestamp: datetime | None = None
    for run_timestamp in sorted(by_run):
        if last_kept_timestamp is not None and run_timestamp < last_kept_timestamp + horizon:
            continue

        run_return = _weighted_run_return(by_run[run_timestamp], market_data, horizon_days, prices_by_ticker)
        if run_return is not None:
            results.append((run_timestamp, run_return))
            last_kept_timestamp = run_timestamp

    return results


@dataclass
class PaperBook:
    """A synthetic, run-compounded equity curve fed by matured decision outcomes."""

    equity: float
    high_water_mark: float
    last_run_timestamp: datetime | None = None
    runs_applied: set[str] = field(default_factory=set)
    rebased_at: datetime | None = None

    def apply_run(self, run_timestamp: datetime, run_return: float) -> bool:
        """Compound one run's weighted return onto the curve, unless already applied."""
        key = run_timestamp.isoformat()
        if key in self.runs_applied:
            return False
        # Floor equity at zero against data anomalies because the drawdown gate consumes this safety state.
        self.equity = max(0.0, self.equity * (1.0 + run_return))
        self.high_water_mark = max(self.high_water_mark, self.equity)
        self.last_run_timestamp = run_timestamp
        self.runs_applied.add(key)
        return True

    def drawdown_from_peak(self) -> float:
        """Return peak-to-trough drawdown as a non-negative fraction."""
        return max(0.0, drawdown_fraction(self.high_water_mark, self.equity))

    def rebase(self, new_inception_value: float, rebased_at: datetime | None = None) -> None:
        """Rebase equity and its high-water mark while preserving already-applied run identities."""
        self.equity = max(0.0, new_inception_value)
        self.high_water_mark = self.equity
        self.rebased_at = rebased_at or now_local()

    def prune_runs_applied(self, cutoff: datetime) -> int:
        """Drop runs_applied entries older than cutoff, so the set doesn't grow forever."""
        before = len(self.runs_applied)
        self.runs_applied = {key for key in self.runs_applied if datetime.fromisoformat(key) >= cutoff}
        return before - len(self.runs_applied)


def _fresh_book() -> PaperBook:
    """Return an empty book seeded at settings.ARGUS_TOTAL_WEALTH."""
    return PaperBook(equity=settings.ARGUS_TOTAL_WEALTH, high_water_mark=settings.ARGUS_TOTAL_WEALTH)


def load(path: str) -> PaperBook:
    """Load persisted paper equity, starting fresh only when absent and raising on corrupt history."""
    file_path = Path(path)
    if not file_path.exists():
        return _fresh_book()

    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)
    last_run_raw = raw.get("last_run_timestamp")
    rebased_at_raw = raw.get("rebased_at")
    return PaperBook(
        equity=raw["equity"],
        high_water_mark=raw["high_water_mark"],
        last_run_timestamp=datetime.fromisoformat(last_run_raw) if last_run_raw else None,
        runs_applied=set(raw.get("runs_applied", [])),
        rebased_at=datetime.fromisoformat(rebased_at_raw) if rebased_at_raw else None,
    )


def save(book: PaperBook, path: str) -> None:
    """Persist a PaperBook to JSON, creating parent directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "equity": book.equity,
        "high_water_mark": book.high_water_mark,
        "last_run_timestamp": (book.last_run_timestamp.isoformat() if book.last_run_timestamp else None),
        "runs_applied": sorted(book.runs_applied),
        "rebased_at": book.rebased_at.isoformat() if book.rebased_at else None,
    }
    atomic_write_text(file_path, json.dumps(payload, indent=2))
