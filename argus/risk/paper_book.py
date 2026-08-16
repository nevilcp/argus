"""
argus/risk/paper_book.py

A synthetic paper equity curve derived from matured decision outcomes, giving
the kill switch a real loss signal to gate on.

ARGUS tracks no positions and executes nothing — there is no broker, no book,
no mark-to-market. Each graph invocation ("run") is treated as one notional
rebalance: within a run, decisions that took a position are weighted by
allocation_pct and averaged into that run's realized return once at least one
matured decision's outcome is known, then compounded onto the running equity
curve. A run with only some decisions matured is not diluted with fabricated
zeros for the rest — it's averaged over the matured subset only.

Responsibilities:
  - compute_run_returns: group decisions by run (shared session_timestamp),
    weight-average each run's matured outcomes into one compounding return
  - PaperBook: equity/high-water-mark/applied-runs state, with idempotent
    run application
  - load / save: JSON round trip at a caller-supplied path

Not responsible for:
  - Reconciling individual decisions into cultural memory (see
    orchestration/reconciliation.py, which this module calls into)
  - Deciding when the kill switch halts (see risk/kill_switch.py)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from argus.config import settings
from argus.orchestration.reconciliation import _needs_reconciliation, compute_realized_return
from argus.schemas.signals import ARGUSDecision
from argus.seams import MarketDataProvider

logger = logging.getLogger("argus.paper_book")


def compute_run_returns(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
) -> list[tuple[datetime, float]]:
    """Groups decisions into runs and computes each matured run's weighted return.

    A run is every decision sharing one session_timestamp (one graph
    invocation logs all of its decisions with the same timestamp — see
    graph.py's node_log_decisions). Fetches each ticker's price history at
    most once across the whole call, mirroring reconcile_decisions.

    Args:
        decisions: Candidate decisions, e.g. from load_decisions_from_jsonl.
        market_data: Source of each ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.

    Returns:
        (run_timestamp, weighted_return) pairs, one per run with at least one
        matured, priceable decision, sorted by run_timestamp. A run with no
        matured decisions yet, or whose tickers all failed to fetch, is
        omitted rather than reported as a 0.0 return.
    """
    by_run: dict[datetime, list[ARGUSDecision]] = defaultdict(list)
    for decision in decisions:
        if _needs_reconciliation(decision):
            by_run[decision.session_timestamp].append(decision)

    prices_by_ticker: dict[str, Optional[pd.Series]] = {}
    results: list[tuple[datetime, float]] = []
    for run_timestamp in sorted(by_run):
        weighted_sum = 0.0
        weight_total = 0.0
        for decision in by_run[run_timestamp]:
            ticker = decision.ticker
            if ticker not in prices_by_ticker:
                try:
                    prices_by_ticker[ticker] = market_data.ohlcv_daily(ticker)["close"]
                except Exception as exc:
                    logger.warning(
                        "[PaperBook] failed to fetch price history for %s: %s", ticker, exc
                    )
                    prices_by_ticker[ticker] = None

            prices = prices_by_ticker[ticker]
            if prices is None:
                continue
            outcome = compute_realized_return(decision, market_data, horizon_days, prices=prices)
            if outcome is None:
                continue

            actual_return_pct, _, _ = outcome
            assert decision.allocation is not None, "_needs_reconciliation already confirmed this"
            weight = decision.allocation.allocation_pct
            weighted_sum += actual_return_pct * weight
            weight_total += weight

        if weight_total > 0:
            results.append((run_timestamp, weighted_sum / weight_total))

    return results


@dataclass
class PaperBook:
    """A synthetic, run-compounded equity curve fed by matured decision outcomes."""

    equity: float
    high_water_mark: float
    last_run_timestamp: Optional[datetime] = None
    runs_applied: set[str] = field(default_factory=set)
    """ISO-formatted session_timestamps of runs already compounded in, so a
    re-run of compute_run_returns over a growing decision history doesn't
    double-apply one already reflected in equity."""

    def apply_run(self, run_timestamp: datetime, run_return: float) -> bool:
        """Compounds one run's weighted return onto the curve, unless already applied.

        Args:
            run_timestamp: The run's session_timestamp.
            run_return: Weighted-average matured return for that run, from
                compute_run_returns.

        Returns:
            True if the run was newly applied; False if run_timestamp was
            already in runs_applied.
        """
        key = run_timestamp.isoformat()
        if key in self.runs_applied:
            return False
        self.equity *= 1.0 + run_return
        self.high_water_mark = max(self.high_water_mark, self.equity)
        self.last_run_timestamp = run_timestamp
        self.runs_applied.add(key)
        return True

    def drawdown_from_peak(self) -> float:
        """Returns peak-to-trough drawdown as a non-negative fraction."""
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, (self.high_water_mark - self.equity) / self.high_water_mark)


def load(path: str) -> PaperBook:
    """Loads a persisted PaperBook, or starts a fresh one at ARGUS_TOTAL_WEALTH.

    A missing or unparseable file is treated as "no history yet" rather than
    an error — the first-ever reconcile pass on a fresh deployment has
    nothing to load.

    Args:
        path: Path a prior save() wrote to.

    Returns:
        The loaded PaperBook, or a fresh one seeded at settings.ARGUS_TOTAL_WEALTH.
    """
    file_path = Path(path)
    if not file_path.exists():
        return PaperBook(
            equity=settings.ARGUS_TOTAL_WEALTH, high_water_mark=settings.ARGUS_TOTAL_WEALTH
        )

    try:
        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)
        last_run_raw = raw.get("last_run_timestamp")
        return PaperBook(
            equity=raw["equity"],
            high_water_mark=raw["high_water_mark"],
            last_run_timestamp=datetime.fromisoformat(last_run_raw) if last_run_raw else None,
            runs_applied=set(raw.get("runs_applied", [])),
        )
    except Exception as exc:
        logger.warning("[PaperBook] failed to load %s, starting fresh: %s", path, exc)
        return PaperBook(
            equity=settings.ARGUS_TOTAL_WEALTH, high_water_mark=settings.ARGUS_TOTAL_WEALTH
        )


def save(book: PaperBook, path: str) -> None:
    """Persists a PaperBook to JSON, creating parent directories as needed.

    Args:
        book: The PaperBook to persist.
        path: Destination file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "equity": book.equity,
        "high_water_mark": book.high_water_mark,
        "last_run_timestamp": (
            book.last_run_timestamp.isoformat() if book.last_run_timestamp else None
        ),
        "runs_applied": sorted(book.runs_applied),
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
