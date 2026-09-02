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
zeros for the rest — it's averaged over the matured subset only. Runs are
compounded at most once per horizon_days window, so a decision made every
session doesn't compound the same overlapping days many times over.

Responsibilities:
  - compute_run_returns: group decisions by run (shared session_timestamp),
    weight-average each non-overlapping run's matured outcomes into one
    compounding return
  - PaperBook: equity/high-water-mark/applied-runs state, with idempotent
    run application, manual rebasing (see rebase(), used by
    KillSwitch.reset()), and bounded runs_applied retention (prune_runs_applied)
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from argus.config import settings
from argus.orchestration.reconciliation import _needs_reconciliation, compute_realized_return
from argus.schemas.signals import ARGUSDecision
from argus.seams import MarketDataProvider

logger = logging.getLogger("argus.paper_book")


def _close_prices(
    ticker: str,
    market_data: MarketDataProvider,
    cache: dict[str, Optional[pd.Series]],
) -> Optional[pd.Series]:
    """Returns a ticker's daily close series, fetching each ticker at most once per cache.

    Args:
        ticker: Ticker to price.
        market_data: Source of the daily close price series.
        cache: Per-call memo, mutated in place; a failed fetch is cached as
            None so a broken ticker isn't retried for every decision on it.

    Returns:
        The close price series, or None if the fetch failed.
    """
    if ticker not in cache:
        try:
            cache[ticker] = market_data.ohlcv_daily(ticker)["close"]
        except Exception as exc:
            logger.warning(
                "[PaperBook] failed to fetch price history for %s (%s): %s",
                ticker,
                type(exc).__name__,
                exc,
            )
            cache[ticker] = None
    return cache[ticker]


def _weighted_run_return(
    run_decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
    prices_by_ticker: dict[str, Optional[pd.Series]],
) -> Optional[float]:
    """Averages one run's matured decision outcomes, weighted by allocation_pct.

    Args:
        run_decisions: Decisions sharing one session_timestamp, all already
            confirmed by _needs_reconciliation.
        market_data: Source of each ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.
        prices_by_ticker: Price memo shared across runs, see _close_prices.

    Returns:
        The weighted-average return, or None if nothing in the run has
        matured or priced yet — never a fabricated 0.0.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for decision in run_decisions:
        prices = _close_prices(decision.ticker, market_data, prices_by_ticker)
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

    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


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
        omitted rather than reported as a 0.0 return. Runs falling inside a
        previously kept run's horizon window are also omitted (not zeroed) so
        overlapping runs are never compounded together.
    """
    by_run: dict[datetime, list[ARGUSDecision]] = defaultdict(list)
    for decision in decisions:
        if _needs_reconciliation(decision):
            by_run[decision.session_timestamp].append(decision)

    horizon = timedelta(days=horizon_days)
    prices_by_ticker: dict[str, Optional[pd.Series]] = {}
    results: list[tuple[datetime, float]] = []
    last_kept_timestamp: Optional[datetime] = None
    for run_timestamp in sorted(by_run):
        if last_kept_timestamp is not None and run_timestamp < last_kept_timestamp + horizon:
            continue

        run_return = _weighted_run_return(
            by_run[run_timestamp], market_data, horizon_days, prices_by_ticker
        )
        if run_return is not None:
            results.append((run_timestamp, run_return))
            last_kept_timestamp = run_timestamp

    return results


@dataclass
class PaperBook:
    """A synthetic, run-compounded equity curve fed by matured decision outcomes.

    Attributes:
        equity: Current notional equity value.
        high_water_mark: Highest equity value observed so far.
        last_run_timestamp: session_timestamp of the most recently applied
            run, or None if no run has been applied yet.
        runs_applied: ISO-formatted session_timestamps of runs already
            compounded in, so a re-run of compute_run_returns over a growing
            decision history doesn't double-apply one already reflected in
            equity.
        rebased_at: When this book was last manually rebased via rebase(),
            if ever.
    """

    equity: float
    high_water_mark: float
    last_run_timestamp: Optional[datetime] = None
    runs_applied: set[str] = field(default_factory=set)
    rebased_at: Optional[datetime] = None

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

    def rebase(self, new_inception_value: float, rebased_at: Optional[datetime] = None) -> None:
        """Rebases equity and high-water mark to a new inception value in place.

        Called alongside KillSwitch.reset() so an operator's manual reset is
        reflected in the persisted curve, not just the in-memory kill switch —
        otherwise the next reconcile pass calls update_portfolio_value with the
        stale, already-drawn-down equity and re-halts within
        check_interval_seconds. runs_applied is left untouched, so runs already
        compounded into the old curve are never re-applied against the new base.

        Args:
            new_inception_value: Replacement equity and high-water-mark value.
            rebased_at: Timestamp of the rebase; defaults to now.
        """
        self.equity = new_inception_value
        self.high_water_mark = new_inception_value
        self.rebased_at = rebased_at or datetime.now()  # noqa: DTZ005

    def prune_runs_applied(self, cutoff: datetime) -> int:
        """Drops runs_applied entries older than cutoff, so the set doesn't grow forever.

        Safe to prune independently of equity/high_water_mark: runs_applied
        exists only to make apply_run idempotent against re-processing a run
        compute_run_returns has already returned once, and a run this old will
        never be recomputed again — decisions that far back have either
        already been compacted out of decisions.jsonl or fall outside every
        horizon window compute_run_returns groups by.

        Args:
            cutoff: Runs applied before this timestamp are dropped from
                tracking. Should be tz-naive, matching the naive
                session_timestamps runs_applied entries are derived from.

        Returns:
            Count of entries dropped.
        """
        before = len(self.runs_applied)
        self.runs_applied = {
            key for key in self.runs_applied if datetime.fromisoformat(key) >= cutoff
        }
        return before - len(self.runs_applied)


def _fresh_book() -> PaperBook:
    """Returns an empty book seeded at settings.ARGUS_TOTAL_WEALTH."""
    return PaperBook(
        equity=settings.ARGUS_TOTAL_WEALTH, high_water_mark=settings.ARGUS_TOTAL_WEALTH
    )


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
        return _fresh_book()

    try:
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
    except Exception as exc:
        logger.warning("[PaperBook] failed to load %s, starting fresh: %s", path, exc)
        return _fresh_book()


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
        "rebased_at": book.rebased_at.isoformat() if book.rebased_at else None,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
