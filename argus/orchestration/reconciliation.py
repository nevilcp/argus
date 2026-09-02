"""
argus/orchestration/reconciliation.py

Closes the decision -> outcome loop: leave-one-out credit assignment,
realized-return computation against a MarketDataProvider, and pulling
completed decisions back out of the LangGraph checkpoint so
cultural.store_trade_outcome has something real to persist.

Decisions are read back from the existing LangGraph checkpoint rather than a
new archive, credit assignment uses leave-one-out ablation rather than exact
Shapley, and horizon_days is provisional pending a pre-registered evaluation.

Responsibilities:
  - credit_primary_driver: leave-one-out ablation over HybridSignalAggregator
  - compute_realized_return: entry/exit close prices -> (return, holding_days, reason)
  - reconcile_decision / reconcile_decisions: orchestrate the above into
    cultural.store_trade_outcome calls
  - load_decisions_from_checkpoints: read ARGUSDecision objects back out of
    argus_graph.db
  - load_decisions_from_jsonl: read ARGUSDecision objects back out of a
    decisions.jsonl log (the unattended collector's lighter-weight alternative)
  - prune_checkpoints / compact_decisions_jsonl: bound the two decision
    stores above so neither grows forever
  - run_reconciliation_pass: compose the above plus the paper-book update
    into the one reconciliation sequence both api/main.py and
    scripts/reconcile_outcomes.py run

Not responsible for:
  - Deciding what a "good" outcome is, or evaluation metrics (see
    backtesting/evaluation.py and backtesting/metrics.py)
  - Scheduling reconciliation runs (see scripts/reconcile_outcomes.py)

Dependencies:
  - argus.orchestration.aggregator (HybridSignalAggregator)
  - argus.orchestration.graph (build_checkpoint_serde)
  - argus.memory.cultural (CulturalMemoryManager)
  - argus.seams (MarketDataProvider)
  - argus.risk.paper_book (PaperBook; imported inside run_reconciliation_pass
    to break the cycle, since paper_book imports back into this module)
  - langgraph (SqliteSaver, to read argus_graph.db)
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.graph import build_checkpoint_serde
from argus.params import RECONCILIATION
from argus.schemas.signals import ARGUSDecision
from argus.seams import MarketDataProvider

logger = logging.getLogger("argus.reconciliation")

_ABLATABLE_AGENTS = ("technical", "fundamental", "sentiment")


def credit_primary_driver(
    decision: ARGUSDecision, aggregator: Optional[HybridSignalAggregator] = None
) -> str:
    """Credits whichever specialist agent's signal moved the aggregated result the most.

    Leave-one-out ablation: reruns HybridSignalAggregator.aggregate() once per
    present agent with that agent's signal removed. Each removal is scored as
    (did the consensus direction flip, how large was this agent's own
    baseline weighted vote) and the largest-scoring removal is credited.

    Direction first, magnitude second, rather than a scalar delta on the
    ablated call's final `conviction`: it is the consensus *direction* an
    ablation settles on that decides whether the resulting trade changes, so an
    agent whose removal flips the direction had a categorically larger effect
    than one whose removal only moved the number. Among removals that don't
    flip the outcome, the agent that cast more raw vote was the bigger
    contributor — read straight off the baseline `weighted_votes`, since the
    pools are additive sums of independent per-agent votes and a removed
    agent's marginal effect on its pool is exactly its own vote. This avoids
    exact Shapley over the 2^3 - 1 coalitions.

    Each ablated rerun replays `decision.aggregated.reliability` — the same
    reliability dict the baseline aggregation used — rather than an
    unweighted aggregate() call. Without it, every ablation would silently
    use reliability_mult=1.0, so the baseline (computed with reliability)
    and every ablated rerun (computed without it) could disagree on
    direction before any agent is even removed, spuriously flipping credit
    to argmax(baseline_votes) for every agent regardless of which one was
    actually removed.

    Args:
        decision: Completed ARGUSDecision with technical/fundamental/sentiment
            signals and its `aggregated` result already populated.
        aggregator: HybridSignalAggregator to re-run; defaults to a fresh
            instance (aggregate() is a pure function of its arguments — there
            is no state for two calls to share).

    Returns:
        'technical', 'fundamental', or 'sentiment'; 'unknown' if fewer than
        two agents contributed a signal (ablation needs something to compare
        against) or the decision has no aggregated result to ablate from.
    """
    present = [name for name in _ABLATABLE_AGENTS if getattr(decision, name) is not None]

    if decision.aggregated is None or len(present) < 2:
        return present[0] if len(present) == 1 else "unknown"

    agg = aggregator or HybridSignalAggregator()
    baseline_signal = decision.aggregated.signal
    baseline_votes = decision.aggregated.weighted_votes

    best_name = "unknown"
    best_score = (False, -1.0)
    for name in present:
        result = agg.aggregate(
            None if name == "technical" else decision.technical,
            decision.macro,
            None if name == "fundamental" else decision.fundamental,
            None if name == "sentiment" else decision.sentiment,
            reliability=decision.aggregated.reliability,
        )
        flipped = result.signal != baseline_signal
        score = (flipped, baseline_votes.get(name, 0.0))
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def _as_naive_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalizes a datetime/Timestamp to tz-naive so entry/exit comparisons never raise."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _needs_reconciliation(decision: ARGUSDecision) -> bool:
    """Whether a decision took a position that could have a realized outcome."""
    return (
        decision.technical is not None
        and decision.allocation is not None
        and decision.allocation.allocation_pct > 0
    )


def _realized_return_from_prices(
    decision: ARGUSDecision, prices: pd.Series, horizon_days: int
) -> Optional[tuple[float, int, str]]:
    """Pairs a decision's entry price with a later close drawn from an already-fetched series.

    Args:
        decision: Completed ARGUSDecision; caller has already confirmed
            _needs_reconciliation(decision).
        prices: Close-price Series for decision.ticker.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.

    Returns:
        (actual_return_pct, holding_days, exit_reason), or None when the
        price series doesn't yet extend past the target exit date — horizon
        not reached, deferred rather than fabricated.
    """
    assert decision.technical is not None, "caller must confirm _needs_reconciliation(decision)"
    entry_price = decision.technical.current_price
    entry_date = _as_naive_timestamp(decision.session_timestamp)
    target_exit_date = entry_date + timedelta(days=horizon_days)

    prices = prices.copy()
    prices.index = pd.DatetimeIndex([_as_naive_timestamp(ts) for ts in prices.index])

    on_or_after = prices[prices.index >= target_exit_date].sort_index()
    if on_or_after.empty:
        return None

    exit_date = on_or_after.index[0]
    exit_price = float(on_or_after.iloc[0])

    actual_return_pct = (exit_price - entry_price) / entry_price
    holding_days = (exit_date - entry_date).days
    exit_reason = f"horizon reached ({horizon_days}d target, {holding_days}d actual)"

    return actual_return_pct, holding_days, exit_reason


def compute_realized_return(
    decision: ARGUSDecision,
    market_data: MarketDataProvider,
    horizon_days: int,
    *,
    prices: Optional[pd.Series] = None,
) -> Optional[tuple[float, int, str]]:
    """Computes realized return by pairing the decision's entry price with a later close.

    Entry price is decision.technical.current_price — the close the
    technical agent scored against at decision time. Exit price is the first
    close on or after session_timestamp + horizon_days, read from
    market_data.ohlcv_daily().

    Args:
        decision: Completed ARGUSDecision.
        market_data: Source of the ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.
        prices: Pre-fetched close-price Series for decision.ticker. Passed by
            callers batching one fetch per ticker across several decisions
            (see reconcile_decisions, risk/paper_book.py); fetched here when
            omitted.

    Returns:
        (actual_return_pct, holding_days, exit_reason), or None when there is
        nothing to reconcile: no technical signal (no entry price), no
        allocation or a zero/negative-weight allocation (no position was
        taken), or the price series doesn't yet extend past the target exit
        date — horizon not reached, deferred rather than fabricated.
    """
    if not _needs_reconciliation(decision):
        return None
    if prices is None:
        prices = market_data.ohlcv_daily(decision.ticker)["close"]
    return _realized_return_from_prices(decision, prices, horizon_days)


def reconcile_decision(
    decision: ARGUSDecision,
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    horizon_days: int = RECONCILIATION.horizon_days,
    *,
    prices: Optional[pd.Series] = None,
) -> bool:
    """Reconciles a single decision: computes its outcome and stores it if the horizon has passed.

    Args:
        decision: Completed ARGUSDecision to reconcile.
        market_data: Source of the ticker's daily close price series.
        cultural: CulturalMemoryManager to persist the outcome to.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.
        prices: Pre-fetched close-price Series for decision.ticker. Passed by
            reconcile_decisions() so decisions sharing a ticker issue one
            market_data fetch between them instead of one each; fetched here
            when omitted.

    Returns:
        True if an outcome was computed and handed to
        cultural.store_trade_outcome (which itself still drops |return| <=
        RECONCILIATION.min_abs_return_for_storage trades — see its
        docstring). False if the decision had nothing to reconcile or its
        horizon hasn't passed yet.
    """
    outcome = compute_realized_return(decision, market_data, horizon_days, prices=prices)
    if outcome is None:
        return False

    actual_return_pct, holding_days, exit_reason = outcome
    primary_driver = credit_primary_driver(decision)

    cultural.store_trade_outcome(
        decision=decision,
        actual_return_pct=actual_return_pct,
        holding_days=holding_days,
        exit_reason=exit_reason,
        primary_driver=primary_driver,
    )
    logger.info(
        "[Reconcile] %s (%s): %+.2f%% over %d days, primary_driver=%s",
        decision.ticker,
        decision.decision_id,
        actual_return_pct * 100,
        holding_days,
        primary_driver,
    )
    return True


def reconcile_decisions(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    horizon_days: int = RECONCILIATION.horizon_days,
) -> int:
    """Reconciles every not-yet-reconciled decision whose horizon has passed.

    Skips decisions cultural already has a stored outcome for (see
    CulturalMemoryManager.already_reconciled) and fetches each ticker's price
    history at most once regardless of how many decisions share it. This runs
    on a daily schedule against a decision history that only grows — without
    both, every past decision gets re-fetched and re-processed on every run.

    Args:
        decisions: Decisions to attempt reconciliation for, e.g. from
            load_decisions_from_checkpoints().
        market_data: Source of each ticker's daily close price series.
        cultural: CulturalMemoryManager to persist outcomes to.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.

    Returns:
        Count of decisions actually stored via cultural.store_trade_outcome.
    """
    already_done = cultural.already_reconciled([d.decision_id for d in decisions])
    candidates = [
        d for d in decisions if d.decision_id not in already_done and _needs_reconciliation(d)
    ]

    prices_by_ticker: dict[str, Optional[pd.Series]] = {}
    stored = 0
    for decision in candidates:
        if decision.ticker not in prices_by_ticker:
            try:
                prices_by_ticker[decision.ticker] = market_data.ohlcv_daily(decision.ticker)["close"]
            except Exception as exc:
                logger.warning(
                    "[Reconcile] failed to fetch price history for %s (%s): %s",
                    decision.ticker,
                    type(exc).__name__,
                    exc,
                )
                prices_by_ticker[decision.ticker] = None

        prices = prices_by_ticker[decision.ticker]
        if prices is None:
            continue
        if reconcile_decision(decision, market_data, cultural, horizon_days, prices=prices):
            stored += 1

    return stored


@contextmanager
def _checkpoint_saver(db_path: str) -> Iterator[SqliteSaver]:
    """Opens a SqliteSaver over the checkpoint database, closing its connection on exit."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=build_checkpoint_serde())
    finally:
        conn.close()


def _newest_checkpoint_per_thread(saver: SqliteSaver) -> Iterator[tuple[str, Checkpoint]]:
    """Yields each thread's newest checkpoint exactly once, as (thread_id, checkpoint).

    SqliteSaver.list() yields checkpoints newest-first, so a thread's first tuple
    is its most recent — and so most complete — one.
    """
    seen_threads: set[str] = set()
    for tup in saver.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        if thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)
        yield thread_id, tup.checkpoint


def load_decisions_from_checkpoints(db_path: str = "argus_graph.db") -> list[ARGUSDecision]:
    """Reads every session's decisions back out of the LangGraph checkpoint database.

    Each session runs under its own thread_id (see build_graph()'s callers),
    and SqliteSaver.list() yields checkpoints newest-first; this keeps each
    thread's first (= most recent, most complete) checkpoint's `decisions`
    channel value and skips the rest, reading the checkpoint the graph
    already writes rather than maintaining a dedicated decision archive.

    Args:
        db_path: Path to the SQLite file build_graph() checkpoints to.

    Returns:
        Every ARGUSDecision found across all threads. Empty list if the
        database doesn't exist yet or holds no checkpoints.
    """
    decisions: list[ARGUSDecision] = []
    with _checkpoint_saver(db_path) as saver:
        for _thread_id, checkpoint in _newest_checkpoint_per_thread(saver):
            channel_values = checkpoint.get("channel_values", {})
            decisions.extend(channel_values.get("decisions") or [])
    return decisions


def load_decisions_from_jsonl(path: str) -> list[ARGUSDecision]:
    """Reads decisions back out of a JSONL decision log, one ARGUSDecision per line.

    The lighter-weight counterpart to load_decisions_from_checkpoints, for
    deployments that don't carry the full LangGraph checkpoint database
    around — the unattended collector (argus/orchestration/collector.py)
    appends each decision here instead, so the state a scheduled run has to
    persist stays small.

    Args:
        path: Path to a JSONL file of ARGUSDecision.model_dump_json() lines.

    Returns:
        Every ARGUSDecision found in the file. Empty list if the file
        doesn't exist yet. A line that fails to parse is logged and skipped
        rather than aborting the whole read.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    decisions: list[ARGUSDecision] = []
    with open(file_path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                decisions.append(ARGUSDecision.model_validate_json(line))
            except Exception as exc:
                logger.warning(
                    "load_decisions_from_jsonl: skipping malformed line %d in %s: %s",
                    line_no,
                    path,
                    exc,
                )
    return decisions


@dataclass
class CompactionResult:
    """Outcome of one compact_decisions_jsonl() call.

    Attributes:
        retained: Count of decisions written back to the log.
        retired_unresolved: Count of decisions dropped for aging past
            RECONCILIATION.unresolved_retirement_days without ever having
            received an outcome — distinct from `retained`'s complement,
            which also includes decisions dropped as routine, already-resolved
            cleanup.
    """

    retained: int
    retired_unresolved: int = 0


def compact_decisions_jsonl(
    path: str,
    cutoff: datetime,
    unresolved_cutoff: datetime,
    resolved_ids: Collection[str],
) -> CompactionResult:
    """Rewrites a decisions.jsonl log, dropping resolved sessions older than cutoff.

    A decision is resolved once it has a stored outcome (its decision_id is in
    resolved_ids) or could never produce one (see _needs_reconciliation) —
    anything else has no recorded outcome yet and is not eligible for this
    routine prune, however old it is: a reconcile pass that never ran, or
    never found a matured price, must not cost it the evidence it would
    otherwise have produced. Such a decision is instead retired once it also
    passes the much wider `unresolved_cutoff`, at which point it is declared
    permanently unresolvable and dropped anyway, counted separately from a
    routine, resolved prune.

    Meant to run right after a reconcile_decisions() pass over the same log
    (see api/main.py's _reconcile_once and scripts/reconcile_outcomes.py),
    with resolved_ids reflecting that same pass's outcome stores.

    Args:
        path: decisions.jsonl path.
        cutoff: Resolved decisions with session_timestamp before this are
            dropped. Should be tz-naive, matching the naive timestamps this
            log stores.
        unresolved_cutoff: Unresolved decisions with session_timestamp before
            this are retired anyway. Should be tz-naive and at or before
            cutoff (i.e. further in the past), so an unresolved decision
            always gets at least as long as a resolved one.
        resolved_ids: decision_id values that already have a stored outcome.

    Returns:
        A CompactionResult. Both fields are 0 (and no file written) if the
        file doesn't exist yet.
    """
    file_path = Path(path)
    if not file_path.exists():
        return CompactionResult(retained=0)

    decisions = load_decisions_from_jsonl(path)
    cutoff_naive = _as_naive_timestamp(cutoff)
    unresolved_cutoff_naive = _as_naive_timestamp(unresolved_cutoff)

    kept: list[ARGUSDecision] = []
    retired_unresolved = 0
    for decision in decisions:
        session_ts = _as_naive_timestamp(decision.session_timestamp)
        if session_ts >= cutoff_naive:
            kept.append(decision)
            continue

        resolved = decision.decision_id in resolved_ids or not _needs_reconciliation(decision)
        if resolved:
            continue
        if session_ts >= unresolved_cutoff_naive:
            kept.append(decision)
        else:
            retired_unresolved += 1

    if len(kept) == len(decisions):
        return CompactionResult(retained=len(kept), retired_unresolved=retired_unresolved)

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for decision in kept:
            f.write(decision.model_dump_json() + "\n")
    tmp_path.replace(file_path)

    logger.info(
        "compact_decisions_jsonl: kept %d/%d decision(s) in %s (cutoff=%s, retired %d "
        "unresolved past %s)",
        len(kept),
        len(decisions),
        path,
        cutoff_naive.isoformat(),
        retired_unresolved,
        unresolved_cutoff_naive.isoformat(),
    )
    return CompactionResult(retained=len(kept), retired_unresolved=retired_unresolved)


def prune_checkpoints(db_path: str, cutoff: datetime) -> int:
    """Deletes every checkpoint thread whose most recent checkpoint predates cutoff, then VACUUMs.

    Each build_graph() invocation checkpoints under a fresh, never-reused
    thread_id (see build_graph()'s callers) and nothing resumes a thread by
    id, so a thread's checkpoints exist only as load_decisions_from_checkpoints'
    read path — a fallback now that decisions.jsonl is the durable
    record both /analyze and the collector write to. Retention, not removal:
    the checkpoint DB stays available as that fallback for anything within the
    reconciliation window, it just no longer grows without bound.

    Args:
        db_path: Path to the SQLite file build_graph() checkpoints to.
        cutoff: Threads whose newest checkpoint timestamp is before this are
            deleted. Should be tz-naive, matching the naive timestamps
            LangGraph checkpoint metadata is compared against here.

    Returns:
        Count of threads deleted. 0 if the database doesn't exist yet.
    """
    if not Path(db_path).exists():
        return 0

    cutoff_naive = _as_naive_timestamp(cutoff)
    with _checkpoint_saver(db_path) as saver:
        stale_threads: list[str] = []
        for thread_id, checkpoint in _newest_checkpoint_per_thread(saver):
            ts_raw = checkpoint.get("ts")
            if ts_raw is None:
                continue
            if _as_naive_timestamp(datetime.fromisoformat(ts_raw)) < cutoff_naive:
                stale_threads.append(thread_id)

        if not stale_threads:
            return 0

        params = [(thread_id,) for thread_id in stale_threads]
        conn = saver.conn
        cur = conn.cursor()
        cur.executemany("DELETE FROM checkpoints WHERE thread_id = ?", params)
        cur.executemany("DELETE FROM writes WHERE thread_id = ?", params)
        conn.commit()
        conn.execute("VACUUM")

    logger.info(
        "prune_checkpoints: deleted %d thread(s) from %s (cutoff=%s)",
        len(stale_threads),
        db_path,
        cutoff_naive.isoformat(),
    )
    return len(stale_threads)


@dataclass
class ReconciliationReport:
    """Outcome of one run_reconciliation_pass() call.

    Attributes:
        decisions_loaded: Count of decisions read from the configured
            decision source (decisions.jsonl or the checkpoint database).
        outcomes_stored: Count of decisions actually reconciled and stored
            via cultural.store_trade_outcome.
        paper_book_updated: Whether the paper-book update step completed.
            False if it raised — equity/drawdown are then still their unset
            defaults, not a real (zeroed-out) portfolio value, so a caller
            syncing a kill switch off `equity` must gate on this first.
        equity: Paper-book equity after this pass, or 0.0 if
            paper_book_updated is False.
        drawdown: Paper-book drawdown from peak after this pass, or 0.0 if
            paper_book_updated is False.
        runs_applied_pruned: Count of applied-run entries pruned from the
            paper book.
        pending_snapshots_expired: Count of PENDING cultural-memory
            snapshots expired by this pass.
        decisions_compacted: Retained decisions.jsonl count, or None if
            decisions_log_path wasn't given.
        decisions_retired_unresolved: Count of decisions.jsonl decisions
            dropped for aging past RECONCILIATION.unresolved_retirement_days
            without ever receiving an outcome, or None if decisions_log_path
            wasn't given. Distinct from decisions_compacted, which counts what
            survived — this counts a deliberate, unresolved retirement rather
            than a routine, resolved prune.
        checkpoints_pruned: Deleted checkpoint-thread count, or None if
            checkpoint_db_path wasn't given.
        errors: One entry per independent step that failed; every other
            step still ran.
    """

    decisions_loaded: int = 0
    outcomes_stored: int = 0
    paper_book_updated: bool = False
    equity: float = 0.0
    drawdown: float = 0.0
    runs_applied_pruned: int = 0
    pending_snapshots_expired: int = 0
    decisions_compacted: Optional[int] = None
    decisions_retired_unresolved: Optional[int] = None
    checkpoints_pruned: Optional[int] = None
    errors: list[str] = field(default_factory=list)


@contextmanager
def _pass_step(report: ReconciliationReport, label: str) -> Iterator[None]:
    """Runs one independently-failable step of a reconciliation pass.

    A raise is logged and recorded on the report instead of propagating, so one
    store failing to bound itself never skips the next one.

    Args:
        report: Report the failure is recorded on.
        label: Step name, used verbatim in both the log line and the recorded error.
    """
    try:
        yield
    except Exception as exc:
        logger.exception("[Reconcile] %s failed", label)
        report.errors.append(f"{label} failed: {exc}")


def run_reconciliation_pass(
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    paper_book_path: str,
    *,
    decisions_log_path: Optional[str] = None,
    checkpoint_db_path: Optional[str] = None,
    horizon_days: int = RECONCILIATION.horizon_days,
) -> ReconciliationReport:
    """Runs the one reconciliation sequence shared by api/main.py and scripts/reconcile_outcomes.py.

    Loads decisions, reconciles the matured ones against market_data, updates
    the paper book, and bounds every growing store whose path was supplied.
    Kill-switch sync is deliberately not part of this: it stays at the API's
    call site, fed by the returned report's equity, so a scheduled runner
    with no daemon to sync simply doesn't call it — that difference is one
    line at the call site rather than an argument silently omitted here.

    The decisions log wins as the read source when decisions_log_path is
    given; the checkpoint database otherwise. Every store whose path is
    known is bounded regardless of which one was the read source, so a
    caller supplying both (api/main.py) gets both bounded and a caller
    supplying only one (a collector-only deployment) gets exactly that one.

    Args:
        market_data: Source of each ticker's daily close price series.
        cultural: CulturalMemoryManager to persist outcomes to and expire
            PENDING snapshots from.
        paper_book_path: Path the paper equity curve is loaded from and
            saved to.
        decisions_log_path: Path to a decisions.jsonl log. Read source when
            given, and compacted at the end of the pass. Omit for a
            deployment with no jsonl log.
        checkpoint_db_path: Path to the LangGraph checkpoint database. Read
            source when decisions_log_path is omitted; pruned at the end of
            the pass whenever given. Omit for a deployment with no
            checkpoint database.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.

    Returns:
        A ReconciliationReport describing what happened. Failures in the
        paper-book update and in each independently-bounded store
        (decisions.jsonl compaction, checkpoint pruning, PENDING snapshot
        expiry) are caught, logged, and recorded in `errors` — one store
        failing never skips another. The paper-book update is one atomic
        load/compute/apply/prune/save step: on failure nothing from it is
        saved, so the persisted book is never left partially applied.

    Raises:
        ValueError: Neither decisions_log_path nor checkpoint_db_path was
            given, so there is no decision source to read.
        Exception: Whatever loading decisions or reconcile_decisions raises —
            unlike the bounded stores below, these aren't independent of the
            rest of the pass, so a failure here is the caller's to handle.
    """
    # Deferred: argus.risk.paper_book imports this module for
    # _needs_reconciliation/compute_realized_return, so importing it at
    # module level here would be circular.
    from argus.risk import paper_book

    if decisions_log_path:
        decisions = load_decisions_from_jsonl(decisions_log_path)
    elif checkpoint_db_path:
        decisions = load_decisions_from_checkpoints(checkpoint_db_path)
    else:
        raise ValueError(
            "run_reconciliation_pass requires decisions_log_path or checkpoint_db_path"
        )

    report = ReconciliationReport(decisions_loaded=len(decisions))
    report.outcomes_stored = reconcile_decisions(
        decisions, market_data=market_data, cultural=cultural, horizon_days=horizon_days
    )

    cutoff = datetime.now() - timedelta(  # noqa: DTZ005
        days=horizon_days + RECONCILIATION.retention_margin_days
    )
    unresolved_cutoff = datetime.now() - timedelta(  # noqa: DTZ005
        days=horizon_days + RECONCILIATION.unresolved_retirement_days
    )

    with _pass_step(report, "paper-book update"):
        book = paper_book.load(paper_book_path)
        for run_timestamp, run_return in paper_book.compute_run_returns(
            decisions, market_data, horizon_days
        ):
            book.apply_run(run_timestamp, run_return)
        # unresolved_cutoff, not cutoff: an unresolved decision can now outlive
        # cutoff in decisions.jsonl (see compact_decisions_jsonl), so a run
        # sharing its session_timestamp must stay in runs_applied at least as
        # long — otherwise a later pass that recomputes the run from the
        # decisions still on disk would compound it onto equity a second time.
        report.runs_applied_pruned = book.prune_runs_applied(unresolved_cutoff)
        paper_book.save(book, paper_book_path)
        report.equity = book.equity
        report.drawdown = book.drawdown_from_peak()
        report.paper_book_updated = True

    if decisions_log_path:
        with _pass_step(report, "decisions.jsonl compaction"):
            resolved_ids = cultural.already_reconciled([d.decision_id for d in decisions])
            compaction = compact_decisions_jsonl(
                decisions_log_path, cutoff, unresolved_cutoff, resolved_ids
            )
            report.decisions_compacted = compaction.retained
            report.decisions_retired_unresolved = compaction.retired_unresolved

    if checkpoint_db_path:
        with _pass_step(report, "checkpoint pruning"):
            report.checkpoints_pruned = prune_checkpoints(checkpoint_db_path, cutoff)

    with _pass_step(report, "PENDING snapshot expiry"):
        report.pending_snapshots_expired = cultural.expire_pending_snapshots(cutoff)

    return report
