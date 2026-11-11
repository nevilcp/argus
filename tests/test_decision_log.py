"""Tests for orchestration/decision_log.py.

Covers the decisions.jsonl log (append, load, compaction under its lock) and
reading decisions back out of the LangGraph checkpoint database.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from argus.config import settings
from argus.orchestration.decision_log import (
    CompactionResult,
    append_decisions_jsonl,
    compact_decisions_jsonl,
    load_decisions_from_checkpoints,
    load_decisions_from_jsonl,
)
from argus.orchestration.graph import build_graph
from argus.orchestration.state import ARGUSState
from argus.schemas.signals import (
    ARGUSDecision,
    Signal,
)
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider
from tests.helpers.builders import (
    argus_decision,
    neutral_cultural_memory,
    position_allocation,
    technical_signal,
)
from tests.helpers.decision_store import write_decisions_log
from tests.helpers.paths import FIXTURES_DIR

# ---------------------------------------------------------------------------
# load_decisions_from_checkpoints
# ---------------------------------------------------------------------------


def test_load_decisions_from_checkpoints_on_a_missing_db_returns_empty_and_creates_nothing(
    tmp_path,
):
    """A missing checkpoint database reads as empty without sqlite creating the file."""
    db_path = tmp_path / "does_not_exist.db"

    assert load_decisions_from_checkpoints(str(db_path)) == []
    assert not db_path.exists()


def test_load_decisions_from_checkpoints_round_trips_a_real_graph_run(tmp_path):
    """Checkpoints round-trip real ARGUSDecision objects, not degraded dicts."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"]

    def per_ticker_llm(fixture_file: str) -> FixtureLLMClient:
        with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
            responses = json.load(f)

        def key_fn(user_prompt: str) -> str:
            for ticker in universe:
                if ticker in user_prompt:
                    return ticker
            raise KeyError(user_prompt[:200])

        return FixtureLLMClient(responses, key_fn=key_fn)

    def portfolio_llm() -> FixtureLLMClient:
        with open(FIXTURES_DIR / "llm_responses" / "portfolio.json") as f:
            response = json.load(f)
        return FixtureLLMClient({"only": json.dumps(response)}, key_fn=lambda _p: "only")

    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)

    db_path = str(tmp_path / "argus_graph_test.db")
    memory = neutral_cultural_memory()
    graph = build_graph(
        market_data=FixtureMarketDataProvider(fixtures_dir=FIXTURES_DIR / "market_data"),
        fundamental_llm=per_ticker_llm("fundamental.json"),
        sentiment_llm=per_ticker_llm("sentiment.json"),
        portfolio_llm=portfolio_llm(),
        checkpoint_db_path=db_path,
        cultural_memory_provider=lambda: memory,
    )

    state = ARGUSState(
        ticker=universe[0],
        universe=universe,
        total_wealth=100_000.0,
        invest_pct=0.8,
        risk_tolerance="MODERATE",
        backtest_mode=False,
        session_seed=None,
        as_of=None,
        price_history={},
        session_states=session_states,
        macro_context=None,
        technical_signals={},
        fundamental_signals={},
        sentiment_signals={},
        cultural_memory={"wisdom": [], "warnings": []},
        risk_assessments={},
        aggregated_signals={},
        portfolio_allocation=None,
        decisions=[],
        errors=[],
    )

    graph.invoke({**state}, {"configurable": {"thread_id": "test-thread"}})

    decisions = load_decisions_from_checkpoints(db_path)

    assert decisions, "expected at least one decision to round-trip"
    assert all(isinstance(d, ARGUSDecision) for d in decisions)
    assert {d.ticker for d in decisions} <= set(universe)


# ---------------------------------------------------------------------------
# compact_decisions_jsonl
# ---------------------------------------------------------------------------


def _compact(
    path: Path | str,
    cutoff: datetime,
    unresolved_cutoff: datetime = datetime(1900, 1, 1),
    resolved_ids: frozenset[str] = frozenset(),
) -> CompactionResult:
    """compact_decisions_jsonl with a default unresolved_cutoff far enough in the past
    to be a no-op.

    Lets tests that don't exercise unresolved-retention behavior omit it.
    """
    return compact_decisions_jsonl(str(path), cutoff, unresolved_cutoff, resolved_ids)


def test_compact_decisions_jsonl_drops_sessions_before_cutoff(tmp_path):
    """Decisions older than cutoff are dropped; everything at or after it is kept."""
    path = write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="OLD", session_timestamp=datetime(2026, 1, 1)),
        ARGUSDecision(ticker="EXACT", session_timestamp=datetime(2026, 1, 10)),
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )

    result = _compact(path, cutoff=datetime(2026, 1, 10))

    assert result.retained == 2
    assert result.retired_unresolved == 0
    tickers = {d.ticker for d in load_decisions_from_jsonl(str(path))}
    assert tickers == {"EXACT", "NEW"}


def test_compact_decisions_jsonl_preserves_malformed_lines_and_counts_them(tmp_path):
    """A line that fails to parse is written back verbatim rather than silently deleted."""
    path = write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="OLD", session_timestamp=datetime(2026, 1, 1)),
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"ticker": "BROKEN", "session_timestamp": \n')

    result = _compact(path, cutoff=datetime(2026, 1, 10))

    assert result.retained == 1
    assert result.malformed_preserved == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert '{"ticker": "BROKEN", "session_timestamp":' in lines
    assert {d.ticker for d in load_decisions_from_jsonl(str(path))} == {"NEW"}


def test_append_issued_during_compaction_is_not_lost(tmp_path, monkeypatch):
    """An append that arrives mid-compaction lands after it instead of being overwritten."""
    from argus.orchestration import decision_log

    path = write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="OLD", session_timestamp=datetime(2026, 1, 1)),
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )
    mid_compaction = threading.Event()
    release = threading.Event()
    real_needs = decision_log.needs_reconciliation

    def _held(decision):
        # Called per decision after the log was read and before it is rewritten
        mid_compaction.set()
        assert release.wait(timeout=10)
        return real_needs(decision)

    monkeypatch.setattr(decision_log, "needs_reconciliation", _held)

    compactor = threading.Thread(target=_compact, args=(path, datetime(2026, 1, 10)))
    compactor.start()
    assert mid_compaction.wait(timeout=10)

    appender = threading.Thread(
        target=append_decisions_jsonl,
        args=([ARGUSDecision(ticker="LATE", session_timestamp=datetime(2026, 1, 21))], str(path)),
    )
    appender.start()
    appender.join(timeout=0.5)  # blocked on the lock when the fix is in; done at once without it
    release.set()
    compactor.join(timeout=10)
    appender.join(timeout=10)

    assert {d.ticker for d in load_decisions_from_jsonl(str(path))} == {"NEW", "LATE"}


def test_compact_decisions_jsonl_missing_file_is_a_noop():
    """A path that doesn't exist yet returns 0 and writes nothing."""
    result = _compact("/nonexistent/decisions.jsonl", cutoff=datetime(2026, 1, 1))
    assert result.retained == 0
    assert result.retired_unresolved == 0


def test_compact_decisions_jsonl_nothing_to_drop_leaves_the_file_untouched(tmp_path):
    """When every decision is at or after cutoff, the file is not rewritten."""
    path = write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )
    original_mtime = path.stat().st_mtime_ns

    result = _compact(path, cutoff=datetime(2026, 1, 1))

    assert result.retained == 1
    assert path.stat().st_mtime_ns == original_mtime


def test_compact_decisions_jsonl_never_drops_an_unresolved_position_decision_at_routine_cutoff(
    tmp_path,
):
    """A decision that took a position but has no stored outcome survives routine compaction.

    Reproduces the observed bug: a reconcile pass that finds nothing to
    reconcile (e.g. a missed schedule tick, or a price fetch failure) must
    not cost the decision its one chance at being scored, however old it is
    relative to the ordinary retention cutoff.
    """
    old = ARGUSDecision(
        ticker="UNRESOLVED",
        session_timestamp=datetime(2020, 1, 1),
        technical=technical_signal(Signal.BULLISH, 0.8),
        allocation=position_allocation(ticker="UNRESOLVED"),
    )
    path = write_decisions_log(tmp_path / "decisions.jsonl", old)

    result = _compact(
        path,
        cutoff=datetime(2026, 1, 1),
        unresolved_cutoff=datetime(2019, 1, 1),
        resolved_ids=frozenset(),
    )

    assert result.retained == 1
    assert result.retired_unresolved == 0
    assert {d.ticker for d in load_decisions_from_jsonl(str(path))} == {"UNRESOLVED"}


def test_compact_decisions_jsonl_retires_an_unresolved_decision_past_the_wider_cutoff(
    tmp_path,
):
    """An unresolved decision is dropped, and counted separately, once it ages past
    unresolved_cutoff."""
    old = ARGUSDecision(
        ticker="STALE",
        session_timestamp=datetime(2020, 1, 1),
        technical=technical_signal(Signal.BULLISH, 0.8),
        allocation=position_allocation(ticker="STALE"),
    )
    path = write_decisions_log(tmp_path / "decisions.jsonl", old)

    result = _compact(
        path,
        cutoff=datetime(2026, 1, 1),
        unresolved_cutoff=datetime(2025, 1, 1),
        resolved_ids=frozenset(),
    )

    assert result.retained == 0
    assert result.retired_unresolved == 1
    assert load_decisions_from_jsonl(str(path)) == []


def test_compact_decisions_jsonl_drops_a_resolved_decision_at_the_routine_cutoff(tmp_path):
    """A decision with a stored outcome is pruned at the ordinary cutoff, not held past it."""
    old = ARGUSDecision(
        ticker="RESOLVED",
        session_timestamp=datetime(2020, 1, 1),
        technical=technical_signal(Signal.BULLISH, 0.8),
        allocation=position_allocation(ticker="RESOLVED"),
    )
    path = write_decisions_log(tmp_path / "decisions.jsonl", old)

    result = _compact(
        path,
        cutoff=datetime(2026, 1, 1),
        unresolved_cutoff=datetime(2019, 1, 1),
        resolved_ids=frozenset({old.decision_id}),
    )

    assert result.retained == 0
    assert result.retired_unresolved == 0
    assert load_decisions_from_jsonl(str(path)) == []


def test_compact_decisions_jsonl_a_no_position_decision_is_resolved_by_definition(tmp_path):
    """A decision with no position is dropped at the routine cutoff even with no stored outcome.

    It never needed reconciliation, so it can never appear in resolved_ids —
    the routine cutoff must still apply to it, or a HOLD-only history would
    never shrink.
    """
    old = ARGUSDecision(
        ticker="HOLD",
        session_timestamp=datetime(2020, 1, 1),
        technical=technical_signal(Signal.NEUTRAL, 0.0),
    )
    path = write_decisions_log(tmp_path / "decisions.jsonl", old)

    result = _compact(
        path,
        cutoff=datetime(2026, 1, 1),
        unresolved_cutoff=datetime(2019, 1, 1),
        resolved_ids=frozenset(),
    )

    assert result.retained == 0
    assert result.retired_unresolved == 0


def test_append_decisions_jsonl_stamps_running_image_tag(tmp_path, monkeypatch):
    """Issue #95: every logged decision carries the tag of the image that produced it."""
    monkeypatch.setattr(settings, "ARGUS_IMAGE_TAG", "sha-abc123def456")
    log_path = tmp_path / "decisions.jsonl"
    decision = argus_decision("AAPL", datetime(2024, 1, 2))
    assert decision.image_tag is None

    append_decisions_jsonl([decision], str(log_path))

    logged = [ARGUSDecision.model_validate_json(line) for line in log_path.read_text().splitlines()]
    assert logged[0].image_tag == "sha-abc123def456"
    assert decision.image_tag is None  # the caller's own object is left untouched
