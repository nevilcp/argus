"""
Tests for argus/backtesting/replay.py, which replaces the deleted walk-forward engine.
See argus/backtesting/replay.py's docstring for what a "session" is.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from argus.backtesting.replay import _normalize_as_of, replay_session, replay_sessions

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_replay_session_produces_a_valid_allocation():
    """A replayed session allocates every ticker in its universe."""
    result = replay_session(FIXTURES_DIR)

    assert sorted(result.universe) == ["AAPL", "GOOGL", "JPM", "MSFT", "NVDA", "XOM"]

    alloc = result.final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(result.universe)


def test_replay_sessions_preserves_order():
    """replay_sessions() invokes session N+1 only after session N has returned."""
    results = replay_sessions([FIXTURES_DIR, FIXTURES_DIR])

    assert [r.session_dir for r in results] == [FIXTURES_DIR, FIXTURES_DIR]
    assert all(r.final_state.get("portfolio_allocation") is not None for r in results)


def test_replay_session_never_touches_the_production_checkpoint_db(tmp_path, monkeypatch):
    """Regression test for X4: replay must checkpoint into a temp db, never argus_graph.db.

    build_graph()'s checkpoint_db_path defaults to the production store; replay
    previously omitted the argument, so replayed sessions (with their rebased,
    fake-looking-genuine timestamps) landed in the same store
    reconciliation.load_decisions_from_checkpoints() reads.
    """
    monkeypatch.chdir(tmp_path)

    replay_session(FIXTURES_DIR)

    assert not (tmp_path / "argus_graph.db").exists()


def test_replay_session_stamps_decisions_with_the_fixture_capture_date():
    """Every replayed decision's session_timestamp is the fixture's own capture date.

    Regression test for LD-2: node_log_decisions previously stamped
    datetime.now() regardless of as_of, so a closed_loop=True replay wrote
    wall-clock dates straight into the real ./chroma_db via
    store_decision_snapshot before any post-hoc rebase could run. as_of is
    now always passed to the graph, and node_log_decisions uses it directly
    (or falls back to wall-clock only when it is unset, as in a live session).
    """
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    expected_date = datetime.fromisoformat(next(iter(session_states.values()))["timestamp"])

    result = replay_session(FIXTURES_DIR, closed_loop=False)

    decisions = result.final_state.get("decisions") or []
    assert decisions
    assert all(d.session_timestamp == expected_date for d in decisions)


def test_replay_session_closed_loop_scopes_cultural_memory_to_the_session_date():
    """closed_loop=True must pass the fixture's own capture date as `as_of`.

    Regression test for the PR 7 plan's closed-loop item: without it, replayed
    sessions could read outcomes that, relative to the fixture's point in time,
    haven't happened yet.
    """
    mock_memory = mock.Mock(
        retrieve_wisdom=mock.Mock(return_value=[]),
        retrieve_warnings=mock.Mock(return_value=[]),
        get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
        store_decision_snapshot=mock.Mock(),
    )
    with mock.patch("argus.orchestration.graph.get_cultural_memory", return_value=mock_memory):
        replay_session(FIXTURES_DIR, closed_loop=True)

    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    expected_as_of = datetime.fromisoformat(next(iter(session_states.values()))["timestamp"])

    assert mock_memory.retrieve_wisdom.call_args.kwargs["as_of"] == expected_as_of
    assert mock_memory.retrieve_warnings.call_args.kwargs["as_of"] == expected_as_of
    assert mock_memory.get_agent_accuracy.call_args.kwargs["as_of"] == expected_as_of


def test_normalize_as_of_strips_tzinfo_from_an_aware_timestamp():
    """An aware fixture timestamp (as a canonical-timestamp buffer would now produce) is
    converted to ET wall-clock and returned naive, so it can't mix with a naive
    datetime.now() fallback in node_log_decisions and break compute_run_returns' sort.
    """
    aware_utc = datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))

    result = _normalize_as_of(aware_utc)

    assert result.tzinfo is None
    assert result == datetime(2024, 1, 2, 9, 30)


def test_normalize_as_of_passes_a_naive_timestamp_through_unchanged():
    """A naive timestamp (today's fixture format) is returned exactly as given."""
    naive = datetime(2024, 1, 2, 9, 30)

    assert _normalize_as_of(naive) is naive
