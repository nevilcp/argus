"""Tests for argus/backtesting/replay.py, the walk-forward engine's replacement."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from argus.backtesting.replay import _normalize_as_of, replay_session, replay_sessions
from tests.helpers.builders import neutral_cultural_memory
from tests.helpers.paths import FIXTURES_DIR


def _fixture_capture_date() -> datetime:
    """Return the capture timestamp of the first session in the fixture's session_states.json."""
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    return datetime.fromisoformat(next(iter(session_states.values()))["timestamp"])


def test_replay_session_produces_a_valid_allocation():
    result = replay_session(FIXTURES_DIR)

    assert sorted(result.universe) == ["AAPL", "GOOGL", "JPM", "MSFT", "NVDA", "XOM"]

    alloc = result.final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(result.universe)


def test_replay_sessions_preserves_order():
    results = replay_sessions([FIXTURES_DIR, FIXTURES_DIR])

    assert [r.session_dir for r in results] == [FIXTURES_DIR, FIXTURES_DIR]
    assert all(r.final_state.get("portfolio_allocation") is not None for r in results)


def test_replay_session_never_touches_the_production_checkpoint_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    replay_session(FIXTURES_DIR)

    assert not (tmp_path / "argus_graph.db").exists()


def test_replay_session_stamps_decisions_with_the_fixture_capture_date():
    expected_date = _fixture_capture_date()

    result = replay_session(FIXTURES_DIR, closed_loop=False)

    decisions = result.final_state.get("decisions") or []
    assert decisions
    assert all(d.session_timestamp == expected_date for d in decisions)


def test_replay_session_closed_loop_scopes_cultural_memory_to_the_session_date():
    mock_memory = neutral_cultural_memory()
    replay_session(FIXTURES_DIR, closed_loop=True, cultural_memory_provider=lambda: mock_memory)

    expected_as_of = _fixture_capture_date()

    assert mock_memory.retrieve_wisdom.call_args.kwargs["as_of"] == expected_as_of
    assert mock_memory.retrieve_warnings.call_args.kwargs["as_of"] == expected_as_of
    assert mock_memory.get_agent_accuracy.call_args.kwargs["as_of"] == expected_as_of


def test_normalize_as_of_strips_tzinfo_from_an_aware_timestamp():
    aware_utc = datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))

    result = _normalize_as_of(aware_utc)

    assert result.tzinfo is None
    assert result == datetime(2024, 1, 2, 9, 30)


def test_normalize_as_of_passes_a_naive_timestamp_through_unchanged():
    naive = datetime(2024, 1, 2, 9, 30)

    assert _normalize_as_of(naive) is naive
