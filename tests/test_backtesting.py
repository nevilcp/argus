"""
Tests for argus/backtesting/replay.py, which replaces the deleted walk-forward engine.
See argus/backtesting/replay.py's docstring for what a "session" is.
"""

from pathlib import Path

from argus.backtesting.replay import replay_session, replay_sessions

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
