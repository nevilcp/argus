"""
argus/backtesting/replay.py

Minimal intraday backtest: replays recorded fixture "sessions" through the
real ARGUS graph (build_graph()), in strict time order.

This exists instead of a multi-year walk-forward backtest because Yahoo's
5-minute intraday data is only available for the trailing 60 days, and the
MFT session_states dict (the technical-indicator snapshot
TechnicalStatisticalAgent needs) is itself only ever captured live from the
running pipeline, not reconstructible for an arbitrary past date. There is
no historical 5-minute-resolution data to backtest against beyond ~60 days,
and no way to fabricate session_states for older dates without violating
the system's "return None rather than fabricate defaults" rule.

A "session" here is one fixture snapshot directory shaped like
tests/fixtures/ (market_data/ + llm_responses/ subdirectories) — one real
point-in-time capture of everything the graph needs, in the same format
scripts/capture_fixtures.py produces. Point-in-time correctness is
structural here, not runtime-enforced (contrast the deleted
PointInTimeEnforcer): each session's FixtureMarketDataProvider is scoped to
that session's own directory, so a later session's data cannot leak into an
earlier one — there is nothing to leak, since each session only ever sees
its own files, and replay_sessions() only invokes session N+1 after N has
returned.

Today exactly one such session exists (tests/fixtures/). Scaling this to
the ~60-day / ~780-decision-point window the design targets requires
capturing that many more sessions from a live MFT pipeline run — future
work, with the pre-registered evaluation as the first real consumer. This
module's job is to prove the replay mechanism itself is correct and
reusable, not to fabricate a backlog of sessions that were never recorded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import uuid4

from argus.orchestration.graph import build_graph
from argus.orchestration.state import ARGUSState
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

logger = logging.getLogger("argus.backtesting.replay")


@dataclass
class SessionResult:
    """One replayed session's identity and raw graph output."""

    session_dir: Path
    universe: list[str]
    final_state: dict[str, Any]


def _per_ticker_llm(session_dir: Path, fixture_file: str, universe: list[str]) -> FixtureLLMClient:
    """Keys FixtureLLMClient by whichever universe ticker appears in the prompt.

    Matches tests/test_golden_dag.py's approach — fundamental/sentiment
    prompts embed the real ticker as their subject (build_compact_prompt,
    backtest_mode=False).
    """
    with open(session_dir / "llm_responses" / fixture_file) as f:
        responses = json.load(f)

    def key_fn(user_prompt: str) -> str:
        for ticker in universe:
            if ticker in user_prompt:
                return ticker
        raise KeyError(f"no known ticker found in prompt: {user_prompt[:200]!r}")

    return FixtureLLMClient(responses, key_fn=key_fn)


def _portfolio_llm(session_dir: Path) -> FixtureLLMClient:
    """PortfolioManagerAgent.allocate() makes exactly one LLM call per session."""
    with open(session_dir / "llm_responses" / "portfolio.json") as f:
        response = json.load(f)
    return FixtureLLMClient({"only": json.dumps(response)}, key_fn=lambda _prompt: "only")


def _load_session_states(session_dir: Path) -> tuple[list[str], dict[str, dict]]:
    """Loads the fixture's session_states.json and returns its sorted ticker universe.

    Args:
        session_dir: Directory shaped like tests/fixtures/.

    Returns:
        Tuple of (sorted ticker list, raw ticker -> session-state dict).
    """
    with open(session_dir / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    return sorted(session_states), session_states


def replay_session(
    session_dir: Path,
    total_wealth: float = 100_000.0,
    invest_pct: float = 0.8,
    risk_tolerance: str = "MODERATE",
    closed_loop: bool = False,
) -> SessionResult:
    """Runs the real graph once against a single captured fixture session.

    Every agent here is fixture-backed (FixtureLLMClient never reaches the
    governor — only GroqLLMClient does), so replay is inherently governor-free
    without needing to patch anything. Returns the raw final ARGUSState dict;
    this function makes no judgment about what a "good" outcome looks like —
    that's the evaluation module's job.

    Args:
        session_dir: Directory shaped like tests/fixtures/ (market_data/ +
            llm_responses/ subdirectories).
        total_wealth: Simulated investable capital for this session.
        invest_pct: Fraction of total_wealth eligible for allocation.
        risk_tolerance: 'CONSERVATIVE', 'MODERATE', or 'AGGRESSIVE'.
        closed_loop: When False (default), cultural memory is mocked to
            return neutral wisdom/warnings/accuracy — it isn't a
            fixture-replay candidate — so reliability weighting is fixed at
            the 0.5 prior. When True, the real `get_cultural_memory()` is
            used, so `wisdom`/`warnings`/reliability weighting read
            whatever outcome history actually exists in `chroma_db`, scoped
            via `as_of` to what existed as of this session's own capture
            date — never outcomes from sessions replayed "later" in this
            call. The evaluation runs both so it can compare the two
            conditions. Either way, `as_of` is also what node_log_decisions
            stamps each decision's session_timestamp with, so a replayed
            decision's date is always the fixture's capture date, never
            replay wall-clock time.

    Returns:
        SessionResult with the session's universe and the graph's final state.
    """
    universe, session_states = _load_session_states(session_dir)
    session_date = datetime.fromisoformat(next(iter(session_states.values()))["timestamp"])

    state = ARGUSState(
        ticker=universe[0],
        universe=universe,
        total_wealth=total_wealth,
        invest_pct=invest_pct,
        risk_tolerance=risk_tolerance,
        backtest_mode=False,
        session_seed=None,
        as_of=session_date,
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
    config = {"configurable": {"thread_id": str(uuid4())}}

    # A dedicated temp checkpoint db, never the production argus_graph.db: replay
    # is a read-only simulation over fixtures, and writing into the store
    # reconciliation.py's load_decisions_from_checkpoints() reads would make
    # fabricated replay sessions indistinguishable from genuine live ones.
    with tempfile.TemporaryDirectory() as tmp_dir:
        graph = build_graph(
            market_data=FixtureMarketDataProvider(fixtures_dir=session_dir / "market_data"),
            fundamental_llm=_per_ticker_llm(session_dir, "fundamental.json", universe),
            sentiment_llm=_per_ticker_llm(session_dir, "sentiment.json", universe),
            portfolio_llm=_portfolio_llm(session_dir),
            checkpoint_db_path=str(Path(tmp_dir) / "replay_checkpoint.db"),
        )

        with contextlib.ExitStack() as stack:
            if not closed_loop:
                mock_get_cultural_memory = stack.enter_context(
                    mock.patch("argus.orchestration.graph.get_cultural_memory")
                )
                mock_get_cultural_memory.return_value = mock.Mock(
                    retrieve_wisdom=mock.Mock(return_value=[]),
                    retrieve_warnings=mock.Mock(return_value=[]),
                    get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
                    store_decision_snapshot=mock.Mock(),
                )
            final_state = graph.invoke(state, config)

    return SessionResult(session_dir=session_dir, universe=universe, final_state=final_state)


def replay_sessions(session_dirs: list[Path], **kwargs: Any) -> list[SessionResult]:
    """Replays each session directory in the given order.

    Order is the whole point: session N+1 is never invoked until session N
    has returned, which is what makes point-in-time correctness structural
    rather than something a runtime enforcer has to check.

    Args:
        session_dirs: Session directories in chronological order.
        **kwargs: Forwarded to replay_session (total_wealth, invest_pct, risk_tolerance).

    Returns:
        One SessionResult per session_dir, in the same order.
    """
    results = []
    for session_dir in session_dirs:
        logger.info("[Replay] Running session %s", session_dir)
        results.append(replay_session(session_dir, **kwargs))
    return results
