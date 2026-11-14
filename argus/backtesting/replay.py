"""Ordered fixture replay through the production decision graph."""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

from argus.infra.clock import MARKET_TZ
from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.checkpoint import new_run_config
from argus.orchestration.graph import build_graph
from argus.orchestration.state import initial_state
from argus.params import EVALUATION
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

logger = logging.getLogger(__name__)


def _normalize_as_of(ts: datetime) -> datetime:
    """Convert aware timestamps to naive Eastern wall time and leave naive timestamps unchanged."""
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(MARKET_TZ).replace(tzinfo=None)


@dataclass
class SessionResult:
    """One replayed session's identity and raw graph output."""

    session_dir: Path
    universe: list[str]
    final_state: dict[str, Any]


def _per_ticker_llm(session_dir: Path, fixture_file: str, universe: list[str]) -> FixtureLLMClient:
    """Key FixtureLLMClient by whichever universe ticker appears in the prompt."""

    def key_fn(user_prompt: str) -> str:
        for ticker in universe:
            if ticker in user_prompt:
                return ticker
        raise KeyError(f"no known ticker found in prompt: {user_prompt[:200]!r}")

    return FixtureLLMClient.from_fixture_file(session_dir / "llm_responses" / fixture_file, key_fn=key_fn)


def _portfolio_llm(session_dir: Path) -> FixtureLLMClient:
    """Build the FixtureLLMClient for PortfolioManagerAgent's one LLM call."""
    with open(session_dir / "llm_responses" / "portfolio.json") as f:
        response = json.load(f)
    return FixtureLLMClient({"only": json.dumps(response)}, key_fn=lambda _prompt: "only")


def _neutral_cultural_memory() -> mock.Mock:
    """Build a neutral mock cultural-memory client for open-loop replay."""
    return mock.Mock(
        retrieve_wisdom=mock.Mock(return_value=[]),
        retrieve_warnings=mock.Mock(return_value=[]),
        get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
        store_decision_snapshot=mock.Mock(),
    )


def _load_session_states(session_dir: Path) -> tuple[list[str], dict[str, dict]]:
    """Load the fixture's session_states.json and return its sorted ticker universe."""
    with open(session_dir / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    return sorted(session_states), session_states


def replay_session(
    session_dir: Path,
    total_wealth: float = EVALUATION.replay_total_wealth,
    invest_pct: float = EVALUATION.replay_invest_pct,
    risk_tolerance: str = "MODERATE",
    closed_loop: bool = False,
    cultural_memory_provider: Callable[[], CulturalMemoryManager] | None = None,
) -> SessionResult:
    """Run the real graph once against a single captured fixture session."""
    universe, session_states = _load_session_states(session_dir)
    captured_at = next(iter(session_states.values()))["timestamp"]
    session_date = _normalize_as_of(datetime.fromisoformat(captured_at))

    state = initial_state(
        universe,
        session_states=session_states,
        total_wealth=total_wealth,
        invest_pct=invest_pct,
        risk_tolerance=risk_tolerance,
        as_of=session_date,
    )
    config = new_run_config("replay")

    # Isolate replay checkpoints so simulated decisions cannot enter the production reconciliation history.
    if cultural_memory_provider is None and not closed_loop:
        neutral_memory = _neutral_cultural_memory()

        def cultural_memory_provider() -> CulturalMemoryManager:
            return neutral_memory  # type: ignore[return-value]  # a Mock standing in

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph = build_graph(
            market_data=FixtureMarketDataProvider(fixtures_dir=session_dir / "market_data"),
            fundamental_llm=_per_ticker_llm(session_dir, "fundamental.json", universe),
            sentiment_llm=_per_ticker_llm(session_dir, "sentiment.json", universe),
            portfolio_llm=_portfolio_llm(session_dir),
            checkpoint_db_path=str(Path(tmp_dir) / "replay_checkpoint.db"),
            cultural_memory_provider=cultural_memory_provider,
        )
        final_state = graph.invoke(state, config)

    return SessionResult(session_dir=session_dir, universe=universe, final_state=final_state)


def replay_sessions(session_dirs: list[Path], **kwargs: Any) -> list[SessionResult]:
    """Replay each session directory in the given order."""
    results: list[SessionResult] = []
    for session_dir in session_dirs:
        logger.info("Running session %s", session_dir)
        results.append(replay_session(session_dir, **kwargs))
    return results
