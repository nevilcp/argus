"""
tests/test_golden_dag.py

Runs the complete ARGUS LangGraph DAG (build_graph()) with every network and
LLM boundary served from tests/fixtures/ — zero network calls, zero LLM API
calls, zero torch/transformers import — and asserts the output is stable
across repeated runs.

Cultural memory is mocked, not fixture-backed: it isn't part of the
MarketDataProvider/LLMClient seam (it's inherently stateful, not a
replay candidate), and leaving it real would pull in sentence-transformers
(the optional `[models]` extra) and write to a real ./chroma_db directory on
every test run — the same reasoning tests/test_integration.py already
applies to its graph smoke test.

Nondeterministic fields (timestamps, session_id/UUIDs, wall-clock-derived
`data_as_of_date`) are stripped before comparison; everything else — signal
values, convictions, verdicts, allocations — must be byte-identical between
two independent invocations of the same fixture-backed graph.
"""

import json
from pathlib import Path
from unittest import mock
from uuid import uuid4

from argus.orchestration.graph import build_graph
from argus.orchestration.state import ARGUSState
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"]


def _per_ticker_llm(fixture_file: str) -> FixtureLLMClient:
    """Keys FixtureLLMClient by whichever universe ticker appears in the prompt.

    Fundamental/sentiment prompts embed the real ticker as their subject
    (build_compact_prompt, backtest_mode=False) — see argus/agents/fundamental.py.
    """
    with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
        responses = json.load(f)

    def key_fn(user_prompt: str) -> str:
        for ticker in UNIVERSE:
            if ticker in user_prompt:
                return ticker
        raise KeyError(f"no known ticker found in prompt: {user_prompt[:200]!r}")

    return FixtureLLMClient(responses, key_fn=key_fn)


def _portfolio_llm() -> FixtureLLMClient:
    """PortfolioManagerAgent.allocate() makes exactly one LLM call per session."""
    with open(FIXTURES_DIR / "llm_responses" / "portfolio.json") as f:
        response = json.load(f)
    return FixtureLLMClient({"only": json.dumps(response)}, key_fn=lambda _prompt: "only")


def _initial_state() -> ARGUSState:
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)

    return ARGUSState(
        ticker=UNIVERSE[0],
        universe=UNIVERSE,
        total_wealth=100_000.0,
        invest_pct=0.8,
        risk_tolerance="MODERATE",
        backtest_mode=False,
        session_seed=None,
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


# Excluded from the stability comparison (not from the run itself) since they vary run-to-run
_VOLATILE_KEYS = {"timestamp", "session_id", "data_as_of_date"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _run_fixture_graph() -> dict:
    """Invoke the fixture-backed DAG once, with cultural memory mocked out.

    The governor needs no patch: every LLM call here is fixture-backed, and
    FixtureLLMClient never reaches it (only GroqLLMClient does).

    Returns:
        The final ARGUSState dict produced by the graph.
    """
    graph = build_graph(
        market_data=FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(_initial_state(), config)

    return final_state


class _NoneMacroMarketData(FixtureMarketDataProvider):
    """Wraps the real fixture provider but forces a total FRED data outage.

    Every other method (fred_series, ohlcv_daily, ...) is inherited unchanged,
    so if a downstream node ran anyway it would get real fixture data rather
    than crashing for an unrelated reason.
    """

    def macro_bundle(self) -> dict:
        """Returns an all-None bundle, matching a total FRED outage."""
        return {"vix": None, "fed_funds": None, "t10y2y": None}


def test_macro_context_none_short_circuits_to_end():
    """A None macro_context routes straight to END; no fan-out node runs."""
    graph = build_graph(
        market_data=_NoneMacroMarketData(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(_initial_state(), config)

    assert final_state.get("macro_context") is None
    assert final_state.get("technical_signals") == {}
    assert final_state.get("fundamental_signals") == {}
    assert final_state.get("sentiment_signals") == {}
    assert final_state.get("risk_assessments") == {}
    assert final_state.get("portfolio_allocation") is None


def test_golden_dag_runs_offline_and_produces_a_valid_allocation():
    """The fixture-backed DAG runs end to end with zero network/LLM/torch dependencies."""
    final_state = _run_fixture_graph()

    assert final_state.get("macro_context") is not None
    assert final_state.get("technical_signals")
    assert final_state.get("fundamental_signals")
    assert final_state.get("sentiment_signals")
    assert final_state.get("risk_assessments")
    assert final_state.get("aggregated_signals")
    assert final_state.get("errors") == []

    alloc = final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(UNIVERSE)


def test_missing_indicator_is_reported_in_errors_not_silently_dropped():
    """A ticker missing a required indicator is excluded from technical_signals

    and named in state["errors"], rather than vanishing with no trace.
    """
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    del session_states["NVDA"]["adx_14"]

    state = _initial_state()
    state["session_states"] = session_states

    graph = build_graph(
        market_data=FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(state, config)

    assert "NVDA" not in final_state["technical_signals"]
    assert any("NVDA" in e for e in final_state["errors"])


def test_missing_indicator_still_reaches_aggregation_with_evidence_surfaced():
    """A ticker missing one specialist still reaches aggregated_signals

    with the gap named in agents_present, and the allocator's prompt shows
    the reduced evidence rather than treating the ticker as fully evidenced.
    """
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    del session_states["NVDA"]["adx_14"]

    state = _initial_state()
    state["session_states"] = session_states

    with open(FIXTURES_DIR / "llm_responses" / "portfolio.json") as f:
        portfolio_response = json.load(f)
    captured_prompts = []

    def _capture_key_fn(user_prompt: str) -> str:
        captured_prompts.append(user_prompt)
        return "only"

    portfolio_llm = FixtureLLMClient(
        {"only": json.dumps(portfolio_response)}, key_fn=_capture_key_fn
    )

    graph = build_graph(
        market_data=FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=portfolio_llm,
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(state, config)

    agg = final_state["aggregated_signals"]["NVDA"]
    assert agg.agents_present == ["fundamental", "sentiment"]

    assert len(captured_prompts) == 1
    assert "NVDA:" in captured_prompts[0]
    assert "Evidence=2/3" in captured_prompts[0]


def test_golden_dag_output_is_stable_across_runs():
    """Two independent invocations of the same fixture-backed graph are byte-identical."""
    first = _run_fixture_graph()
    second = _run_fixture_graph()

    first_alloc = _strip_volatile(first["portfolio_allocation"].model_dump(mode="json"))
    second_alloc = _strip_volatile(second["portfolio_allocation"].model_dump(mode="json"))
    assert first_alloc == second_alloc

    first_agg = {
        t: _strip_volatile(sig.model_dump(mode="json"))
        for t, sig in first["aggregated_signals"].items()
    }
    second_agg = {
        t: _strip_volatile(sig.model_dump(mode="json"))
        for t, sig in second["aggregated_signals"].items()
    }
    assert first_agg == second_agg

    first_risk = {
        t: _strip_volatile(sig.model_dump(mode="json"))
        for t, sig in first["risk_assessments"].items()
    }
    second_risk = {
        t: _strip_volatile(sig.model_dump(mode="json"))
        for t, sig in second["risk_assessments"].items()
    }
    assert first_risk == second_risk
