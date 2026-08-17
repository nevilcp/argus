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

import numpy as np
import pandas as pd

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


# Excluded from the stability comparison (not from the run itself) since they vary run-to-run
_VOLATILE_KEYS = {"timestamp", "session_id", "data_as_of_date"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _run_fixture_graph(tmp_path: Path) -> dict:
    """Invoke the fixture-backed DAG once, with cultural memory mocked out.

    The governor needs no patch: every LLM call here is fixture-backed, and
    FixtureLLMClient never reaches it (only GroqLLMClient does).

    Args:
        tmp_path: Per-test checkpoint directory (X-5) — keeps this run's
            checkpoints out of the real argus_graph.db, not just the fixture
            autouse `_isolated_data_dir` build_graph()'s default would also
            resolve against.

    Returns:
        The final ARGUSState dict produced by the graph.
    """
    graph = build_graph(
        market_data=FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
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


def test_macro_context_none_still_produces_a_degraded_allocation(tmp_path):
    """A None macro_context must not gate the three specialists or the allocator.

    Regression test for X1: no specialist agent consumes MacroContext, so a FRED
    outage must not zero out a session that needs no FRED data. The three
    specialists, aggregation, risk evaluation, and portfolio allocation must all
    still run, degraded rather than empty, with the gap named in errors.
    """
    graph = build_graph(
        market_data=_NoneMacroMarketData(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(_initial_state(), config)

    assert final_state.get("macro_context") is None
    assert final_state.get("technical_signals")
    assert final_state.get("fundamental_signals")
    assert final_state.get("sentiment_signals")
    assert final_state.get("risk_assessments")
    assert final_state.get("aggregated_signals")

    alloc = final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(UNIVERSE)

    assert any("macro_context unavailable" in e for e in final_state.get("errors", []))


class _SpyCountingMarketData(FixtureMarketDataProvider):
    """Wraps the fixture provider to count ``ohlcv_daily`` calls and fabricate a SPY series.

    The shared price_history fixture predates RE-5 and has no "SPY" entry, and adding
    one there would perturb every other golden_dag test's carefully-tuned VETO/REDUCE
    expectations (real beta feeding risk.py's beta gate). Fabricating it only here keeps
    this test's SPY data isolated from the rest of the fixture-backed suite.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ohlcv_daily_calls: list[str] = []
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        prices = 400 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(dates)))
        self._spy_df = pd.DataFrame({"close": prices}, index=dates)

    def ohlcv_daily(self, ticker: str, period: str = "2y"):
        """Records the call and returns fabricated SPY data, else defers to the fixture."""
        self.ohlcv_daily_calls.append(ticker)
        if ticker == "SPY":
            return self._spy_df
        return super().ohlcv_daily(ticker, period=period)

    def multiple_daily(self, tickers: list[str], period: str = "1y"):
        """Same as the base multiple_daily, but routes SPY through ohlcv_daily for counting."""
        result = super().multiple_daily([t for t in tickers if t != "SPY"], period=period)
        if "SPY" in tickers:
            result["SPY"] = self.ohlcv_daily("SPY", period=period)
        return result


def test_spy_fetched_once_per_graph_run_not_once_per_risk_call(tmp_path):
    """RE-5 regression: SPY rides along with node_fetch_price_history's universe fetch.

    Before the fix, node_fetch_price_history never carried SPY in price_history, so
    risk.py's ols_portfolio_beta fell back to its own live fetch on every one of the
    N+1 risk_engine.evaluate() calls a session makes (one portfolio-level plus one per
    ticker) — 7 fetches for this 6-ticker universe instead of 1.
    """
    provider = _SpyCountingMarketData()
    graph = build_graph(
        market_data=provider,
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        graph.invoke(_initial_state(), config)

    assert provider.ohlcv_daily_calls.count("SPY") == 1


class _CountingVixMarketData(_NoneMacroMarketData):
    """_NoneMacroMarketData with a vix() call counter, isolated from other tests."""

    def __init__(self) -> None:
        super().__init__()
        self.vix_calls = 0

    def vix(self) -> float:
        """Records the call, then defers to the inherited fixture-backed reading."""
        self.vix_calls += 1
        return super().vix()


def test_macro_context_none_still_reaches_a_real_vix_reading(tmp_path):
    """RE-6 regression: a FRED outage must not silently default the blackout gate to 20.0.

    _NoneMacroMarketData forces macro_bundle() (hence macro_context) to None but leaves
    vix() — read straight from yfinance in production, independent of FRED — intact, so
    the risk engine's blackout gate should reach a real VIX reading via one explicit call
    rather than defaulting.
    """
    provider = _CountingVixMarketData()
    graph = build_graph(
        market_data=provider,
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(_initial_state(), config)

    assert final_state.get("macro_context") is None
    assert provider.vix_calls == 1


def test_golden_dag_runs_offline_and_produces_a_valid_allocation(tmp_path):
    """The fixture-backed DAG runs end to end with zero network/LLM/torch dependencies."""
    final_state = _run_fixture_graph(tmp_path)

    assert final_state.get("macro_context") is not None
    assert final_state.get("technical_signals")
    assert final_state.get("fundamental_signals")
    assert final_state.get("sentiment_signals")
    assert final_state.get("risk_assessments")
    assert final_state.get("aggregated_signals")
    assert all(agg.model_healthy is True for agg in final_state["aggregated_signals"].values())

    # The fixture LLM response proposes allocations for three VETO'd tickers;
    # code-level enforcement (PR 4, PA-1) zeroes them rather than trusting the
    # prompt, and records each correction here instead of leaving errors empty.
    errors = final_state.get("errors")
    assert errors is not None and len(errors) == 3
    for ticker in ("MSFT", "JPM", "GOOGL"):
        assert any(
            f"zeroed {ticker} allocation (risk verdict VETO)" in e for e in errors
        )

    alloc = final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(UNIVERSE)
    for pos in alloc.portfolio:
        if pos.ticker in ("MSFT", "JPM", "GOOGL"):
            assert pos.allocation_pct == 0.0


def test_missing_indicator_is_reported_in_errors_not_silently_dropped(tmp_path):
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
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(state, config)

    assert "NVDA" not in final_state["technical_signals"]
    assert any("NVDA" in e for e in final_state["errors"])


def test_missing_indicator_still_reaches_aggregation_with_evidence_surfaced(tmp_path):
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
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        final_state = graph.invoke(state, config)

    agg = final_state["aggregated_signals"]["NVDA"]
    assert agg.agents_present == ["fundamental", "sentiment"]

    assert len(captured_prompts) == 1
    assert "NVDA:" in captured_prompts[0]
    assert "Evidence=2/3" in captured_prompts[0]


def test_cultural_memory_import_error_degrades_instead_of_crashing_the_run(tmp_path):
    """A missing sentence-transformers/torch [models] extra must not abort the session.

    Regression test for C1: get_cultural_memory() raising ImportError must degrade
    retrieve_cultural_memory to an empty result and signal_aggregation's reliability
    lookup to the 0.5 prior, rather than crashing the whole graph run.
    """
    graph = build_graph(
        market_data=FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=_portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.side_effect = ImportError("sentence-transformers not installed")
        final_state = graph.invoke(_initial_state(), config)

    assert final_state.get("cultural_memory") == {"wisdom": [], "warnings": []}
    assert final_state.get("portfolio_allocation") is not None
    assert len(final_state["portfolio_allocation"].portfolio) == len(UNIVERSE)

    # SA-5: reliability fell back to the 0.5 prior because cultural memory was
    # unavailable, not because no outcome history exists yet — model_healthy
    # must say so rather than carrying full multiplier authority silently.
    aggs = final_state.get("aggregated_signals") or {}
    assert aggs
    assert all(agg.model_healthy is False for agg in aggs.values())


def test_golden_dag_output_is_stable_across_runs(tmp_path):
    """Two independent invocations of the same fixture-backed graph are byte-identical."""
    first = _run_fixture_graph(tmp_path)
    second = _run_fixture_graph(tmp_path)

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
