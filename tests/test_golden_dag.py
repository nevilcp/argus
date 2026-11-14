"""Runs the complete ARGUS LangGraph DAG (build_graph())."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from argus.orchestration.graph import build_graph
from argus.orchestration.state import ARGUSState
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider
from tests.helpers.builders import neutral_cultural_memory
from tests.helpers.paths import FIXTURES_DIR

UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"]


def _per_ticker_llm(fixture_file: str) -> FixtureLLMClient:
    """Key FixtureLLMClient by whichever universe ticker appears in the prompt."""
    with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
        responses = json.load(f)

    def key_fn(user_prompt: str) -> str:
        for ticker in UNIVERSE:
            if ticker in user_prompt:
                return ticker
        raise KeyError(f"no known ticker found in prompt: {user_prompt[:200]!r}")

    return FixtureLLMClient(responses, key_fn=key_fn)


def _portfolio_llm() -> FixtureLLMClient:
    """Return the captured portfolio response for its single session-level call."""
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


_VOLATILE_KEYS = {"timestamp", "session_id", "data_as_of_date"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _strip_volatile_signals(signals: dict) -> dict:
    """Dump a ticker -> pydantic signal mapping to JSON with the volatile keys removed."""
    return {ticker: _strip_volatile(sig.model_dump(mode="json")) for ticker, sig in signals.items()}


def _build_fixture_graph(
    tmp_path: Path,
    market_data: FixtureMarketDataProvider | None = None,
    portfolio_llm: FixtureLLMClient | None = None,
    cultural_memory_provider: Callable[[], Any] | None = None,
):
    """Compile the DAG with every seam fixture-backed."""
    memory = neutral_cultural_memory()
    return build_graph(
        market_data=market_data or FixtureMarketDataProvider(),
        fundamental_llm=_per_ticker_llm("fundamental.json"),
        sentiment_llm=_per_ticker_llm("sentiment.json"),
        portfolio_llm=portfolio_llm or _portfolio_llm(),
        checkpoint_db_path=str(tmp_path / "argus_graph_test.db"),
        cultural_memory_provider=cultural_memory_provider or (lambda: memory),
    )


def _invoke(graph, state: ARGUSState | None = None) -> dict:
    """Invoke the graph once on a fresh thread."""
    config = {"configurable": {"thread_id": str(uuid4())}}

    return graph.invoke(_initial_state() if state is None else state, config)


class _NoneMacroMarketData(FixtureMarketDataProvider):
    """Wraps the real fixture provider but forces a total FRED data outage."""

    def macro_bundle(self) -> dict:
        """Return an all-None bundle, matching a total FRED outage."""
        return {"vix": None, "fed_funds": None, "t10y2y": None}


def test_macro_context_none_still_produces_a_degraded_allocation(tmp_path):
    final_state = _invoke(_build_fixture_graph(tmp_path, market_data=_NoneMacroMarketData()))

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
    """Wraps the fixture provider to count `ohlcv_daily` calls and fabricate a SPY series."""

    def __init__(self) -> None:
        super().__init__()
        self.ohlcv_daily_calls: list[str] = []
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        prices = 400 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(dates)))
        self._spy_df = pd.DataFrame({"close": prices}, index=dates)

    def ohlcv_daily(self, ticker: str, period: str = "2y"):
        """Record the call and return fabricated SPY data, else defers to the fixture."""
        self.ohlcv_daily_calls.append(ticker)
        if ticker == "SPY":
            return self._spy_df
        return super().ohlcv_daily(ticker, period=period)

    def multiple_daily(self, tickers: list[str], period: str = "1y"):
        """Route benchmark reads through the counted daily fetch."""
        result = super().multiple_daily([t for t in tickers if t != "SPY"], period=period)
        if "SPY" in tickers:
            result["SPY"] = self.ohlcv_daily("SPY", period=period)
        return result


def test_spy_fetched_once_per_graph_run_not_once_per_risk_call(tmp_path):
    provider = _SpyCountingMarketData()
    _invoke(_build_fixture_graph(tmp_path, market_data=provider))

    assert provider.ohlcv_daily_calls.count("SPY") == 1


class _CountingVixMarketData(_NoneMacroMarketData):
    """_NoneMacroMarketData with a vix() call counter, isolated from other tests."""

    def __init__(self) -> None:
        super().__init__()
        self.vix_calls = 0

    def vix(self) -> float:
        """Record the call, then defers to the inherited fixture-backed reading."""
        self.vix_calls += 1
        return super().vix()


def test_macro_context_none_still_reaches_a_real_vix_reading(tmp_path):
    provider = _CountingVixMarketData()
    final_state = _invoke(_build_fixture_graph(tmp_path, market_data=provider))

    assert final_state.get("macro_context") is None
    assert provider.vix_calls == 1


def test_golden_dag_runs_offline_and_produces_a_valid_allocation(tmp_path):
    final_state = _invoke(_build_fixture_graph(tmp_path))

    assert final_state.get("macro_context") is not None
    assert final_state.get("technical_signals")
    assert final_state.get("fundamental_signals")
    assert final_state.get("sentiment_signals")
    assert final_state.get("risk_assessments")
    assert final_state.get("aggregated_signals")
    assert all(agg.model_healthy is True for agg in final_state["aggregated_signals"].values())

    # Enforce fixture vetoes and recompute residual cash, recording every advisory proposal correction.
    errors = final_state.get("errors")
    assert errors is not None
    assert len(errors) == 4
    for ticker in ("MSFT", "JPM", "GOOGL"):
        assert any(f"zeroed {ticker} allocation (risk verdict VETO)" in e for e in errors)
    assert any("cash_reserve_pct forced to residual" in e for e in errors)

    alloc = final_state.get("portfolio_allocation")
    assert alloc is not None
    assert len(alloc.portfolio) == len(UNIVERSE)
    for pos in alloc.portfolio:
        if pos.ticker in ("MSFT", "JPM", "GOOGL"):
            assert pos.allocation_pct == 0.0


def _state_missing_nvda_adx() -> ARGUSState:
    """Build a session whose NVDA indicators lack required ADX evidence."""
    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)
    del session_states["NVDA"]["adx_14"]

    state = _initial_state()
    state["session_states"] = session_states
    return state


def test_missing_indicator_is_reported_in_errors_not_silently_dropped(tmp_path):
    final_state = _invoke(_build_fixture_graph(tmp_path), _state_missing_nvda_adx())

    assert "NVDA" not in final_state["technical_signals"]
    assert any("NVDA" in e for e in final_state["errors"])


def test_missing_indicator_still_reaches_aggregation_with_evidence_surfaced(tmp_path):
    with open(FIXTURES_DIR / "llm_responses" / "portfolio.json") as f:
        portfolio_response = json.load(f)
    captured_prompts = []

    def _capture_key_fn(user_prompt: str) -> str:
        captured_prompts.append(user_prompt)
        return "only"

    portfolio_llm = FixtureLLMClient({"only": json.dumps(portfolio_response)}, key_fn=_capture_key_fn)

    graph = _build_fixture_graph(tmp_path, portfolio_llm=portfolio_llm)
    final_state = _invoke(graph, _state_missing_nvda_adx())

    agg = final_state["aggregated_signals"]["NVDA"]
    assert agg.agents_present == ["fundamental", "sentiment"]

    assert len(captured_prompts) == 1
    assert "NVDA:" in captured_prompts[0]
    assert "Evidence=2/3" in captured_prompts[0]


def test_cultural_memory_import_error_degrades_instead_of_crashing_the_run(tmp_path):

    def unavailable() -> Any:
        raise ImportError("sentence-transformers not installed")

    graph = _build_fixture_graph(tmp_path, cultural_memory_provider=unavailable)
    config = {"configurable": {"thread_id": str(uuid4())}}

    final_state = graph.invoke(_initial_state(), config)

    assert final_state.get("cultural_memory") == {"wisdom": [], "warnings": []}
    assert final_state.get("portfolio_allocation") is not None
    assert len(final_state["portfolio_allocation"].portfolio) == len(UNIVERSE)

    # Mark reliability unhealthy when memory is unavailable; a neutral prior alone cannot disclose the outage.
    aggs = final_state.get("aggregated_signals") or {}
    assert aggs
    assert all(agg.model_healthy is False for agg in aggs.values())


def test_golden_dag_output_is_stable_across_runs(tmp_path):
    first = _invoke(_build_fixture_graph(tmp_path))
    second = _invoke(_build_fixture_graph(tmp_path))

    first_alloc = _strip_volatile(first["portfolio_allocation"].model_dump(mode="json"))
    second_alloc = _strip_volatile(second["portfolio_allocation"].model_dump(mode="json"))
    assert first_alloc == second_alloc

    for key in ("aggregated_signals", "risk_assessments"):
        assert _strip_volatile_signals(first[key]) == _strip_volatile_signals(second[key])


def _round_floats(obj, ndigits: int = 8):
    """Round every float so the snapshot tolerates last-digit libm/numpy differences."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _snapshot_text(final_state: dict) -> str:
    """Serialize the behavior-relevant slice of a final state to stable JSON text."""
    macro = final_state.get("macro_context")
    payload = {
        "portfolio_allocation": final_state["portfolio_allocation"].model_dump(mode="json"),
        "aggregated_signals": {t: s.model_dump(mode="json") for t, s in final_state["aggregated_signals"].items()},
        "risk_assessments": {t: s.model_dump(mode="json") for t, s in final_state["risk_assessments"].items()},
        "macro_context": macro.model_dump(mode="json") if macro is not None else None,
        "errors": final_state["errors"],
    }
    return json.dumps(_round_floats(_strip_volatile(payload)), indent=2, sort_keys=True) + "\n"


def test_golden_dag_matches_snapshot(tmp_path, assert_golden):
    final_state = _invoke(_build_fixture_graph(tmp_path))

    assert_golden("dag_snapshot.json", _snapshot_text(final_state))


class _MissingReadingsMarketData(FixtureMarketDataProvider):
    def macro_bundle(self):
        bundle = super().macro_bundle()
        return {**bundle, "vix": None, "t10y2y": None}

    def ticker_calendar(self, ticker):
        raise OSError("calendar unavailable")

    def vix(self):
        raise OSError("VIX unavailable")


def test_golden_dag_preserves_unavailable_measurements(tmp_path):
    state = _invoke(_build_fixture_graph(tmp_path, market_data=_MissingReadingsMarketData()))
    macro = state["macro_context"]
    assert macro.vix_level is None
    assert macro.vix_regime is None
    assert macro.yield_curve_shape is None
    for risk in state["risk_assessments"].values():
        assert risk.var_99 is None
        assert risk.cvar is None
        assert risk.marginal_var is None
        assert risk.portfolio_beta is None
        assert risk.stop_loss is None
    for sentiment in state["sentiment_signals"].values():
        assert sentiment.catalyst_data_available is False
