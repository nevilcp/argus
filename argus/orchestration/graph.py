"""
argus/orchestration/graph.py

LangGraph execution orchestrator for the ARGUS multi-agent decision network.

Coordinates asynchronous fetch pipelines, parallel specialized analyst agents,
centralized risk assessment, dynamic signal aggregation, and long-term cultural
memory ingestion into a stateful execution graph.

Responsibilities:
  - Define node functions that map ARGUSState → partial state updates
  - Build and compile the LangGraph StateGraph DAG via build_graph()

Not responsible for:
  - Agent implementation logic (see individual agent modules)
  - Schema validation (see schemas/signals.py)
  - API serving (see api/main.py)

Dependencies:
  - langgraph
  - argus.agents.* (all six specialist agents)
  - SQLite checkpointer (argus_graph.db) for resumable runs

Agent wiring is a build_graph() parameter, not a module-level singleton:
every agent that touches the network or an LLM accepts a
MarketDataProvider/LLMClient, defaulting to the real implementation, so a
test can build a fixture-backed graph without patching this module's
internals. Callers — api/main.py, orchestration/collector.py — call
build_graph() themselves rather than importing a shared instance.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from chromadb.errors import ChromaError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import RiskStatisticalEngine
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.config import settings
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.state import ARGUSState
from argus.params import RISK
from argus.schemas.signals import AggregatedSignal, ARGUSDecision, RiskAssessment, RiskVerdict, Signal
from argus.seams import LiveMarketDataProvider, LLMClient, MarketDataProvider

logger = logging.getLogger("argus.graph")

# Every schemas.signals model nested in ARGUSState must be listed here or the
# checkpointer's msgpack deserializer degrades it to a plain dict; shared with
# reconciliation.py's checkpoint loader so the two allowlists can't drift apart
_CHECKPOINT_MODEL_CLASSES = (
    "MacroContext",
    "TechnicalSignal",
    "FundamentalSignal",
    "SentimentSignal",
    "RiskAssessment",
    "AggregatedSignal",
    "PositionAllocation",
    "PortfolioAllocation",
    "ARGUSDecision",
)


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Returns the JsonPlusSerializer used to read/write ARGUSState checkpoints."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("argus.schemas.signals", cls) for cls in _CHECKPOINT_MODEL_CLASSES
        ]
    )


def _open_checkpointer(checkpoint_db_path: str) -> Optional[SqliteSaver]:
    """Opens a fresh SqliteSaver connection, or None if the path can't be opened.

    Args:
        checkpoint_db_path: SQLite file to checkpoint to.

    Returns:
        A new SqliteSaver, or None on any I/O failure (a checkpointer-less
        compile still runs the graph, just without resumability).
    """
    try:
        if checkpoint_db_path != ":memory:":
            Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        return SqliteSaver(conn, serde=build_checkpoint_serde())
    except Exception:
        return None


class _CheckpointedGraph:
    """Compiles the DAG fresh, with its own SQLite connection, on every invoke().

    StateGraph.compile() does no I/O beyond opening the checkpoint connection;
    the expensive part — constructing the six agents (LLM clients, the macro
    HMM classifier load from disk) — happens once in build_graph() and is
    captured in the node closures this wraps. Concurrent callers (parallel
    /analyze requests, the unattended collector loop) therefore each get their
    own sqlite3.Connection instead of contending on one shared checkpointer.
    """

    def __init__(self, builder: StateGraph, checkpoint_db_path: str):
        """Stores the built (but not compiled) graph and its checkpoint path."""
        self._builder = builder
        self._checkpoint_db_path = checkpoint_db_path

    def invoke(self, state: ARGUSState, config: Optional[RunnableConfig] = None) -> dict:
        """Compiles with a fresh checkpointer and invokes the graph once.

        Args:
            state: Initial ARGUSState for this session.
            config: LangGraph run config, e.g. ``{"configurable": {"thread_id": ...}}``.

        Returns:
            The final ARGUSState dict produced by the run.
        """
        checkpointer = _open_checkpointer(self._checkpoint_db_path)
        compiled = self._builder.compile(checkpointer=checkpointer)
        try:
            return compiled.invoke(state, config)
        finally:
            if checkpointer is not None:
                checkpointer.conn.close()


def _summarize_technical_posture(aggregated_signals: dict[str, AggregatedSignal]) -> str:
    """Builds retrieve_wisdom's free-text technical-posture argument from this session's own signals.

    Replaces a hardcoded "mixed technicals" placeholder that never reflected the
    session it was queried for.

    Args:
        aggregated_signals: This session's ticker → AggregatedSignal mapping.

    Returns:
        A short directional-breakdown summary, or a fixed string when empty.
    """
    if not aggregated_signals:
        return "no aggregated signals available"

    bullish = sum(1 for s in aggregated_signals.values() if s.signal == Signal.BULLISH)
    bearish = sum(1 for s in aggregated_signals.values() if s.signal == Signal.BEARISH)
    neutral = len(aggregated_signals) - bullish - bearish
    avg_conviction = sum(s.conviction for s in aggregated_signals.values()) / len(aggregated_signals)
    return (
        f"{bullish} bullish, {bearish} bearish, {neutral} neutral across "
        f"{len(aggregated_signals)} tickers (avg conviction {avg_conviction:.2f})"
    )


def _apply_portfolio_cap(single: RiskAssessment, cap: float) -> dict:
    """Maps an SLSQP-derived per-ticker cap onto a single-ticker risk verdict.

    Args:
        single: The ticker's own (non-portfolio) risk assessment.
        cap: SLSQP-optimal weight ceiling for this ticker.

    Returns:
        Partial RiskAssessment field updates: ``verdict``, ``approved_weight``,
        ``veto_reasons``.
    """
    if cap < RISK.slsqp_zero_cap_epsilon:
        # A cap this small is no room to allocate, not a REDUCE — REDUCE
        # still counts as approved (portfolio.py:236) and the prompt would
        # demand invest_pct deployment against a 0% cap
        return {
            "verdict": RiskVerdict.VETO,
            "approved_weight": 0.0,
            "veto_reasons": single.veto_reasons
            + [
                f"[Portfolio] SLSQP cap {cap:.2%} below the "
                f"{RISK.slsqp_zero_cap_epsilon:.2%} allocation floor"
            ],
        }
    # Downgrade verdict monotonically: never upgrade a VETO to REDUCE or APPROVE
    verdict = (
        single.verdict
        if single.verdict == RiskVerdict.VETO
        else (RiskVerdict.REDUCE if cap < single.approved_weight else single.verdict)
    )
    return {
        "verdict": verdict,
        "approved_weight": min(cap, single.approved_weight),
        "veto_reasons": single.veto_reasons + [f"[Portfolio] SLSQP cap: {cap:.1%}"],
    }


def build_graph(
    market_data: Optional[MarketDataProvider] = None,
    fundamental_llm: Optional[LLMClient] = None,
    sentiment_llm: Optional[LLMClient] = None,
    portfolio_llm: Optional[LLMClient] = None,
    checkpoint_db_path: Optional[str] = None,
):
    """Constructs and compiles the ARGUS decision graph.

    Args:
        market_data: Shared data source injected into every agent that fetches
            market/macro data (macro, fundamental, sentiment, risk). Defaults
            to a live LiveMarketDataProvider for each agent independently.
        fundamental_llm: LLMClient for FundamentalAgent. Defaults to a real
            Groq client.
        sentiment_llm: LLMClient for SentimentAgent. Defaults to a real Groq
            client.
        portfolio_llm: LLMClient for PortfolioManagerAgent. Defaults to a real
            Groq client.
        checkpoint_db_path: SQLite file each invocation checkpoints ARGUSState
            to. Exposed (rather than hardcoded) so tests — and
            argus/orchestration/reconciliation.py's own tests — can point it
            at a temp file instead of the real store production callers use.
            None (default) resolves to ``<ARGUS_DATA_DIR>/argus_graph.db`` at
            call time (X-5), so tests/conftest.py's per-test ARGUS_DATA_DIR
            override also isolates a call site that omits this argument.

    Returns:
        A _CheckpointedGraph, ready for .invoke(). Recompiles with a fresh
        checkpointer connection on every call rather than baking one
        connection into a single compiled graph — see _CheckpointedGraph.
    """
    if checkpoint_db_path is None:
        checkpoint_db_path = str(Path(settings.ARGUS_DATA_DIR) / "argus_graph.db")

    market_data = market_data or LiveMarketDataProvider()

    macro_agent = MacroStatisticalAgent(market_data=market_data)
    tech_agent = TechnicalStatisticalAgent()
    fund_agent = FundamentalAgent(llm_client=fundamental_llm, market_data=market_data)
    sent_agent = SentimentAgent(llm_client=sentiment_llm, market_data=market_data)
    risk_engine = RiskStatisticalEngine(market_data=market_data)
    aggregator = HybridSignalAggregator()
    portfolio_agent = PortfolioManagerAgent(llm_client=portfolio_llm)

    def node_fetch_price_history(state: ARGUSState) -> dict:
        """Fetches raw daily pricing data for risk covariance calculation.

        Session states (technical indicators) are sourced exclusively from the MFT
        live cache injected by the API layer. Daily compression is intentionally
        omitted here — computing indicators from daily closes produces a different
        signal resolution than the intraday 5-minute data the TechnicalAgent expects.
        Tickers absent from the MFT cache will carry no session_state and will be
        excluded from downstream technical and portfolio allocation nodes.

        Args:
            state: ARGUSState containing ``universe`` list of tickers and
                ``session_states`` pre-populated from the MFT live cache.

        Returns:
            Partial state update with ``price_history``, which also carries the SPY
            benchmark series alongside the universe. ``session_states`` is passed
            through unchanged from the incoming state.
        """
        # SPY rides along with the universe fetch so risk.py's ols_portfolio_beta finds it
        # already in price_history, instead of issuing its own uncached fetch on every one
        # of the N+1 risk_engine.evaluate() calls node_risk_evaluation makes this session
        fetch_tickers = list(state["universe"])
        if "SPY" not in fetch_tickers:
            fetch_tickers.append("SPY")
        history_dfs = market_data.multiple_daily(fetch_tickers, period="1y")
        history_serializable = {}

        for ticker, df in history_dfs.items():
            history_serializable[ticker] = {
                "dates": df.index.astype(str).tolist(),
                "prices": df["close"].tolist(),
            }

        # session_states already populated from MFT cache by API layer; pass through unchanged
        return {
            "price_history": history_serializable,
            "session_states": state.get("session_states") or {},
        }

    def node_macro_analysis(state: ARGUSState) -> dict:
        """Classifies current macroeconomic regimes and volatility indices.

        Args:
            state: ARGUSState (no required fields; macro agent uses its own data sources).

        Returns:
            Partial state update with ``macro_context``.
        """
        macro_ctx = macro_agent.analyze()
        if macro_ctx is None:
            logger.error(
                "node_macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable). "
                "The three specialist agents run independently and will still produce a "
                "degraded, macro-agnostic allocation this session."
            )
            return {
                "macro_context": None,
                "errors": ["macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable)"],
            }
        return {"macro_context": macro_ctx}

    def node_technical_analysis(state: ARGUSState) -> dict:
        """Computes statistical technical indicator metrics across active assets.

        Args:
            state: ARGUSState with ``session_states`` populated by ``node_fetch_price_history``.

        Returns:
            Partial state update with ``technical_signals``.
        """
        session_states = state.get("session_states", {})
        signals, errors = tech_agent.batch_analyze(session_states)
        return {"technical_signals": signals, "errors": errors}

    def node_fundamental_analysis(state: ARGUSState) -> dict:
        """Audits corporate health metrics and qualitative economic moat vectors.

        Args:
            state: ARGUSState with ``universe``, ``backtest_mode``, and ``session_seed``.

        Returns:
            Partial state update with ``fundamental_signals``.
        """
        signals, errors = fund_agent.batch_analyze(
            state["universe"],
            backtest_mode=state.get("backtest_mode", False),
            session_seed=state.get("session_seed"),
        )
        return {"fundamental_signals": signals, "errors": errors}

    def node_sentiment_analysis(state: ARGUSState) -> dict:
        """Scores news articles and social media sentiment signals.

        Args:
            state: ARGUSState with ``universe`` list of tickers.

        Returns:
            Partial state update with ``sentiment_signals``.
        """
        signals, errors = sent_agent.batch_analyze(state["universe"])
        return {"sentiment_signals": signals, "errors": errors}

    def node_retrieve_cultural_memory(state: ARGUSState) -> dict:
        """Retrieves semantic trading wisdom and historical trade patterns matching the current regime.

        Runs after signal_aggregation (not beside the specialist fan-out) so it can
        summarize this session's own aggregated_signals into the similarity query,
        rather than a hardcoded placeholder.

        Args:
            state: ARGUSState with ``macro_context`` to scope the semantic similarity
                query and ``aggregated_signals`` to summarize into it.

        Returns:
            Partial state update with ``cultural_memory`` dict containing 'wisdom' and 'warnings' lists.
        """
        macro = state.get("macro_context")
        if not macro:
            return {"cultural_memory": {"wisdom": [], "warnings": []}}

        try:
            memory = get_cultural_memory()
        except (ImportError, OSError, ChromaError) as exc:
            # ImportError: sentence-transformers/torch are an optional [models]
            # extra. OSError/ChromaError: persist_dir unwritable or an
            # incompatible on-disk Chroma schema. All three degrade to the same
            # empty result as no macro context, rather than crashing the node
            logger.warning("node_retrieve_cultural_memory: cultural memory unavailable: %s", exc)
            return {"cultural_memory": {"wisdom": [], "warnings": []}}

        as_of = state.get("as_of")
        technical_summary = _summarize_technical_posture(state.get("aggregated_signals", {}))
        wisdom = memory.retrieve_wisdom(macro, technical_summary, as_of=as_of)
        warnings = memory.retrieve_warnings(macro, as_of=as_of)

        return {
            "cultural_memory": {
                "wisdom": wisdom if wisdom else [],
                "warnings": warnings if warnings else [],
            }
        }

    def node_signal_aggregation(state: ARGUSState) -> dict:
        """Consolidates specialized analyst signals into weighted consensus indicators.

        Runs whenever at least one specialist produced a signal, independent of
        whether macro_context is populated — HybridSignalAggregator.aggregate()
        already handles macro=None (no multiplier scaling), so a macro data outage
        must not suppress aggregation of the specialists that did succeed.

        Args:
            state: ARGUSState with ``technical_signals``, ``fundamental_signals``,
                and ``sentiment_signals`` populated; ``macro_context`` optional.

        Returns:
            Partial state update with ``aggregated_signals``.
        """
        aggs: dict[str, AggregatedSignal] = {}
        macro = state.get("macro_context")
        as_of = state.get("as_of")
        agent_names = ("technical", "fundamental", "sentiment")
        model_healthy = True

        if macro is not None:
            # Per-regime reliability doesn't depend on ticker, so compute it once
            # per session rather than once per ticker.
            regime = macro.macro_regime.value
            try:
                memory = get_cultural_memory()
                accuracy = {
                    name: memory.get_agent_accuracy(name, regime=regime, as_of=as_of)
                    for name in agent_names
                }
                reliability = {name: accuracy[name][0] for name in agent_names}
                reliability_n = {name: accuracy[name][1] for name in agent_names}
            except (ImportError, OSError, ChromaError) as exc:
                # Same guard as node_retrieve_cultural_memory; degrade every
                # agent to the 0.5 neutral prior rather than crash
                logger.warning("node_signal_aggregation: cultural memory unavailable: %s", exc)
                reliability = {name: 0.5 for name in agent_names}
                reliability_n = {name: 0 for name in agent_names}
                model_healthy = False
        else:
            reliability = {name: 0.5 for name in agent_names}
            reliability_n = {name: 0 for name in agent_names}

        for ticker in state["universe"]:
            tech = state.get("technical_signals", {}).get(ticker)
            fund = state.get("fundamental_signals", {}).get(ticker)
            sent = state.get("sentiment_signals", {}).get(ticker)
            if tech is None and fund is None and sent is None:
                continue
            agg = aggregator.aggregate(
                tech, macro, fund, sent, reliability=reliability, reliability_n=reliability_n
            )
            aggs[ticker] = (
                agg if model_healthy else agg.model_copy(update={"model_healthy": False})
            )
        return {"aggregated_signals": aggs}

    def node_risk_evaluation(state: ARGUSState) -> dict:
        """Enforces structural, statistical, and sector concentration risk caps.

        Evaluates the full portfolio covariance first, then maps SLSQP-derived
        allocation caps back to per-ticker RiskAssessments. Portfolio-level VETO
        verdicts propagate down to suppress all individual asset approvals.

        Args:
            state: ARGUSState with ``price_history``, ``macro_context``, ``universe``,
                and ``aggregated_signals`` populated.

        Returns:
            Partial state update with ``risk_assessments`` and ``errors``. On an
            unexpected failure, ``risk_assessments`` is empty and the failure is
            named in ``errors`` rather than raised (RE-10) — nothing downstream
            can safely allocate against a risk session that half-completed.
        """
        import pandas as pd

        errors: list[str] = []

        try:
            history_raw = state.get("price_history", {})
            history = {
                ticker: pd.Series(data["prices"], index=pd.to_datetime(data["dates"]))
                for ticker, data in history_raw.items()
            }
            macro_ctx = state.get("macro_context")
            if macro_ctx is not None:
                vix = macro_ctx.vix_level
            else:
                # A FRED outage only takes out the macro bundle; the VIX blackout gate
                # needs just the index level, which yfinance can still serve directly
                try:
                    vix = market_data.vix()
                except Exception as exc:
                    logger.warning(
                        "node_risk_evaluation: VIX fetch failed with no macro_context; "
                        "falling back to vix=20.0: %s",
                        exc,
                    )
                    errors.append(
                        f"risk_evaluation: VIX fetch failed, used fallback vix=20.0 ({exc})"
                    )
                    vix = 20.0
            universe = state["universe"]

            # Extract signed convictions: positive = BULLISH, negative = BEARISH, zero = NEUTRAL
            convictions = {}
            aggs = state.get("aggregated_signals", {})
            if aggs:
                for t, sig in aggs.items():
                    sign = (
                        1.0
                        if sig.signal == Signal.BULLISH
                        else (-1.0 if sig.signal == Signal.BEARISH else 0.0)
                    )
                    convictions[t] = sig.conviction * sign

            # Joint evaluation captures inter-asset covariance and global diversification limits
            base_weight = min(0.15, 1.0 / len(universe)) if universe else 0.15
            full_portfolio = [{"ticker": t, "weight": base_weight} for t in universe]
            portfolio_result = risk_engine.evaluate(
                full_portfolio, history, vix, convictions=convictions
            )

            portfolio_vetoed = portfolio_result.verdict == RiskVerdict.VETO
            ticker_cap_map: dict[str, float] = portfolio_result.optimal_weights

            if portfolio_result.optimizer_converged is False:
                errors.append(
                    "risk_evaluation: SLSQP solve did not converge; portfolio-level caps "
                    "were not applied to any ticker"
                )
            # RE-4: below-floor diversification is informational on the assessment's own
            # veto_reasons (risk.py), not a hard VETO — surface it here too so it's visible
            # without inspecting every per-ticker assessment
            errors.extend(
                f"risk_evaluation: {r}"
                for r in portfolio_result.veto_reasons
                if r.startswith("Below diversification floor")
            )

            assessments = {}

            for ticker in universe:
                single = risk_engine.evaluate(
                    [{"ticker": ticker, "weight": settings.MAX_SINGLE_POSITION_PCT}], history, vix
                )

                updates: dict = {}

                # Propagate portfolio-level correlation stats for uniform telemetry, but only
                # when the portfolio call actually measured them — a structural-violation
                # short-circuit reports None, not a fabricated 0.0
                if portfolio_result.avg_correlation is not None:
                    updates["avg_correlation"] = portfolio_result.avg_correlation

                if ticker in ticker_cap_map and not portfolio_vetoed:
                    updates.update(_apply_portfolio_cap(single, ticker_cap_map[ticker]))

                # Portfolio-level veto overrides all individual verdicts to prevent partial allocation
                if portfolio_vetoed:
                    updates["verdict"] = RiskVerdict.VETO
                    updates["approved_weight"] = 0.0
                    updates["veto_reasons"] = single.veto_reasons + [
                        f"[Portfolio] {r}" for r in portfolio_result.veto_reasons
                    ]

                # A single model_validate reconstruction, not sequential model_copy calls —
                # model_copy skips validators, so it was bypassing approved_le_proposed
                assessments[ticker] = (
                    RiskAssessment.model_validate({**single.model_dump(), **updates})
                    if updates
                    else single
                )

            return {"risk_assessments": assessments, "errors": errors}
        except Exception as exc:
            logger.exception("node_risk_evaluation: unexpected failure")
            errors.append(f"risk_evaluation: unexpected failure: {exc}")
            return {"risk_assessments": {}, "errors": errors}

    def node_portfolio_allocation(state: ARGUSState) -> dict:
        """Determines capital allocations using specialist ratings, risk targets, and memory heuristics.

        Args:
            state: ARGUSState with all signal, risk, and cultural memory fields populated.

        Returns:
            Partial state update with ``portfolio_allocation``.
        """
        # Scoped to tickers with an aggregated signal, not the whole universe: a
        # ticker every specialist failed on has nothing to allocate against, and
        # an all-None entry would still pass build_signal_table's is-approved
        # filter through to _is_risk_approved(None) -> False silently rather than
        # never appearing in all_signals at all
        signals_dict = {}
        for ticker in state.get("aggregated_signals", {}):
            signals_dict[ticker] = {
                "technical": state.get("technical_signals", {}).get(ticker),
                "fundamental": state.get("fundamental_signals", {}).get(ticker),
                "sentiment": state.get("sentiment_signals", {}).get(ticker),
                "risk": state.get("risk_assessments", {}).get(ticker),
                "aggregated": state.get("aggregated_signals", {}).get(ticker),
            }

        profile = {
            "total_wealth": state.get("total_wealth"),
            "invest_pct": state.get("invest_pct"),
            "risk_tolerance": state.get("risk_tolerance"),
        }

        macro = state.get("macro_context")
        errors: list[str] = []
        if macro is None:
            errors.append(
                "portfolio_allocation: macro_context unavailable this session; "
                "allocating on specialist signals alone"
            )

        mem = state.get("cultural_memory", {})
        wisdom = mem.get("wisdom", [])
        warnings = mem.get("warnings", [])

        alloc = portfolio_agent.allocate(
            profile, signals_dict, macro, wisdom, warnings, adjustments=errors
        )
        if alloc is None:
            # allocate() has already appended the specific failing stage (LLM call,
            # JSON parse, response enforcement, or schema validation) to errors
            logger.error(
                "node_portfolio_allocation: PortfolioManagerAgent returned None. "
                "Skipping portfolio_allocation to allow graceful exit."
            )
            return {"errors": errors}
        return {"portfolio_allocation": alloc, "errors": errors}

    def node_log_decisions(state: ARGUSState) -> dict:
        """Persists structural decision profiles to vector database collections.

        Args:
            state: ARGUSState with all fields populated from prior nodes.

        Returns:
            Partial state update with ``decisions`` list of completed ARGUSDecision
            objects and ``errors`` naming any per-ticker construction failures.
        """
        allocation = state.get("portfolio_allocation")
        if not allocation:
            logger.warning("[log_decisions] No portfolio_allocation in state; skipping.")
            return {"decisions": []}

        positions = {pos.ticker: pos for pos in allocation.portfolio}
        # A replayed session's as_of is the fixture's own capture date; falling back
        # to wall-clock only applies to genuinely live sessions
        now = state.get("as_of") or datetime.now()
        decisions: list[ARGUSDecision] = []
        errors: list[str] = []

        try:
            memory = get_cultural_memory()
        except (ImportError, OSError, ChromaError) as exc:
            # Same guard as node_retrieve_cultural_memory
            logger.warning("node_log_decisions: cultural memory unavailable: %s", exc)
            memory = None

        snapshots_stored = 0
        for ticker in state.get("universe", []):
            try:
                decision = ARGUSDecision(
                    ticker=ticker,
                    session_timestamp=now,
                    technical=state.get("technical_signals", {}).get(ticker),
                    macro=state.get("macro_context"),
                    fundamental=state.get("fundamental_signals", {}).get(ticker),
                    sentiment=state.get("sentiment_signals", {}).get(ticker),
                    risk=state.get("risk_assessments", {}).get(ticker),
                    aggregated=state.get("aggregated_signals", {}).get(ticker),
                    allocation=positions.get(ticker),
                )
                decisions.append(decision)

                # Snapshot stored pre-confirmation to enable pre-settlement similarity retrieval
                if memory is not None and memory.store_decision_snapshot(decision):
                    snapshots_stored += 1

            except Exception as exc:
                msg = f"log_decisions: failed to build decision for {ticker}: {exc}"
                logger.warning("[log_decisions] %s", msg)
                errors.append(msg)

        logger.info(
            "[log_decisions] Logged %d/%d decisions to cultural memory.",
            snapshots_stored,
            len(state.get("universe", [])),
        )
        return {"decisions": decisions, "errors": errors}

    builder = StateGraph(ARGUSState)

    builder.add_node("fetch_price_history", node_fetch_price_history)
    builder.add_node("macro_analysis", node_macro_analysis)
    builder.add_node("technical_analysis", node_technical_analysis)
    builder.add_node("fundamental_analysis", node_fundamental_analysis)
    builder.add_node("sentiment_analysis", node_sentiment_analysis)
    builder.add_node("retrieve_cultural_memory", node_retrieve_cultural_memory)
    builder.add_node("signal_aggregation", node_signal_aggregation)
    builder.add_node("risk_evaluation", node_risk_evaluation)
    builder.add_node("portfolio_allocation", node_portfolio_allocation)
    builder.add_node("log_decisions", node_log_decisions)

    builder.set_entry_point("fetch_price_history")

    # macro_analysis runs in parallel with the three specialists, not ahead of
    # them: none of them consumes macro_context (see state.py's docstring), so
    # gating them behind it needlessly serializes fundamental's ~20 sequential
    # Groq round-trips behind the HMM window and drops a whole session's output
    # whenever only the FRED bundle is unavailable.
    builder.add_edge("fetch_price_history", "macro_analysis")
    builder.add_edge("fetch_price_history", "technical_analysis")
    builder.add_edge("fetch_price_history", "fundamental_analysis")
    builder.add_edge("fetch_price_history", "sentiment_analysis")

    builder.add_edge(
        ["macro_analysis", "technical_analysis", "fundamental_analysis", "sentiment_analysis"],
        "signal_aggregation",
    )

    # retrieve_cultural_memory runs after signal_aggregation (not beside it) so it
    # can summarize this session's own aggregated_signals into its similarity
    # query; wisdom/warnings are consumed only by portfolio_allocation, so this
    # is safe to run in parallel with risk_evaluation.
    builder.add_edge("signal_aggregation", "retrieve_cultural_memory")
    builder.add_edge("signal_aggregation", "risk_evaluation")
    builder.add_edge(["retrieve_cultural_memory", "risk_evaluation"], "portfolio_allocation")
    builder.add_edge("portfolio_allocation", "log_decisions")
    builder.add_edge("log_decisions", END)

    return _CheckpointedGraph(builder, checkpoint_db_path)
