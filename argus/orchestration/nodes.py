"""
argus/orchestration/nodes.py

The ten LangGraph node functions, each mapping ARGUSState → a partial state
update, plus the helpers that only they use.

Responsibilities:
  - Define one module-level function per graph node, taking the state and a
    GraphDeps bundle of the already-constructed agents and seams
  - Hold the pure helpers the nodes share (VIX resolution, portfolio-cap
    mapping, per-session reliability, technical-posture summary)

Not responsible for:
  - Constructing the agents or wiring the DAG (see orchestration/graph.py)
  - Agent logic (see the individual agent modules)
  - State schema (see orchestration/state.py)

Dependencies:
  - argus.agents.* (the six specialist agents, held in GraphDeps)
  - argus.memory.cultural (the CulturalMemoryManager type)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from chromadb.errors import ChromaError

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import (
    COVARIANCE_NOTE_PREFIX,
    DIVERSIFICATION_FLOOR_PREFIX,
    RiskStatisticalAgent,
)
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.config import settings
from argus.infra.clock import now_local
from argus.infra.diagnostics import Diagnostics
from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.aggregator import SPECIALIST_AGENTS, HybridSignalAggregator
from argus.orchestration.state import ARGUSState, build_ticker_snapshots
from argus.params import AGGREGATOR, RISK, SYSTEM
from argus.ports import MarketDataProvider
from argus.schemas.signals import (
    AggregatedSignal,
    ARGUSDecision,
    MacroContext,
    RiskAssessment,
    RiskVerdict,
    Signal,
)

logger = logging.getLogger(__name__)

# Signed conviction convention shared with risk.py: positive is long-side pressure
_SIGNAL_SIGN = {Signal.BULLISH: 1.0, Signal.BEARISH: -1.0, Signal.NEUTRAL: 0.0}


@dataclass(frozen=True)
class GraphDeps:
    """Everything a node needs besides the state, built once per graph.

    Attributes:
        market_data: Shared data source, used directly by the price-history
            and VIX-fallback paths.
        macro_agent: Classifies the macro regime.
        tech_agent: Computes technical signals.
        fund_agent: Produces fundamental signals.
        sent_agent: Produces sentiment signals.
        risk_agent: Evaluates per-ticker and portfolio risk.
        aggregator: Combines the specialist signals.
        portfolio_agent: Allocates capital.
        cultural_memory_provider: Returns the cultural memory store. Called on
            each use, because constructing the real store can fail (see
            ``_cultural_memory_or_none``).
    """

    market_data: MarketDataProvider
    macro_agent: MacroStatisticalAgent
    tech_agent: TechnicalStatisticalAgent
    fund_agent: FundamentalAgent
    sent_agent: SentimentAgent
    risk_agent: RiskStatisticalAgent
    aggregator: HybridSignalAggregator
    portfolio_agent: PortfolioManagerAgent
    cultural_memory_provider: Callable[[], CulturalMemoryManager]


def _cultural_memory_or_none(deps: GraphDeps) -> CulturalMemoryManager | None:
    """Returns the cultural memory store, or None when it cannot be opened.

    Args:
        deps: The graph's dependencies.

    Returns:
        The store, or None after logging why it is unavailable. ImportError:
        sentence-transformers/torch are an optional [models] extra. OSError and
        ChromaError: the persist directory is unwritable or holds an
        incompatible on-disk Chroma schema. Every caller degrades to the same
        empty result rather than crashing the node.
    """
    try:
        return deps.cultural_memory_provider()
    except (ImportError, OSError, ChromaError) as exc:
        logger.warning("cultural memory unavailable: %s", exc)
        return None


def _summarize_technical_posture(aggregated_signals: dict[str, AggregatedSignal]) -> str:
    """Builds this session's technical-posture summary for retrieve_wisdom's query.

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
    avg_conviction = sum(s.conviction for s in aggregated_signals.values()) / len(
        aggregated_signals
    )
    return (
        f"{bullish} bullish, {bearish} bearish, {neutral} neutral across "
        f"{len(aggregated_signals)} tickers (avg conviction {avg_conviction:.2f})"
    )


def _session_reliability(
    macro: MacroContext | None, as_of: datetime | None, deps: GraphDeps
) -> tuple[dict[str, float], dict[str, int], bool]:
    """Reads each specialist's per-regime historical win rate once for the whole session.

    Reliability is a function of the regime, not of the ticker, so this runs
    once per session rather than once per aggregated ticker.

    Args:
        macro: This session's MacroContext, or None if the macro node degraded.
        as_of: Point-in-time cutoff for cultural-memory reads.
        deps: The graph's dependencies.

    Returns:
        ``(reliability, reliability_n, model_healthy)`` for
        HybridSignalAggregator.aggregate(). With no regime to scope the lookup,
        or with cultural memory unavailable, every agent falls back to the 0.5
        neutral prior at a zero sample count; only the latter is unhealthy.
    """
    neutral_reliability = dict.fromkeys(SPECIALIST_AGENTS, AGGREGATOR.neutral_reliability)
    neutral_counts = dict.fromkeys(SPECIALIST_AGENTS, 0)
    if macro is None:
        return neutral_reliability, neutral_counts, True

    memory = _cultural_memory_or_none(deps)
    if memory is None:
        return neutral_reliability, neutral_counts, False

    accuracy = {
        name: memory.get_agent_accuracy(name, regime=macro.macro_regime.value, as_of=as_of)
        for name in SPECIALIST_AGENTS
    }
    return (
        {name: rate for name, (rate, _) in accuracy.items()},
        {name: n for name, (_, n) in accuracy.items()},
        True,
    )


def _resolve_vix(
    macro_ctx: MacroContext | None, market_data: MarketDataProvider
) -> tuple[float, str | None]:
    """Resolves the VIX level the risk engine's blackout gate reads.

    Args:
        macro_ctx: This session's MacroContext, or None if the macro node degraded.
        market_data: Direct fallback source for the index level.

    Returns:
        ``(vix, error)``, where ``error`` is an ARGUSState["errors"] entry naming
        the fallback that was used, or None when a real level was resolved.
    """
    if macro_ctx is not None:
        return macro_ctx.vix_level, None

    # A FRED outage only takes out the macro bundle; the VIX blackout gate needs
    # just the index level, which yfinance can still serve directly
    try:
        return market_data.vix(), None
    except Exception as exc:
        logger.warning(
            "VIX fetch failed with no macro_context; falling back to vix=20.0: %s",
            exc,
        )
        return 20.0, f"risk_evaluation: VIX fetch failed, used fallback vix=20.0 ({exc})"


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
        # A cap this small is no room to allocate, not a REDUCE — REDUCE still
        # counts as approved (see schemas/signals.py's TickerSnapshot.risk_approved) and the
        # prompt would demand invest_pct deployment against a 0% cap
        return {
            "verdict": RiskVerdict.VETO,
            "approved_weight": 0.0,
            "veto_reasons": [
                *single.veto_reasons,
                (
                    f"[Portfolio] SLSQP cap {cap:.2%} below the "
                    f"{RISK.slsqp_zero_cap_epsilon:.2%} allocation floor"
                ),
            ],
        }
    # Downgrade verdict monotonically: never upgrade a VETO to REDUCE or APPROVE
    if single.verdict != RiskVerdict.VETO and cap < single.approved_weight:
        verdict = RiskVerdict.REDUCE
    else:
        verdict = single.verdict
    return {
        "verdict": verdict,
        "approved_weight": min(cap, single.approved_weight),
        "veto_reasons": [
            *single.veto_reasons,
            f"[Portfolio] SLSQP cap: {cap:.1%}",
        ],
    }


def node_fetch_price_history(state: ARGUSState, deps: GraphDeps) -> dict:
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
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``price_history``, which also carries the SPY
        benchmark series alongside the universe. ``session_states`` is passed
        through unchanged from the incoming state.
    """
    # SPY rides along with the universe fetch so risk.py's ols_portfolio_beta finds it
    # already in price_history, instead of issuing its own uncached fetch on every one
    # of the N+1 risk_agent.evaluate() calls node_risk_evaluation makes this session
    fetch_tickers = list(state["universe"])
    if "SPY" not in fetch_tickers:
        fetch_tickers.append("SPY")
    history_dfs = deps.market_data.multiple_daily(fetch_tickers, period="1y")
    history_serializable = {
        ticker: {
            "dates": df.index.astype(str).tolist(),
            "prices": df["close"].tolist(),
        }
        for ticker, df in history_dfs.items()
    }
    return {
        "price_history": history_serializable,
        "session_states": state.get("session_states") or {},
    }


def node_macro_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Classifies current macroeconomic regimes and volatility indices.

    Args:
        state: ARGUSState (no required fields; macro agent uses its own data sources).
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``macro_context``.
    """
    macro_ctx = deps.macro_agent.analyze()
    if macro_ctx is None:
        logger.error(
            "MacroStatisticalAgent returned None (FRED data unavailable). "
            "The three specialist agents run independently and will still produce a "
            "degraded, macro-agnostic allocation this session."
        )
        return {
            "macro_context": None,
            "errors": [
                "macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable)"
            ],
        }
    return {"macro_context": macro_ctx}


def node_technical_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Computes statistical technical indicator metrics across active assets.

    Args:
        state: ARGUSState with ``session_states`` populated by ``node_fetch_price_history``.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``technical_signals``.
    """
    session_states = state.get("session_states", {})
    signals, errors = deps.tech_agent.batch_analyze(session_states)
    return {"technical_signals": signals, "errors": errors}


def node_fundamental_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Audits corporate health metrics and qualitative economic moat vectors.

    Args:
        state: ARGUSState with ``universe``, ``backtest_mode``, and ``session_seed``.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``fundamental_signals``.
    """
    signals, errors = deps.fund_agent.batch_analyze(
        state["universe"],
        backtest_mode=state.get("backtest_mode", False),
        session_seed=state.get("session_seed"),
    )
    return {"fundamental_signals": signals, "errors": errors}


def node_sentiment_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Scores news article sentiment signals.

    Args:
        state: ARGUSState with ``universe`` list of tickers.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``sentiment_signals``.
    """
    signals, errors = deps.sent_agent.batch_analyze(state["universe"])
    return {"sentiment_signals": signals, "errors": errors}


def node_retrieve_cultural_memory(state: ARGUSState, deps: GraphDeps) -> dict:
    """Retrieves cultural-memory wisdom and warnings matching the current regime.

    Runs after signal_aggregation (not beside the specialist fan-out) so it can
    summarize this session's own aggregated_signals into the similarity query,
    rather than a hardcoded placeholder.

    Args:
        state: ARGUSState with ``macro_context`` to scope the semantic similarity
            query and ``aggregated_signals`` to summarize into it.
        deps: The graph's dependencies.

    Returns:
        Partial state update with a ``cultural_memory`` dict of 'wisdom' and 'warnings' lists.
    """
    empty: dict = {"cultural_memory": {"wisdom": [], "warnings": []}}
    macro = state.get("macro_context")
    if not macro:
        return empty

    memory = _cultural_memory_or_none(deps)
    if memory is None:
        return empty

    as_of = state.get("as_of")
    technical_summary = _summarize_technical_posture(state.get("aggregated_signals", {}))
    wisdom = memory.retrieve_wisdom(macro, technical_summary, as_of=as_of)
    warnings = memory.retrieve_warnings(macro, as_of=as_of)

    return {"cultural_memory": {"wisdom": wisdom or [], "warnings": warnings or []}}


def node_signal_aggregation(state: ARGUSState, deps: GraphDeps) -> dict:
    """Consolidates specialized analyst signals into weighted consensus indicators.

    Runs whenever at least one specialist produced a signal, independent of
    whether macro_context is populated — HybridSignalAggregator.aggregate()
    already handles macro=None (no multiplier scaling), so a macro data outage
    must not suppress aggregation of the specialists that did succeed.

    Args:
        state: ARGUSState with ``technical_signals``, ``fundamental_signals``,
            and ``sentiment_signals`` populated; ``macro_context`` optional.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``aggregated_signals``.
    """
    macro = state.get("macro_context")
    reliability, reliability_n, model_healthy = _session_reliability(
        macro, state.get("as_of"), deps
    )

    aggs: dict[str, AggregatedSignal] = {}
    for ticker, snap in build_ticker_snapshots(state).items():
        if snap.technical is None and snap.fundamental is None and snap.sentiment is None:
            continue
        agg = deps.aggregator.aggregate(
            snap.technical,
            macro,
            snap.fundamental,
            snap.sentiment,
            reliability=reliability,
            reliability_n=reliability_n,
        )
        # Reconstructed through validation: model_copy would skip the schema's validators
        aggs[ticker] = (
            agg
            if model_healthy
            else AggregatedSignal.model_validate({**agg.model_dump(), "model_healthy": False})
        )
    return {"aggregated_signals": aggs}


def node_risk_evaluation(state: ARGUSState, deps: GraphDeps) -> dict:
    """Enforces structural, statistical, and sector concentration risk caps.

    Evaluates the full portfolio covariance first, then maps SLSQP-derived
    allocation caps back to per-ticker RiskAssessments. Portfolio-level VETO
    verdicts propagate down to suppress all individual asset approvals.

    Args:
        state: ARGUSState with ``price_history``, ``macro_context``, ``universe``,
            and ``aggregated_signals`` populated.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``risk_assessments`` and ``errors``. On an
        unexpected failure, ``risk_assessments`` is empty and the failure is
        named in ``errors`` rather than raised — nothing downstream can
        safely allocate against a risk session that half-completed.
    """
    errors: list[str] = []

    try:
        history = {
            ticker: pd.Series(data["prices"], index=pd.to_datetime(data["dates"]))
            for ticker, data in state.get("price_history", {}).items()
        }
        vix, vix_error = _resolve_vix(state.get("macro_context"), deps.market_data)
        if vix_error is not None:
            errors.append(vix_error)
        universe = state["universe"]

        convictions = {
            ticker: sig.conviction * _SIGNAL_SIGN[sig.signal]
            for ticker, sig in state.get("aggregated_signals", {}).items()
        }

        # Joint evaluation captures inter-asset covariance and global diversification limits
        base_weight = (
            min(SYSTEM.max_single_position_pct, 1.0 / len(universe))
            if universe
            else SYSTEM.max_single_position_pct
        )
        full_portfolio = [{"ticker": t, "weight": base_weight} for t in universe]
        portfolio_result = deps.risk_agent.evaluate(
            full_portfolio, history, vix, convictions=convictions
        )

        portfolio_vetoed = portfolio_result.verdict == RiskVerdict.VETO
        ticker_cap_map: dict[str, float] = portfolio_result.optimal_weights

        if portfolio_result.optimizer_converged is False:
            errors.append(
                "risk_evaluation: SLSQP solve did not converge; portfolio-level caps "
                "were not applied to any ticker"
            )
        # Below-floor diversification is informational on the assessment's own
        # veto_reasons (risk.py), not a hard VETO — surface it here too so it's
        # visible without inspecting every per-ticker assessment
        errors.extend(
            f"risk_evaluation: {r}"
            for r in portfolio_result.veto_reasons
            if r.startswith((DIVERSIFICATION_FLOOR_PREFIX, COVARIANCE_NOTE_PREFIX))
        )

        assessments = {}

        for ticker in universe:
            single = deps.risk_agent.evaluate(
                [{"ticker": ticker, "weight": settings.MAX_SINGLE_POSITION_PCT}], history, vix
            )

            updates: dict = {}

            # Propagate portfolio-level correlation stats for uniform telemetry, but only
            # when the portfolio call actually measured them — a structural-violation
            # short-circuit reports None, not a fabricated 0.0
            if portfolio_result.avg_correlation is not None:
                updates["avg_correlation"] = portfolio_result.avg_correlation

            if portfolio_vetoed:
                # A portfolio-level veto overrides every individual verdict, so its
                # own caps never apply — partial allocation is what it exists to stop
                updates["verdict"] = RiskVerdict.VETO
                updates["approved_weight"] = 0.0
                updates["veto_reasons"] = single.veto_reasons + [
                    f"[Portfolio] {r}" for r in portfolio_result.veto_reasons
                ]
            elif ticker in ticker_cap_map:
                updates.update(_apply_portfolio_cap(single, ticker_cap_map[ticker]))

            # A single model_validate reconstruction, not sequential model_copy calls —
            # model_copy skips validators, so it was bypassing approved_le_proposed
            assessments[ticker] = (
                RiskAssessment.model_validate({**single.model_dump(), **updates})
                if updates
                else single
            )

        return {"risk_assessments": assessments, "errors": errors}
    except Exception as exc:
        logger.exception("unexpected failure")
        errors.append(f"risk_evaluation: unexpected failure: {exc}")
        return {"risk_assessments": {}, "errors": errors}


def node_portfolio_allocation(state: ARGUSState, deps: GraphDeps) -> dict:
    """Allocates capital using specialist signals, risk targets, and cultural-memory heuristics.

    Args:
        state: ARGUSState with all signal, risk, and cultural memory fields populated.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``portfolio_allocation``.
    """
    # Scoped to tickers with an aggregated signal, not the whole universe: a
    # ticker every specialist failed on has nothing to allocate against.
    snapshots = {
        ticker: snap
        for ticker, snap in build_ticker_snapshots(state).items()
        if snap.aggregated is not None
    }

    profile = {
        "total_wealth": state.get("total_wealth"),
        "invest_pct": state.get("invest_pct"),
        "risk_tolerance": state.get("risk_tolerance"),
    }

    macro = state.get("macro_context")
    diagnostics = Diagnostics()
    if macro is None:
        diagnostics.record(
            "portfolio_allocation: macro_context unavailable this session; "
            "allocating on specialist signals alone"
        )

    mem = state.get("cultural_memory", {})
    wisdom = mem.get("wisdom", [])
    warnings = mem.get("warnings", [])

    alloc = deps.portfolio_agent.allocate(
        profile, snapshots, macro, wisdom, warnings, diagnostics=diagnostics
    )
    if alloc is None:
        # allocate() has already appended the specific failing stage (LLM call,
        # JSON parse, response enforcement, or schema validation) to diagnostics
        logger.error(
            "PortfolioManagerAgent returned None. "
            "Skipping portfolio_allocation to allow graceful exit."
        )
        return {"errors": diagnostics.messages}
    return {"portfolio_allocation": alloc, "errors": diagnostics.messages}


def node_log_decisions(state: ARGUSState, deps: GraphDeps) -> dict:
    """Persists structural decision profiles to vector database collections.

    Args:
        state: ARGUSState with all fields populated from prior nodes.
        deps: The graph's dependencies.

    Returns:
        Partial state update with ``decisions`` list of completed ARGUSDecision
        objects and ``errors`` naming any per-ticker construction failures.
    """
    allocation = state.get("portfolio_allocation")
    if not allocation:
        logger.warning("No portfolio_allocation in state; skipping.")
        return {"decisions": []}

    positions = {pos.ticker: pos for pos in allocation.portfolio}
    # A replayed session's as_of is the fixture's own capture date; falling back
    # to wall-clock only applies to genuinely live sessions
    now = state.get("as_of") or now_local()
    decisions: list[ARGUSDecision] = []
    errors: list[str] = []

    memory = _cultural_memory_or_none(deps)

    ticker_snapshots = build_ticker_snapshots(state)
    universe = state.get("universe", [])
    stored_count = 0
    for ticker in universe:
        try:
            snap = ticker_snapshots[ticker]
            decision = ARGUSDecision(
                ticker=ticker,
                session_timestamp=now,
                technical=snap.technical,
                macro=state.get("macro_context"),
                fundamental=snap.fundamental,
                sentiment=snap.sentiment,
                risk=snap.risk,
                aggregated=snap.aggregated,
                allocation=positions.get(ticker),
            )
            decisions.append(decision)

            # Snapshot stored pre-confirmation to enable pre-settlement similarity retrieval
            if memory is not None and memory.store_decision_snapshot(decision):
                stored_count += 1

        except Exception as exc:
            msg = f"log_decisions: failed to build decision for {ticker}: {exc}"
            logger.warning("%s", msg)
            errors.append(msg)

    logger.info(
        "Logged %d/%d decisions to cultural memory.",
        stored_count,
        len(universe),
    )
    return {"decisions": decisions, "errors": errors}
