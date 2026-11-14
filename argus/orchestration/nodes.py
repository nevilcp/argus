"""Decision graph nodes and their injected dependencies."""

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
    """Everything a node needs besides the state, built once per graph."""

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
    """Return the cultural memory store, or None when it cannot be opened."""
    try:
        return deps.cultural_memory_provider()
    except (ImportError, OSError, ChromaError) as exc:
        logger.warning("cultural memory unavailable: %s", exc)
        return None


def _summarize_technical_posture(aggregated_signals: dict[str, AggregatedSignal]) -> str:
    """Build this session's technical-posture summary for retrieve_wisdom's query."""
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


def _session_reliability(
    macro: MacroContext | None, as_of: datetime | None, deps: GraphDeps
) -> tuple[dict[str, float], dict[str, int], bool]:
    """Read each specialist's per-regime historical win rate once for the whole session."""
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


def _resolve_vix(macro_ctx: MacroContext | None, market_data: MarketDataProvider) -> tuple[float | None, str | None]:
    """Resolve the VIX level the risk engine's blackout gate reads."""
    if macro_ctx is not None and macro_ctx.vix_level is not None:
        return macro_ctx.vix_level, None

    # Fetch VIX directly when the FRED bundle fails so risk can still evaluate the volatility gate.
    try:
        return market_data.vix(), None
    except Exception as exc:
        logger.warning(
            "VIX unavailable; risk blackout check will fail closed: %s",
            exc,
        )
        return None, f"risk_evaluation: VIX unavailable ({exc})"


def _apply_portfolio_cap(single: RiskAssessment, cap: float) -> dict:
    """Map an SLSQP-derived per-ticker cap onto a single-ticker risk verdict."""
    if cap < RISK.slsqp_zero_cap_epsilon:
        # Veto unusably small caps because REDUCE still counts as risk-approved in the allocation prompt.
        return {
            "verdict": RiskVerdict.VETO,
            "approved_weight": 0.0,
            "veto_reasons": [
                *single.veto_reasons,
                (f"[Portfolio] SLSQP cap {cap:.2%} below the {RISK.slsqp_zero_cap_epsilon:.2%} allocation floor"),
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
    """Fetch raw daily pricing data for risk covariance calculation."""
    # Fetch SPY with the universe so joint and per-ticker beta evaluations share one benchmark history.
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
    """Classify current macroeconomic regimes and volatility indices."""
    macro_ctx = deps.macro_agent.analyze()
    if macro_ctx is None:
        logger.error(
            "MacroStatisticalAgent returned None (FRED data unavailable). "
            "The three specialist agents run independently and will still produce a "
            "degraded, macro-agnostic allocation this session."
        )
        return {
            "macro_context": None,
            "errors": ["macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable)"],
        }
    return {"macro_context": macro_ctx}


def node_technical_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Compute statistical technical indicator metrics across active assets."""
    session_states = state.get("session_states", {})
    signals, errors = deps.tech_agent.batch_analyze(session_states)
    return {"technical_signals": signals, "errors": errors}


def node_fundamental_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Audit corporate health metrics and qualitative economic moat vectors."""
    signals, errors = deps.fund_agent.batch_analyze(
        state["universe"],
        backtest_mode=state.get("backtest_mode", False),
        session_seed=state.get("session_seed"),
    )
    return {"fundamental_signals": signals, "errors": errors}


def node_sentiment_analysis(state: ARGUSState, deps: GraphDeps) -> dict:
    """Score news article sentiment signals."""
    signals, errors = deps.sent_agent.batch_analyze(state["universe"])
    return {"sentiment_signals": signals, "errors": errors}


def node_retrieve_cultural_memory(state: ARGUSState, deps: GraphDeps) -> dict:
    """Retrieve cultural-memory wisdom and warnings matching the current regime."""
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
    """Consolidate specialized analyst signals into weighted consensus indicators."""
    macro = state.get("macro_context")
    reliability, reliability_n, model_healthy = _session_reliability(macro, state.get("as_of"), deps)

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
            agg if model_healthy else AggregatedSignal.model_validate({**agg.model_dump(), "model_healthy": False})
        )
    return {"aggregated_signals": aggs}


def node_risk_evaluation(state: ARGUSState, deps: GraphDeps) -> dict:
    """Enforce structural, statistical, and sector concentration risk caps."""
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
            min(SYSTEM.max_single_position_pct, 1.0 / len(universe)) if universe else SYSTEM.max_single_position_pct
        )
        full_portfolio = [{"ticker": t, "weight": base_weight} for t in universe]
        portfolio_result = deps.risk_agent.evaluate(full_portfolio, history, vix, convictions=convictions)

        portfolio_vetoed = portfolio_result.verdict == RiskVerdict.VETO
        ticker_cap_map: dict[str, float] = portfolio_result.optimal_weights

        if portfolio_result.optimizer_converged is False:
            errors.append(
                "risk_evaluation: SLSQP solve did not converge; portfolio-level caps were not applied to any ticker"
            )
        # Surface informational diversification notes in graph diagnostics without changing their risk verdict.
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

            # Propagate portfolio correlation only when the joint evaluation actually measured it.
            if portfolio_result.avg_correlation is not None:
                updates["avg_correlation"] = portfolio_result.avg_correlation

            if portfolio_vetoed:
                # Propagate a portfolio veto to every ticker because partial allocation would defeat the joint gate.
                updates["verdict"] = RiskVerdict.VETO
                updates["approved_weight"] = 0.0
                updates["veto_reasons"] = single.veto_reasons + [
                    f"[Portfolio] {r}" for r in portfolio_result.veto_reasons
                ]
            elif ticker in ticker_cap_map:
                updates.update(_apply_portfolio_cap(single, ticker_cap_map[ticker]))

            # Reconstruct through validation because model_copy would bypass approved-weight invariants.
            assessments[ticker] = (
                RiskAssessment.model_validate({**single.model_dump(), **updates}) if updates else single
            )

        return {"risk_assessments": assessments, "errors": errors}
    except Exception as exc:
        logger.exception("unexpected failure")
        errors.append(f"risk_evaluation: unexpected failure: {exc}")
        return {"risk_assessments": {}, "errors": errors}


def node_portfolio_allocation(state: ARGUSState, deps: GraphDeps) -> dict:
    """Allocate capital using specialist signals, risk targets, and cultural-memory heuristics."""
    # Allocate only against tickers with aggregated evidence; a failed specialist set supplies no posture.
    snapshots = {ticker: snap for ticker, snap in build_ticker_snapshots(state).items() if snap.aggregated is not None}

    profile = {
        "total_wealth": state.get("total_wealth"),
        "invest_pct": state.get("invest_pct"),
        "risk_tolerance": state.get("risk_tolerance"),
    }

    macro = state.get("macro_context")
    diagnostics = Diagnostics()
    if macro is None:
        diagnostics.record(
            "portfolio_allocation: macro_context unavailable this session; allocating on specialist signals alone"
        )

    mem = state.get("cultural_memory", {})
    wisdom = mem.get("wisdom", [])
    warnings = mem.get("warnings", [])

    alloc = deps.portfolio_agent.allocate(profile, snapshots, macro, wisdom, warnings, diagnostics=diagnostics)
    if alloc is None:
        # Preserve the decoder or enforcement diagnostics already recorded by allocate.
        logger.error("PortfolioManagerAgent returned None. Skipping portfolio_allocation to allow graceful exit.")
        return {"errors": diagnostics.messages}
    return {"portfolio_allocation": alloc, "errors": diagnostics.messages}


def node_log_decisions(state: ARGUSState, deps: GraphDeps) -> dict:
    """Persist structural decision profiles to vector database collections."""
    allocation = state.get("portfolio_allocation")
    if not allocation:
        logger.warning("No portfolio_allocation in state; skipping.")
        return {"decisions": []}

    positions = {pos.ticker: pos for pos in allocation.portfolio}
    # Use the fixture capture date for replay and the wall clock only for live decisions.
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
