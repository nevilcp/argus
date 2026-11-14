"""Generative portfolio construction and allocation manager agent."""

from __future__ import annotations

import logging
from uuid import uuid4

import numpy as np

from argus.config import settings
from argus.infra.clock import now_local
from argus.infra.diagnostics import Diagnostics
from argus.params import AGGREGATOR, PORTFOLIO
from argus.ports import LLMClient
from argus.schemas.prompting import UNTRUSTED_GUARD, schema_block, wrap_untrusted
from argus.schemas.signals import (
    MacroContext,
    PortfolioAllocation,
    PortfolioProposal,
    Signal,
    TickerSnapshot,
)
from argus.seams import default_groq_client
from argus.structured_output import decode_or_degrade

logger = logging.getLogger(__name__)


def half_kelly_weight(
    win_probability: float,
    avg_win_pct: float = PORTFOLIO.kelly_avg_win_pct,
    avg_loss_pct: float = PORTFOLIO.kelly_avg_loss_pct,
    max_position: float = PORTFOLIO.kelly_max_position,
) -> float:
    """Compute a capped Half-Kelly weight from assumed win and loss magnitudes."""
    b = avg_win_pct / avg_loss_pct
    p = win_probability
    q = 1.0 - p

    full_kelly = (b * p - q) / b
    half_kelly = full_kelly / PORTFOLIO.kelly_divisor
    # Floor at 0.0 so negative Kelly (bearish expectation) does not force a 2% allocation.
    return float(np.clip(half_kelly, 0.0, max_position))


def build_signal_table(snapshots: dict[str, TickerSnapshot], macro: MacroContext | None) -> str:
    """Construct a consolidated prompt table mapping approved specialist signal values."""
    lines = []
    if macro is not None:
        vix_display = (
            f"VIX {macro.vix_level:.2f} ({macro.vix_regime.value})"
            if macro.vix_level is not None and macro.vix_regime is not None
            else "VIX unavailable"
        )
        lines.append(f"MACRO: {macro.macro_regime.value} | {vix_display} | {macro.sector_rotation_signal.value}")
    else:
        lines.append("MACRO: unavailable this session (data source failure) — allocate on specialist signals alone")
    lines.append("")

    for ticker, snap in snapshots.items():
        if not snap.risk_approved:
            continue
        risk = snap.risk
        if risk is None:
            raise RuntimeError("Required risk assessment is missing")

        f = snap.fundamental
        t = snap.technical
        s = snap.sentiment
        agg = snap.aggregated

        fsig = f"{f.signal.value}({f.conviction:.2f})" if f else "N/A"
        tsig = f"{t.signal.value}({t.conviction:.2f})" if t else "N/A"
        ssig = f"{s.signal.value}({s.conviction:.2f})" if s else "N/A"
        asig = f"{agg.signal.value}({agg.conviction:.2f})" if agg else "N/A"
        stop_display = f"{risk.stop_loss:.2f}" if isinstance(risk.stop_loss, float) else "N/A"
        var_display = f"{risk.var_99:.2%}" if isinstance(risk.var_99, float) else "N/A"
        beta_display = f"{risk.portfolio_beta:.2f}" if risk.portfolio_beta is not None else "N/A"
        evidence = f"{len(agg.agents_present)}/3" if agg else "0/3"

        line = (
            f"{ticker}: FUND={fsig} TECH={tsig} SENT={ssig} AGG={asig} Evidence={evidence} "
            f"VaR={var_display} Beta={beta_display} Stop={stop_display} Cap={risk.approved_weight:.1%}"
        )
        lines.append(line)

    return "\n".join(lines)


_PROPOSAL_SCHEMA_JSON = schema_block(PortfolioProposal)


SYSTEM_PROMPT = (
    "You are a systematic portfolio construction specialist at a quantitative investment fund. "
    "You receive pre-computed, risk-approved signals from four independent analytical agents. "
    "You do not generate investment advice. All outputs are research inputs requiring human review "
    "before any capital is deployed.\n"
    "\n"
    "EPISTEMIC STANDARD: Allocate only to tickers present in the supplied signal table. "
    "Do not impute signals for absent tickers. "
    "Do not recall historical prices or news. "
    "If signal quality is uniformly poor, hold cash rather than force allocations.\n"
    "\n"
    "ALLOCATION RULES (non-negotiable):\n"
    "  1. Only allocate to tickers listed in the signal table.\n"
    "  2. cash_reserve_pct = 1.0 - sum(all allocation_pct values). "
    "     It is the arithmetic residual, not a free variable.\n"
    "  3. Target equity deployment of deployment_ceiling (see portfolio_context — the achievable "
    "     ceiling given invest_pct and this universe's approved position caps, not raw invest_pct "
    "     itself). Acceptable range: [deployment_ceiling - 0.15, deployment_ceiling]. "
    "     Do not force allocations when dominant signals are BEARISH.\n"
    "  4. allocation_pct is a decimal fraction in [0.0, 0.15]. "
    "     e.g. 10% = 0.10, 15% = 0.15. Never output raw integers like 10 or 7.5.\n"
    "  5. allocation_pct must not exceed the ticker's Cap value from the signal table.\n"
    "  6. Include every approved ticker in the portfolio list, even if allocation_pct = 0.0.\n"
    "  7. Signal priority: BULLISH → allocate up to Cap; NEUTRAL → reduced diversifier; "
    "     BEARISH → 0.0 unless required to reach minimum deployment target.\n"
    "  8. thesis: ≤20 words stating the single primary reason to hold "
    '     (or "Skipped: bearish signal" for zero-allocation entries).\n'
    "  9. advisor_note: 2-4 professional sentences for a non-expert client. "
    "     State the rationale, key risks, and what to monitor. "
    "     For zero-allocation entries, explain why the position was avoided. "
    f"     Maximum {PORTFOLIO.advisor_note_char_limit} characters. Spell out all acronyms on first use.\n"
    "  10. Before returning: verify cash_reserve_pct = 1.0 - sum(allocation_pct), "
    "      all allocation_pct values are decimals in [0.0, 0.15], "
    "      and every approved ticker is present.\n"
    "  11. Evidence is agents_present/3 for that ticker's AGG signal. When Evidence is "
    "      1/3, allocation_pct must not exceed half the ticker's Cap — thin evidence "
    "      warrants a smaller position, not a skip.\n"
    "\n"
    "OUTPUT: Return ONLY a valid JSON object — no preamble, no markdown, no trailing text."
)


class PortfolioManagerAgent:
    """Model portfolio proposals with risk limits enforced by code."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or default_groq_client(
            settings.ARGUS_PORTFOLIO_MODEL,
            PORTFOLIO.llm_temperature,
            PORTFOLIO.llm_max_tokens,
        )
        self._session_count = 0
        self.max_position_pct = settings.MAX_SINGLE_POSITION_PCT

    def allocate(
        self,
        user_profile: dict,
        snapshots: dict[str, TickerSnapshot],
        macro: MacroContext | None,
        cultural_wisdom: list[str] | None = None,
        cultural_warnings: list[str] | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> PortfolioAllocation | None:
        """Aggregate all specialist agent verdicts to construct a consolidated PortfolioAllocation."""
        diagnostics = diagnostics or Diagnostics()

        total_wealth = user_profile.get("total_wealth")
        # Capital base is total wealth; invest_pct (RULE 3) is applied once by the LLM, not pre-scaled here.
        investable = float(total_wealth) if total_wealth is not None else 0.0

        if investable <= 0:
            # Reject nonpositive capital before even the all-cash allocation would fail schema validation.
            diagnostics.record(
                "portfolio_allocation: total_wealth missing or not positive "
                f"({total_wealth!r}); cannot build an allocation",
                "error",
            )
            return None

        approved_tickers = [t for t, s in snapshots.items() if s.risk_approved]

        if not approved_tickers:
            logger.debug("No approved tickers. Reverting to all-cash.")
            return self._all_cash_allocation(investable)

        # invest_pct can be unreachable for a small universe; state the achievable ceiling instead.
        invest_pct = float(user_profile.get("invest_pct", 1.0))
        deployment_ceiling = min(invest_pct, self._approved_cap_sum(approved_tickers, snapshots))

        prompt = self._build_allocation_prompt(
            investable=investable,
            invest_pct=invest_pct,
            deployment_ceiling=deployment_ceiling,
            user_profile=user_profile,
            signal_table=build_signal_table(snapshots, macro),
            kelly_suggestions=self._kelly_anchors(approved_tickers, snapshots),
            cultural_wisdom=cultural_wisdom,
            cultural_warnings=cultural_warnings,
        )

        proposal = decode_or_degrade(
            self.llm_client,
            SYSTEM_PROMPT,
            prompt,
            PortfolioProposal,
            repair=False,
            diagnostics=diagnostics,
            label="portfolio_allocation",
            # No per-ticker fallback here; propagate so the API layer surfaces a 429/503, not a generic 500.
            reraise_governor=True,
            decode_level="error",
            api_error_text="PortfolioManagerAgent API error",
        )
        if proposal is None:
            return None

        allocation = self._build_allocation(proposal, snapshots, investable, diagnostics)
        if allocation is None:
            return None

        self._session_count += 1
        return allocation

    @staticmethod
    def _approved_cap_sum(approved_tickers: list[str], snapshots: dict[str, TickerSnapshot]) -> float:
        """Sum the risk-approved weight caps of the approved tickers."""
        total = 0.0
        for t in approved_tickers:
            risk = snapshots[t].risk
            if risk is None:
                raise RuntimeError("Required risk assessment is missing")
            total += risk.approved_weight
        return total

    def _build_allocation(
        self,
        proposal: PortfolioProposal,
        snapshots: dict[str, TickerSnapshot],
        investable: float,
        diagnostics: Diagnostics,
    ) -> PortfolioAllocation | None:
        """Enforce risk limits on the LLM proposal and build the validated allocation."""
        stage = "response_enforcement"
        try:
            enforced_portfolio = self._enforce_risk(proposal, snapshots, investable, diagnostics)

            total_equity = sum(p["allocation_pct"] for p in enforced_portfolio)
            cash_reserve_pct = round(max(PORTFOLIO.cash_reserve_floor_pct, 1.0 - total_equity), 6)
            if abs(cash_reserve_pct - proposal.cash_reserve_pct) > 1e-6:
                diagnostics.record(
                    "portfolio_allocation: cash_reserve_pct forced to residual "
                    f"{cash_reserve_pct:.4f} (LLM proposed {proposal.cash_reserve_pct:.4f})"
                )

            data = {
                "session_id": str(uuid4()),
                "user_investable_capital": investable,
                "portfolio": enforced_portfolio,
                "cash_reserve_pct": cash_reserve_pct,
                "rebalance_trigger": proposal.rebalance_trigger,
                "timestamp": now_local().isoformat(),
            }

            stage = "schema_validation"
            return PortfolioAllocation.model_validate(data)
        except Exception as e:
            diagnostics.record(f"portfolio_allocation: PortfolioManagerAgent failed at {stage}: {e}", "error")
            return None

    def _kelly_anchors(self, approved_tickers: list[str], snapshots: dict[str, TickerSnapshot]) -> dict[str, float]:
        """Compute the Half-Kelly weight to anchor each approved ticker's sizing."""
        reliability_n: dict[str, int] = {}
        for ticker in approved_tickers:
            agg = snapshots[ticker].aggregated
            if agg and agg.reliability_n:
                reliability_n = agg.reliability_n
                break
        if not any(n > 0 for n in reliability_n.values()):
            return {}

        kelly_suggestions = {}
        for ticker in approved_tickers:
            agg = snapshots[ticker].aggregated
            risk_signal = snapshots[ticker].risk
            max_pos = risk_signal.approved_weight if risk_signal else self.max_position_pct
            if not agg or agg.signal == Signal.BEARISH or not agg.weighted_votes:
                continue
            driver = max(agg.weighted_votes, key=lambda k: agg.weighted_votes[k])
            win_prob = agg.reliability.get(driver, AGGREGATOR.neutral_reliability)
            kelly_suggestions[ticker] = half_kelly_weight(win_probability=win_prob, max_position=max_pos)
        return kelly_suggestions

    def _build_allocation_prompt(
        self,
        investable: float,
        invest_pct: float,
        deployment_ceiling: float,
        user_profile: dict,
        signal_table: str,
        kelly_suggestions: dict[str, float],
        cultural_wisdom: list[str] | None,
        cultural_warnings: list[str] | None,
    ) -> str:
        """Assemble the human-turn allocation prompt from this session's state."""
        wisdom_text = "\n".join(f"- {w}" for w in (cultural_wisdom or [])[:3])
        warnings_text = "\n".join(f"- {w}" for w in (cultural_warnings or [])[:3])
        if kelly_suggestions:
            kelly_text = "\n".join(f"{t}: half-Kelly suggests {w:.1%}" for t, w in kelly_suggestions.items())
            kelly_block = (
                "HALF-KELLY ANCHORS (mathematical starting point for NEUTRAL/BULLISH sizing; "
                "override only with explicit signal justification):\n"
                f"{kelly_text}\n"
            )
        else:
            kelly_block = (
                "HALF-KELLY ANCHORS: omitted — no agent has recorded outcome history yet. "
                "Size NEUTRAL/BULLISH positions from signal strength and Cap alone.\n"
            )

        floor_pct = max(0.0, deployment_ceiling - PORTFOLIO.equity_floor_adjustment)
        context = wrap_untrusted(
            "portfolio_context",
            f"  capital_usd: ${investable:,.0f}\n"
            f"  invest_pct: {invest_pct:.0%}\n"
            f"  deployment_ceiling: {deployment_ceiling:.0%} (min of invest_pct and the sum of "
            f"every approved ticker's Cap below — the most equity this universe can absorb "
            f"under position caps)\n"
            f"  risk_tolerance: {user_profile.get('risk_tolerance', 'MODERATE')}\n"
            f"  minimum_equity_usd: ${investable * floor_pct:,.0f} "
            f"({floor_pct:.0%} of investable capital)",
            {"session": self._session_count, "as_of": "intraday"},
        )
        table = wrap_untrusted(
            "signal_table",
            "Columns: FUND=fundamental_signal(conviction) TECH=technical_signal(conviction) "
            "SENT=sentiment_signal(conviction) AGG=aggregated_signal(conviction) "
            "Evidence=agents_present/3 (specialists that actually voted into AGG) "
            "VaR=99%_VaR Beta=portfolio_beta Stop=stop_loss_level Cap=risk_engine_ceiling\n"
            "AGG conviction is scaled against the full three-agent vote mass, not just the "
            "specialists present — it falls with disagreement or missing coverage rather than "
            "saturating near 1.0 whenever the agents who voted happen to agree.\n"
            f"{signal_table}",
        )
        return (
            f"{context}\n"
            f"{UNTRUSTED_GUARD}\n"
            "\n"
            f"{table}\n"
            "\n"
            f"{kelly_block}"
            "\n"
            "CULTURAL WISDOM (from prior sessions with similar macro regime):\n"
            f"{wisdom_text if wisdom_text else 'None available.'}\n"
            "\n"
            "CULTURAL WARNINGS (failed trades in this macro regime — avoid repeating these patterns):\n"
            f"{warnings_text if warnings_text else 'None available.'}\n"
            "\n"
            "Step 1 — For each approved ticker: compare AGG signal against Half-Kelly anchor and Cap.\n"
            "Step 2 — Assign allocation_pct respecting all 11 ALLOCATION RULES from the system prompt.\n"
            "Step 3 — Compute cash_reserve_pct = 1.0 - sum(allocation_pct). Do not set it independently.\n"
            "Step 4 — Verify: (1) all allocation_pct values are decimals in [0.0, Cap], "
            "(2) cash_reserve_pct + sum(allocation_pct) = 1.0, "
            "(3) every approved ticker appears in portfolio.\n"
            "\n"
            "Output JSON schema exactly:\n"
            f"{_PROPOSAL_SCHEMA_JSON}"
        )

    def _enforce_risk(
        self,
        proposal: PortfolioProposal,
        snapshots: dict[str, TickerSnapshot],
        investable: float,
        diagnostics: Diagnostics,
    ) -> list[dict]:
        """Clamp, zero, or drop each proposed position against its risk verdict."""
        enforced_portfolio = []
        for pos in proposal.portfolio:
            ticker = pos.ticker
            if ticker not in snapshots:
                diagnostics.record(f"portfolio_allocation: dropped {ticker} (not in this session's signal universe)")
                continue

            snap = snapshots[ticker]
            risk = snap.risk
            proposed_pct = pos.allocation_pct
            pos_data = pos.model_dump()
            if not snap.risk_approved:
                if proposed_pct:
                    verdict = risk.verdict.value if risk else "MISSING"
                    diagnostics.record(f"portfolio_allocation: zeroed {ticker} allocation (risk verdict {verdict})")
                pos_data["allocation_pct"] = 0.0
            else:
                if risk is None:
                    raise RuntimeError("Required risk assessment is missing")
                cap = risk.approved_weight
                if proposed_pct > cap + 1e-9:
                    diagnostics.record(
                        f"portfolio_allocation: clamped {ticker} allocation from "
                        f"{proposed_pct:.4f} to risk cap {cap:.4f}"
                    )
                    pos_data["allocation_pct"] = cap

            pos_data["allocation_usd"] = round(investable * pos_data["allocation_pct"], 2)
            pos_data["thesis"] = pos_data["thesis"][: PORTFOLIO.thesis_char_limit]
            if pos_data["advisor_note"] is not None:
                pos_data["advisor_note"] = pos_data["advisor_note"][: PORTFOLIO.advisor_note_char_limit]
            if risk is not None and risk.stop_loss is not None:
                pos_data["stop_loss"] = float(risk.stop_loss)

            enforced_portfolio.append(pos_data)

        return enforced_portfolio

    def _all_cash_allocation(self, investable: float) -> PortfolioAllocation:
        """Build an all-cash allocation when no ticker is risk-approved."""
        return PortfolioAllocation(
            session_id=str(uuid4()),
            user_investable_capital=investable,
            portfolio=[],
            cash_reserve_pct=1.0,
            rebalance_trigger="MONTHLY",
            timestamp=now_local(),
        )
