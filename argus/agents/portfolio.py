"""
argus/agents/portfolio.py
=========================
Final Portfolio Manager Agent.
Uses Groq Llama 3.3 70B Versatile to synthesize pre-computed quantitative and generative
signals into optimal allocations, bounded by Half-Kelly sizing logic.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from argus.config import settings
from argus.orchestration.governor import governor
from argus.schemas.signals import MacroContext, PortfolioAllocation, RiskVerdict

logger = logging.getLogger("argus.portfolio")

# ──────────────────────────────────────────────────────────────────────────────
# Half-Kelly Position Sizer
# ──────────────────────────────────────────────────────────────────────────────

def half_kelly_weight(
    win_probability: float,
    avg_win_pct: float = 0.08,     # Default: 8% avg win
    avg_loss_pct: float = 0.04,    # Default: 4% avg loss (2:1 reward/risk)
    max_position: float = 0.15     # Hard cap from config
) -> float:
    """
    Computes a conservative Half-Kelly suggested weight given a win probability.
    """
    b = avg_win_pct / avg_loss_pct  # Reward-to-risk ratio
    p = win_probability
    q = 1.0 - p
    
    if b <= 0:
        return 0.02
        
    full_kelly = (b * p - q) / b
    half_kelly = full_kelly / 2.0
    return float(np.clip(half_kelly, 0.02, max_position))

# ──────────────────────────────────────────────────────────────────────────────
# Signal Table Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_signal_table(all_signals: dict[str, dict], macro: MacroContext) -> str:
    """
    Builds a compact prompt string representing all APPROVED asset signals.
    """
    lines = []
    lines.append(
        f"MACRO: {macro.macro_regime.value} | "
        f"VIX {macro.vix_level:.2f} ({macro.vix_regime}) | "
        f"{macro.sector_rotation_signal}"
    )
    lines.append("")
    
    for ticker, sigs in all_signals.items():
        risk = sigs.get("risk")
        if not risk or risk.verdict != RiskVerdict.APPROVE:
            continue
            
        f = sigs.get("fundamental")
        t = sigs.get("technical")
        s = sigs.get("sentiment")
        agg = sigs.get("aggregated")
        
        # Format strings with safety fallbacks
        fsig = f"{f.signal.value}({f.conviction:.2f})" if f else "N/A"
        tsig = f"{t.signal.value}({t.conviction:.2f})" if t else "N/A"
        ssig = f"{s.signal.value}({s.conviction:.2f})" if s else "N/A"
        asig = f"{agg.signal.value}({agg.conviction:.2f})" if agg else "N/A"
        stop = risk.stop_losses.get(ticker, "N/A")
        if isinstance(stop, float):
            stop = f"{stop:.2f}"
            
        line = (
            f"{ticker}: FUND={fsig} TECH={tsig} SENT={ssig} AGG={asig} "
            f"VaR={risk.var_99:.2%} Beta={risk.portfolio_beta:.2f} Stop={stop}"
        )
        lines.append(line)
        
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent Class
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a portfolio allocation specialist. You receive 
pre-computed, risk-approved signals from 4 analytical agents. Your sole task 
is to allocate capital optimally.

STRICT RULES:
1. Only allocate to tickers present in the signal table.
2. Total allocation_pct values must sum to exactly (1 - cash_reserve_pct).
3. Set cash_reserve_pct between 0.10 and 0.30 based on macro regime.
4. Each allocation_pct must not exceed 0.15 (enforced externally too).
5. thesis must be under 20 words describing the single best reason to hold.
6. Output ONLY valid JSON. No preamble, no explanation."""

class PortfolioManagerAgent:
    def __init__(self) -> None:
        api_key = settings.groq_api_key or "DUMMY_KEY_FOR_TESTING"
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.05,
            max_tokens=1200,
            groq_api_key=api_key,
        )
        self._session_count = 0
        self.max_position_pct = getattr(settings, "MAX_SINGLE_POSITION_PCT", 0.15)

    def allocate(
        self,
        user_profile: dict,
        all_signals: dict[str, dict],
        macro: MacroContext,
        cultural_wisdom: Optional[list[str]] = None
    ) -> PortfolioAllocation:
        """
        Synthesize all agent outputs into a final PortfolioAllocation.
        """
        investable = float(user_profile.get("total_wealth", 0.0)) * float(user_profile.get("invest_pct", 1.0))
        
        approved_tickers = [
            t for t, s in all_signals.items() 
            if s.get("risk") and s["risk"].verdict == RiskVerdict.APPROVE
        ]
        
        if not approved_tickers:
            logger.info("[Portfolio] No approved tickers. Reverting to all-cash.")
            return self._all_cash_allocation(investable)

        # Compute Half-Kelly suggested weights as context for LLM
        kelly_suggestions = {}
        for ticker in approved_tickers:
            agg = all_signals[ticker].get("aggregated")
            if agg:
                kelly_suggestions[ticker] = half_kelly_weight(
                    win_probability=agg.conviction,
                    max_position=self.max_position_pct
                )

        # Build prompt
        signal_table = build_signal_table(all_signals, macro)
        wisdom_text = "\n".join(f"- {w}" for w in (cultural_wisdom or [])[:3])
        kelly_text = "\n".join(f"{t}: half-Kelly suggests {w:.1%}" 
                               for t, w in kelly_suggestions.items())
        
        prompt = f"""PORTFOLIO ALLOCATION — Session #{self._session_count}

User capital: ${investable:,.0f} | Risk tolerance: {user_profile.get('risk_tolerance', 'MODERATE')}

SIGNALS:
{signal_table}

HALF-KELLY SIZE SUGGESTIONS (use as starting point):
{kelly_text}

CULTURAL WISDOM (from past successful trades in similar regime):
{wisdom_text if wisdom_text else "None available yet."}

Output JSON schema exactly:
{{"portfolio":[{{"ticker":"","allocation_pct":0.0,"allocation_usd":0.0,"stop_loss":0.0,"target_price":null,"thesis":"<20 words","composite_conviction":0.0,"time_horizon":"3-6 months"}}],"cash_reserve_pct":0.0,"expected_sharpe":null,"rebalance_trigger":"MONTHLY"}}
"""
        estimated_tokens = int(len(prompt.split()) * 1.3) + 1200
        governor.wait_if_needed("llama-3.3-70b-versatile", estimated_tokens)

        for attempt in range(3):
            try:
                response = self.llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ])
                raw = response.content
                if isinstance(raw, list):
                    raw = "".join(item.get("text", "") for item in raw if isinstance(item, dict) and item.get("type") == "text")
                elif not isinstance(raw, str):
                    raw = str(raw)
                
                raw = raw.strip()
                if raw.startswith("```"):
                    parts = raw.split("```")
                    if len(parts) >= 3:
                        raw = parts[1]
                    else:
                        raw = parts[-1]
                    raw = raw.strip()
                    if raw.startswith("json"):
                        raw = raw[4:].strip()
                
                data = json.loads(raw)
                
                # Fill allocation_usd from allocation_pct and override stop_loss
                for pos in data.get("portfolio", []):
                    pos["allocation_usd"] = round(investable * pos["allocation_pct"], 2)
                    ticker = pos["ticker"]
                    if "thesis" in pos and isinstance(pos["thesis"], str):
                        pos["thesis"] = pos["thesis"][:120]
                    if ticker in all_signals and all_signals[ticker].get("risk"):
                        engine_stop = all_signals[ticker]["risk"].stop_losses.get(ticker)
                        if engine_stop is not None:
                            pos["stop_loss"] = float(engine_stop)
                
                data["session_id"] = str(uuid4())
                data["user_investable_capital"] = investable
                data["timestamp"] = datetime.now().isoformat()
                
                allocation = PortfolioAllocation.model_validate(data)
                self._session_count += 1
                return allocation
                
            except Exception as e:
                logger.warning("[Portfolio] Attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    return self._all_cash_allocation(investable)
                time.sleep(2 ** attempt)

        return self._all_cash_allocation(investable)

    def _all_cash_allocation(self, investable: float) -> PortfolioAllocation:
        """Fallback when no positions pass risk or API repeatedly fails."""
        return PortfolioAllocation(
            session_id=str(uuid4()),
            user_investable_capital=investable,
            portfolio=[],
            cash_reserve_pct=1.0,
            expected_sharpe=0.0,
            rebalance_trigger="MONTHLY",
            timestamp=datetime.now()
        )
