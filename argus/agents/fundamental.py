"""
argus/agents/fundamental.py
===========================
Generative Fundamental Analyst using Gemini 3.1 Flash Lite.

Analyses pre-fetched fundamental ratios and outputs a strict JSON signal.
Implements ticker anonymisation to prevent LLM parametric memory contamination
during backtesting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from argus.config import settings
from argus.data.fetchers import fetch_fundamentals
from argus.orchestration.governor import governor
from argus.schemas.signals import FundamentalSignal, Signal

logger = logging.getLogger("argus.fundamental")

# ──────────────────────────────────────────────────────────────────────────────
# Anonymiser and Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

def anonymize_ticker(ticker: str, session_seed: int) -> str:
    """
    Generate a deterministic anonymous code like 'COMP_A7' to prevent
    parametric memory contamination during backtests.
    """
    h = hashlib.md5(f"{ticker}{session_seed}".encode()).hexdigest()[:4].upper()
    return f"COMP_{h}"

def build_compact_prompt(ticker: str, pit_data: dict, anon_id: Optional[str] = None) -> str:
    """Build a token-efficient prompt for the LLM."""
    subject = anon_id if anon_id else ticker
    fundamentals = pit_data.get("fundamentals", {})
    as_of = pit_data.get("as_of_date", "")

    # For the prompt, we format the dictionary into a compact bullet list
    metrics_str = "\n".join(f"- {k}: {v}" for k, v in fundamentals.items())

    # Sector context handling
    sector = fundamentals.get("sector", "Unknown")
    sector_str = f"[{sector} sector company]" if anon_id else f"{ticker} operates in the {sector} sector."

    # Using dummy median P/E as it isn't dynamically fetched per industry yet
    industry_median_pe = 20.0

    prompt = f"""
Analyze the following financial metrics for {subject}.
{sector_str}
Industry Median P/E context: ~{industry_median_pe}
Data as of: {as_of}

Metrics:
{metrics_str}

Output ONLY a valid JSON object matching the FundamentalSignal schema.
Fields required: 
"signal" (string: BULLISH, BEARISH, NEUTRAL), 
"conviction" (float 0.0 to 1.0), 
"pe_ttm" (float or null), 
"revenue_growth_yoy" (float or null), 
"operating_margin" (float or null),
"fcf_yield" (float or null), 
"debt_to_equity" (float or null), 
"roic" (float or null), 
"moat_score" (int 1 to 10), 
"reasoning" (string).

reasoning must be under 80 words. No Markdown. No preamble.
"""
    return prompt.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Weekly Cache
# ──────────────────────────────────────────────────────────────────────────────

class FundamentalCache:
    """7-day local cache to prevent redundant LLM calls for static fundamentals."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[FundamentalSignal, datetime]] = {}
        self._ttl_days = 7

    def get(self, ticker: str) -> Optional[FundamentalSignal]:
        """Return cached signal if fresh, else None."""
        if ticker in self._cache:
            signal, cached_at = self._cache[ticker]
            if (datetime.now() - cached_at).days < self._ttl_days:
                return signal
        return None

    def set(self, ticker: str, signal: FundamentalSignal) -> None:
        """Store signal in cache with current timestamp."""
        self._cache[ticker] = (signal, datetime.now())

    def is_stale(self, ticker: str) -> bool:
        """Return True if the ticker is missing from cache or older than 7 days."""
        return self.get(ticker) is None


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent Class
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a pure Fundamental Analyst. Analyze ONLY the 
structured financial data provided. Do NOT use your parametric memory of this 
company's stock price history or past news events. 

ANTI-CONFORMITY: Your conviction score must reflect YOUR analysis of the data, 
not what you think is a "safe" answer. A moat_score of 9+ requires clear 
evidence of durable competitive advantage from the ratios provided.

Output ONLY a valid JSON object. No Markdown code blocks. No explanation text.
If any field cannot be determined from the data, output null for that field."""

class FundamentalAgent:
    """Generative Fundamental Analyst powered by Gemini 3.1 Flash Lite."""

    def __init__(self) -> None:
        api_key = settings.google_ai_api_key or "DUMMY_KEY_FOR_TESTING"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.1,
            max_output_tokens=600,
            google_api_key=api_key
        )
        self.cache = FundamentalCache()

    def analyze(
        self, 
        ticker: str, 
        backtest_mode: bool = False,
        session_seed: Optional[int] = None
    ) -> FundamentalSignal:
        """Analyze fundamentals and return a fully validated FundamentalSignal."""
        
        # ── 1. Check Cache ──
        if not self.cache.is_stale(ticker):
            cached = self.cache.get(ticker)
            if cached:
                logger.debug("FundamentalAgent.analyze: Cache hit for %s", ticker)
                return cached

        # ── 2. Governor Capacity Check ──
        if governor.get_remaining_capacity("gemini-3.1-flash-lite") < 5:
            logger.warning("[Fundamental] Low capacity for gemini-3.1-flash-lite, using NEUTRAL fallback")
            return self._neutral_fallback(ticker)

        # ── 3. Fetch Data ──
        try:
            fundamentals = fetch_fundamentals(ticker)
        except Exception as e:
            logger.warning("[Fundamental] Failed to fetch fundamentals for %s: %s", ticker, e)
            return self._neutral_fallback(ticker)

        pit_data = {
            "fundamentals": fundamentals,
            "as_of_date": date.today().isoformat(),
        }

        # ── 4. Build Prompt ──
        anon_id = anonymize_ticker(ticker, session_seed) if backtest_mode and session_seed else None
        prompt = build_compact_prompt(ticker, pit_data, anon_id)
        estimated_tokens = len(prompt.split()) * 1.3

        # ── 5. Governor Wait ──
        governor.wait_if_needed("gemini-3.1-flash-lite", int(estimated_tokens + 600))

        # ── 6. LLM Call with Retry ──
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
                
                # Strip markdown code blocks
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
                
                # Enforce system fields
                data["ticker"] = ticker
                data["anon_id"] = anon_id
                data["temporal_horizon"] = "LONG_TERM"
                data["data_as_of_date"] = pit_data["as_of_date"]
                data["timestamp"] = datetime.now().isoformat()
                data["api_calls_used"] = 1
                
                # Truncate reasoning to comply with Pydantic 400-char limit
                if "reasoning" in data and isinstance(data["reasoning"], str):
                    data["reasoning"] = data["reasoning"][:400]
                
                signal = FundamentalSignal.model_validate(data)
                self.cache.set(ticker, signal)
                logger.info("[Fundamental] Analysis complete for %s -> %s", ticker, signal.signal.value)
                return signal
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning("[Fundamental] Attempt %d parse error for %s: %s", attempt + 1, ticker, e)
                if attempt == 2:
                    return self._neutral_fallback(ticker)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error("[Fundamental] Attempt %d API error for %s: %s", attempt + 1, ticker, e)
                if attempt == 2:
                    return self._neutral_fallback(ticker)
                time.sleep(2 ** attempt)

        return self._neutral_fallback(ticker)

    def _neutral_fallback(self, ticker: str) -> FundamentalSignal:
        """Return a safe NEUTRAL signal when API or parsing fails."""
        return FundamentalSignal(
            ticker=ticker,
            signal=Signal.NEUTRAL,
            conviction=0.35,
            temporal_horizon="LONG_TERM",
            reasoning="Insufficient data or API capacity limit reached.",
            pe_ttm=None,
            revenue_growth_yoy=None,
            operating_margin=None,
            fcf_yield=None,
            debt_to_equity=None,
            roic=None,
            moat_score=5,
            data_as_of_date=date.today().isoformat(),
            api_calls_used=0,
            timestamp=datetime.now()
        )

    def batch_analyze(self, tickers: list[str], backtest_mode: bool = False, session_seed: Optional[int] = None) -> dict[str, FundamentalSignal]:
        """Analyze multiple tickers sequentially to respect RPM limits."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze(ticker, backtest_mode, session_seed)
        return results
