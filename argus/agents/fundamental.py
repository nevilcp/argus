"""Generative fundamental analysis agent powered by Groq.

Responsibilities:
  - Ingest and evaluate financial ratio payloads (valuation multiples, capital efficiency, growth rates)
  - Generate structured directional signals via LLM analysis
  - Apply deterministic ticker anonymization to isolate reasoning models from parametric memory
    contamination during backtests

Not responsible for:
  - Fetching raw price data (see data/fetchers.py)
  - Risk assessment (see agents/risk.py)
  - Portfolio allocation (see agents/portfolio.py)

Dependencies:
  - langchain_groq
  - GROQ_API_KEY env var must be set (see .env.example)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from argus.config import settings
from argus.data.cache import TTLCache
from argus.infra.clock import now_local
from argus.infra.diagnostics import Diagnostics
from argus.params import FUNDAMENTAL, STRUCTURED_OUTPUT, SYSTEM
from argus.ports import LLMClient, MarketDataProvider
from argus.schemas.prompting import UNTRUSTED_GUARD, field_list, wrap_untrusted
from argus.schemas.signals import UNKNOWN_SECTOR, FundamentalSignal, FundamentalVerdict
from argus.seams import LiveMarketDataProvider, default_groq_client
from argus.structured_output import decode_or_degrade

logger = logging.getLogger(__name__)

# Approximate sector-median P/E anchors for the LLM prompt; rough averages, not precise thresholds.
# Provenance: ARBITRARY — no cited source or as-of date, so treat them as rough anchors.
_SECTOR_PE_MEDIANS: dict[str, float] = {
    "Technology": 32.0,
    "Consumer Cyclical": 24.0,
    "Communication Services": 22.0,
    "Healthcare": 23.0,
    "Financial Services": 15.0,
    "Industrials": 20.0,
    "Consumer Defensive": 22.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 35.0,
    "Basic Materials": 16.0,
}

# Measured from market_data.fundamentals(); LLM only supplies signal/conviction/moat/reasoning.
_MEASURED_FUNDAMENTAL_FIELDS = (
    "sector",
    "industry",
    "marketCap",
    "pe_ttm",
    "p_fcf",
    "revenue_growth_yoy",
    "operating_margin",
    "net_margin",
    "fcf_yield",
    "debt_to_equity",
    "current_ratio",
    "roe",
    "roic",
)


def _use_backtest_seed(backtest_mode: bool, session_seed: int | None) -> bool:
    """Decides whether a session should anonymize and derive as_of_date from session_seed.

    Args:
        backtest_mode: Whether the caller requested backtest anonymization.
        session_seed: Session seed, if any. ``0`` is a legal seed — checked via
            ``is not None``, not truthiness, since ``bool(0)`` is False.

    Returns:
        True if backtest anonymization/date-seeding should apply.
    """
    return backtest_mode and session_seed is not None


def _session_seed_to_date(session_seed: int) -> date:
    """Parses a session_seed integer date stamp into a calendar date.

    Args:
        session_seed: Integer date stamp (e.g. 20240115), as documented on
            ``anonymize_ticker``.

    Returns:
        The corresponding calendar date.
    """
    return datetime.strptime(str(session_seed), "%Y%m%d").date()  # noqa: DTZ007


def anonymize_ticker(ticker: str, session_seed: int) -> str:
    """Generates a deterministic hash-based identifier to mask real tickers during backtesting.

    Uses MD5 (not cryptographic — just for deterministic obfuscation) to produce
    a short code that is stable per ticker+session combination, enabling the LLM
    to reason about metrics without accessing parametric memory of the real company.

    Args:
        ticker: Real equity ticker symbol (e.g. 'AAPL').
        session_seed: Integer date stamp (e.g. 20240115) that scopes anonymization
            to a specific simulation date.

    Returns:
        Anonymized identifier string in the format ``COMP_XXXX``.
    """
    h = hashlib.md5(f"{ticker}{session_seed}".encode()).hexdigest()[:4].upper()
    return f"COMP_{h}"


def build_compact_prompt(ticker: str, pit_data: dict, anon_id: str | None = None) -> str:
    """Constructs a structured, token-minimized evaluation prompt with metric unit classifications.

    Formats metric values with their units and context thresholds so the LLM can
    interpret raw numbers without needing domain knowledge embedded in the question.

    Args:
        ticker: Real equity ticker symbol; used as subject when not anonymized.
        pit_data: Point-in-time dict with keys ``fundamentals`` and ``as_of_date``.
        anon_id: If set, replaces the ticker in the prompt for backtest anonymization.

    Returns:
        Stripped prompt string ready for LLM invocation.
    """
    subject = anon_id if anon_id else ticker
    fundamentals = pit_data.get("fundamentals", {})
    as_of = pit_data.get("as_of_date", "")

    sector = fundamentals.get("sector", UNKNOWN_SECTOR)
    industry_median_pe = _SECTOR_PE_MEDIANS.get(sector, FUNDAMENTAL.default_sector_pe_median)

    METRIC_LABELS = {
        "pe_ttm": ("P/E Ratio", f"x earnings   [industry median ~{industry_median_pe:.1f}x]"),
        "revenue_growth_yoy": ("Revenue Growth YoY", "decimal  (0.12 = 12%)"),
        "operating_margin": ("Operating Margin", "decimal  (0.32 = 32%)"),
        "net_margin": ("Net Margin", "decimal  (0.20 = 20%)"),
        "fcf_yield": ("FCF Yield", "decimal  (0.03 = 3%)   [higher = cheaper valuation]"),
        "debt_to_equity": ("Debt/Equity Ratio", "ratio    (0.80 = 0.80x; >2.0x = high leverage)"),
        "current_ratio": ("Current Ratio", "ratio    (1.5 = 1.5x;  <1.0 = liquidity risk)"),
        "roe": ("Return on Equity", "decimal  (0.35 = 35%)"),
        "roic": ("ROIC (proxy)", "decimal  (0.20 = 20%;  >0.15 = strong)"),
        "p_fcf": ("Price/FCF Multiple", "x FCF    [lower = cheaper]"),
        "marketCap": ("Market Cap", "USD"),
    }

    lines = []
    for k, v in fundamentals.items():
        if k in ("sector", "industry", "as_of_date"):
            continue
        if k in METRIC_LABELS:
            label, unit_hint = METRIC_LABELS[k]
            lines.append(f"- {label}: {v}  [{unit_hint}]")
        else:
            lines.append(f"- {k}: {v}")
    metrics_str = "\n".join(lines)

    prompt = (
        f"{wrap_untrusted('fundamental_data', metrics_str, {'ticker': subject, 'as_of': as_of, 'sector': sector})}\n"
        "\n"
        "You are reasoning as of the as_of date above. Do not use parametric memory of this company.\n"
        f"{UNTRUSTED_GUARD}\n"
        "\n"
        "Step 1 — Recall domain rules:\n"
        "  • pe_ttm > 40x with FCF Yield < 0.02 → elevated valuation risk; moat must be demonstrated.\n"
        "  • ROIC > 0.15 consistently signals durable competitive advantage.\n"
        "  • Debt/Equity > 2.0x AND Current Ratio < 1.0 → financial distress flag.\n"
        "  • Revenue Growth YoY > 0.20 with Operating Margin > 0.20 → high-quality growth.\n"
        "  • moat_score of 8+ requires at least two structural advantages visible in the ratios.\n"
        "\n"
        "Step 2 — Assess each metric against its domain threshold.\n"
        "Step 3 — Derive signal (BULLISH / BEARISH / NEUTRAL) and conviction [0.0, 1.0].\n"
        "Step 4 — Score moat_score [1, 10]; cite the ratio evidence that justifies the score.\n"
        "\n"
        "Before returning: verify (1) all ratios are internally consistent, "
        "(2) conviction reflects data quality not assumed reputation, "
        "(3) null fields do not drive the primary signal.\n"
        "\n"
        "Output ONLY a valid JSON object — no markdown, no preamble, no trailing text.\n"
        "Fields required: " + field_list(FundamentalVerdict) + "."
    )
    return prompt.strip()


SYSTEM_PROMPT = (
    "You are a senior fundamental analyst at a systematic equity fund. "
    "You analyze ONLY the structured financial data provided in each request. "
    "You do not use parametric memory of the company's stock price history, media coverage, or past events. "
    "You do not generate investment advice. All outputs are research inputs requiring human review.\n"
    "\n"
    "EPISTEMIC STANDARD: Conviction must reflect the supplied data exclusively. "
    "If a field is null, do not impute a value. "
    "If fewer than four metrics are available, cap conviction at 0.50.\n"
    "\n"
    "ANTI-CONFORMITY: moat_score ≥ 8 requires at least two structural advantages "
    "explicitly evidenced by the provided ratios (e.g. ROIC > 0.15 + Operating Margin > 0.25). "
    "Do not assign high moat_score based on assumed brand reputation.\n"
    "\n"
    "DATA SCHEMA — all fields use the following units and thresholds:\n"
    "  pe_ttm              : raw multiple  (e.g. 25.0 = 25x earnings; industry median ~20x)\n"
    "  revenue_growth_yoy  : decimal       (e.g. 0.12 = 12% YoY growth)\n"
    "  operating_margin    : decimal       (e.g. 0.32 = 32%; threshold for 'strong': > 0.20)\n"
    "  net_margin          : decimal       (e.g. 0.20 = 20% net margin)\n"
    "  fcf_yield           : decimal       (e.g. 0.03 = 3%; higher = cheaper valuation)\n"
    "  debt_to_equity      : ratio         (e.g. 0.80 = 0.80x leverage; > 2.0x = high risk)\n"
    "  current_ratio       : ratio         (e.g. 1.5 = 1.5x liquidity; < 1.0 = distress signal)\n"
    "  roe                 : decimal       (e.g. 0.35 = 35% return on equity)\n"
    "  roic                : decimal       (e.g. 0.20 = 20% return on invested capital; > 0.15 = strong)\n"
    "  p_fcf               : raw multiple  (e.g. 30.0 = 30x free cash flow; lower = cheaper)\n"
    "  marketCap           : USD absolute dollar value\n"
    "\n"
    "DOMAIN RULES (apply before deriving signal):\n"
    "  1. pe_ttm > 40x with fcf_yield < 0.02 → elevated valuation risk.\n"
    "  2. roic > 0.15 → evidence of durable competitive advantage.\n"
    "  3. debt_to_equity > 2.0 AND current_ratio < 1.0 → financial distress flag.\n"
    "  4. revenue_growth_yoy > 0.20 AND operating_margin > 0.20 → high-quality growth.\n"
    "\n"
    "OUTPUT: Return ONLY a valid JSON object — no markdown, no preamble, no trailing text. "
    "For any field that cannot be determined from the supplied data, output null."
)


def _signal_payload(
    verdict: FundamentalVerdict, ticker: str, pit_data: dict[str, Any]
) -> dict[str, Any]:
    """Merges an LLM verdict with the measured ratios into a FundamentalSignal payload.

    Args:
        verdict: Decoded LLM verdict, supplying only signal/conviction/moat_score/reasoning.
        ticker: Equity ticker symbol.
        pit_data: Point-in-time dict with keys ``fundamentals`` and ``as_of_date``.

    Returns:
        Dict ready for ``FundamentalSignal.model_validate``.
    """
    data: dict[str, Any] = verdict.model_dump()
    data["ticker"] = ticker
    data["data_as_of_date"] = pit_data["as_of_date"]
    data["timestamp"] = now_local().isoformat()
    data["api_calls_used"] = 1

    # Measured ratios come from the fetched payload, never the LLM echo.
    f = pit_data.get("fundamentals", {})
    for key in _MEASURED_FUNDAMENTAL_FIELDS:
        data[key] = f.get(key)
    data["sector"] = data["sector"] or UNKNOWN_SECTOR
    data["industry"] = data["industry"] or "Unknown"

    data["reasoning"] = data["reasoning"][: FUNDAMENTAL.reasoning_char_limit]
    return data


class FundamentalAgent:
    """Agent coordinating LLM valuation and economic moat auditing.

    Uses an injected LLMClient (Groq by default) to construct structured
    investment theses from ratio payloads fetched via an injected
    MarketDataProvider, applying local caches, the client's remaining-capacity
    reserve, and the shared structured-output decoder's parse/validate/retry
    (with repair).

    Attributes:
        llm_client: Injected LLM backend used to generate verdicts.
        market_data: Injected provider used to fetch fundamentals.
        cache: TTL-bounded cache of prior signals, keyed by
            ``(ticker, session_seed)``.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        market_data: MarketDataProvider | None = None,
    ) -> None:
        """Constructs Groq/live defaults for any provider not injected.

        Args:
            llm_client: LLM backend; defaults to a Groq-backed client.
            market_data: Provider for fundamentals lookups; defaults to live fetches.
        """
        self.llm_client = llm_client or default_groq_client(
            settings.ARGUS_FUNDAMENTAL_MODEL,
            STRUCTURED_OUTPUT.llm_temperature,
            FUNDAMENTAL.max_tokens,
        )
        self.market_data = market_data or LiveMarketDataProvider()
        # Keyed on (ticker, session_seed) so backtest sessions and live calls never share a cached signal.
        self.cache: TTLCache[tuple[str, int | None], FundamentalSignal] = TTLCache(
            ttl=timedelta(days=7), max_entries=SYSTEM.max_tracked_tickers
        )

    def analyze(
        self,
        ticker: str,
        backtest_mode: bool = False,
        session_seed: int | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> FundamentalSignal | None:
        """Audits fundamentals for a single ticker and returns a validated Pydantic signal.

        Checks the local cache first, enforces the injected LLM client's
        remaining capacity, fetches current financial ratios, and decodes a
        FundamentalVerdict from the LLM via argus.structured_output.decode
        (with repair enabled).

        Args:
            ticker: Equity ticker symbol.
            backtest_mode: When True, anonymizes the ticker to prevent LLM parametric recall.
            session_seed: Integer date seed for deterministic anonymization; required when
                backtest_mode is True.
            diagnostics: Receives a reason on every path that returns None, so
                callers can surface the failure instead of only logging it.
                Defaults to a throwaway collector.

        Returns:
            A validated FundamentalSignal, or None if decoding ultimately fails.
        """
        diagnostics = diagnostics or Diagnostics()
        label = f"fundamental_analysis[{ticker}]"
        cached = self.cache.get((ticker, session_seed))
        if cached:
            logger.debug("Cache hit for %s", ticker)
            return cached

        if not self._has_spare_capacity():
            diagnostics.record(f"{label}: LLM capacity too low, skipped")
            return None

        try:
            fundamentals = self.market_data.fundamentals(ticker)
        except Exception as e:
            diagnostics.record(f"{label}: failed to fetch fundamentals: {e}")
            return None

        as_of_date = date.today()  # noqa: DTZ011
        anon_id = None
        if _use_backtest_seed(backtest_mode, session_seed):
            if session_seed is None:
                raise ValueError("Required session seed is missing")
            as_of_date = _session_seed_to_date(session_seed)
            anon_id = anonymize_ticker(ticker, session_seed)

        pit_data: dict[str, Any] = {
            "fundamentals": fundamentals,
            "as_of_date": as_of_date.isoformat(),
        }
        prompt = build_compact_prompt(ticker, pit_data, anon_id)

        verdict = decode_or_degrade(
            self.llm_client,
            SYSTEM_PROMPT,
            prompt,
            FundamentalVerdict,
            repair=True,
            diagnostics=diagnostics,
            label=label,
        )
        if verdict is None:
            return None

        data = _signal_payload(verdict, ticker, pit_data)

        try:
            signal = FundamentalSignal.model_validate(data)
        except ValidationError as e:
            # Merged measured data (e.g. negative debt_to_equity) can fail schema even after a clean decode.
            diagnostics.record(f"{label}: measured data failed validation: {e}")
            return None

        self.cache.set((ticker, session_seed), signal)
        logger.debug("Analysis complete for %s -> %s", ticker, signal.signal.value)
        return signal

    def _has_spare_capacity(self) -> bool:
        """Reports whether the LLM client has room left beyond the held-back reserve.

        Returns:
            True when the client's remaining capacity still covers the reserve kept
            for the same-model portfolio agent (see ``FUNDAMENTAL.capacity_reserve_fraction``).
        """
        capacity_reserve = max(
            1, int(settings.ARGUS_GROQ_RPM * FUNDAMENTAL.capacity_reserve_fraction)
        )
        return self.llm_client.remaining_capacity() >= capacity_reserve

    def batch_analyze(
        self, tickers: list[str], backtest_mode: bool = False, session_seed: int | None = None
    ) -> tuple[dict[str, FundamentalSignal], list[str]]:
        """Performs fundamental evaluations sequentially across a set of tickers.

        Args:
            tickers: List of equity ticker symbols.
            backtest_mode: Passed through to each ``analyze`` call for anonymization.
            session_seed: Passed through to each ``analyze`` call for anonymization.

        Returns:
            Tuple of (signals, errors). ``signals`` maps ticker → FundamentalSignal
            for each successfully analyzed ticker. ``errors`` names each ticker
            omitted and why.
        """
        results = {}
        diagnostics = Diagnostics()
        for ticker in tickers:
            res = self.analyze(ticker, backtest_mode, session_seed, diagnostics=diagnostics)
            if res is not None:
                results[ticker] = res
        return results, diagnostics.messages
