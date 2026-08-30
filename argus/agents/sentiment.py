"""
argus/agents/sentiment.py

Multi-tier media sentiment analysis agent for the ARGUS platform.

Responsibilities:
  - Fetch and aggregate raw financial news data
  - Classify news headlines locally using a HuggingFace FinBERT pipeline
  - Synthesize discrete sentiment metrics into a unified SentimentSignal via LLM

Not responsible for:
  - Technical indicator computation (see agents/technical.py)
  - Fundamental ratio analysis (see agents/fundamental.py)
  - Portfolio allocation decisions (see agents/portfolio.py)

Dependencies:
  - transformers >= 4.0 (ProsusAI/finbert)
  - langchain_groq
  - yfinance (for catalyst event detection)
  - GROQ_API_KEY env var must be set (see .env.example)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from argus.config import settings
from argus.data import fetchers
from argus.data.cache import TTLCache
from argus.orchestration.governor import RateLimitExceeded, UnregisteredModel
from argus.schemas.signals import SentimentSignal, SentimentVerdict
from argus.seams import GroqLLMClient, LiveMarketDataProvider, LLMClient, MarketDataProvider
from argus.structured_output import StructuredOutputError, decode

logger = logging.getLogger("argus.sentiment")

# Rough estimate of one fundamental_analysis Groq 70B round-trip, used only to
# pace batch_analyze's scraper calls against a live provider — see its docstring.
# Not tuned against real latency measurements; revisit if fundamental_analysis's
# actual per-ticker duration diverges enough to reintroduce burst risk.
_FANOUT_PACE_SECONDS_PER_TICKER = 2.5

# Covers config.py's default ARGUS_UNIVERSE so fetch_news's "TICKER OR company_name"
# query has a real second term — for short tickers like "V" it otherwise becomes
# "V OR V", which barely narrows the search versus the ticker alone. Unmapped
# tickers fall back to the ticker itself.
_TICKER_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "MA": "Mastercard",
    "HD": "Home Depot",
    "MRK": "Merck",
    "ABBV": "AbbVie",
    "LLY": "Eli Lilly",
    "AVGO": "Broadcom",
    "CVX": "Chevron",
}

# Module-level singleton prevents repeated model initialization (~30s first load)
_FINBERT_PIPELINE = None


def get_finbert():
    """Returns the singleton FinBERT text-classification pipeline, initializing it on first call.

    Returns:
        A HuggingFace transformers Pipeline configured for CPU sentiment classification.
    """
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline

        logger.debug("[FinBERT] Loading ProsusAI/finbert pipeline (first load may take ~30s)...")
        _FINBERT_PIPELINE = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            # CPU forces broad compatibility; GPU acceleration requires explicit opt-in
            device=-1,
            max_length=512,
            truncation=True,
        )
        logger.debug("[FinBERT] Loaded successfully.")
    return _FINBERT_PIPELINE


def score_headlines_with_finbert(articles: list[dict]) -> list[dict]:
    """Classifies headline articles using FinBERT and returns signed numeric scores.

    Processes up to 25 articles, in the order given, to bound CPU blocking time
    on large fetches. Positive labels map to +score, negative to -score, neutral
    to 0.0.

    Args:
        articles: List of article dicts with ``title`` and ``published_at`` keys
            (as returned by ``MarketDataProvider.news``), in relevance order —
            the caller decides ordering before truncation.

    Returns:
        List of dicts with keys ``headline``, ``published_at``, ``label``,
        ``raw_score``, ``numeric``. Returns an empty list if articles is empty.
    """
    if not articles:
        return []

    finbert = get_finbert()
    results = []
    # Cap at 25 to limit CPU blocking duration on large fetches
    for article in articles[:25]:
        headline = article.get("title", "")
        if not headline:
            continue
        try:
            out = finbert(headline[:512])[0]
            if out["label"] == "positive":
                numeric = out["score"]
            elif out["label"] == "negative":
                numeric = -out["score"]
            else:
                numeric = 0.0

            results.append(
                {
                    "headline": headline[:100],
                    "published_at": article.get("published_at", ""),
                    "label": out["label"],
                    "raw_score": out["score"],
                    "numeric": numeric,
                }
            )
        except Exception as e:
            logger.debug("[FinBERT] Error on headline: %s", e)
    return results


def aggregate_finbert_scores(scored: list[dict], decay_rate: float = 0.95) -> dict:
    """Aggregates FinBERT headline scores using exponential recency decay.

    Exponential decay prioritizes recent context over stale news; articles
    earlier in the list are weighted less than later ones. Callers must sort
    ``scored`` chronologically (oldest first) before calling this — the
    decay is positional, not date-aware. Confidence is capped at 10 articles
    to normalize behavior on small datasets.

    Args:
        scored: List of scored dicts from ``score_headlines_with_finbert``,
            sorted oldest to newest.
        decay_rate: Decay factor applied per position from oldest to newest.

    Returns:
        Dict with keys ``net``, ``pct_pos``, ``pct_neg``, ``confidence``.
    """
    if not scored:
        return {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.3}

    n = len(scored)
    weights = [decay_rate ** (n - 1 - i) for i in range(n)]

    numerics = [s["numeric"] for s in scored]
    net = float(np.average(numerics, weights=weights))
    pct_pos = sum(1 for s in scored if s["label"] == "positive") / n
    pct_neg = sum(1 for s in scored if s["label"] == "negative") / n
    confidence = min(n / 10.0, 1.0)

    return {
        "net": round(net, 4),
        "pct_pos": round(pct_pos, 3),
        "pct_neg": round(pct_neg, 3),
        "confidence": round(confidence, 3),
    }


def _check_earnings_calendar(ticker: str) -> bool:
    """Returns True if an earnings date falls within the next 14 days.

    Uses yfinance's calendar API via fetchers.fetch_ticker_calendar (retried,
    rate-limit-classified) rather than calling yfinance directly. The response
    shape (DataFrame or dict) varies across yfinance versions; both are parsed
    for cross-version compatibility.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        True if an upcoming earnings event is within 14 days, False otherwise
        (including when the calendar can't be fetched at all).
    """
    try:
        cal = fetchers.fetch_ticker_calendar(ticker)
        import pandas as pd

        if isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            earnings_dates = cal.loc["Earnings Date"]
            if not earnings_dates.empty:
                dt = pd.to_datetime(earnings_dates.iloc[0])
                if 0 <= (dt.tz_localize(None) - datetime.now()).days <= 14:  # noqa: DTZ005
                    return True
        elif isinstance(cal, dict) and "Earnings Date" in cal:
            dt = pd.to_datetime(cal["Earnings Date"][0])
            if 0 <= (dt.tz_localize(None) - datetime.now()).days <= 14:  # noqa: DTZ005
                return True
    except Exception as e:
        logger.debug("[EarningsCalendar] %s check failed: %s", ticker, e)
    return False


# Maps metric keys to human-readable descriptions used in the LLM synthesis prompt
_SENTIMENT_METRIC_LABELS: dict[str, str] = {
    "net_finbert_score": "[-1.0 to +1.0]  weighted average FinBERT score across headlines "
    "(+1.0 = fully positive, -1.0 = fully negative, 0.0 = neutral)",
    "pct_positive": "[0.0 to 1.0]    fraction of headlines labelled positive "
    "(e.g. 0.60 = 60% positive)",
    "pct_negative": "[0.0 to 1.0]    fraction of headlines labelled negative "
    "(e.g. 0.25 = 25% negative)",
    "finbert_confidence": "[0.0 to 1.0]    confidence proxy based on article volume "
    "(caps at 1.0 when ≥10 articles)",
    "news_volume_7d": "[integer]        total news articles fetched in the last 7 days",
    "news_scored_count": "[integer]        of those, how many were actually scored by FinBERT "
    "(<=25) — the denominator behind pct_positive/pct_negative",
    "news_data_available": "[bool]           False means the news fetch failed or hit its daily "
    "quota — news_volume_7d is a placeholder 0, not a real zero",
    "upcoming_catalyst": "[bool]           True if earnings are expected within 14 days",
}


# Field descriptions for the prompt's declared output schema, keyed by the same
# names as SentimentVerdict's schema — a drift test (test_sentiment.py) asserts
# the two stay equal, so a schema change that isn't mirrored here fails loudly
# instead of leaving the prompt asking for fields the model can't supply. Shared
# between _build_synthesis_prompt and SYSTEM_PROMPT so their two schema restatements
# can't drift apart from each other either.
_VERDICT_FIELD_DESCRIPTIONS: dict[str, str] = {
    "signal": '"BULLISH|BEARISH|NEUTRAL"',
    "conviction": "<float 0.0–1.0>",
    "sentiment_decay_risk": '"LOW|MEDIUM|HIGH"',
    "reasoning": '"<≤60 words citing the primary driver>"',
}
_VERDICT_SCHEMA_JSON = (
    "{" + ", ".join(f'"{k}": {v}' for k, v in _VERDICT_FIELD_DESCRIPTIONS.items()) + "}"
)


def _build_synthesis_prompt(ticker: str, metrics: dict) -> str:
    """Constructs the human-turn message for LLM sentiment synthesis.

    Args:
        ticker: Equity ticker symbol.
        metrics: Dict of pre-computed sentiment metrics.

    Returns:
        Stripped prompt string ready for LLM invocation.
    """
    lines = []
    for k, v in metrics.items():
        hint = _SENTIMENT_METRIC_LABELS.get(k, "")
        if hint:
            lines.append(f"- {k}: {v}  {hint}")
        else:
            lines.append(f"- {k}: {v}")
    metrics_str = "\n".join(lines)

    prompt = (
        f"<sentiment_data ticker=\"{ticker}\" as_of=\"intraday\">\n"
        f"{metrics_str}\n"
        "</sentiment_data>\n"
        "\n"
        "Step 1 — Recall domain rules:\n"
        "  • net_finbert_score > +0.30 with pct_positive > 0.55 → strong positive regime.\n"
        "  • net_finbert_score < -0.20 with pct_negative > 0.40 → negative pressure.\n"
        "  • finbert_confidence < 0.30 (< 3 articles) → evidence too thin; cap conviction at 0.45.\n"
        "  • upcoming_catalyst=True → sentiment_decay_risk must be at least MEDIUM.\n"
        "\n"
        "Step 2 — Classify the sentiment regime: BULLISH, BEARISH, or NEUTRAL.\n"
        "Step 3 — Assign conviction [0.0, 1.0]; cite the single most determinative metric.\n"
        "Step 4 — Set sentiment_decay_risk (LOW | MEDIUM | HIGH) and state the mechanism.\n"
        "\n"
        "Output ONLY valid JSON matching this exact schema — no preamble, no trailing text:\n"
        f"{_VERDICT_SCHEMA_JSON}"
    )
    return prompt.strip()


SYSTEM_PROMPT = (
    "You are a quantitative sentiment analyst at a systematic equity fund. "
    "You work exclusively with pre-computed FinBERT news metrics provided in each request. "
    "You do not generate investment advice. All outputs are research inputs requiring downstream human review.\n"
    "\n"
    "EPISTEMIC STANDARD: If the provided data is insufficient to determine a signal with confidence, "
    "assign signal=NEUTRAL and cap conviction at 0.40. Do not infer or recall any information "
    "about the ticker beyond what is explicitly supplied. When news_data_available is False, "
    "treat the news metrics as unknown, not as zero — a failed fetch is not evidence of a "
    "neutral or absent signal.\n"
    "\n"
    "DATA SCHEMA — interpret each field strictly per the following units and ranges:\n"
    "  net_finbert_score    : float [-1.0, +1.0]  Exponential-decay-weighted mean FinBERT polarity\n"
    "                         over the last 7 days. +1.0 = fully positive, -1.0 = fully negative.\n"
    "  pct_positive         : float [0.0, 1.0]    Fraction of evaluated headlines classified positive.\n"
    "  pct_negative         : float [0.0, 1.0]    Fraction of evaluated headlines classified negative.\n"
    "  finbert_confidence   : float [0.0, 1.0]    Article-volume confidence proxy; caps at 1.0 for ≥10 articles.\n"
    "                         Values below 0.30 indicate thin evidence — reduce conviction accordingly.\n"
    "  news_volume_7d       : integer              Total articles fetched in the prior 7-day window.\n"
    "  upcoming_catalyst    : boolean              True if an earnings event falls within 14 calendar days.\n"
    "                         Presence elevates sentiment_decay_risk because event-driven sentiment\n"
    "                         typically reverts rapidly post-announcement.\n"
    "\n"
    "DOMAIN RULES (apply before classifying):\n"
    "  1. net_finbert_score > +0.30 AND pct_positive > 0.55 → positive sentiment regime.\n"
    "  2. net_finbert_score < -0.20 AND pct_negative > 0.40 → negative sentiment pressure.\n"
    "  3. finbert_confidence < 0.30 → evidence too thin; cap conviction at 0.45 regardless of score.\n"
    "  4. upcoming_catalyst=True → sentiment_decay_risk must be MEDIUM or HIGH.\n"
    "\n"
    "OUTPUT: Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.\n"
    "Schema: " + _VERDICT_SCHEMA_JSON
)


class SentimentAgent:
    """LLM-backed sentiment analysis agent combining FinBERT headline scores with
    LLM synthesis.

    Uses a daily cache to prevent redundant inference for the same ticker within a
    trading session. Obtains its verdict via the shared structured-output decoder
    with repair disabled — see analyze()'s docstring for why.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        market_data: Optional[MarketDataProvider] = None,
    ) -> None:
        """Constructs Groq/live defaults for any provider not injected.

        Args:
            llm_client: LLM backend; defaults to a Groq-backed client.
            market_data: Provider for price lookups; defaults to live fetches.
        """
        if llm_client is None:
            api_key = settings.groq_api_key
            if not api_key:
                logger.warning(
                    "SentimentAgent: GROQ_API_KEY is not set — LLM calls will fail at invocation time."
                )
            llm_client = GroqLLMClient(
                model=settings.ARGUS_SENTIMENT_MODEL,
                temperature=0.1,
                # Measured against gpt-oss-20b at reasoning_effort="low" on
                # reconstructed fixture prompts (AAPL/MSFT/NVDA): completion_tokens
                # peaked at 417, of which up to 357 was reasoning. 550 leaves ~30%
                # headroom above that peak.
                max_tokens=550,
                api_key=api_key,
            )
        self.llm_client = llm_client
        self.market_data = market_data or LiveMarketDataProvider()
        self.cache: TTLCache[str, SentimentSignal] = TTLCache(ttl=timedelta(days=1))

    def analyze(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        errors: Optional[list[str]] = None,
    ) -> Optional[SentimentSignal]:
        """Generates a SentimentSignal for a single ticker using FinBERT + LLM synthesis.

        Args:
            ticker: Equity ticker symbol.
            company_name: Optional display name used in news queries (defaults to ticker).
            errors: If given, a reason is appended here on every path that
                returns None, so callers can surface the failure instead of
                only logging it.

        Decodes a SentimentVerdict from the LLM via argus.structured_output.decode
        with repair disabled: a failed decode degrades this ticker rather than
        re-prompting with the failure appended, since repair would roughly double
        token spend for a ticker whose measured completion peak already sits close
        to its budget, against a governor that is the binding constraint on the
        whole system (see #71).

        Returns:
            A validated SentimentSignal, or None if decoding ultimately fails.
        """
        cached = self.cache.get(ticker)
        if cached:
            return cached

        try:
            news = self.market_data.news(ticker, company_name or ticker, days_back=7)
        except Exception as e:
            logger.warning("[Sentiment] %s news fetch raised: %s", ticker, e)
            news = None

        news_available = news is not None
        news_list = news or []
        # Keep news_list in its fetched (relevance) order for the [:25] truncation,
        # then sort the scored subset chronologically so aggregate_finbert_scores's
        # positional decay actually favors the newest articles
        articles = [a for a in news_list if a.get("title")]
        scored = score_headlines_with_finbert(articles)
        scored.sort(key=lambda s: s["published_at"] or "")
        agg = aggregate_finbert_scores(scored)

        metrics = {
            "net_finbert_score": agg["net"],
            "pct_positive": agg["pct_pos"],
            "pct_negative": agg["pct_neg"],
            "finbert_confidence": agg["confidence"],
            "news_volume_7d": len(news_list),
            "news_scored_count": len(scored),
            "news_data_available": news_available,
            "upcoming_catalyst": _check_earnings_calendar(ticker),
        }

        prompt = _build_synthesis_prompt(ticker, metrics)

        try:
            verdict = decode(self.llm_client, SYSTEM_PROMPT, prompt, SentimentVerdict, repair=False)
        except StructuredOutputError as e:
            logger.warning("[Sentiment] Decode failed for %s: %s", ticker, e)
            if errors is not None:
                errors.append(f"sentiment_analysis[{ticker}]: {e}")
            return None
        except (RateLimitExceeded, UnregisteredModel) as e:
            # The governor already exhausted its own bounded wait before raising
            # either of these — degrade this ticker immediately rather than
            # retrying into the same wall.
            logger.warning("[Sentiment] Governor rejected call for %s: %s", ticker, e)
            if errors is not None:
                errors.append(f"sentiment_analysis[{ticker}]: rate limited: {e}")
            return None
        except Exception as e:
            logger.error("[Sentiment] API error for %s: %s", ticker, e)
            if errors is not None:
                errors.append(f"sentiment_analysis[{ticker}]: API error: {e}")
            return None

        signal = SentimentSignal(
            ticker=ticker,
            # A failed fetch's 0.0 is indistinguishable from a real neutral
            # read; persist None rather than fabricate a measured score
            finbert_net_score=metrics["net_finbert_score"] if news_available else None,
            pct_positive=metrics["pct_positive"],
            pct_negative=metrics["pct_negative"],
            news_volume_7d=metrics["news_volume_7d"],
            news_scored_count=metrics["news_scored_count"],
            news_data_available=metrics["news_data_available"],
            upcoming_catalyst=metrics["upcoming_catalyst"],
            signal=verdict.signal,
            conviction=verdict.conviction,
            sentiment_decay_risk=verdict.sentiment_decay_risk,
            reasoning=verdict.reasoning[:400],
            timestamp=datetime.now(),  # noqa: DTZ005
            api_calls_used=1,
        )
        self.cache.set(ticker, signal)
        return signal

    def batch_analyze(self, tickers: list[str]) -> tuple[dict[str, SentimentSignal], list[str]]:
        """Generates SentimentSignals sequentially for a list of tickers.

        Against a live market-data provider, paces each ticker's scraper calls
        (NewsAPI + the yfinance earnings-calendar lookup) rather than
        bursting all of them in the first few seconds. In graph.py's fan-out,
        this node runs concurrently with fundamental_analysis's ~20 sequential
        Groq round-trips — the slower node either way — so spreading sentiment's
        requests across that same slack costs no wall-clock time while sharply
        cutting 429 risk on the scrapers. Fixture-backed providers skip pacing:
        there's no live burst to smooth, and it would only slow down tests.

        Looks up each ticker's company name (see ``_TICKER_COMPANY_NAMES``) so the
        news query isn't just the ticker duplicated against itself.

        Args:
            tickers: List of equity ticker symbols.

        Returns:
            Tuple of (signals, errors). ``signals`` maps ticker → SentimentSignal
            for each successfully analyzed ticker. ``errors`` names each ticker
            omitted and why.
        """
        pace = isinstance(self.market_data, LiveMarketDataProvider)
        results = {}
        errors: list[str] = []
        for i, ticker in enumerate(tickers):
            if pace and i > 0:
                time.sleep(_FANOUT_PACE_SECONDS_PER_TICKER)
            company_name = _TICKER_COMPANY_NAMES.get(ticker, ticker)
            try:
                res = self.analyze(ticker, company_name=company_name, errors=errors)
            except Exception as exc:
                logger.warning(
                    "batch_analyze: %s failed — %s: %s", ticker, type(exc).__name__, exc
                )
                errors.append(f"sentiment_analysis[{ticker}]: {type(exc).__name__}: {exc}")
                continue
            if res is not None:
                results[ticker] = res
        return results, errors
