"""
argus/agents/sentiment.py

Multi-tier media and social sentiment analysis agent for the ARGUS platform.

Responsibilities:
  - Fetch and aggregate raw financial news and social sentiment data
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

import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import ValidationError

from argus.config import settings
from argus.orchestration.governor import governor
from argus.schemas.signals import SentimentSignal
from argus.seams import GroqLLMClient, LiveMarketDataProvider, LLMClient, MarketDataProvider

logger = logging.getLogger("argus.sentiment")

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


def score_headlines_with_finbert(headlines: list[str]) -> list[dict]:
    """Classifies headlines using FinBERT and returns signed numeric scores.

    Processes up to 25 headlines to bound CPU blocking time on large fetches.
    Positive labels map to +score, negative to -score, neutral to 0.0.

    Args:
        headlines: List of raw news headline strings.

    Returns:
        List of dicts with keys ``headline``, ``label``, ``raw_score``, ``numeric``.
        Returns an empty list if headlines is empty.
    """
    if not headlines:
        return []

    finbert = get_finbert()
    results = []
    # Cap at 25 to limit CPU blocking duration on large fetches
    for headline in headlines[:25]:
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
    earlier in the list are weighted less than later ones. Confidence is
    capped at 10 articles to normalize behavior on small datasets.

    Args:
        scored: List of scored dicts from ``score_headlines_with_finbert``.
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


class SentimentDailyCache:
    """In-memory per-ticker cache for SentimentSignals with a 1-day TTL.

    Avoids redundant FinBERT inference and LLM calls for the same ticker
    within a single trading session.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[SentimentSignal, datetime]] = {}
        self._ttl_days = 1

    def get(self, ticker: str) -> Optional[SentimentSignal]:
        """Returns a cached signal if it exists and is within the TTL window.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            Cached SentimentSignal, or None if absent or stale.
        """
        if ticker in self._cache:
            signal, cached_at = self._cache[ticker]
            if (datetime.now() - cached_at).total_seconds() < self._ttl_days * 86400:
                return signal
        return None

    def set(self, ticker: str, signal: SentimentSignal) -> None:
        """Stores a signal with the current timestamp.

        Args:
            ticker: Equity ticker symbol.
            signal: Validated SentimentSignal to cache.
        """
        self._cache[ticker] = (signal, datetime.now())

    def is_stale(self, ticker: str) -> bool:
        """Returns True if the ticker has no valid cached signal.

        Args:
            ticker: Equity ticker symbol.
        """
        return self.get(ticker) is None


def _check_earnings_calendar(ticker: str) -> bool:
    """Returns True if an earnings date falls within the next 14 days.

    Uses yfinance calendar API, which returns varying structures across versions;
    both DataFrame and dict formats are parsed to ensure cross-version compatibility.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        True if an upcoming earnings event is within 14 days, False otherwise.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        cal = t.calendar
        import pandas as pd

        if isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            earnings_dates = cal.loc["Earnings Date"]
            if not earnings_dates.empty:
                dt = pd.to_datetime(earnings_dates.iloc[0])
                if (dt.tz_localize(None) - datetime.now()).days <= 14:
                    return True
        elif isinstance(cal, dict) and "Earnings Date" in cal:
            dt = pd.to_datetime(cal["Earnings Date"][0])
            if (dt.tz_localize(None) - datetime.now()).days <= 14:
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
    "social_mention_surge": "[bool]           True if social mention volume is unusually high",
    "upcoming_catalyst": "[bool]           True if earnings are expected within 14 days",
}


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
        "  • social_mention_surge=True alone is insufficient for BULLISH unless FinBERT confirms.\n"
        "\n"
        "Step 2 — Classify the sentiment regime: BULLISH, BEARISH, or NEUTRAL.\n"
        "Step 3 — Assign conviction [0.0, 1.0]; cite the single most determinative metric.\n"
        "Step 4 — Set sentiment_decay_risk (LOW | MEDIUM | HIGH) and state the mechanism.\n"
        "\n"
        "Output ONLY valid JSON matching this exact schema — no preamble, no trailing text:\n"
        '{"signal": "<BULLISH|BEARISH|NEUTRAL>", "conviction": <float 0.0–1.0>, '
        '"sentiment_decay_risk": "<LOW|MEDIUM|HIGH>", "reasoning": "<≤60 words citing key driver>"}'
    )
    return prompt.strip()


SYSTEM_PROMPT = (
    "You are a quantitative sentiment analyst at a systematic equity fund. "
    "You work exclusively with pre-computed FinBERT and social media metrics provided in each request. "
    "You do not generate investment advice. All outputs are research inputs requiring downstream human review.\n"
    "\n"
    "EPISTEMIC STANDARD: If the provided data is insufficient to determine a signal with confidence, "
    "assign signal=NEUTRAL and cap conviction at 0.40. Do not infer or recall any information "
    "about the ticker beyond what is explicitly supplied.\n"
    "\n"
    "DATA SCHEMA — interpret each field strictly per the following units and ranges:\n"
    "  net_finbert_score    : float [-1.0, +1.0]  Exponential-decay-weighted mean FinBERT polarity\n"
    "                         over the last 7 days. +1.0 = fully positive, -1.0 = fully negative.\n"
    "  pct_positive         : float [0.0, 1.0]    Fraction of evaluated headlines classified positive.\n"
    "  pct_negative         : float [0.0, 1.0]    Fraction of evaluated headlines classified negative.\n"
    "  finbert_confidence   : float [0.0, 1.0]    Article-volume confidence proxy; caps at 1.0 for ≥10 articles.\n"
    "                         Values below 0.30 indicate thin evidence — reduce conviction accordingly.\n"
    "  news_volume_7d       : integer              Total articles fetched in the prior 7-day window.\n"
    "  social_mention_surge : boolean              True if 7-day social volume is statistically abnormal.\n"
    "  upcoming_catalyst    : boolean              True if an earnings event falls within 14 calendar days.\n"
    "                         Presence elevates sentiment_decay_risk because event-driven sentiment\n"
    "                         typically reverts rapidly post-announcement.\n"
    "\n"
    "DOMAIN RULES (apply before classifying):\n"
    "  1. net_finbert_score > +0.30 AND pct_positive > 0.55 → positive sentiment regime.\n"
    "  2. net_finbert_score < -0.20 AND pct_negative > 0.40 → negative sentiment pressure.\n"
    "  3. finbert_confidence < 0.30 → evidence too thin; cap conviction at 0.45 regardless of score.\n"
    "  4. upcoming_catalyst=True → sentiment_decay_risk must be MEDIUM or HIGH.\n"
    "  5. social_mention_surge without FinBERT confirmation is insufficient for BULLISH.\n"
    "\n"
    "OUTPUT: Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.\n"
    'Schema: {"signal": "BULLISH|BEARISH|NEUTRAL", "conviction": <float 0.0–1.0>, '
    '"sentiment_decay_risk": "LOW|MEDIUM|HIGH", "reasoning": "<≤60 words citing the primary driver>"}'
)


class SentimentAgent:
    """LLM-backed sentiment analysis agent combining FinBERT and social media signals.

    Uses a daily cache to prevent redundant inference for the same ticker within a
    trading session. Retries the LLM call up to 3 times with exponential back-off
    on parse or validation failures.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        market_data: Optional[MarketDataProvider] = None,
    ) -> None:
        if llm_client is None:
            api_key = settings.groq_api_key
            if not api_key:
                logger.warning(
                    "SentimentAgent: GROQ_API_KEY is not set — LLM calls will fail at invocation time."
                )
            llm_client = GroqLLMClient(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=350,
                api_key=api_key,
            )
        self.llm_client = llm_client
        self.market_data = market_data or LiveMarketDataProvider()
        self.cache = SentimentDailyCache()

    def analyze(self, ticker: str, company_name: Optional[str] = None) -> Optional[SentimentSignal]:
        """Generates a SentimentSignal for a single ticker using FinBERT + LLM synthesis.

        Args:
            ticker: Equity ticker symbol.
            company_name: Optional display name used in news queries (defaults to ticker).

        Returns:
            A validated SentimentSignal, or None if all retry attempts fail.
        """
        if not self.cache.is_stale(ticker):
            cached = self.cache.get(ticker)
            if cached:
                return cached

        news = self.market_data.news(ticker, company_name or ticker, days_back=7)
        social = self.market_data.social_sentiment(ticker)

        headlines = [a.get("title", "") for a in news if a.get("title")]
        scored = score_headlines_with_finbert(headlines)
        agg = aggregate_finbert_scores(scored)

        metrics = {
            "net_finbert_score": agg["net"],
            "pct_positive": agg["pct_pos"],
            "pct_negative": agg["pct_neg"],
            "finbert_confidence": agg["confidence"],
            "news_volume_7d": len(news),
            "social_mention_surge": social.get("mention_surge", False),
            "social_volume_change_pct": social.get("volume_change_pct", 0.0),
            "upcoming_catalyst": _check_earnings_calendar(ticker),
        }

        prompt = _build_synthesis_prompt(ticker, metrics)
        estimated_tokens = 450

        governor.wait_if_needed("llama-3.1-8b-instant", estimated_tokens)

        for attempt in range(3):
            try:
                raw = self.llm_client.complete(SYSTEM_PROMPT, prompt).strip()
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

                signal = SentimentSignal(
                    ticker=ticker,
                    finbert_net_score=metrics["net_finbert_score"],
                    pct_positive=metrics["pct_positive"],
                    pct_negative=metrics["pct_negative"],
                    news_volume_7d=metrics["news_volume_7d"],
                    social_volume_change_pct=metrics["social_volume_change_pct"],
                    social_mention_surge=metrics["social_mention_surge"],
                    upcoming_catalyst=metrics["upcoming_catalyst"],
                    signal=data["signal"],
                    conviction=float(data["conviction"]),
                    sentiment_decay_risk=data.get("sentiment_decay_risk", "MEDIUM"),
                    reasoning=data.get("reasoning", "")[:400],
                    timestamp=datetime.now(),
                    api_calls_used=1,
                )
                self.cache.set(ticker, signal)
                return signal

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning("[Sentiment] Attempt %d parse error: %s", attempt + 1, e)
                if attempt == 2:
                    return None
                time.sleep(2**attempt)
            except Exception as e:
                logger.warning("[Sentiment] Attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    return None
                time.sleep(2**attempt)

        return None

    def batch_analyze(self, tickers: list[str]) -> dict[str, SentimentSignal]:
        """Generates SentimentSignals sequentially for a list of tickers.

        Args:
            tickers: List of equity ticker symbols.

        Returns:
            Mapping of ticker → SentimentSignal for each successfully analyzed ticker.
            Failed tickers are omitted silently (already logged inside analyze).
        """
        results = {}
        for ticker in tickers:
            res = self.analyze(ticker)
            if res is not None:
                results[ticker] = res
        return results
