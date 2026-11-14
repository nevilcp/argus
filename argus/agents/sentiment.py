"""Multi-tier media sentiment analysis agent for the ARGUS platform."""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from argus.config import settings
from argus.data.cache import TTLCache
from argus.infra.clock import now_local
from argus.infra.diagnostics import Diagnostics
from argus.params import SENTIMENT, STRUCTURED_OUTPUT, SYSTEM
from argus.ports import LLMClient, MarketDataProvider
from argus.schemas.prompting import UNTRUSTED_GUARD, schema_block, wrap_untrusted
from argus.schemas.signals import SentimentSignal, SentimentVerdict
from argus.seams import LiveMarketDataProvider, default_groq_client
from argus.structured_output import decode_or_degrade

logger = logging.getLogger(__name__)

# Company names for fetch_news's query term; short tickers like V need one to narrow the search.
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

# Expose the deployed FinBERT identifier in health reports.
FINBERT_MODEL = "ProsusAI/finbert"

# Module-level singleton prevents repeated model initialization (~30s first load).
_FINBERT_PIPELINE = None


def get_finbert():
    """Return the singleton FinBERT text-classification pipeline."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline

        logger.debug("Loading %s pipeline (first load may take ~30s)...", FINBERT_MODEL)
        _FINBERT_PIPELINE = pipeline(
            task="text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=-1,
            max_length=512,
            truncation=True,
        )
        logger.debug("Loaded successfully.")
    return _FINBERT_PIPELINE


def score_headlines_with_finbert(articles: list[dict]) -> list[dict]:
    """Classify headlines with FinBERT and return signed numeric scores."""
    if not articles:
        return []

    finbert = get_finbert()
    results = []
    for article in articles[: SENTIMENT.max_articles_scored]:
        headline = article.get("title", "")
        if not headline:
            continue
        try:
            out = finbert(headline[: SENTIMENT.headline_char_limit_for_model])[0]
            if out["label"] == "positive":
                numeric = out["score"]
            elif out["label"] == "negative":
                numeric = -out["score"]
            else:
                numeric = 0.0

            results.append(
                {
                    "headline": headline[: SENTIMENT.headline_char_limit_stored],
                    "published_at": article.get("published_at", ""),
                    "label": out["label"],
                    "raw_score": out["score"],
                    "numeric": numeric,
                }
            )
        except Exception as e:
            logger.debug("Error on headline: %s", e)
    return results


def aggregate_finbert_scores(scored: list[dict], decay_rate: float = SENTIMENT.news_decay_rate) -> dict:
    """Aggregate oldest-first FinBERT scores with positional decay and capped confidence."""
    if not scored:
        return {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.0}

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


def _check_earnings_calendar(ticker: str, market_data: MarketDataProvider) -> bool | None:
    """Return True if an earnings date falls within SENTIMENT.catalyst_window_days of now."""
    try:
        cal = market_data.ticker_calendar(ticker)
        if not isinstance(cal, (dict, pd.DataFrame)):
            return None

        if isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            earnings_dates = cal.loc["Earnings Date"]
            if not earnings_dates.empty:
                dt = pd.to_datetime(earnings_dates.iloc[0])
                if 0 <= (dt.tz_localize(None) - now_local()).days <= SENTIMENT.catalyst_window_days:
                    return True
        elif isinstance(cal, dict) and "Earnings Date" in cal:
            dt = pd.to_datetime(cal["Earnings Date"][0])
            if 0 <= (dt.tz_localize(None) - now_local()).days <= SENTIMENT.catalyst_window_days:
                return True
    except Exception as e:
        logger.warning("%s earnings calendar unavailable: %s", ticker, e)
        return None
    return False


_SENTIMENT_METRIC_LABELS: dict[str, str] = {
    "net_finbert_score": "[-1.0 to +1.0]  weighted average FinBERT score across headlines "
    "(+1.0 = fully positive, -1.0 = fully negative, 0.0 = neutral)",
    "pct_positive": "[0.0 to 1.0]    fraction of headlines labeled positive (e.g. 0.60 = 60% positive)",
    "pct_negative": "[0.0 to 1.0]    fraction of headlines labeled negative (e.g. 0.25 = 25% negative)",
    "finbert_confidence": "[0.0 to 1.0]    confidence proxy based on article volume (caps at 1.0 when ≥10 articles)",
    "news_volume_7d": "[integer]        total news articles fetched in the last 7 days",
    "news_scored_count": "[integer]        of those, how many were actually scored by FinBERT "
    "(<=25) — the denominator behind pct_positive/pct_negative",
    "news_data_available": "[bool]           False means the news fetch failed or hit its daily "
    "quota — news_volume_7d is a placeholder 0, not a real zero",
    "catalyst_data_available": "[bool] False means the earnings calendar is unavailable; upcoming_catalyst is a placeholder",
    "upcoming_catalyst": "[bool]           True if earnings are expected within 14 days",
}


_VERDICT_SCHEMA_JSON = schema_block(SentimentVerdict)


def _build_synthesis_prompt(ticker: str, metrics: dict) -> str:
    """Construct the human-turn message for LLM sentiment synthesis."""
    lines = []
    for k, v in metrics.items():
        hint = _SENTIMENT_METRIC_LABELS.get(k, "")
        if hint:
            lines.append(f"- {k}: {v}  {hint}")
        else:
            lines.append(f"- {k}: {v}")
    metrics_str = "\n".join(lines)

    prompt = (
        f"{wrap_untrusted('sentiment_data', metrics_str, {'ticker': ticker, 'as_of': 'intraday'})}\n"
        f"{UNTRUSTED_GUARD}\n"
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
    """LLM-backed sentiment agent combining FinBERT scores with LLM synthesis."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        market_data: MarketDataProvider | None = None,
    ) -> None:
        self.llm_client = llm_client or default_groq_client(
            settings.ARGUS_SENTIMENT_MODEL,
            STRUCTURED_OUTPUT.llm_temperature,
            SENTIMENT.max_tokens,
        )
        self.market_data = market_data or LiveMarketDataProvider()
        self.cache: TTLCache[str, SentimentSignal] = TTLCache(
            ttl=timedelta(days=1), max_entries=SYSTEM.max_tracked_tickers
        )

    def analyze(
        self,
        ticker: str,
        company_name: str | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> SentimentSignal | None:
        """Generate a ticker's SentimentSignal via FinBERT and LLM synthesis."""
        cached = self.cache.get(ticker)
        if cached:
            return cached

        metrics = self._news_metrics(ticker, company_name)
        prompt = _build_synthesis_prompt(ticker, metrics)

        verdict = decode_or_degrade(
            self.llm_client,
            SYSTEM_PROMPT,
            prompt,
            SentimentVerdict,
            repair=False,
            diagnostics=diagnostics or Diagnostics(),
            label=f"sentiment_analysis[{ticker}]",
        )
        if verdict is None:
            return None

        signal = SentimentSignal(
            ticker=ticker,
            # A failed fetch's 0.0 looks like a real neutral; persist None instead of a fabricated score.
            finbert_net_score=(metrics["net_finbert_score"] if metrics["news_data_available"] else None),
            pct_positive=metrics["pct_positive"],
            pct_negative=metrics["pct_negative"],
            news_volume_7d=metrics["news_volume_7d"],
            news_scored_count=metrics["news_scored_count"],
            news_data_available=metrics["news_data_available"],
            upcoming_catalyst=metrics["upcoming_catalyst"],
            catalyst_data_available=metrics["catalyst_data_available"],
            signal=verdict.signal,
            conviction=verdict.conviction,
            sentiment_decay_risk=verdict.sentiment_decay_risk,
            reasoning=verdict.reasoning[: SENTIMENT.reasoning_char_limit],
            timestamp=now_local(),
            api_calls_used=1,
        )
        self.cache.set(ticker, signal)
        return signal

    def _news_metrics(self, ticker: str, company_name: str | None) -> dict:
        """Fetch this ticker's news and reduces it to LLM-facing metrics."""
        try:
            news = self.market_data.news(ticker, company_name or ticker, days_back=7)
        except Exception as e:
            logger.warning("%s news fetch raised: %s", ticker, e)
            news = None

        news_list = news or []
        articles = [a for a in news_list if a.get("title")]
        scored = score_headlines_with_finbert(articles)
        scored.sort(key=lambda s: s["published_at"] or "")
        agg = aggregate_finbert_scores(scored)

        catalyst = _check_earnings_calendar(ticker, self.market_data)
        return {
            "net_finbert_score": agg["net"],
            "pct_positive": agg["pct_pos"],
            "pct_negative": agg["pct_neg"],
            "finbert_confidence": agg["confidence"],
            "news_volume_7d": len(news_list),
            "news_scored_count": len(scored),
            "news_data_available": news is not None,
            "upcoming_catalyst": catalyst is True,
            "catalyst_data_available": catalyst is not None,
        }

    def batch_analyze(self, tickers: list[str]) -> tuple[dict[str, SentimentSignal], list[str]]:
        """Generate SentimentSignals sequentially for a list of tickers."""
        pace = self.market_data.is_live
        results = {}
        diagnostics = Diagnostics()
        for i, ticker in enumerate(tickers):
            if pace and i > 0:
                time.sleep(SENTIMENT.fanout_pace_seconds_per_ticker)
            company_name = _TICKER_COMPANY_NAMES.get(ticker, ticker)
            try:
                res = self.analyze(ticker, company_name=company_name, diagnostics=diagnostics)
            except Exception as exc:
                diagnostics.record(f"sentiment_analysis[{ticker}]: {type(exc).__name__}: {exc}")
                continue
            if res is not None:
                results[ticker] = res
        return results, diagnostics.messages
