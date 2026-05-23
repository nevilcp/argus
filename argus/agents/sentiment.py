"""
argus/agents/sentiment.py
=========================
Two-stage Sentiment Agent.
Stage 1: FinBERT (ProsusAI/finbert) for zero-cost, local NLP classification.
Stage 2: Groq Llama 3.1 8B for fast synthesis of aggregated metrics.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError
from transformers import pipeline

from argus.config import settings
from argus.data.fetchers import fetch_news, fetch_social_sentiment
from argus.orchestration.governor import governor
from argus.schemas.signals import SentimentSignal, Signal

logger = logging.getLogger("argus.sentiment")


# ──────────────────────────────────────────────────────────────────────────────
# FinBERT Module-Level Setup
# ──────────────────────────────────────────────────────────────────────────────

_FINBERT_PIPELINE = None  # Lazy-loaded singleton

def get_finbert():
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        logger.info("[FinBERT] Loading ProsusAI/finbert pipeline (first load may take ~30s)...")
        _FINBERT_PIPELINE = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU
            max_length=512,
            truncation=True,
        )
        logger.info("[FinBERT] Loaded successfully.")
    return _FINBERT_PIPELINE


# ──────────────────────────────────────────────────────────────────────────────
# FinBERT Scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_headlines_with_finbert(headlines: list[str]) -> list[dict]:
    finbert = get_finbert()
    if not headlines:
        return []
    
    results = []
    for headline in headlines[:25]:  # Cap at 25 to bound compute time
        try:
            out = finbert(headline[:512])[0]
            if out["label"] == "positive":
                numeric = out["score"]
            elif out["label"] == "negative":
                numeric = -out["score"]
            else:
                numeric = 0.0
                
            results.append({
                "headline": headline[:100], 
                "label": out["label"], 
                "raw_score": out["score"], 
                "numeric": numeric
            })
        except Exception as e:
            logger.debug("[FinBERT] Error on headline: %s", e)
    return results

def aggregate_finbert_scores(scored: list[dict], decay_rate: float = 0.95) -> dict:
    if not scored:
        return {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.3}
    
    # Recency decay: most recent = weight 1.0, oldest = weight decay_rate^(n-1)
    n = len(scored)
    weights = [decay_rate ** (n - 1 - i) for i in range(n)]
    
    numerics = [s["numeric"] for s in scored]
    net = float(np.average(numerics, weights=weights))
    pct_pos = sum(1 for s in scored if s["label"] == "positive") / n
    pct_neg = sum(1 for s in scored if s["label"] == "negative") / n
    confidence = min(n / 10.0, 1.0)  # More articles = higher confidence (caps at 10)
    
    return {
        "net": round(net, 4), 
        "pct_pos": round(pct_pos, 3), 
        "pct_neg": round(pct_neg, 3), 
        "confidence": round(confidence, 3)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Daily Cache
# ──────────────────────────────────────────────────────────────────────────────

class SentimentDailyCache:
    """1-day local cache to prevent redundant LLM calls for static sentiment."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[SentimentSignal, datetime]] = {}
        self._ttl_days = 1

    def get(self, ticker: str) -> Optional[SentimentSignal]:
        if ticker in self._cache:
            signal, cached_at = self._cache[ticker]
            if (datetime.now() - cached_at).days < self._ttl_days:
                return signal
        return None

    def set(self, ticker: str, signal: SentimentSignal) -> None:
        self._cache[ticker] = (signal, datetime.now())

    def is_stale(self, ticker: str) -> bool:
        return self.get(ticker) is None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_earnings_calendar(ticker: str) -> bool:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        # Check based on dict or dataframe return type
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
    except Exception:
        pass
    return False

def _build_synthesis_prompt(ticker: str, metrics: dict) -> str:
    metrics_str = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    prompt = f"""
Synthesize the following sentiment metrics for {ticker}:

{metrics_str}

Assign a sentiment signal (BULLISH, BEARISH, NEUTRAL) and a conviction score (0.0 to 1.0).
Determine the sentiment_decay_risk (LOW, MEDIUM, HIGH) based on how ephemeral the catalysts are.
Reasoning must be short (under 60 words).
Output ONLY valid JSON.
"""
    return prompt.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent Class
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a pure Sentiment Analyst. You receive pre-computed 
sentiment metrics from FinBERT and social data. Synthesize them into a signal.
Do NOT add qualitative market commentary. Your job is to classify the 
sentiment regime and assign conviction.

Output ONLY valid JSON. Fields: signal, conviction, reasoning (max 60 words), 
sentiment_decay_risk. No other text."""

class SentimentAgent:
    def __init__(self) -> None:
        api_key = settings.groq_api_key or "DUMMY_KEY_FOR_TESTING"
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=350,
            groq_api_key=api_key,
        )
        self.cache = SentimentDailyCache()

    def analyze(self, ticker: str, company_name: Optional[str] = None) -> SentimentSignal:
        if not self.cache.is_stale(ticker):
            cached = self.cache.get(ticker)
            if cached:
                return cached

        # Stage 1: Fetch data
        news = fetch_news(ticker, company_name or ticker, days_back=7)
        social = fetch_social_sentiment(ticker)
        
        # Stage 1: FinBERT scoring
        headlines = [a.get("title", "") for a in news if a.get("title")]
        scored = score_headlines_with_finbert(headlines)
        agg = aggregate_finbert_scores(scored)
        
        # Build metrics dict
        metrics = {
            "net_finbert_score": agg["net"],
            "pct_positive": agg["pct_pos"],
            "pct_negative": agg["pct_neg"],
            "finbert_confidence": agg["confidence"],
            "news_volume_7d": len(news),
            "social_mention_surge": social.get("mention_surge", False),
            "social_mention_count": social.get("mention_count_7d", 0),
            "upcoming_catalyst": _check_earnings_calendar(ticker),
        }
        
        # Stage 2: Groq 8B synthesis
        prompt = _build_synthesis_prompt(ticker, metrics)
        estimated_tokens = 450
        
        governor.wait_if_needed("llama-3.1-8b-instant", estimated_tokens)
        
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
                
                signal = SentimentSignal(
                    ticker=ticker,
                    signal=data["signal"],
                    conviction=float(data["conviction"]),
                    finbert_net_score=metrics["net_finbert_score"],
                    pct_positive=metrics["pct_positive"],
                    pct_negative=metrics["pct_negative"],
                    news_volume_7d=metrics["news_volume_7d"],
                    social_mention_surge=metrics["social_mention_surge"],
                    upcoming_catalyst=metrics["upcoming_catalyst"],
                    sentiment_decay_risk=data.get("sentiment_decay_risk", "MEDIUM"),
                    reasoning=data.get("reasoning", "")[:400],
                    temporal_horizon="EVENT_DRIVEN",
                    timestamp=datetime.now(),
                    api_calls_used=1
                )
                self.cache.set(ticker, signal)
                return signal
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning("[Sentiment] Attempt %d parse error: %s", attempt + 1, e)
                if attempt == 2:
                    return self._neutral_fallback(ticker)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("[Sentiment] Attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    return self._neutral_fallback(ticker)
                time.sleep(2 ** attempt)

        return self._neutral_fallback(ticker)

    def _neutral_fallback(self, ticker: str) -> SentimentSignal:
        return SentimentSignal(
            ticker=ticker,
            signal=Signal.NEUTRAL,
            conviction=0.35,
            finbert_net_score=0.0,
            pct_positive=0.0,
            pct_negative=0.0,
            news_volume_7d=0,
            social_mention_surge=False,
            upcoming_catalyst=False,
            sentiment_decay_risk="HIGH",
            reasoning="Fallback: Insufficient data or API failure.",
            temporal_horizon="EVENT_DRIVEN",
            timestamp=datetime.now(),
            api_calls_used=0
        )

    def batch_analyze(self, tickers: list[str]) -> dict[str, SentimentSignal]:
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze(ticker)
        return results
