"""
Tests for the Sentiment Agent and its pure-Python helpers.
"""

from datetime import datetime, timedelta

import pytest

from argus.agents.sentiment import aggregate_finbert_scores, SentimentDailyCache
from argus.schemas.signals import SentimentSignal, Signal

def test_aggregate_finbert_scores_empty():
    res = aggregate_finbert_scores([])
    assert res == {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.3}

def test_aggregate_finbert_scores_decay():
    scored = [
        {"numeric": 1.0, "label": "positive"}, # Oldest
        {"numeric": -1.0, "label": "negative"}, # Newest
    ]
    # decay_rate is 0.95. Oldest gets weight 0.95^1 = 0.95. Newest gets weight 0.95^0 = 1.0.
    # net = (1.0 * 0.95 + -1.0 * 1.0) / 1.95 = -0.05 / 1.95 = -0.0256
    res = aggregate_finbert_scores(scored)
    assert res["net"] < 0.0 # Newer negative article outweighs older positive article
    assert res["pct_pos"] == 0.5
    assert res["pct_neg"] == 0.5
    assert res["confidence"] == 0.2 # 2 articles = 0.2

def test_aggregate_finbert_scores_confidence_cap():
    scored = [{"numeric": 1.0, "label": "positive"} for _ in range(15)]
    res = aggregate_finbert_scores(scored)
    assert res["confidence"] == 1.0 # Capped at 1.0

def test_sentiment_daily_cache():
    cache = SentimentDailyCache()
    assert cache.is_stale("AAPL")

    signal = SentimentSignal(
        ticker="AAPL",
        finbert_net_score=0.5,
        pct_positive=0.7,
        pct_negative=0.1,
        news_volume_7d=12,
        social_volume_change_pct=10.0,
        social_mention_surge=True,
        upcoming_catalyst=False,
        signal=Signal.BULLISH,
        conviction=0.8,
        sentiment_decay_risk="LOW",
        reasoning="Test",
        api_calls_used=0,
        timestamp=datetime.now(),
    )

    cache.set("AAPL", signal)
    assert not cache.is_stale("AAPL")
    assert cache.get("AAPL") == signal

    # Manually expire the cache entry to test TTL
    # It checks (now - cached_at).total_seconds() < 86400
    cache._cache["AAPL"] = (signal, datetime.now() - timedelta(days=2))
    assert cache.is_stale("AAPL")
    assert cache.get("AAPL") is None
