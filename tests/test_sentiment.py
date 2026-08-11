"""
Tests for the Sentiment Agent and its pure-Python helpers.
"""

from datetime import datetime, timedelta


from argus.agents.sentiment import aggregate_finbert_scores, SentimentDailyCache
from argus.schemas.signals import SentimentSignal, Signal

def test_aggregate_finbert_scores_empty():
    """An empty article list returns the neutral, low-confidence default."""
    res = aggregate_finbert_scores([])
    assert res == {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.3}

def test_aggregate_finbert_scores_decay():
    """A newer negative article outweighs an equal-magnitude older positive one."""
    scored = [
        {"numeric": 1.0, "label": "positive"}, # Oldest
        {"numeric": -1.0, "label": "negative"}, # Newest
    ]
    # decay_rate 0.95 weights oldest at 0.95^1, newest at 0.95^0, so recency wins the net score
    res = aggregate_finbert_scores(scored)
    assert res["net"] < 0.0
    assert res["pct_pos"] == 0.5
    assert res["pct_neg"] == 0.5
    assert res["confidence"] == 0.2 # 2 articles = 0.2

def test_aggregate_finbert_scores_confidence_cap():
    """Confidence saturates at 1.0 regardless of how many articles are scored."""
    scored = [{"numeric": 1.0, "label": "positive"} for _ in range(15)]
    res = aggregate_finbert_scores(scored)
    assert res["confidence"] == 1.0 # capped at 1.0

def test_sentiment_daily_cache():
    """A cache entry becomes stale, and unretrievable, once its TTL elapses."""
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

    # Backdate the entry past the cache's 86400s TTL to force staleness
    cache._cache["AAPL"] = (signal, datetime.now() - timedelta(days=2))
    assert cache.is_stale("AAPL")
    assert cache.get("AAPL") is None
