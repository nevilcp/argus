"""
Tests for the Sentiment Agent and its pure-Python helpers.
"""

from datetime import datetime, timedelta


from argus.agents.sentiment import SentimentAgent, aggregate_finbert_scores, SentimentDailyCache
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


class _StubMarketData:
    """Minimal MarketDataProvider stub exposing only what SentimentAgent.analyze uses."""

    def __init__(self, news, social):
        self._news = news
        self._social = social

    def news(self, ticker, company_name, days_back=7):
        return self._news

    def social_sentiment(self, ticker):
        return self._social

    def ohlcv_daily(self, ticker, period="2y"):
        raise NotImplementedError

    def multiple_daily(self, tickers, period="1y"):
        raise NotImplementedError

    def fundamentals(self, ticker):
        raise NotImplementedError

    def fred_series(self, series_id, start="2018-01-01"):
        raise NotImplementedError

    def macro_bundle(self):
        raise NotImplementedError

    def vix(self):
        raise NotImplementedError


class _RecordingLLMClient:
    """Stub LLMClient returning a fixed response while recording the prompt it received."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.last_user_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return self._response_text


def test_analyze_flags_unavailable_news_and_social_in_the_llm_prompt(monkeypatch):
    """A None news result and an unavailable social result surface as explicit False flags, not silent zeros."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    market_data = _StubMarketData(
        news=None,
        social={
            "mention_surge": False,
            "volume_change_pct": 0.0,
            "top_posts": [],
            "earnings_within_14d": False,
            "social_data_available": False,
        },
    )
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "no data"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_volume_7d == 0
    assert "news_data_available: False" in llm.last_user_prompt
    assert "social_data_available: False" in llm.last_user_prompt


def test_analyze_reports_availability_true_with_real_data(monkeypatch):
    """Genuine news/social data is flagged available, not confused with the absence placeholder."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)
    # Headlines are non-empty here, which would otherwise load the real FinBERT
    # pipeline; stub it out since this test only cares about the availability flags.
    monkeypatch.setattr(
        "argus.agents.sentiment.score_headlines_with_finbert",
        lambda headlines: [{"headline": h[:100], "label": "neutral", "raw_score": 0.5, "numeric": 0.0} for h in headlines],
    )

    market_data = _StubMarketData(
        news=[{"title": "Real headline", "description": "", "published_at": "", "source": ""}],
        social={
            "mention_surge": True,
            "volume_change_pct": 0.4,
            "top_posts": [],
            "earnings_within_14d": False,
            "social_data_available": True,
        },
    )
    llm = _RecordingLLMClient(
        '{"signal": "BULLISH", "conviction": 0.6, "sentiment_decay_risk": "LOW", "reasoning": "positive"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_volume_7d == 1
    assert "news_data_available: True" in llm.last_user_prompt
    assert "social_data_available: True" in llm.last_user_prompt


def test_batch_analyze_paces_only_against_a_live_market_data_provider(monkeypatch):
    """Fixture-backed providers skip fan-out pacing so the test suite stays fast."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    market_data = _StubMarketData(news=[], social={"social_data_available": False})
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    sleep_calls = []
    monkeypatch.setattr("argus.agents.sentiment.time.sleep", lambda s: sleep_calls.append(s))

    agent.batch_analyze(["AAPL", "MSFT", "GOOGL"])

    assert sleep_calls == []
