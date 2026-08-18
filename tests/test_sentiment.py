"""
Tests for the Sentiment Agent and its pure-Python helpers.
"""

from datetime import datetime, timedelta


from argus.agents.sentiment import (
    SentimentAgent,
    aggregate_finbert_scores,
    SentimentDailyCache,
    _check_earnings_calendar,
)
from argus.schemas.signals import SentimentSignal, Signal

def test_aggregate_finbert_scores_empty():
    """An empty article list returns the neutral, low-confidence default."""
    res = aggregate_finbert_scores([])
    assert res == {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.3}

def test_aggregate_finbert_scores_decay():
    """A later-position negative article outweighs an equal-magnitude earlier positive one.

    aggregate_finbert_scores itself only knows position, not real dates — callers
    are responsible for sorting ``scored`` chronologically before calling it.
    """
    scored = [
        {"numeric": 1.0, "label": "positive"},  # earlier position
        {"numeric": -1.0, "label": "negative"},  # later position
    ]
    # decay_rate 0.95 weights the earlier position at 0.95^1, the later at 0.95^0,
    # so the later position wins the net score
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

    def __init__(self, news):
        self._news = news

    def news(self, ticker, company_name, days_back=7):
        return self._news

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


def test_analyze_flags_unavailable_news_in_the_llm_prompt(monkeypatch):
    """A None news result surfaces as an explicit False flag, not a silent zero."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    market_data = _StubMarketData(news=None)
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "no data"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_volume_7d == 0
    assert signal.news_scored_count == 0
    assert signal.news_data_available is False
    # A failed fetch's 0.0 is indistinguishable from a real neutral read;
    # the persisted signal must not fabricate a measured score
    assert signal.finbert_net_score is None
    assert "news_data_available: False" in llm.last_user_prompt


def test_analyze_reports_availability_true_with_real_data(monkeypatch):
    """Genuine news data is flagged available, not confused with the absence placeholder."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)
    # Headlines are non-empty here, which would otherwise load the real FinBERT
    # pipeline; stub it out since this test only cares about the availability flags.
    monkeypatch.setattr(
        "argus.agents.sentiment.score_headlines_with_finbert",
        lambda articles: [
            {
                "headline": a["title"][:100],
                "published_at": a.get("published_at", ""),
                "label": "neutral",
                "raw_score": 0.5,
                "numeric": 0.0,
            }
            for a in articles
        ],
    )

    market_data = _StubMarketData(
        news=[{"title": "Real headline", "description": "", "published_at": "", "source": ""}],
    )
    llm = _RecordingLLMClient(
        '{"signal": "BULLISH", "conviction": 0.6, "sentiment_decay_risk": "LOW", "reasoning": "positive"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_volume_7d == 1
    assert signal.news_scored_count == 1
    assert signal.news_data_available is True
    assert signal.finbert_net_score == 0.0
    assert "news_data_available: True" in llm.last_user_prompt


def test_batch_analyze_paces_only_against_a_live_market_data_provider(monkeypatch):
    """Fixture-backed providers skip fan-out pacing so the test suite stays fast."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    market_data = _StubMarketData(news=[])
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    sleep_calls = []
    monkeypatch.setattr("argus.agents.sentiment.time.sleep", lambda s: sleep_calls.append(s))

    agent.batch_analyze(["AAPL", "MSFT", "GOOGL"])

    assert sleep_calls == []


def test_scored_articles_are_sorted_by_published_at_before_aggregation(monkeypatch):
    """Recency decay follows publish date, not NewsAPI's relevance rank.

    NewsAPI is queried with sort_by="relevancy", so the fetched article order is a
    relevance rank, not a timeline. Feeding that order straight into the
    position-based decay would let a highly-relevant-but-stale article dominate.
    Here the most relevant (first) article is old and negative; the least
    relevant (last) article is new and positive. Decay must favor the new one.
    """
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    def stub_score(articles):
        polarity = {"stale bad news": -1.0, "fresh good news": 1.0}
        return [
            {
                "headline": a["title"],
                "published_at": a["published_at"],
                "label": "negative" if polarity[a["title"]] < 0 else "positive",
                "raw_score": 1.0,
                "numeric": polarity[a["title"]],
            }
            for a in articles
        ]

    monkeypatch.setattr("argus.agents.sentiment.score_headlines_with_finbert", stub_score)

    market_data = _StubMarketData(
        # Relevance order (as NewsAPI returns it): stale/negative ranks first.
        news=[
            {"title": "stale bad news", "published_at": "2024-01-01T00:00:00Z"},
            {"title": "fresh good news", "published_at": "2024-06-01T00:00:00Z"},
        ],
    )
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.finbert_net_score > 0.0


def test_analyze_tolerates_a_raising_news_provider(monkeypatch):
    """A news provider that raises mid-fetch degrades to unavailable rather than crashing."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    class _RaisingNewsMarketData(_StubMarketData):
        def news(self, ticker, company_name, days_back=7):
            raise RuntimeError("provider outage")

    market_data = _RaisingNewsMarketData(news=None)
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_data_available is False
    assert signal.finbert_net_score is None


def test_batch_analyze_tolerates_a_single_ticker_raising(monkeypatch):
    """One ticker's uncaught exception must not abort the batch; the rest still complete."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    market_data = _StubMarketData(news=[])
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)
    original_analyze = agent.analyze

    def flaky_analyze(ticker, company_name=None, errors=None):
        if ticker == "MSFT":
            raise RuntimeError("boom")
        return original_analyze(ticker, company_name=company_name, errors=errors)

    monkeypatch.setattr(agent, "analyze", flaky_analyze)

    results, errors = agent.batch_analyze(["AAPL", "MSFT", "GOOGL"])

    assert "MSFT" not in results
    assert "AAPL" in results
    assert "GOOGL" in results
    assert any("MSFT" in e for e in errors)


def test_batch_analyze_passes_company_name_for_known_tickers(monkeypatch):
    """A short ticker like 'V' gets its full company name so the news query isn't 'V OR V'."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker: False)

    received = {}

    class _RecordingMarketData(_StubMarketData):
        def news(self, ticker, company_name, days_back=7):
            received[ticker] = company_name
            return []

    market_data = _RecordingMarketData(news=[])
    llm = _RecordingLLMClient(
        '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'
    )
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    agent.batch_analyze(["V", "ZZZZ"])

    assert received["V"] == "Visa"
    assert received["ZZZZ"] == "ZZZZ"  # unmapped ticker falls back to itself


def test_check_earnings_calendar_past_date_is_not_upcoming(monkeypatch):
    """An earnings date in the past must not report an upcoming catalyst."""
    past_date = datetime.now() - timedelta(days=15)
    monkeypatch.setattr(
        "argus.agents.sentiment.fetchers.fetch_ticker_calendar",
        lambda ticker: {"Earnings Date": [past_date]},
    )
    assert _check_earnings_calendar("AAPL") is False


def test_check_earnings_calendar_future_date_within_window_is_upcoming(monkeypatch):
    """An earnings date within the next 14 days reports an upcoming catalyst."""
    future_date = datetime.now() + timedelta(days=5)
    monkeypatch.setattr(
        "argus.agents.sentiment.fetchers.fetch_ticker_calendar",
        lambda ticker: {"Earnings Date": [future_date]},
    )
    assert _check_earnings_calendar("AAPL") is True
