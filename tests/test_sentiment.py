from datetime import datetime, timedelta

import pytest

from argus.agents.sentiment import (
    SentimentAgent,
    _build_synthesis_prompt,
    _check_earnings_calendar,
    aggregate_finbert_scores,
)
from argus.governor import RateLimitExceeded
from argus.infra.diagnostics import Diagnostics
from argus.schemas.prompting import UNTRUSTED_GUARD
from tests.helpers.market_data import FakeMarketData

_NEUTRAL_RESPONSE = '{"signal": "NEUTRAL", "conviction": 0.3, "sentiment_decay_risk": "LOW", "reasoning": "n/a"}'


def _no_upcoming_earnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the calendar lookup, which would otherwise reach yfinance on every analyze()."""
    monkeypatch.setattr("argus.agents.sentiment._check_earnings_calendar", lambda ticker, market_data: False)


def test_aggregate_finbert_scores_empty():
    res = aggregate_finbert_scores([])
    assert res == {"net": 0.0, "pct_pos": 0.0, "pct_neg": 0.0, "confidence": 0.0}


def test_aggregate_finbert_scores_decay():
    scored = [
        {"numeric": 1.0, "label": "positive"},
        {"numeric": -1.0, "label": "negative"},
    ]
    # Weight later chronological positions more heavily; positional decay does not inspect timestamps.
    res = aggregate_finbert_scores(scored)
    assert res["net"] < 0.0
    assert res["pct_pos"] == 0.5
    assert res["pct_neg"] == 0.5
    assert res["confidence"] == 0.2


def test_aggregate_finbert_scores_confidence_cap():
    scored = [{"numeric": 1.0, "label": "positive"} for _ in range(15)]
    res = aggregate_finbert_scores(scored)
    assert res["confidence"] == 1.0


class _RecordingLLMClient:
    """Stub LLMClient returning a fixed response while recording the prompt it received."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.last_user_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return self._response_text


def test_analyze_degrades_the_ticker_on_governor_exhaustion_without_retrying(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

    market_data = FakeMarketData(news=[])

    class _RateLimitedLLMClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            raise RateLimitExceeded("gpt-oss-20b quota exhausted")

    llm = _RateLimitedLLMClient()
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    diagnostics = Diagnostics()
    signal = agent.analyze("AAPL", diagnostics=diagnostics)

    assert signal is None
    assert llm.calls == 1
    assert len(diagnostics.messages) == 1
    assert "rate limited" in diagnostics.messages[0]


def test_analyze_degrades_the_ticker_without_repairing_a_bad_response(monkeypatch):
    _no_upcoming_earnings(monkeypatch)
    monkeypatch.setattr("argus.structured_output.time.sleep", lambda s: None)

    market_data = FakeMarketData(news=[])
    llm = _RecordingLLMClient("not json")
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    diagnostics = Diagnostics()
    signal = agent.analyze("AAPL", diagnostics=diagnostics)

    assert signal is None
    assert len(diagnostics.messages) == 1
    assert "Your previous response was invalid" not in llm.last_user_prompt


def test_analyze_flags_unavailable_news_in_the_llm_prompt(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

    market_data = FakeMarketData(news=None)
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_volume_7d == 0
    assert signal.news_scored_count == 0
    assert signal.news_data_available is False
    # Persist unavailable sentiment as None so a failed fetch cannot resemble a measured neutral score.
    assert signal.finbert_net_score is None
    assert "news_data_available: False" in llm.last_user_prompt


def test_analyze_reports_availability_true_with_real_data(monkeypatch):
    _no_upcoming_earnings(monkeypatch)
    # Stub FinBERT because nonempty headlines would otherwise load the real model in an availability test.
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

    market_data = FakeMarketData(
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
    _no_upcoming_earnings(monkeypatch)

    market_data = FakeMarketData(news=[])
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    sleep_calls = []
    monkeypatch.setattr("argus.agents.sentiment.time.sleep", lambda s: sleep_calls.append(s))

    agent.batch_analyze(["AAPL", "MSFT", "GOOGL"])

    assert sleep_calls == []


def test_scored_articles_are_sorted_by_published_at_before_aggregation(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

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

    market_data = FakeMarketData(
        news=[
            {"title": "stale bad news", "published_at": "2024-01-01T00:00:00Z"},
            {"title": "fresh good news", "published_at": "2024-06-01T00:00:00Z"},
        ],
    )
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.finbert_net_score > 0.0


def test_analyze_tolerates_a_raising_news_provider(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

    market_data = FakeMarketData(news=RuntimeError("provider outage"))
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert signal is not None
    assert signal.news_data_available is False
    assert signal.finbert_net_score is None


def test_batch_analyze_tolerates_a_single_ticker_raising(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

    market_data = FakeMarketData(news=[])
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)
    original_analyze = agent.analyze

    def flaky_analyze(ticker, company_name=None, diagnostics=None):
        if ticker == "MSFT":
            raise RuntimeError("boom")
        return original_analyze(ticker, company_name=company_name, diagnostics=diagnostics)

    monkeypatch.setattr(agent, "analyze", flaky_analyze)

    results, errors = agent.batch_analyze(["AAPL", "MSFT", "GOOGL"])

    assert "MSFT" not in results
    assert "AAPL" in results
    assert "GOOGL" in results
    assert any("MSFT" in e for e in errors)


def test_batch_analyze_passes_company_name_for_known_tickers(monkeypatch):
    _no_upcoming_earnings(monkeypatch)

    received = {}

    def record_news(ticker, company_name, days_back=7):
        received[ticker] = company_name
        return []

    market_data = FakeMarketData(news=record_news)
    llm = _RecordingLLMClient(_NEUTRAL_RESPONSE)
    agent = SentimentAgent(llm_client=llm, market_data=market_data)

    agent.batch_analyze(["V", "ZZZZ"])

    assert received["V"] == "Visa"
    assert received["ZZZZ"] == "ZZZZ"


def test_check_earnings_calendar_past_date_is_not_upcoming():
    past_date = datetime.now() - timedelta(days=15)
    market_data = FakeMarketData(news=[], ticker_calendar={"Earnings Date": [past_date]})
    assert _check_earnings_calendar("AAPL", market_data) is False


def test_check_earnings_calendar_future_date_within_window_is_upcoming():
    future_date = datetime.now() + timedelta(days=5)
    market_data = FakeMarketData(news=[], ticker_calendar={"Earnings Date": [future_date]})
    assert _check_earnings_calendar("AAPL", market_data) is True


def test_synthesis_prompt_wraps_the_metrics_and_tells_the_model_not_to_obey_them():
    prompt = _build_synthesis_prompt("AAPL", {"net_finbert_score": 0.2})

    assert prompt.startswith('<sentiment_data ticker="AAPL" as_of="intraday">\n')
    assert "</sentiment_data>\n" + UNTRUSTED_GUARD in prompt


def test_failed_calendar_is_distinguished_from_no_catalyst(monkeypatch):
    monkeypatch.undo()
    provider = FakeMarketData(ticker_calendar=OSError("unavailable"))
    assert _check_earnings_calendar("AAPL", provider) is None
