"""
Tests exercising argus/seams.py's fixture-backed implementations against the
real agent classes — the point of the injection seam (ADR 0007) is that
FundamentalAgent/SentimentAgent produce a real, schema-valid signal from
fixture data with zero network or LLM calls. If these fixtures ever go
stale or the seam breaks, these tests fail; nothing here is decorative.
"""

import json
from pathlib import Path

from argus.agents.fundamental import FundamentalAgent
from argus.agents.sentiment import SentimentAgent
from argus.schemas.signals import FundamentalSignal, SentimentSignal
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _single_response_llm(fixture_file: str, ticker: str) -> FixtureLLMClient:
    with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
        responses = json.load(f)
    # Fixture only handles one ticker per instance — every call in a single
    # analyze() invocation gets the same recorded response for that ticker.
    return FixtureLLMClient({"only": responses[ticker]}, key_fn=lambda _prompt: "only")


def test_fundamental_agent_with_fixtures_produces_valid_signal():
    market_data = FixtureMarketDataProvider()
    llm_client = _single_response_llm("fundamental.json", "AAPL")
    agent = FundamentalAgent(llm_client=llm_client, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert isinstance(signal, FundamentalSignal)
    assert signal.ticker == "AAPL"
    assert signal.signal.value == "BEARISH"
    assert signal.conviction == 0.75
    assert signal.moat_score == 8
    # Fetched (not LLM-produced) fields came through the fixture market data provider
    assert signal.sector == "Technology"
    assert signal.marketCap is not None


def test_sentiment_agent_with_fixtures_produces_valid_signal():
    market_data = FixtureMarketDataProvider()
    llm_client = _single_response_llm("sentiment.json", "AAPL")
    agent = SentimentAgent(llm_client=llm_client, market_data=market_data)

    signal = agent.analyze("AAPL")

    assert isinstance(signal, SentimentSignal)
    assert signal.ticker == "AAPL"
    assert signal.signal.value == "NEUTRAL"
    assert signal.conviction == 0.4
    assert signal.sentiment_decay_risk == "LOW"


def test_fixture_market_data_provider_ohlcv_daily_matches_fixture():
    market_data = FixtureMarketDataProvider()
    df = market_data.ohlcv_daily("AAPL")
    assert not df.empty
    assert "close" in df.columns


def test_fixture_market_data_provider_missing_ticker_raises():
    market_data = FixtureMarketDataProvider()
    try:
        market_data.fundamentals("NOT_A_REAL_TICKER")
        assert False, "expected KeyError for a ticker with no fixture"
    except KeyError:
        pass
