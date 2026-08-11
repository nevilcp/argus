"""
scripts/capture_fixtures.py

One-off script converting the ten `langsmith/run-*.json` node traces (one
real end-to-end pipeline run, captured node-by-node) into deterministic
test fixtures under tests/fixtures/. Run manually — not part of CI, not
imported by any test:

    .venv/bin/python scripts/capture_fixtures.py

Each langsmith trace is a single LangGraph node's real input/output for one
session (universe: AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, XOM, ...). This script
picks the traces for `fetch_price_history`, `macro_analysis`,
`fundamental_analysis`, `sentiment_analysis`, and `portfolio_allocation`, and
reshapes their outputs into the schema argus/seams.py's FixtureMarketDataProvider and
FixtureLLMClient expect: existing traces already contain a real session's
worth of ARGUS output, so this reuses evidence already sitting in the repo
instead of hand-authoring synthetic fixtures.

For the LLM-response fixtures, only the fields that genuinely come from the
model (not the ones the calling code overwrites after `json.loads`, per
FundamentalAgent.analyze / SentimentAgent.analyze) are kept — see the field
lists below, each with a comment pointing at the source line that decides
the split.
"""

from __future__ import annotations

import json
from pathlib import Path

LANGSMITH_DIR = Path(__file__).resolve().parent.parent / "langsmith"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

TRACE_FETCH_PRICE_HISTORY = "run-019e639a-4827-7c31-912c-45706a3854e4.json"
TRACE_MACRO_ANALYSIS = "run-019e639a-4e5e-75d3-bf34-c3d9084953b1.json"
TRACE_FUNDAMENTAL_ANALYSIS = "run-019e639a-5bbb-70a1-8d90-a262c600572d.json"
TRACE_SENTIMENT_ANALYSIS = "run-019e639a-5bbc-7511-b692-0adc24c2fe5a.json"
TRACE_PORTFOLIO_ALLOCATION = "run-019e639b-41cf-7741-be49-ab683c2adb65.json"


def _load_trace(filename: str) -> dict:
    with open(LANGSMITH_DIR / filename) as f:
        return json.load(f)


def capture_price_history() -> None:
    """tests/fixtures/market_data/price_history.json — feeds FixtureMarketDataProvider.ohlcv_daily."""
    trace = _load_trace(TRACE_FETCH_PRICE_HISTORY)
    price_history = trace["outputs"]["price_history"]
    out = FIXTURES_DIR / "market_data" / "price_history.json"
    out.write_text(json.dumps(price_history, indent=2))
    print(f"wrote {out} ({len(price_history)} tickers)")


def capture_fundamentals_and_llm_responses() -> None:
    """tests/fixtures/market_data/fundamentals.json and llm_responses/fundamental.json."""
    trace = _load_trace(TRACE_FUNDAMENTAL_ANALYSIS)
    signals = trace["outputs"]["fundamental_signals"]

    # Fields fetch_fundamentals() itself returns (argus/data/fetchers.py:264-277).
    fundamentals_fields = [
        "pe_ttm", "revenue_growth_yoy", "operating_margin", "net_margin",
        "fcf_yield", "debt_to_equity", "current_ratio", "roe", "roic",
        "sector", "industry", "marketCap", "p_fcf",
    ]
    # Everything else in the parsed JSON is overwritten by fetched data (fundamental.py:340-351)
    llm_fields = ["signal", "conviction", "moat_score", "reasoning"]

    fundamentals_fixture = {}
    llm_responses_fixture = {}
    for ticker, sig in signals.items():
        record = {k: sig.get(k) for k in fundamentals_fields}
        record["as_of_date"] = sig.get("data_as_of_date")
        fundamentals_fixture[ticker] = record
        llm_responses_fixture[ticker] = json.dumps({k: sig[k] for k in llm_fields})

    out1 = FIXTURES_DIR / "market_data" / "fundamentals.json"
    out1.write_text(json.dumps(fundamentals_fixture, indent=2))
    print(f"wrote {out1} ({len(fundamentals_fixture)} tickers)")

    out2 = FIXTURES_DIR / "llm_responses" / "fundamental.json"
    out2.write_text(json.dumps(llm_responses_fixture, indent=2))
    print(f"wrote {out2} ({len(llm_responses_fixture)} tickers)")


def capture_sentiment_llm_responses() -> None:
    """tests/fixtures/llm_responses/sentiment.json."""
    trace = _load_trace(TRACE_SENTIMENT_ANALYSIS)
    signals = trace["outputs"]["sentiment_signals"]

    # Everything else is FinBERT/news/social metrics computed before the call (sentiment.py:405-418)
    llm_fields = ["signal", "conviction", "sentiment_decay_risk", "reasoning"]

    llm_responses_fixture = {
        ticker: json.dumps({k: sig[k] for k in llm_fields}) for ticker, sig in signals.items()
    }
    out = FIXTURES_DIR / "llm_responses" / "sentiment.json"
    out.write_text(json.dumps(llm_responses_fixture, indent=2))
    print(f"wrote {out} ({len(llm_responses_fixture)} tickers)")

    # No trace captured these raw; empty maps make the fixture provider degrade the same
    # way fetchers.py does for an unrecognized ticker, rather than raising
    for name in ("news", "social_sentiment"):
        empty_out = FIXTURES_DIR / "market_data" / f"{name}.json"
        if not empty_out.exists():
            empty_out.write_text(json.dumps({}))
            print(f"wrote {empty_out} (placeholder — not captured by any trace)")


def capture_macro_bundle() -> None:
    """tests/fixtures/market_data/macro_bundle.json.

    macro_analysis's trace captures the *derived* MacroContext, not the raw
    fetch_macro_bundle() dict (MacroStatisticalAgent.analyze() fetches and
    derives in one step — there's no separate node for the raw fetch). This
    reconstructs the subset of macro_bundle()'s schema recoverable from the
    derived context; `t10yie` isn't present in MacroContext and is left
    None, which fetch_macro_bundle() callers already handle (macro.py
    treats any missing bundle field as None, not an error).
    """
    trace = _load_trace(TRACE_MACRO_ANALYSIS)
    ctx = trace["outputs"]["macro_context"]
    bundle = {
        "fed_funds": ctx.get("fed_funds"),
        "unemployment": ctx.get("unemployment"),
        "t10y2y": ctx.get("t10y2y"),
        "t10yie": None,
        "consumer_sentiment": ctx.get("consumer_sentiment"),
        "cpi_yoy": ctx.get("cpi_yoy"),
        "vix": ctx.get("vix_level"),
    }
    out = FIXTURES_DIR / "market_data" / "macro_bundle.json"
    out.write_text(json.dumps(bundle, indent=2))
    print(f"wrote {out}")


def capture_session_states() -> None:
    """tests/fixtures/market_data/session_states.json.

    node_fetch_price_history doesn't compute session_states in code (they're
    pre-populated by the API layer from the MFT live cache — see the node's
    docstring in graph.py); this trace happens to have captured them
    alongside price_history for the same session, so they're the one thing
    this script can't get from a MarketDataProvider fixture at all — a
    golden-path test has to pass them directly as part of the initial
    ARGUSState, keyed by ticker exactly as TechnicalStatisticalAgent.analyze
    expects.
    """
    trace = _load_trace(TRACE_FETCH_PRICE_HISTORY)
    session_states = trace["outputs"]["session_states"]
    out = FIXTURES_DIR / "market_data" / "session_states.json"
    out.write_text(json.dumps(session_states, indent=2))
    print(f"wrote {out} ({len(session_states)} tickers)")


def capture_portfolio_llm_response() -> None:
    """tests/fixtures/llm_responses/portfolio.json.

    Unlike fundamental/sentiment, PortfolioManagerAgent.allocate() makes one
    LLM call for the whole universe, not one per ticker — so this fixture is
    a single JSON blob, not a per-ticker map. Only fields the model actually
    produces are kept (argus/agents/portfolio.py:296-314): allocation_usd and
    stop_loss are recomputed server-side from investable capital and the risk
    engine's own stop, and cash_reserve_pct is forced to
    1 - sum(allocation_pct) regardless of what the model returned.
    """
    trace = _load_trace(TRACE_PORTFOLIO_ALLOCATION)
    alloc = trace["outputs"]["portfolio_allocation"]

    position_fields = ["ticker", "allocation_pct", "composite_conviction", "thesis", "time_horizon"]
    llm_response = {
        "portfolio": [{k: pos.get(k) for k in position_fields} for pos in alloc["portfolio"]],
        "cash_reserve_pct": alloc.get("cash_reserve_pct"),
        "expected_sharpe": alloc.get("expected_sharpe"),
        "rebalance_trigger": alloc.get("rebalance_trigger"),
    }

    out = FIXTURES_DIR / "llm_responses" / "portfolio.json"
    out.write_text(json.dumps(llm_response, indent=2))
    print(f"wrote {out}")


def main() -> None:
    """Runs all fixture-capture steps in sequence, writing outputs under tests/fixtures/."""
    (FIXTURES_DIR / "market_data").mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "llm_responses").mkdir(parents=True, exist_ok=True)
    capture_price_history()
    capture_fundamentals_and_llm_responses()
    capture_sentiment_llm_responses()
    capture_macro_bundle()
    capture_session_states()
    capture_portfolio_llm_response()


if __name__ == "__main__":
    main()
