# ADR 0007: Injection seam for market data and LLM calls

**Status:** Accepted (predictive for the wiring this ADR describes — written
alongside the code, ahead of PR 6's deterministic test suite, which is the
first consumer that depends on this seam existing).

## Context

Before this PR, every agent that touches the network or an LLM did so by
calling module-level functions or constructing clients directly:
`FundamentalAgent`, `SentimentAgent`, `MacroStatisticalAgent`, and
`RiskStatisticalEngine` imported functions from `argus/data/fetchers.py` and
called them inline; `FundamentalAgent`, `SentimentAgent`, and
`PortfolioManagerAgent` each constructed their own `ChatGroq` client in
`__init__`. `argus/orchestration/graph.py` then built one hard-coded instance
of each agent as a module-level singleton (`macro_agent = MacroStatisticalAgent()`,
etc.) at import time.

This made most of the system untestable without live credentials and network
access: importing `argus.orchestration.graph` — which `api/main.py` and
every integration test need — constructed a real `ChatGroq` client (requiring
`GROQ_API_KEY`) and, transitively through `argus.memory.cultural`, a real
ChromaDB embedding function. There was no way to hand an agent a canned
response and assert on its behavior; the closest available tool was
`unittest.mock.patch` on module-level names, which breaks on refactors and
tests implementation structure rather than behavior.

## Decision

Two Protocols in the new `argus/seams.py`, each with a real and a
fixture-backed implementation:

- **`MarketDataProvider`** wraps the subset of `argus/data/fetchers.py`
  that agent classes call directly: `ohlcv_daily`, `multiple_daily`,
  `fundamentals`, `fred_series`, `macro_bundle`, `news`,
  `social_sentiment`, `vix`. `LiveMarketDataProvider` is a thin delegating
  wrapper (zero behavior change); `FixtureMarketDataProvider` reads plain
  JSON from `tests/fixtures/market_data/`, raising `KeyError` for an
  unrecognized ticker rather than returning fabricated zeros (consistent
  with ADR 0002).
- **`LLMClient`** wraps exactly the part of the three agents' Groq usage
  that was duplicated three times: construct a `ChatGroq` with
  `(model, temperature, max_tokens, api_key)`, invoke it with a system +
  user message pair, and normalize the response into a string (`.content`
  is sometimes a list of content blocks, sometimes a plain string).
  `GroqLLMClient` does exactly that; `FixtureLLMClient` returns pre-recorded
  text instead. Deliberately narrow: markdown-fence stripping and
  `json.loads`/schema validation stay in each agent, because that's
  response *interpretation*, not the client boundary, and duplicating that
  logic into the seam would hide behavior differences between agents that
  should stay visible in their own files.

Every agent that used either boundary now accepts it via constructor
injection, defaulting to the real implementation:
`FundamentalAgent(llm_client=None, market_data=None)`,
`SentimentAgent(llm_client=None, market_data=None)`,
`PortfolioManagerAgent(llm_client=None)`,
`MacroStatisticalAgent(market_data=None)`,
`RiskStatisticalEngine(market_data=None)`. Calling any of them with no
arguments reproduces the exact prior behavior — this is additive, not a
breaking change to any existing call site outside `graph.py`.

`argus/orchestration/graph.py`'s seven module-level singletons are replaced
by `build_graph(market_data=None, fundamental_llm=None, sentiment_llm=None,
portfolio_llm=None)`, which constructs the agents locally (as closures over
the node functions, which used to be module-level and now live inside
`build_graph`) and returns a compiled graph. `graph = build_graph()` at
module level preserves the existing `from argus.orchestration.graph import
graph` import for `api/main.py` and the integration tests — production
wiring is unchanged — but a test can now call `build_graph(market_data=
FixtureMarketDataProvider(), ...)` to get a graph that makes zero network or
LLM calls, without patching anything.

`argus.memory.cultural`'s module-level `cultural_memory` singleton — which
eagerly constructed a ChromaDB `SentenceTransformerEmbeddingFunction` at
*import* time — becomes `get_cultural_memory()`, a lazy singleton accessor.
This isn't part of the MarketDataProvider/LLMClient seam (cultural memory
isn't a candidate for fixture replay — it's inherently stateful), but it was
a necessary companion change: without it, `import argus.orchestration.graph`
would still hard-fail without `sentence-transformers` installed, defeating
the point of making it optional (see below). Likewise `argus/agents/sentiment.py`'s
top-level `from transformers import pipeline` moves inside `get_finbert()`,
matching the lazy-construction pattern that function already used for the
pipeline object itself.

`torch`, `transformers`, and `sentence-transformers` move from
`pyproject.toml`'s unconditional `dependencies` into a new `[project.optional-dependencies]`
group named `models`. CI installs `pip install -e ".[test]"`, which does not
pull `models` — so from this PR forward, CI (and any default install) never
downloads or imports these three packages. Anything that genuinely needs
FinBERT or the vector-memory embedding function still works when a caller
opts into `pip install -e ".[models]"`; it just isn't a tax on every install
and every CI run.

## Consequences

- `scripts/capture_fixtures.py` is a one-off, not part of CI: it reshapes
  four of the ten `langsmith/run-*.json` node traces (a real captured
  session) into `tests/fixtures/market_data/*.json` and
  `tests/fixtures/llm_responses/*.json`. This reuses evidence already
  sitting in the repo instead of hand-authoring synthetic fixtures, and
  keeps the traces load-bearing (referenced by `tests/test_seams.py`)
  rather than 635 KB of inert clutter. `tests/test_seams.py` runs
  `FundamentalAgent`/`SentimentAgent` end-to-end against
  `FixtureMarketDataProvider` + `FixtureLLMClient` and asserts on the real,
  validated output — this is the seam's proof of life, not a mock of it.
- The seam intentionally does not cover every fetcher call site. `argus/data/pipeline.py`
  (MFT intraday ingestion) and `argus/risk/kill_switch.py` (VIX blackout
  check) still call `argus/data/fetchers.py` directly, as does
  `MacroStatisticalAgent.fit_on_history` (an offline HMM-training utility,
  not part of the live `analyze()` path graph.py invokes, and its direct
  `yf.download` call has no `MarketDataProvider` equivalent). These are
  free functions or one-off utilities, not the constructor-injected agent
  classes `graph.py` wires up — extending the seam to them is future work
  if PR 6 or later needs to test them offline, not a gap this PR silently
  papers over.
- `tests/test_integration.py::test_full_graph_smoke` and `tests/test_macro.py`
  needed updating: the former now mocks `get_cultural_memory` explicitly
  (it already mocked the fundamental/sentiment agents; leaving cultural
  memory real would have made a "smoke test" depend on `[models]` being
  installed and would write to a real `./chroma_db` directory on every
  test run), and the latter now injects a stub `MarketDataProvider` instead
  of three separate `monkeypatch.setattr("argus.agents.macro.fetch_*", ...)`
  calls — the seam replacing exactly the kind of test that motivated it.
