# ARGUS

Multi-agent financial intelligence system orchestrating specialist LLMs and statistical models for quantitative equity research.

![Python Version](https://img.shields.io/badge/python-%E2%89%A53.12-blue)
![License](https://img.shields.io/badge/license-Research_Only-red)

> **RESEARCH PROJECT ONLY — NOT FINANCIAL ADVICE.** ARGUS is not registered with the SEC or any regulatory body. All outputs are for educational and research purposes only. Do not make investment decisions based on this system's output.

## Overview

ARGUS is a multi-agent artificial intelligence system for quantitative equity research. By orchestrating six specialized agents (Technical, Macro, Fundamental, Sentiment, Risk, and Portfolio) as a directed acyclic graph (DAG), ARGUS synthesizes structured investment theses and asset allocations. See [`limitations.md`](limitations.md) for the current, honestly-stated gaps in this system.

- **Parallel Agent Orchestration** — delegates analysis to domain-specific statistical models and LLMs (see [ADR 0001](docs/adr/0001-statistical-vs-llm-agent-split.md) for the split rationale) rather than a single monolithic prompt.
- **Point-in-Time Fixture Replay** — masks future data when replaying recorded sessions through the real graph. There is no multi-year backtest — see [`docs/adr/0009-no-multiyear-backtest.md`](docs/adr/0009-no-multiyear-backtest.md) — and residual risks (e.g. LLM parametric memory of well-known tickers) are tracked in [`limitations.md`](limitations.md).
- **High-Frequency Data Pipeline** — buffers intraday OHLCV data using an in-memory SQLite ring buffer, enabling near-real-time technical indicator derivation.
- **Dual-Gate Rate Limiting** — enforces strict per-minute and per-day token/request quotas, preventing API exhaustion and cost overruns.
- **Automated Drawdown Kill-Switch** — continuously monitors portfolio risk metrics, blocks new positions and halts the system during extreme volatility (e.g., VIX spikes) or excess drawdown, requiring a manual reset before trading resumes.
- **Persistent Cultural Memory** — stores past decision rationale in a ChromaDB vector vault, allowing the Portfolio agent to recall past reasoning across similar market regimes as outcomes accumulate and resolve.

## Architecture

ARGUS operates as a stateful LangGraph workflow. Data ingestion is handled by an asynchronous pipeline, while decision logic is distributed across parallel analyst nodes before being aggregated and subjected to a quantitative risk audit. See [`docs/adr/`](docs/adr/) for the reasoning behind specific design decisions.

```mermaid
flowchart TD
    %% ── External Data Sources ────────────────────────────────────────────
    subgraph DATASOURCES["🌐 External Data Sources"]
        direction LR
        YF["yfinance\nOHLCV · Fundamentals · VIX"]
        FRED["FRED API\nFedfunds · CPI · Yield Curve · Unemployment"]
        NEWS["NewsAPI\nHeadlines"]
        GT["Google Trends\nSearch Interest"]
    end

    %% ── MFT Data Pipeline ────────────────────────────────────────────────
    subgraph PIPELINE["⚙️ MFT Data Pipeline (Background asyncio task)"]
        direction TB
        FL["_fetch_loop\nEvery 5 min · bulk 5m candles\n2-day history on first fetch"]
        BUF["OHLCVBuffer\nIn-memory SQLite · 78-bar rolling\nINSERT OR REPLACE — safe for bulk re-insert"]
        SL["_session_loop\nEvery 30 min → compress_all\nRSI · MACD · BB · ATR · ADX · VWAP · Momentum"]
        CACHE["_live_session_cache\nIn-memory dict · ticker → feature dict\nUpdated via on_session_ready callback"]
    end

    YF -->|"intraday 5m · 2-day window"| FL
    FL --> BUF --> SL --> CACHE

    %% ── FastAPI Gateway ──────────────────────────────────────────────────
    subgraph API["🚪 FastAPI Gateway  (api/main.py)"]
        direction TB
        GATE["Market Hours Gate\n503 if market closed\n503 if cache not populated"]
        REG["register_tickers()\nAdds new tickers to pipeline\nuniverse on first call"]
        INJ["Inject session_states\nfrom _live_session_cache\ninto ARGUSState"]
    end

    CACHE -->|"live intraday states"| INJ
    INJ --> GATE --> REG

    %% ── LangGraph DAG ────────────────────────────────────────────────────
    subgraph GRAPH["🔁 LangGraph DAG  ·  SQL-Checkpointed  (argus_graph.db)"]
        direction TB

        N0["node: fetch_price_history\nyfinance 1y daily · ThreadPool x5\nPopulates price_history only\n(session_states passed through from API)"]
        N1["node: macro_analysis\nMacroStatisticalAgent\nGaussian HMM · 3-state regime\nFRED + VIX features"]

        subgraph PARALLEL["Parallel Fan-Out  (Send API)"]
            direction LR
            N2["node: technical_analysis\nTechnicalStatisticalAgent\nReads MFT session_states\nRSI · MACD · BB · ADX · VWAP · Momentum\n→ score ∈ [-1,+1]"]
            N3["node: fundamental_analysis\nFundamentalAgent\nLlama-3.3-70b-versatile\n7-day cache · yfinance.info"]
            N4["node: sentiment_analysis\nSentimentAgent\nFinBERT + Llama-3.1-8b\nNews · Google Trends"]
            N5["node: retrieve_cultural_memory\nChromaDB vault\nRetrieve past wisdom and warnings\nscoped by macro regime"]
        end

        N6["node: signal_aggregation\nHybridSignalAggregator\nWeighted conviction voting\ntech 0.35 · fund 0.35 · sent 0.30\nMacro multipliers applied\nContraction regime override"]

        N7["node: risk_evaluation\nRiskStatisticalEngine\nVaR · CVaR · Beta · SLSQP optimizer\nPortfolio covariance from daily price_history\nHalf-Kelly sizing · VIX scaling"]

        N8["node: portfolio_allocation\nPortfolioManagerAgent\nLlama-3.3-70b-versatile\nHalf-Kelly sizing · Pydantic-validated output\nCultural wisdom injected"]

        N9["node: log_decisions\nARGUSDecision snapshots\nPersisted to ChromaDB cultural memory"]

        DONE(["END"])
    end

    %% ── Graph Edges ──────────────────────────────────────────────────────
    REG --> N0
    N0 --> N1
    N1 -->|"Send()"| N2 & N3 & N4 & N5
    N2 & N3 & N4 & N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9 --> DONE

    %% ── Data source wiring ───────────────────────────────────────────────
    YF -->|"1y daily OHLCV"| N0
    FRED -->|"FEDFUNDS · CPIAUCSL\nT10Y2Y · UNRATE"| N1
    YF -->|"^VIX"| N1
    YF -->|"Ticker.info"| N3
    NEWS & GT --> N4

    %% ── Safety Layer (Independent of Graph) ─────────────────────────────
    subgraph SAFETY["🛡️ Safety Layer  (Independent of Graph)"]
        direction LR
        KS["KillSwitch Daemon\nBackground thread · 60s poll\nDrawdown: 8/12/18% by tolerance\nVIX Blackout ≥ 35 → block positions\nHalt file → manual reset required"]
        GOV["RateLimitGovernor\nSingleton · thread-safe\nRPM sliding window (sleep)\nRPD hard cap (exception)\nTPM warning"]
    end

    YF -->|"^VIX every 60s"| KS
    KS -->|"is_halted / new_positions_allowed\nchecked before graph invoke"| GATE
    GOV -.-|"wait_if_needed() before\nevery LLM call"| N3 & N4 & N8

    %% ── Styles ───────────────────────────────────────────────────────────
    classDef stat   fill:#1e3a5f,stroke:#4a90d9,color:#e8f4f8
    classDef llm    fill:#2d1b4e,stroke:#9b59b6,color:#f0e6ff
    classDef safety fill:#3d1a1a,stroke:#e74c3c,color:#ffe8e8
    classDef data   fill:#1a3d2b,stroke:#27ae60,color:#e8f5e9
    classDef io     fill:#2c2c1a,stroke:#f39c12,color:#fff9e6
    classDef api    fill:#1a2c3d,stroke:#3498db,color:#e8f4ff

    class N1,N2,N7 stat
    class N3,N4,N8 llm
    class KS,GOV,PIT safety
    class BUF,N5,CACHE data
    class N0,FL,SL io
    class GATE,REG,INJ,N6,N9 api
```

The system runs in two primary modes:
1. **Live Mode**: Subscribes to the intraday buffer and evaluates current market conditions. 
2. **Backtest Mode**: Invokes the pipeline with point-in-time constraints and evaluates an explicitly scoped historical date window.

## Prerequisites

- **Python**: ≥ 3.11
- **Docker**: Optional, for containerized deployments (`Dockerfile.api`, one service).
- **System Memory**: Minimum 4GB RAM only if installing the optional `models` extra (`pip install -e ".[models]"`), which pulls in local FinBERT inference. The default install and the test suite don't need it.

## Installation

### Method 1: Docker (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/nevilcp/argus.git
cd argus

# 2. Configure environment keys
cp .env.example .env
# Edit .env and supply your required API keys (see Configuration section)

# 3. Build and launch
docker compose up --build -d
```
`docker-compose.yml` defines a single `argus-api` service (`Dockerfile.api`) and passes `GROQ_API_KEY`, `GOOGLE_AI_API_KEY`, `FRED_API_KEY`, and `NEWSAPI_KEY` through from `.env`.

### Method 2: Local Source
```bash
# 1. Clone and enter directory
git clone https://github.com/nevilcp/argus.git
cd argus

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev,test]"

# 4. Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Verify Installation
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model_versions": {
    "synthesis": "llama-3.3-70b-versatile",
    "sentiment": "llama-3.1-8b-instant",
    "fundamental": "llama-3.3-70b-versatile",
    "finbert": "ProsusAI/finbert"
  },
  "can_make_calls": true,
  "governor_report": { "...": "per-model request/token usage, see /governor/report" }
}
```

## Configuration

Required environment variables must be placed in a `.env` file at the project root.

| Variable | Required? | Description / Default |
|----------|-----------|-----------------------|
| `GROQ_API_KEY` | Yes | Authenticates Llama 3.3-70b/3.1-8b for core agent synthesis. |
| `FRED_API_KEY` | Yes | Retrieves macroeconomic indicators (CPI, Fed Funds, UNRATE). |
| `NEWSAPI_KEY` | Yes | Fetches recent headlines for FinBERT sentiment scoring. |
| `GOOGLE_AI_API_KEY` | Optional | Gemini fallback for LLM synthesis. |
| `POLYGON_API_KEY` | Declared, unused | Present in `.env.example`/`Settings`; no fetcher reads it yet. |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Declared, unused | Present in `.env.example`/`Settings`; no fetcher reads it yet. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional | Enables Langfuse tracing for LLM observability. |
| `LANGFUSE_HOST` | Optional | Langfuse instance URL. Default: `https://cloud.langfuse.com` |

Every field above defaults to an empty string (see `argus/config.py`) — the process starts without them, but the feature that depends on a missing key degrades or is skipped rather than failing outright (e.g. macro regime fitting logs a warning and continues).

*Note: Google Trends sentiment analysis does not require an API key.*

## Quick Start / Usage

For a guided walkthrough — including a fully offline path and a live-API
path, plus what to do when a demo step fails — see
[`docs/demo-guide.md`](docs/demo-guide.md).

### Replaying a recorded session
There is no `/backtest` API endpoint — see
[`docs/adr/0009-no-multiyear-backtest.md`](docs/adr/0009-no-multiyear-backtest.md)
for why a multi-year walk-forward backtest isn't offered.
`scripts/replay_backtest.py` replays recorded fixture sessions through the
real graph instead. It needs no API keys and makes no required network calls:

```bash
.venv/bin/python -m scripts.replay_backtest
```

### Pre-registered evaluation

`scripts/run_evaluation.py` scores replayed decisions against real forward
returns (rank IC, hit-rate-with-dead-band, open-loop vs. closed-loop
reliability weighting), per
[`docs/adr/0012-pre-registered-evaluation.md`](docs/adr/0012-pre-registered-evaluation.md).
Unlike the other commands on this page, it **requires network access** — it
fetches real forward daily closes to score against. Results are committed at
[`docs/evaluation-results.md`](docs/evaluation-results.md) — including
where reliability weighting did not measurably help.

```bash
.venv/bin/python -m scripts.run_evaluation
```

### Reconciling outcomes

`scripts/reconcile_outcomes.py` is the other half of the decision→outcome
loop described in [ADR 0010](docs/adr/0010-closing-the-decision-outcome-loop.md):
it reads decisions logged to the LangGraph checkpoint DB, resolves the ones
past their horizon against live prices, and writes outcomes back to cultural
memory so reliability weighting has something to learn from. It also
requires network access, and is meant to run on a schedule (e.g. a daily
cron), not as part of a demo:

```bash
.venv/bin/python -m scripts.reconcile_outcomes
```

## Project Structure

```text
.
├── api/                  # FastAPI entrypoint, HTTP routing, and market-hours gating
├── argus/
│   ├── agents/           # Core AI components (Macro, Fundamental, Sentiment, Risk, Portfolio)
│   ├── backtesting/      # Return-series metrics and fixture-session replay (see ADR 0009)
│   ├── data/             # MFT Pipeline, OHLCV SQLite buffer, and external API fetchers
│   ├── memory/           # ChromaDB-backed cultural wisdom retrieval mechanisms
│   ├── orchestration/    # LangGraph definition, aggregator, safety governors, reconciliation
│   ├── risk/             # Kill-switch daemon and drawdown/VIX blackout logic
│   ├── schemas/          # Pydantic data contracts defining all inter-agent signaling
│   ├── config.py         # Typed Settings (env vars) shared across all modules
│   ├── params.py         # Provenance-tagged numeric constants (see ADR 0006)
│   └── seams.py          # Market-data/LLM injection seam used by tests and replay (ADR 0007)
├── scripts/              # replay_backtest, run_evaluation, reconcile_outcomes, capture_fixtures
├── chroma_db/            # (Auto-generated) Local persistent vector database
├── docs/                 # Demo guide, case study, evaluation results, project notes
│   ├── adr/              # Architecture decision records
│   └── historical/       # Pre-rebuild docs, kept as evidence for the case study
├── tests/                # Deterministic unit tests covering agents and pipelines
├── argus_graph.db        # (Auto-generated) SQLite state checkpointer for LangGraph
├── limitations.md        # Honestly-stated data, modeling, and architectural gaps
├── pyproject.toml        # Project dependencies and build configuration
├── Dockerfile.api        # Container image for the FastAPI service
└── docker-compose.yml    # Single-service (argus-api) orchestration definition
```

## Testing

ARGUS includes a suite of deterministic unit tests that execute without triggering external API calls (via `pytest-mock`), covering core agents, rate limit governors, and the intraday MFT pipeline.

**Run the full test suite:**
```bash
pytest tests/ -v
```

- **Categories**: Tests cover Pydantic validation boundaries, mathematical boundaries (e.g., Half-Kelly position sizing constraints), caching TTL expiration logic, and thread-safe rate limit assertions.
- **Approximate Run Time**: ~35 seconds for 85 tests across 21 files (see [ADR 0008](docs/adr/0008-deterministic-test-suite.md) for why the suite makes no network calls).
- **CI gate**: `.github/workflows/ci.yml` additionally runs `ruff check .` and `mypy argus/` (pinned `ruff==0.16.2`, `mypy==2.3.0`) before the test step.

## Contributing

N/A — ARGUS is currently maintained as a private research and portfolio project. External pull requests are closed, though forks for personal experimentation are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the standing agent rules used while developing this repo.

## Roadmap

The rebuild planned in [issue #1](https://github.com/nevilcp/argus/issues/1) is complete: the decision→outcome loop is closed, the eleven unwired mechanisms are wired or deleted, and the result is measured against a pre-registered bar. [`docs/case-study.md`](docs/case-study.md) is the investigation behind it, written up end to end — including where the change did not help.

Next, in rough priority order: accumulate enough resolved outcomes for reliability weighting to have evidence to work with (the evaluation ran at n=6 with zero resolved outcomes, so weighting could not yet differ from the 0.5 prior), then re-run the pre-registered evaluation at a sample size that can actually reject the null.

## License & Acknowledgements

**License**: Released for Educational and Research Purposes Only.

**Acknowledgements**:
- [LangGraph](https://github.com/langchain-ai/langgraph) for cyclic agent orchestration.
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for financial domain sentiment classification.
