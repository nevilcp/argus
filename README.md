# ARGUS v2 — Multi-Agent Financial Intelligence System

> **RESEARCH PROJECT ONLY — NOT FINANCIAL ADVICE.** ARGUS is not registered with the SEC or any regulatory body. All outputs are for educational and research purposes only. Do not make investment decisions based on this system's output.

ARGUS v2 is an institutional-grade, multi-agent AI system that performs parallel quantitative analysis across six specialist agents — technical, macro, fundamental, sentiment, risk, and portfolio — to produce structured investment theses over a configurable equity universe. The system enforces strict look-ahead bias prevention via point-in-time data gating, runs as a LangGraph DAG with SQL-backed checkpointing, and ships with a Backtrader walk-forward validation engine and a FastAPI + Streamlit frontend.

---

## Architecture

```mermaid
flowchart TD
    %% ── External Data Sources ───────────────────────────────────────────
    subgraph DATASOURCES["🌐 External Data Sources"]
        direction LR
        YF["yfinance\nOHLCV · Fundamentals · VIX"]
        FRED["FRED API\nFedfunds · CPI · Yield Curve · Unemployment"]
        NEWS["NewsAPI\nHeadlines"]
        ST["StockTwits\nBull/Bear · Volume"]
        GT["Google Trends\nSearch Interest"]
    end

    %% ── MFT Data Pipeline ───────────────────────────────────────────────
    subgraph PIPELINE["⚙️ MFT Data Pipeline (Async)"]
        direction TB
        FL["_fetch_loop\nEvery 5 min · 5m candles\nBatches of 4 tickers"]
        BUF["OHLCVBuffer\nIn-memory SQLite · 78-bar rolling"]
        SL["_session_loop\nEvery 30 min → compress_all\nRSI · MACD · BB · ATR · ADX · VWAP · Momentum"]
    end

    YF -->|"intraday 5m"| FL
    FL --> BUF --> SL

    %% ── LangGraph DAG ───────────────────────────────────────────────────
    subgraph GRAPH["🔁 LangGraph DAG  ·  SQL-Checkpointed  (argus_graph.db)"]
        direction TB

        N0["node: fetch_price_history\nyfinance 1y daily · ThreadPool x5"]
        N1["node: macro_analysis\nMacroStatisticalAgent\nGaussian HMM · 3-state regime\nFRED + VIX features"]

        subgraph PARALLEL["Parallel Fan-Out  (Send API)"]
            direction LR
            N2["node: technical_analysis\nTechnicalStatisticalAgent\nRSI·MACD·BB·ADX·VWAP·Momentum\n→ score ∈ [-1,+1]"]
            N3["node: fundamental_analysis\nFundamentalAgent\nGemini 3.1 Flash Lite\n7-day cache · yfinance.info"]
            N4["node: sentiment_analysis\nSentimentAgent\nFinBERT + Llama 3.1-8b\nNews · StockTwits · Google Trends"]
            N5["node: retrieve_cultural_memory\nChromaDB vault\nRetrieve past wisdom & warnings"]
        end

        N6["node: risk_evaluation\nRiskStatisticalEngine\nVaR · CVaR · Beta · Half-Kelly\nUses macro VIX level"]

        subgraph AGG["node: signal_aggregation"]
            direction TB
            AGG1["HybridSignalAggregator\nWeighted conviction voting\nMacro multipliers applied"]
            DEBATE{"Fundamental ≠ Sentiment\nAND best_score < 52%?"}
            AGG2["Llama 3.1-8b\nConflict Arbitration\n+0.25 weight to winner"]
        end

        N7["node: portfolio_allocation\nPortfolioManagerAgent\nLlama 3.3-70b\nHalf-Kelly sizing\nPydantic-validated output"]
        N8["node: log_decisions\nDecision logger"]
        DONE(["END"])
    end

    %% ── Graph Edges ─────────────────────────────────────────────────────
    SL -->|"on_session_ready callback"| N0
    YF -->|"daily OHLCV"| N0
    N0 --> N1
    N1 -->|"Send()"| N2 & N3 & N4 & N5
    N2 & N3 & N4 & N5 --> N6
    N6 --> AGG1
    AGG1 --> DEBATE
    DEBATE -->|"No"| N7
    DEBATE -->|"Yes"| AGG2 --> N7
    N7 --> N8 --> DONE

    %% ── Data source wiring ──────────────────────────────────────────────
    FRED -->|"FEDFUNDS · CPIAUCSL\nT10Y2Y · UNRATE"| N1
    YF -->|"^VIX"| N1
    YF -->|"Ticker.info"| N3
    NEWS & ST & GT --> N4

    %% ── Safety Layer (Independent Threads) ──────────────────────────────
    subgraph SAFETY["🛡️ Safety Layer  (Independent of Graph)"]
        direction LR
        KS["KillSwitch Daemon\nBackground thread · 60s poll\nDrawdown: 8/12/18% by tolerance\nVIX Blackout ≥ 35 → block positions\nHalt file → manual reset required"]
        GOV["RateLimitGovernor\nSingleton · thread-safe\nRPM sliding window (sleep)\nRPD hard cap (exception)\nTPM warning"]
        PIT["PointInTimeEnforcer\nBacktest mode gate\nMasks all data after sim date\nLook-ahead bias prevention"]
    end

    YF -->|"^VIX every 60s"| KS
    GOV -.->|"wait_if_needed() before\nevery LLM call"| N3 & N4 & AGG2 & N7
    PIT -.->|"date-gates yfinance\n& FRED fetchers"| N0 & N1

    %% ── Styles ──────────────────────────────────────────────────────────
    classDef stat   fill:#1e3a5f,stroke:#4a90d9,color:#e8f4f8
    classDef llm    fill:#2d1b4e,stroke:#9b59b6,color:#f0e6ff
    classDef safety fill:#3d1a1a,stroke:#e74c3c,color:#ffe8e8
    classDef data   fill:#1a3d2b,stroke:#27ae60,color:#e8f5e9
    classDef io     fill:#2c2c1a,stroke:#f39c12,color:#fff9e6

    class N1,N2,N6 stat
    class N3,N4,AGG2,N7 llm
    class KS,GOV,PIT safety
    class BUF,N5 data
    class N0,FL,SL io
```

---

## Quick Start (Docker)

**Prerequisites:** Docker, Docker Compose, and valid API keys.

```bash
# 1. Clone
git clone https://github.com/your-org/argus-v2.git
cd argus-v2/argus

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in all API keys (see Keys section below)
> StockTwits and Google Trends require no API key. Run `pip install -e .` and both sources work immediately.

# 3. Build and launch
docker compose build
docker compose up -d

# 4. Verify API health
curl http://localhost:8000/health

# 5. Open dashboard
open http://localhost:8501
```

---

## Local Development Setup

```bash
cd argus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start API
uvicorn api.main:app --reload --port 8000

# Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

---

## Required API Keys

| Key | Source | Purpose |
|-----|--------|---------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Llama 3.3-70b synthesis + sentiment |
| `GOOGLE_AI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Gemini fundamental analysis |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro data (CPI, Fed Funds, Yield Curve) |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | News sentiment headlines |

---

## Backtesting

```bash
# Via API (async, polled)
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL","MSFT","NVDA","JPM","XOM"],
    "start_date": "2021-01-04",
    "end_date":   "2024-12-31",
    "initial_cash": 100000,
    "risk_tolerance": "MODERATE",
    "run_bias_audit": true
  }'

# Poll for result
curl http://localhost:8000/backtest/<job_id>
```

Walk-forward validation uses rolling 6-month train / 1-month out-of-sample windows across the full date range. Results include Sharpe, Sortino, Max Drawdown, VaR, and an automated bias audit (look-ahead, survivorship, data-quality).

---

## Kill Switch Reset

If a drawdown halt is triggered, an `argus_halt_<timestamp>.json` file is written to disk. After reviewing the situation:

```bash
# Delete the halt file (manual confirmation step)
rm argus_halt_*.json

# Then reset via API with new inception value
curl -X POST "http://localhost:8000/kill-switch/reset?new_inception_value=100000"
```

---

## Running Tests

```bash
cd argus
pip install -e ".[test]"
python -m pytest tests/ -v --tb=short
```

All 7 integration tests pass against the live statistical pipeline (real yfinance data). LLM agents are automatically mocked via `pytest-mock`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph 0.2+ with SQL checkpointer |
| LLM Inference | Groq (Llama 3.3-70b, Llama 3.1-8b) + Google Gemini Flash Lite |
| Regime Classification | Gaussian HMM (`hmmlearn`) |
| Sentiment | FinBERT (`ProsusAI/finbert`) + Llama synthesis |
| Technical Indicators | `pandas-ta` (RSI, MACD, BB, ATR, ADX, VWAP) |
| Macro Data | FRED API via `fredapi` |
| Market Data | `yfinance` |
| Backtesting | `backtrader` with custom ARGUS strategy wrapper |
| Memory | ChromaDB (cultural memory vault) |
| API | FastAPI + Uvicorn |
| UI | Streamlit + Plotly |
| Deployment | Docker Compose / Render.com |
| CI | GitHub Actions |

---

## Deployment on Render

1. Push repository to GitHub
2. Connect repo in [Render dashboard](https://render.com)
3. Render auto-detects `render.yaml` — click **Deploy**
4. Add all env vars in the Render environment settings
5. First deploy takes ~5 minutes (model downloads)

---

## Disclaimer

**ARGUS v2 is a research and educational project. It is NOT:**
- Registered investment advice
- A licensed financial advisor
- Compliant with any securities regulation

**All generated portfolio allocations, signals, and recommendations are purely synthetic outputs of a machine learning model trained on historical data. Past performance does not predict future results. You could lose money. Always consult a licensed financial professional before making investment decisions.**
