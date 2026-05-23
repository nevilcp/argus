# ARGUS v2 — Running & Testing Guide

> Complete instructions for local development, testing, backtesting, and deployment.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [API Keys — Step by Step](#api-keys)
3. [Installation](#installation)
4. [Environment Configuration](#environment-configuration)
5. [Running Locally](#running-locally)
6. [Test Suite](#test-suite)
7. [Running Backtests (3 Phases)](#backtests)
8. [Running with Docker](#docker)
9. [Cloud Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Check command |
|-------------|---------|--------------|
| Python | 3.11 or 3.12 | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| Docker (optional) | 24+ | `docker --version` |
| ~3 GB disk space | — | For FinBERT model download |
| RAM | 4 GB min, 8 GB recommended | — |

> **Windows users:** Use WSL2 (Ubuntu 22.04) for the best experience. All commands below assume a Unix shell (bash/zsh).

---

## 2. API Keys — Step by Step

Obtain all keys before installation. All are free, no credit card required.

### Groq (for Sentiment + Portfolio Manager agents)
1. Go to **https://console.groq.com**
2. Sign up with Google or email
3. Click **API Keys** in the left sidebar → **Create API Key**
4. Name it `argus-v2`. Copy the key (shown once).
5. Free limits: `llama-3.1-8b-instant` = 14,400 req/day; `llama-3.3-70b-versatile` = 1,000 req/day

### Google AI Studio (for Fundamental Agent)
1. Go to **https://aistudio.google.com**
2. Sign in with a Google account
3. Click **Get API Key** → **Create API key in new project**
4. Copy the key.
5. Model to use: `gemini-3.1-flash-lite` — 500 req/day free
6. ⚠️ Do NOT use gemini-2.5-flash (only 20 req/day) or gemini-2.5-pro (no free tier)

### FRED API (for Macro Agent — macroeconomic data)
1. Go to **https://fred.stlouisfed.org/docs/api/api_key.html**
2. Click **Request API Key** — requires a name and email
3. Approval is instant and automated
4. Free tier: unlimited requests

### NewsAPI (for Sentiment Agent — news headlines)
1. Go to **https://newsapi.org/register**
2. Fill in name, email, password — no CC
3. Your API key is shown immediately on the dashboard
4. Free tier: 100 requests/day, 1-month history only

### StockTwits & Google Trends (for social sentiment)
- No API keys are required for either source. They work automatically out of the box!

### Polygon.io (for 5-min intraday data — optional, yfinance is fallback)
1. Go to **https://polygon.io**
2. Sign up → free plan gives 5 API calls/minute
3. Copy the API key from the dashboard
4. If you skip this, yfinance handles 5-min data automatically (slightly delayed)

---

## 3. Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/argus-v2.git
cd argus-v2

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -e .

# 4. Verify core imports
python -c "import langgraph, langchain_groq, langchain_google_genai, \
           pandas_ta, hmmlearn, backtrader, chromadb, transformers; \
           print('All core dependencies installed successfully.')"
```

**Expected output:**
```
All core dependencies installed successfully.
```

**FinBERT model download (happens automatically on first sentiment run, ~400 MB):**
```bash
# Pre-download to avoid first-run delay:
python -c "from transformers import pipeline; \
           pipeline('text-classification', model='ProsusAI/finbert'); \
           print('FinBERT ready.')"
```

---

## 4. Environment Configuration

```bash
# Create your .env file from the template
cp .env.example .env
```

Open `.env` in your editor and fill in every key:

```env
# --- LLM APIs ---
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_AI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# --- Market Data ---
FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NEWSAPI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
POLYGON_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx        # Optional

# --- Paper Trading (optional, Alpaca) ---
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

**Validate your configuration:**
```bash
python -c "
from argus.config import settings
print(f'Groq key loaded: {bool(settings.GROQ_API_KEY)}')
print(f'Google key loaded: {bool(settings.GOOGLE_AI_API_KEY)}')
print(f'FRED key loaded: {bool(settings.FRED_API_KEY)}')
print(f'NewsAPI key loaded: {bool(settings.NEWSAPI_KEY)}')
"
```

**Expected output:**
```
Groq key loaded: True
Google key loaded: True
FRED key loaded: True
NewsAPI key loaded: True
```

---

## 5. Running Locally

### Start the FastAPI Backend

```bash
# Terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

**Expected startup output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     [ARGUS] MacroStatisticalAgent HMM fitting on startup...
INFO:     [ARGUS] System ready. Groq 70B remaining capacity: 1000/day
```

Wait for `System ready` before starting the UI.

### Start the Streamlit UI

```bash
# Terminal 2
source .venv/bin/activate
streamlit run ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

### Verify the API is Alive

```bash
curl http://localhost:8000/health | python -m json.tool
```

**Expected response:**
```json
{
  "status": "ok",
  "model_versions": {
    "fundamental": "gemini-3.1-flash-lite",
    "sentiment": "llama-3.1-8b-instant",
    "portfolio": "llama-3.3-70b-versatile"
  },
  "governor_report": {
    "llama-3.3-70b-versatile": {"calls_today": 0, "rpd_limit": 1000, "pct_used": 0.0}
  }
}
```

### Run Your First Analysis (Minimal — 2 tickers)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "total_wealth": 10000,
    "invest_pct": 0.80,
    "risk_tolerance": "MODERATE"
  }' | python -m json.tool
```

**Expected (simplified):**
```json
{
  "session_id": "uuid-here",
  "portfolio": [
    {"ticker": "AAPL", "allocation_pct": 0.12, "allocation_usd": 960.0, "stop_loss": 171.20},
    {"ticker": "MSFT", "allocation_pct": 0.10, "allocation_usd": 800.0, "stop_loss": 389.50}
  ],
  "cash_reserve_pct": 0.18,
  "macro_regime": "EXPANSION",
  "vix_level": 18.4
}
```

**Total time for a 2-ticker analysis: ~15-30 seconds.**
Breakdown: Statistical agents (~1s) + Fundamental/Sentiment LLM calls (~10-20s) + Portfolio Manager (~5s).

---

## 6. Test Suite

### Run All Unit Tests

First, ensure you have the test dependencies installed:
```bash
pip install -e ".[test]"
```

Then run the tests:
```bash
python -m pytest tests/ -v
```

**Expected output:**
```
tests/test_technical.py::test_bullish_signal       PASSED
tests/test_technical.py::test_bearish_signal       PASSED
tests/test_technical.py::test_neutral_signal       PASSED
tests/test_technical.py::test_zero_api_calls       PASSED
tests/test_macro.py::test_rule_based_fallback      PASSED
tests/test_macro.py::test_agent_multipliers        PASSED
tests/test_macro.py::test_cache                    PASSED
tests/test_risk.py::test_vix_blackout              PASSED
tests/test_risk.py::test_overweight_position       PASSED
tests/test_risk.py::test_high_var_reduce           PASSED
tests/test_risk.py::test_approve_healthy_portfolio PASSED
tests/test_risk.py::test_zero_api_calls            PASSED
tests/test_integration.py::TestEndToEnd::...       PASSED  (×7)

========================= 19 passed in 12.4s =========================
```

### Run Only Statistical Agent Tests (Zero API Calls)

```bash
python -m pytest tests/test_technical.py tests/test_macro.py tests/test_risk.py -v
```

Use this during development when you don't want to consume API quota.

### Run Integration Tests (Requires API Keys)

```bash
python -m pytest tests/test_integration.py -v -k "not test_full_graph_smoke"
```

The `test_full_graph_smoke` test makes real LLM calls (~3 API calls). Run it separately when ready:

```bash
python -m pytest tests/test_integration.py::TestEndToEnd::test_full_graph_smoke -v -s
```

### Run the Statistical Pipeline Smoke Test (Free, No API)

```bash
python -c "
from argus.agents.technical import TechnicalStatisticalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.risk import RiskStatisticalEngine
from argus.data.fetchers import fetch_ohlcv_daily
import pandas as pd

print('=== Statistical Tier Smoke Test ===')
print('(Zero API calls expected)')

# Fetch price data
df = fetch_ohlcv_daily('AAPL', period='3mo')
print(f'Price data: {len(df)} days for AAPL')

# Technical agent
tech = TechnicalStatisticalAgent()
session_state = {
    'rsi_14': 58.3, 'macd_histogram': 0.12, 'bb_percent_b': 0.62,
    'atr_pct': 0.014, 'adx_14': 31.0, 'vwap_distance': 0.004,
    'volume_ratio': 1.15, 'momentum_30m': 0.006, 'momentum_1d': 0.018,
    'close': 185.0, 'timestamp': '2025-01-15 14:30:00'
}
sig = tech.analyze('AAPL', session_state)
print(f'Technical: {sig.signal} (conviction={sig.conviction:.2f}, api_calls={sig.api_calls_used})')
assert sig.api_calls_used == 0, 'Technical agent used API — this is a bug!'

# Macro agent
macro = MacroStatisticalAgent()
ctx = macro.analyze()
print(f'Macro: {ctx.macro_regime} | VIX={ctx.vix_level:.1f} | {ctx.yield_curve_shape}')
assert ctx.api_calls_used == 0

# Risk engine
risk = RiskStatisticalEngine()
close = df['close'].squeeze()
result = risk.evaluate(
    proposed_positions=[{'ticker': 'AAPL', 'weight': 0.10}],
    price_history={'AAPL': close, 'SPY': fetch_ohlcv_daily('SPY', '1y')['close'].squeeze()},
    current_vix=ctx.vix_level
)
print(f'Risk: {result.verdict} | VaR99={result.var_99:.2%} | Beta={result.portfolio_beta:.2f}')
assert result.api_calls_used == 0

print()
print('All statistical agents: PASS. Total API calls: 0.')
"
```

### Check API Quota Usage at Any Time

```bash
curl http://localhost:8000/governor/report | python -m json.tool
```

---

## 7. Running Backtests (3 Phases)

> ⚠️ **Important:** Run phases in order. Never use Phase 3 (2025) data until Phases 1 and 2 are complete and parameters are locked. This is what makes the forward test scientifically valid.

### Phase 1 — In-Sample Calibration (2021–2022)

**Purpose:** Fit the HMM, calibrate indicator weights, establish baseline.  
**Estimated time:** 60–120 minutes.  
**API calls consumed:** ~50 (fundamental + sentiment, batched weekly).

```bash
python -m argus.backtesting.phase1_calibration
```

Monitor progress — logs appear every few minutes. Key things to watch for:
```
INFO: HMM fitted. Regime mapping: {0: 'EXPANSION', 1: 'CONTRACTION', 2: 'TRANSITIONAL'}
INFO: Baseline Sharpe: 0.XXX
INFO: New best weights: {...} → Sharpe X.XXX
INFO: PHASE 1 COMPLETE. Locked Sharpe: X.XXX
INFO: Calibration saved to: calibration_report.json
```

Verify the output file was created:
```bash
cat calibration_report.json | python -m json.tool
```

**Do not proceed to Phase 2 if** the bias audit returns any `FAIL` status.

### Phase 2 — Walk-Forward Validation (2023–2024)

**Purpose:** 18-window rolling validation with locked parameters. Must pass minimum Sharpe ≥ 0.80.  
**Estimated time:** 90–180 minutes.  
**API calls consumed:** ~100.

```bash
python -c "
from argus.backtesting.walk_forward import run_walk_forward_validation
from argus.config import settings
import json

# Load locked weights from Phase 1
with open('calibration_report.json') as f:
    cal = json.load(f)
settings.TECHNICAL_INDICATOR_WEIGHTS = cal['locked_indicator_weights']

print('Running Phase 2: Walk-forward validation (2023-2024)...')
results = run_walk_forward_validation(
    universe=settings.UNIVERSE_DEFAULT,
    start='2023-01-03',
    end='2024-12-31',
    train_months=6,
    test_months=1,
    risk_tolerance='MODERATE'
)

print(f'Windows completed: {results[\"n_windows\"]}')
print(f'Average Sharpe:    {results[\"avg_sharpe\"]:.3f} (minimum required: 0.80)')
print(f'Sharpe std dev:    {results[\"std_sharpe\"]:.3f} (lower = more consistent)')
print(f'Consistency score: {results[\"consistency_score\"]:.3f}')
print(f'PASS criteria met: {results[\"pass_criteria\"]}')

if not results['pass_criteria']:
    print()
    print('PHASE 2 FAILED minimum criteria.')
    print('Do NOT proceed to Phase 3 or live use.')
    print('Review agent weights and re-run Phase 1 calibration.')
else:
    print()
    print('PHASE 2 PASSED. Safe to proceed to Phase 3 (2025 forward test).')
" 2>&1 | tee phase2_results.log
```

### Phase 3 — 2025 Forward Test (Out-of-Sample)

**Purpose:** True out-of-sample validation. Zero parameter changes from this point forward.  
**Run this continuously** — it generates a monthly performance report.

```bash
python -c "
from argus.backtesting.engine import run_backtest
from argus.backtesting.metrics import compute_all_metrics
from argus.backtesting.bias_auditor import BiasAuditor
from argus.config import settings
import yfinance as yf
import pandas as pd
import json

print('Phase 3: 2025 Forward Test (out-of-sample)')
print('ZERO parameter changes are permitted from this point forward.')
print()

result = run_backtest(
    universe=settings.UNIVERSE_DEFAULT,
    start='2025-01-02',
    end=pd.Timestamp.today().strftime('%Y-%m-%d'),  # Run through today
    initial_cash=100_000.0,
    risk_tolerance='MODERATE'
)

print(f'Cumulative return:   {result[\"cumulative_return\"]:.1%}')
print(f'Sharpe ratio:        {result[\"sharpe\"]:.3f} (minimum required: 0.70)')
print(f'Max drawdown:        {result[\"max_drawdown\"]:.1%} (limit: 22%)')
print(f'Alpha vs SPY:        {result[\"alpha_annualized\"]:.2%} (minimum: >1%)')
print(f'Win rate:            {result[\"win_rate\"]:.1%}')
print()
print('Phase 3 performance report saved to: phase3_results.json')
with open('phase3_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
" 2>&1 | tee phase3_results.log
```

### Quick Backtest Smoke Test (5 stocks, 3 months — ~10 minutes)

Use this to verify the backtest engine works before running the full 4-year test:

```bash
python -c "
from argus.backtesting.engine import run_backtest
result = run_backtest(
    universe=['AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM'],
    start='2023-01-03',
    end='2023-03-31',
    initial_cash=100_000.0
)
print(f'Smoke test passed. Sharpe: {result.get(\"sharpe\", \"N/A\")}')
print(f'Final portfolio value: \${result.get(\"final_value\", 100000):,.0f}')
"
```

---

## 8. Running with Docker

Docker is the recommended way to run ARGUS in production. It packages all dependencies, models, and configuration into isolated containers.

### Build and Start

```bash
# Ensure .env is populated first
docker compose build          # First build: ~5-10 minutes (downloads Python deps)
docker compose up -d          # Start in background
docker compose logs -f        # Stream logs (Ctrl+C to stop)
```

### Verify Both Services are Running

```bash
docker compose ps
```

Expected:
```
NAME           STATUS          PORTS
argus-api      Up (healthy)    0.0.0.0:8000->8000/tcp
argus-ui       Up              0.0.0.0:8501->8501/tcp
```

### Check API Health Inside Docker

```bash
curl http://localhost:8000/health
```

### Run Tests Inside the Container

```bash
docker compose exec argus-api python -m pytest tests/ -v --tb=short
```

### View API Logs

```bash
docker compose logs argus-api --tail 50
```

### Stop Everything

```bash
docker compose down
```

### Persistent Data

Cultural memory (ChromaDB) and decision logs (SQLite) persist in Docker volumes across restarts. To wipe and start fresh:

```bash
docker compose down -v        # WARNING: deletes all stored trade history
```

---

## 9. Cloud Deployment

### Option A — Render (FastAPI Backend) + Streamlit Cloud (UI)

**Step 1: Push to GitHub**
```bash
git add .
git commit -m "ARGUS v2 production build"
git push origin main
```

**Step 2: Deploy FastAPI on Render**
1. Go to **https://render.com** → New → Web Service
2. Connect your GitHub repository
3. Settings:
   - **Build Command:** `pip install -e .`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Instance Type:** Free (512 MB RAM)
4. Click **Environment** → Add each key from your `.env` file one by one
5. Click **Deploy**. First deploy takes ~8 minutes (FinBERT download included).
6. Note your URL: `https://argus-api-xxxx.onrender.com`

> **Free tier caveat:** Render spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds. This is acceptable for a portfolio project.

**Step 3: Deploy Streamlit UI**
1. Go to **https://share.streamlit.io**
2. Connect GitHub → select your repo
3. Main file path: `ui/app.py`
4. Advanced settings → add secret: `API_BASE_URL = https://argus-api-xxxx.onrender.com`
5. Deploy. Available at: `https://YOUR_APP.streamlit.app`

### Option B — Local Network Demo

To share with others on your local network (e.g., for a class demo):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Share `http://YOUR_LOCAL_IP:8501` — works on the same WiFi network.

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'pandas_ta'"
```bash
pip install pandas-ta --upgrade
# If still failing on Python 3.12:
pip install pandas-ta --pre
```

### "hmmlearn installation fails"
```bash
# Install build dependencies first
pip install cython numpy wheel
pip install hmmlearn
```

### FinBERT download is very slow
```bash
# Set HuggingFace mirror (if in China/restricted network):
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"
```

### "RateLimitExceeded: llama-3.3-70b-versatile RPD limit: 1000/1000"
The daily quota is exhausted. Options:
1. Wait until midnight UTC for the quota to reset
2. Reduce your universe size (fewer tickers = fewer portfolio manager calls)
3. Use `llama-4-scout-17b` as a fallback model (also 1K RPD) — edit `portfolio.py`

### Gemini returns 429 (Too Many Requests)
You're hitting the 15 RPM limit. The governor handles this automatically. If it persists:
```bash
# Check that the governor singleton is running
python -c "from argus.orchestration.governor import governor; print(governor.get_usage_report())"
```

### "No data found for ticker" during backtest
The ticker was not publicly traded at the backtest start date (correct behavior — survivorship bias prevention). The bias auditor flags this:
```bash
python -c "
from argus.backtesting.bias_auditor import BiasAuditor
from datetime import date
auditor = BiasAuditor()
result = auditor.check_survivorship_bias(['AAPL','MSFT','META'], date(2021,1,4))
print(result)
"
```

### Docker build fails at FinBERT step
```bash
# Increase Docker memory limit to at least 4GB
# Docker Desktop → Settings → Resources → Memory → 4 GB
docker compose build --no-cache
```

### ChromaDB "could not connect to tenant" error
```bash
rm -rf chroma_db/          # Wipe corrupted DB
# Restart — DB recreates automatically on startup
```

### Streamlit UI shows "Connection refused" to API
Ensure the backend is running and healthy before starting the UI:
```bash
curl http://localhost:8000/health   # Must return {"status":"ok",...}
# Then start UI
streamlit run ui/app.py
```

---

## Quick Reference — All Commands

```bash
# Installation
pip install -e .

# Start (local)
uvicorn api.main:app --reload --port 8000    # Terminal 1
streamlit run ui/app.py --server.port 8501   # Terminal 2

# Tests
python -m pytest tests/ -v                               # All tests
python -m pytest tests/test_technical.py -v             # Stats only (no API)
python -m pytest tests/test_integration.py -v -s        # Integration (uses API)

# Backtesting
python -m argus.backtesting.phase1_calibration          # Phase 1 (2021-22)
python phase2_validation.py                              # Phase 2 (2023-24)
python phase3_forward.py                                 # Phase 3 (2025)

# Docker
docker compose up -d          # Start
docker compose logs -f        # Stream logs
docker compose down           # Stop

# API calls (manual testing)
curl http://localhost:8000/health
curl http://localhost:8000/governor/report
curl http://localhost:8000/memory/stats

# Quota check
python -c "from argus.orchestration.governor import governor; \
           import json; print(json.dumps(governor.get_usage_report(), indent=2))"
```
