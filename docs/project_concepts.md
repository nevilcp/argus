# ARGUS: Core Concepts and Dataset Alignment

This document outlines all the major concepts used in the **ARGUS Multi-Agent Financial Intelligence System** and explains how each concept directly aligns with the underlying datasets utilized by the project.

ARGUS runs as an institutional-grade, multi-agent LangGraph Directed Acyclic Graph (DAG) that performs quantitative analysis in parallel across six specialist domains. 

---

## 1. Data Sources and Datasets Used
The project relies on a diverse set of live and historical data APIs rather than static CSV datasets. This ensures the system acts on point-in-time, real-world data:
* **`yfinance`**: The core source for daily and intraday OHLCV (Open, High, Low, Close, Volume) price data, VIX levels, and company fundamental metrics.
* **`fredapi` (Federal Reserve Economic Data)**: The source for macroeconomic indicators, including the Federal Funds Rate, CPI, Unemployment Rate, and Treasury Yields.
* **`NewsAPI`**: Supplies recent news articles and headlines for equity symbols.
* **`StockTwits` (Public API)**: Provides social sentiment data, message volume, and bull/bear ratios.
* **`Google Trends` (`pytrends`)**: Tracks search interest and volume surges for specific tickers.

---

## 2. Core Concepts and Alignment

### A. Technical Analysis (TechnicalStatistic Agent)
* **Concept**: Evaluates price action and momentum using standard statistical and trading indicators to find short-term and medium-term entry/exit points. The system calculates indicators like Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, Average True Range (ATR), Average Directional Index (ADX), and Volume Weighted Average Price (VWAP).
* **Dataset Alignment**: Directly utilizes the daily and intraday OHLCV (Open, High, Low, Close, Volume) datasets fetched via `yfinance`. The `pandas-ta` library operates directly on these dataframes to compute the deterministic statistical markers without LLM hallucination risks.

### B. Macroeconomic Regime Classification (MacroStatistical Agent)
* **Concept**: Uses a Gaussian Hidden Markov Model (HMM) to classify the overall market environment into hidden states based on macroeconomic data. These states map to human-readable regimes: **EXPANSION**, **CONTRACTION**, and **TRANSITIONAL**. The context generated dictates the conviction multipliers for other agents (e.g., favoring fundamental analysis during an expansion).
* **Dataset Alignment**: Trains the `hmmlearn` model using 5 specific features: 
  1. `FEDFUNDS` (Federal Funds Rate via FRED)
  2. `CPIAUCSL` (CPI YoY % change via FRED)
  3. `T10Y2Y` (10Y-2Y Treasury spread via FRED)
  4. `UNRATE` (Unemployment Rate via FRED)
  5. `^VIX` (Volatility Index via yfinance).

### C. Fundamental Analysis (FundamentalAgent)
* **Concept**: Assesses the intrinsic value and financial health of a company by evaluating its balance sheet, income statement, and cash flows. It involves analyzing profitability, debt, and valuation multiples. 
* **Dataset Alignment**: Uses `yfinance.Ticker.info` to extract precise fundamental metrics, including Trailing P/E, Revenue Growth YoY, Operating Margin, Net Margin, Return on Equity (ROE), Debt-to-Equity, and Free Cash Flow (FCF) Yield. These raw numbers are fed into **Gemini 3.1 Flash Lite** to generate a qualitative fundamental thesis.

### D. Sentiment Analysis (SentimentAgent)
* **Concept**: Gauges the market mood and public perception of a stock. Positive or negative sentiment can act as a leading indicator for retail volume or institutional shifts. 
* **Dataset Alignment**: 
  * **News Sentiment**: Fetches recent headlines and descriptions from `NewsAPI`.
  * **Social Sentiment**: Extracts bull/bear percentage splits and message volumes from `StockTwits`.
  * **Search Trends**: Checks for retail interest surges using `Google Trends`.
  * These text snippets and metadata are scored and synthesized by **FinBERT** and **Llama 3.1-8b**.

### E. Risk Management (RiskStatistic Engine & KillSwitch)
* **Concept**: Quantifies downside risk and tail risk to prevent catastrophic portfolio losses. It calculates Value at Risk (VaR), Conditional VaR (CVaR), and Beta. Additionally, an independent KillSwitch daemon monitors for severe drawdowns.
* **Dataset Alignment**: Calculates VaR and CVaR strictly from the historical daily returns derived from the `yfinance` OHLCV price dataset. The KillSwitch monitors the current `^VIX` level against extreme thresholds.

### F. Portfolio Allocation and Sizing (PortfolioManagerAgent)
* **Concept**: Synthesizes the contradictory or aligned signals from all other agents to decide the final portfolio weighting. It utilizes the **Kelly Criterion** (specifically Half-Kelly sizing) to mathematically optimize the bet size based on the win probability and risk/reward ratio.
* **Dataset Alignment**: Consumes the structured `MacroContext`, fundamental/technical signals, and risk statistics outputted by previous agents. An LLM (**Llama 3.3-70b**) acts as the final arbitrator, strictly formatting its output via Pydantic to ensure mathematically valid portfolio allocations.

### G. System Orchestration and Backtesting Validation
* **Concept**: Ensuring that the multi-agent system does not "cheat" by using future data to make past decisions (Look-Ahead Bias). The system orchestrates agents using a LangGraph Directed Acyclic Graph (DAG) and backtests via walk-forward validation (e.g., train on 6 months, test on 1 month out-of-sample).
* **Dataset Alignment**: Enforced by a `PointInTimEnforcer` which masks all `yfinance` and FRED data after the current backtest simulation date, feeding only the historically available slices to the agents and the `Backtrader` engine.
