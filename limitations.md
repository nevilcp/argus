# Argus Project Limitations

This document tracks identified architectural, technical, data, and modeling limitations of the Argus multi-agent financial intelligence system.

## 1. Data Availability & Freshness
- **Stale Price-Derived Fundamental Metrics**: The 7-day TTL cache for `FundamentalSignal` caches both static accounting metrics (like revenue growth) and price-derived multiples (like P/E ratio, market cap, and FCF yield). In fast-moving markets, the price-derived multiples used by the LLM agent will diverge from real-time market data.
- **ROA Used as ROIC Proxy**: The data fetcher maps Yahoo Finance's `returnOnAssets` directly to the `roic` field (`"roic": info.get("returnOnAssets")`). Return on Invested Capital (ROIC) and Return on Assets (ROA) are mathematically distinct financial ratios, potentially distorting capital efficiency evaluations.
- **Macro Reporting Lags**: The `MacroStatisticalAgent` relies on FRED data (e.g., CPI, GDP, Federal Funds Rate). These indicators suffer from significant reporting lags (often weeks or months after the fact), causing the Gaussian HMM regime classifier to react slowly to real-time economic pivots.
- **Backtesting Historical News Restrictions**: The NewsAPI free tier limits historical news queries to the trailing 28 days. Any backtest simulations set further in the past will fetch empty lists, rendering news/sentiment-based signals inactive.


## 2. LLM Dependencies & Constraints
- **API Rate Limits and Cost**: The system heavily relies on external LLM APIs (Groq and Google AI). Running a large universe of tickers quickly consumes rate limits (tokens per minute), necessitating aggressive `governor` throttles, exponential backoff, and ultimately graceful degradation (skipping tickers or returning cash-only defaults).
- **Non-Determinism in Signal Generation**: While rigid JSON schemas and Domain-Knowledge Chain-of-Thought (DK-CoT) prompts are enforced, the `FundamentalAgent` and `PortfolioManagerAgent` are powered by generative models. Conviction scores and advisor notes can vary slightly between identical inputs.
- **Context Window Limits in Allocation**: The `PortfolioManagerAgent` receives the entire signal table for the universe. For very large universes (e.g., the entire S&P 500), the token count of the XML-delimited state may exceed the context limits of the LLM or degrade reasoning quality.

## 3. Backtesting Limitations
- **Parametric Memory Contamination**: While the system uses deterministic ticker anonymization to combat look-ahead bias, LLMs might still implicitly recognize high-profile companies (e.g., AAPL, NVDA) from unique combinations of their exact market cap, margins, and sector, leading to implicit bias.
- **Shortened Hash Collision Risk**: The deterministic anonymization logic slices the MD5 hash hex digest to 4 characters (`[:4]`) in `fundamental.py` and 6 characters (`[:6]`) in `pit_enforcer.py`. While sufficient for small asset universes, a 4-character hex slice has only $16^4 = 65,536$ unique identifiers, raising collision risks in larger stock databases.

## 4. Financial Modeling & Execution Simulation
- **No Slippage or Market Impact Modeling**: The system calculates theoretical allocations based on end-of-day or intraday prices but does not account for execution slippage, bid-ask spreads, liquidity constraints, or the market impact of trading large sizes.
- **Static Covariance in Crisis**: The `RiskStatisticalEngine` utilizes historical covariance matrices for the SLSQP optimizer. Historical covariance notoriously breaks down during sudden market shocks or black swan events (when asset correlations tend to converge to 1), potentially under-calculating true Value-at-Risk (VaR).
- **Long-Only Constraint**: Currently, the system is designed to either allocate capital to long equity positions or hold cash reserves. It does not support short selling, options, or derivatives hedging.

## 5. Architectural Trade-offs
- **Synchronous Database Bottlenecks**: The LangGraph state machine execution uses `SqliteSaver` checkpointers which operate synchronously. This prevents the use of true `async` graph invocations (`ainvoke`), causing the orchestration layer to block threads heavily during execution.
