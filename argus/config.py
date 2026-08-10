"""
argus/config.py

Environment and system configuration service for ARGUS.

Uses pydantic-settings to bind variables from the local environment registry or .env file.
Enforces structured typing across secrets, technical weights, portfolio limits, and safety
thresholds.

Responsibilities:
  - Expose a single typed Settings instance used across all modules
  - Validate API key and numeric threshold types on startup

Not responsible for:
  - Injecting .env into os.environ (handled by python-dotenv in api/main.py)
  - Key rotation or secret management
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Type-safe application settings resolved from environment variables and .env.

    All fields default to safe empty strings or conservative thresholds so the system
    degrades gracefully when optional keys (e.g. FRED_API_KEY, NEWSAPI_KEY) are absent.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider credentials
    groq_api_key: str = Field(default="", description="Groq LLM API key")
    google_ai_api_key: str = Field(default="", description="Google AI / Gemini API key")

    # Market and macroeconomic data providers
    fred_api_key: str = Field(default="", description="FRED (Federal Reserve) API key")
    polygon_api_key: str = Field(default="", description="Polygon.io market data API key")
    alpaca_api_key: str = Field(default="", description="Alpaca trading API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca trading API secret")

    # News and sentiment data providers
    newsapi_key: str = Field(default="", description="NewsAPI.org key")

    # Default equity universe: 20 liquid large-cap tickers spanning 7 GICS sectors.
    # Broadened to reduce sector concentration risk in Phase 2 walk-forward validation.
    UNIVERSE_DEFAULT: List[str] = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "JPM", "V", "UNH",
        "XOM", "JNJ", "PG", "MA", "HD",
        "MRK", "ABBV", "LLY", "AVGO", "CVX",
    ]

    # MFT pipeline timing parameters
    MFT_CANDLE_INTERVAL: str = "5m"
    MFT_DECISION_INTERVAL_SECONDS: int = 1800
    CANDLE_BUFFER_SIZE: int = 78

    # Technical indicator scoring weights — mutated during Phase 1 grid search;
    # locked values are persisted to calibration_report.json after Phase 1 completes
    TECHNICAL_INDICATOR_WEIGHTS: Dict[str, float] = {
        "rsi": 2.0,
        "macd": 2.0,
        "bb": 1.5,
        "adx": 1.0,
        "vwap": 1.0,
        "momentum": 1.5,
    }

    # Portfolio hard limits enforced by RiskStatisticalEngine
    MAX_SINGLE_POSITION_PCT: float = 0.15
    MAX_SECTOR_CONCENTRATION: float = 0.40
    MAX_PORTFOLIO_BETA: float = 1.50

    # Kill-switch circuit breaker thresholds
    VIX_BLACKOUT_THRESHOLD: float = 35.0
    MAX_DRAWDOWN_HALT: float = 0.15

    # Historical lookback window used in rolling indicator calculations
    LOOKBACK_DAYS: int = 252


# Module-level singleton — import this everywhere rather than instantiating locally
settings = Settings()
