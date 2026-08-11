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

from argus.params import SYSTEM
from argus.params import TECHNICAL_INDICATOR_WEIGHTS as _TECHNICAL_WEIGHTS

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

    # MFT pipeline timing parameters. Numeric defaults live in argus/params.py,
    # tagged with provenance (literature/convention/calibrated/arbitrary); this
    # class only adds env-var override capability on top of them.
    MFT_CANDLE_INTERVAL: str = "5m"
    MFT_DECISION_INTERVAL_SECONDS: int = SYSTEM.mft_decision_interval_seconds
    CANDLE_BUFFER_SIZE: int = SYSTEM.candle_buffer_size

    # Technical indicator scoring weights
    TECHNICAL_INDICATOR_WEIGHTS: Dict[str, float] = {
        "rsi": _TECHNICAL_WEIGHTS.rsi,
        "macd": _TECHNICAL_WEIGHTS.macd,
        "bb": _TECHNICAL_WEIGHTS.bb,
        "adx": _TECHNICAL_WEIGHTS.adx,
        "vwap": _TECHNICAL_WEIGHTS.vwap,
        "momentum": _TECHNICAL_WEIGHTS.momentum,
    }

    # Portfolio hard limits enforced by RiskStatisticalEngine
    MAX_SINGLE_POSITION_PCT: float = SYSTEM.max_single_position_pct
    MAX_SECTOR_CONCENTRATION: float = SYSTEM.max_sector_concentration
    MAX_PORTFOLIO_BETA: float = SYSTEM.max_portfolio_beta

    # Kill-switch circuit breaker thresholds
    VIX_BLACKOUT_THRESHOLD: float = SYSTEM.vix_blackout_threshold
    MAX_DRAWDOWN_HALT: float = SYSTEM.max_drawdown_halt

    # Historical lookback window used in rolling indicator calculations
    LOOKBACK_DAYS: int = SYSTEM.lookback_days


# Module-level singleton — import this everywhere rather than instantiating locally
settings = Settings()
