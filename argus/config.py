"""
Configuration module for ARGUS v2.

Defines a pydantic-settings `Settings` class that reads all secrets from the
`.env` file and exposes system-wide constants (universe, indicator weights,
risk thresholds, backtesting parameters).  Import `settings` from this module
everywhere to access configuration in a type-safe manner.
"""

from __future__ import annotations

from typing import List, Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralised, environment-driven configuration for ARGUS v2."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Providers ─────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq LLM API key")
    google_ai_api_key: str = Field(default="", description="Google AI / Gemini API key")

    # ── Market & Economic Data ────────────────────────────────────────────────
    fred_api_key: str = Field(default="", description="FRED (Federal Reserve) API key")
    polygon_api_key: str = Field(default="", description="Polygon.io market data API key")
    alpaca_api_key: str = Field(default="", description="Alpaca trading API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca trading API secret")

    # ── News & Sentiment ──────────────────────────────────────────────────────
    newsapi_key: str = Field(default="", description="NewsAPI.org key")

    # ── Observability (Langfuse) ──────────────────────────────────────────────
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")

    # ── Equity Universe ───────────────────────────────────────────────────────
    UNIVERSE_DEFAULT: List[str] = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "JPM", "V", "UNH",
        "XOM", "JNJ", "PG", "MA", "HD",
        "MRK", "ABBV", "LLY", "AVGO", "CVX",
    ]

    # ── Mid-Frequency Trading (MFT) ───────────────────────────────────────────
    MFT_CANDLE_INTERVAL: str = "5m"
    MFT_DECISION_INTERVAL_SECONDS: int = 1800
    CANDLE_BUFFER_SIZE: int = 78

    # ── Technical Indicator Weights ───────────────────────────────────────────
    TECHNICAL_INDICATOR_WEIGHTS: Dict[str, float] = {
        "rsi": 2.0,
        "macd": 2.0,
        "bb": 1.5,
        "adx": 1.0,
        "vwap": 1.0,
        "momentum": 1.5,
    }

    # ── Portfolio Risk Limits ─────────────────────────────────────────────────
    MAX_SINGLE_POSITION_PCT: float = 0.15
    MAX_SECTOR_CONCENTRATION: float = 0.40
    MAX_PORTFOLIO_BETA: float = 1.50

    # ── Circuit-Breaker / Kill-Switch Thresholds ──────────────────────────────
    VIX_BLACKOUT_THRESHOLD: float = 35.0
    MAX_DRAWDOWN_HALT: float = 0.15

    # ── Backtesting ───────────────────────────────────────────────────────────
    LOOKBACK_DAYS: int = 252


# Module-level singleton — import this everywhere.
settings = Settings()
