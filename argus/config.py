"""Typed environment settings with defaults from parameter provenance."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from argus.params import (
    COLLECTOR,
    GOVERNOR,
    RECONCILIATION,
    SYSTEM,
)
from argus.params import TECHNICAL_INDICATOR_WEIGHTS as _TECHNICAL_WEIGHTS
from argus.schemas.requests import (
    MAX_INVEST_PCT,
    MIN_INVEST_PCT,
    MIN_TOTAL_WEALTH,
    RiskTolerance,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env before provider imports, using the working directory so installed packages find deployment settings.
load_dotenv()


class Settings(BaseSettings):
    """Type-safe application settings resolved from environment variables and .env."""

    # Resolve env_file relative to the deployment working directory, matching load_dotenv above.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    GROQ_API_KEY: str = Field(default="", description="Groq LLM API key")

    FRED_API_KEY: str = Field(default="", description="FRED (Federal Reserve) API key")

    NEWSAPI_KEY: str = Field(default="", description="NewsAPI.org key")

    ARGUS_UNIVERSE: Annotated[list[str], NoDecode] = Field(
        default=[
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "JPM",
            "V",
            "UNH",
            "XOM",
            "JNJ",
            "PG",
            "MA",
            "HD",
            "MRK",
            "ABBV",
            "LLY",
            "AVGO",
            "CVX",
        ],
        description="Ticker universe the unattended collector tracks and analyzes",
    )

    # Numeric defaults and provenance live in params.py; settings supply environment overrides.
    MFT_CANDLE_INTERVAL: str = "1m"
    # Derive buffer depth from the candle interval unless explicitly overridden.
    CANDLE_BUFFER_SIZE: int | None = None

    ARGUS_DATA_DIR: str = Field(default="data", description="Directory for the persistent intraday buffer and logs")
    # Derive Chroma and runs paths under ARGUS_DATA_DIR unless separate storage is configured.
    ARGUS_CHROMA_DIR: str | None = Field(
        default=None, description="Directory backing the ChromaDB cultural-memory store"
    )
    ARGUS_RUNS_DIR: str | None = Field(default=None, description="Directory for kill-switch halt-event dumps")
    ARGUS_COLLECTOR_ENABLED: bool = Field(
        default=False, description="Run the graph automatically on a schedule during market hours"
    )
    ARGUS_COLLECTOR_INTERVAL_SECONDS: int = Field(
        default=COLLECTOR.interval_seconds,
        description="Seconds between automatic collection cycles",
    )
    ARGUS_RECONCILE_ENABLED: bool = Field(
        default=False, description="Automatically reconcile matured decisions once a day"
    )
    ARGUS_RECONCILE_HOUR_ET: int = Field(
        default=RECONCILIATION.daily_run_hour_et,
        ge=0,
        le=23,
        description="Eastern-time hour to run the daily reconcile pass",
    )
    ARGUS_TOTAL_WEALTH: float = Field(
        default=COLLECTOR.total_wealth,
        gt=MIN_TOTAL_WEALTH,
        description="Total wealth used by unattended collection cycles",
    )
    ARGUS_INVEST_PCT: float = Field(
        default=COLLECTOR.invest_pct,
        gt=MIN_INVEST_PCT,
        le=MAX_INVEST_PCT,
        description="Invest fraction used by unattended collection cycles",
    )
    ARGUS_RISK_TOLERANCE: RiskTolerance = Field(
        default="MODERATE", description="Risk tolerance used by unattended collection cycles"
    )
    ARGUS_CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default=["*"], description="Allowed CORS origins for the FastAPI gateway"
    )
    ARGUS_API_KEY: str = Field(
        default="",
        description=(
            "Shared secret required via the X-API-Key header on /analyze and "
            "/kill-switch/reset. Blank (the default) disables the check — fine for "
            "local development behind a trusted network boundary, not production."
        ),
    )
    ARGUS_LOG_LEVEL: str = Field(default="INFO", description="Log level for the argus, api and scripts loggers")

    ARGUS_IMAGE_TAG: str = Field(
        default="unknown",
        description=(
            "Commit-identified image tag baked in at build time via Dockerfile.api's "
            "ARGUS_IMAGE_TAG build arg (see image.yml); 'unknown' outside a built image"
        ),
    )

    # Use bootstrap quotas until provider response headers reveal the account limits for each model.
    ARGUS_GROQ_RPM: int = Field(
        default=GOVERNOR.bootstrap_rpm,
        gt=0,
        description="Bootstrap requests-per-minute floor before headers are observed",
    )
    ARGUS_GROQ_TPM: int = Field(
        default=GOVERNOR.bootstrap_tpm,
        gt=0,
        description="Bootstrap tokens-per-minute floor before headers are observed",
    )
    ARGUS_ANALYZE_DEADLINE_SECONDS: int = Field(
        default=SYSTEM.analyze_deadline_seconds,
        gt=0,
        description=(
            "Server-side deadline for a single /analyze graph run. At the "
            "6,000 TPM bootstrap floor, a cold 20-ticker sweep spends ~256s in "
            "governor sleeps plus ~50s of sentiment pacing; a proxy that times out "
            "well before this would otherwise leave the server burning quota on a "
            "request nothing is still waiting on."
        ),
    )

    # Require registered model identifiers at startup so configuration errors fail before the first request.
    ARGUS_FUNDAMENTAL_MODEL: str = Field(default="openai/gpt-oss-120b")
    ARGUS_SENTIMENT_MODEL: str = Field(default="openai/gpt-oss-20b")
    ARGUS_PORTFOLIO_MODEL: str = Field(default="openai/gpt-oss-120b")

    ARGUS_HMM_MODEL_PATH: str = Field(
        default=str(BASE_DIR / "argus" / "models" / "macro_hmm.joblib"),
        description=("Path to the persisted RegimeClassifier artifact (scripts/train_macro_hmm.py output)"),
    )

    @field_validator("ARGUS_UNIVERSE", "ARGUS_CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_csv_list(cls, value: object) -> object:
        """Split a comma-separated env string into a list; passes non-strings through."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    TECHNICAL_INDICATOR_WEIGHTS: dict[str, float] = asdict(_TECHNICAL_WEIGHTS)

    MAX_SINGLE_POSITION_PCT: float = SYSTEM.max_single_position_pct
    MAX_SECTOR_CONCENTRATION: float = SYSTEM.max_sector_concentration
    MAX_PORTFOLIO_BETA: float = SYSTEM.max_portfolio_beta

    VIX_BLACKOUT_THRESHOLD: float = SYSTEM.vix_blackout_threshold

    def data_path(self, name: str) -> Path:
        """Return `name` under the current ARGUS_DATA_DIR."""
        return Path(self.ARGUS_DATA_DIR) / name


def web_concurrency() -> str | None:
    """Return the WEB_CONCURRENCY environment value, or None when unset."""
    return os.environ.get("WEB_CONCURRENCY")


settings = Settings()
