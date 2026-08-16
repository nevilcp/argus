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
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from argus.params import SYSTEM
from argus.params import TECHNICAL_INDICATOR_WEIGHTS as _TECHNICAL_WEIGHTS

BASE_DIR = Path(__file__).resolve().parent.parent


def _split_csv(value: object) -> object:
    """Splits a comma-separated env string into a list; passes non-strings through.

    pydantic-settings JSON-decodes env values for list-typed fields by default,
    which rejects a plain `AAPL,MSFT` string — the fields using this validator
    are marked `NoDecode` to skip that and land here instead.

    Args:
        value: Raw field value — a comma-separated string from the environment,
            or an already-parsed list (e.g. the Python-level default).

    Returns:
        A list of stripped, non-empty strings, or `value` unchanged if it
        wasn't a string.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


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

    groq_api_key: str = Field(default="", description="Groq LLM API key")

    fred_api_key: str = Field(default="", description="FRED (Federal Reserve) API key")

    newsapi_key: str = Field(default="", description="NewsAPI.org key")

    # Broadened to 20 tickers across 7 GICS sectors to reduce sector concentration risk
    ARGUS_UNIVERSE: Annotated[List[str], NoDecode] = Field(
        default=[
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "JPM", "V", "UNH",
            "XOM", "JNJ", "PG", "MA", "HD",
            "MRK", "ABBV", "LLY", "AVGO", "CVX",
        ],
        description="Ticker universe the unattended collector tracks and analyzes",
    )

    # Numeric defaults live in argus/params.py, tagged with provenance; this class
    # only adds env-var override capability on top of them
    #
    # yfinance's chart endpoint returns the whole requested period in one HTTP
    # call regardless of interval granularity, so 1m candles cost nothing extra
    # against any rate limit — see data/pipeline.py for how coarser resolutions
    # are derived locally via resampling rather than fetched separately
    MFT_CANDLE_INTERVAL: str = "1m"
    MFT_DECISION_INTERVAL_SECONDS: int = SYSTEM.mft_decision_interval_seconds
    # None means "derive from MFT_CANDLE_INTERVAL" (see data/pipeline.py's
    # _derive_buffer_size); set explicitly only to override that derivation
    CANDLE_BUFFER_SIZE: Optional[int] = None

    # ─── Unattended operation ────────────────────────────────────────────────
    ARGUS_DATA_DIR: str = Field(
        default="data", description="Directory for the persistent intraday buffer and logs"
    )
    ARGUS_COLLECTOR_ENABLED: bool = Field(
        default=False, description="Run the graph automatically on a schedule during market hours"
    )
    ARGUS_COLLECTOR_INTERVAL_SECONDS: int = Field(
        default=3600, description="Seconds between automatic collection cycles"
    )
    ARGUS_RECONCILE_ENABLED: bool = Field(
        default=False, description="Automatically reconcile matured decisions once a day"
    )
    ARGUS_RECONCILE_HOUR_ET: int = Field(
        default=17, ge=0, le=23, description="Eastern-time hour to run the daily reconcile pass"
    )
    ARGUS_TOTAL_WEALTH: float = Field(
        default=100_000.0, gt=1000, description="Total wealth used by unattended collection cycles"
    )
    ARGUS_INVEST_PCT: float = Field(
        default=0.6, gt=0.05, le=0.95, description="Invest fraction used by unattended collection cycles"
    )
    ARGUS_RISK_TOLERANCE: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] = Field(
        default="MODERATE", description="Risk tolerance used by unattended collection cycles"
    )
    ARGUS_CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(
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
    ARGUS_LOG_LEVEL: str = Field(default="INFO", description="Root log level for argus.* loggers")
    ARGUS_HMM_MODEL_PATH: str = Field(
        default=str(BASE_DIR / "argus" / "models" / "macro_hmm.joblib"),
        description="Path to the persisted RegimeClassifier artifact (scripts/train_macro_hmm.py output)",
    )

    @field_validator("ARGUS_UNIVERSE", "ARGUS_CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_csv_list(cls, value: object) -> object:
        """Applies :func:`_split_csv` to comma-separated list fields."""
        return _split_csv(value)

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

    VIX_BLACKOUT_THRESHOLD: float = SYSTEM.vix_blackout_threshold

    # Historical lookback window used in rolling indicator calculations
    LOOKBACK_DAYS: int = SYSTEM.lookback_days


# Module-level singleton — import this everywhere rather than instantiating locally
settings = Settings()
