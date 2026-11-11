"""
api/models.py

Request and response models for the HTTP gateway.

Responsibilities:
  - AnalysisRequest: validate and normalize the /analyze payload
  - AnalysisResponse: the shape /analyze returns

Not responsible for:
  - Routing or admission checks (see api/main.py)
  - The input bounds themselves (see argus/schemas/requests.py)
"""

from pydantic import BaseModel, Field, field_validator

from argus.data.tickers import is_valid_ticker
from argus.schemas.requests import (
    MAX_INVEST_PCT,
    MIN_INVEST_PCT,
    MIN_TOTAL_WEALTH,
    RiskTolerance,
)
from argus.schemas.signals import PositionAllocation


class AnalysisRequest(BaseModel):
    """Request payload defining parameters for stock universe valuation and rebalancing."""

    tickers: list[str] = Field(min_length=1, max_length=20)
    total_wealth: float = Field(gt=MIN_TOTAL_WEALTH)
    invest_pct: float = Field(gt=MIN_INVEST_PCT, le=MAX_INVEST_PCT)
    risk_tolerance: RiskTolerance = "MODERATE"

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, tickers: list[str]) -> list[str]:
        """Strips whitespace, upper-cases, validates ticker shape, then dedupes.

        Runs in that order deliberately: pydantic's StringConstraints checks
        `pattern` against the raw, pre-transform string even when combined
        with `to_upper`/`strip_whitespace`, so a case/whitespace-only field
        constraint can't do this — the transform has to happen before the
        pattern check runs.

        Raises:
            ValueError: If any entry isn't a valid ticker after normalization.
        """
        normalized = [t.strip().upper() for t in tickers]
        invalid = [t for t in normalized if not is_valid_ticker(t)]
        if invalid:
            raise ValueError(f"Invalid ticker symbol(s): {invalid}")
        return list(dict.fromkeys(normalized))


class AnalysisResponse(BaseModel):
    """Synthesized portfolio allocation recommendation and system diagnostics."""

    session_id: str
    portfolio: list[PositionAllocation]
    cash_reserve_pct: float
    macro_regime: str
    vix_level: float
    governor_report: dict
    timestamp: str
    errors: list[str] = []
