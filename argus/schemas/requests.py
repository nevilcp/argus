"""Input types and bounds shared by settings and HTTP request validation."""

from __future__ import annotations

from typing import Literal

RiskTolerance = Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]

MIN_TOTAL_WEALTH = 1000.0

# Require both deployed capital and a cash reserve at the shared input boundary.
MIN_INVEST_PCT = 0.05
MAX_INVEST_PCT = 0.95
