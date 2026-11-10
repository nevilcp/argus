"""
argus/schemas/requests.py

Input types and bounds shared by the environment settings (argus/config.py)
and the HTTP request models (api/main.py), so a value one accepts is never
rejected by the other.

Responsibilities:
  - Define the RiskTolerance tiers
  - Define the exclusive/inclusive bounds on investable wealth and invest fraction

Not responsible for:
  - Validating ticker symbols (see argus/data/tickers.py)
"""

from __future__ import annotations

from typing import Literal

RiskTolerance = Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]

# Total wealth must exceed this (USD): below it, per-position dollar sizes are meaningless.
MIN_TOTAL_WEALTH = 1000.0

# Invest fraction must exceed MIN_INVEST_PCT and be at most MAX_INVEST_PCT, so some
# capital is always deployed and some always held back.
MIN_INVEST_PCT = 0.05
MAX_INVEST_PCT = 0.95
