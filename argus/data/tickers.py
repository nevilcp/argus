"""Equity ticker validation shared by all universe entry points."""

from __future__ import annotations

import re

# Accept equity share classes but exclude indices and currency pairs without company financials or sectors.
TICKER_PATTERN = re.compile(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?")


def is_valid_ticker(ticker: str) -> bool:
    """Check whether ticker matches the accepted equity symbol shape."""
    return bool(TICKER_PATTERN.fullmatch(ticker))
