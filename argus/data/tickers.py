"""
argus/data/tickers.py

Ticker symbol validation shared across every entry point that grows the
tracked universe: /analyze's request schema, MFTDataPipeline.register_tickers,
run_collection_cycle, and collect_session.py's --universe CLI arg.

Responsibilities:
  - Define the accepted equity ticker symbol pattern
  - Validate a candidate ticker string against it

Not responsible for:
  - Case normalization or deduplication (see api/main.py's AnalysisRequest)
  - Universe size limits (see argus/params.py's SYSTEM.max_tracked_tickers)
"""

from __future__ import annotations

import re

# 1-5 letter root, optional one-or-two-letter share-class suffix (BRK.B, BRK-B).
# Deliberately excludes ^GSPC-style indices and BTC-USD-style pairs: neither
# has the GICS sector or financials the risk and fundamental agents fetch.
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")


def is_valid_ticker(ticker: str) -> bool:
    """Checks whether ticker matches the accepted equity symbol shape.

    Args:
        ticker: Candidate ticker symbol, expected already upper-cased.

    Returns:
        True if ticker matches TICKER_PATTERN.
    """
    return bool(TICKER_PATTERN.match(ticker))
