"""The one place ARGUS reads the wall clock and names its market timezone."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def now_local() -> datetime:
    """Return the naive system-local time."""
    return datetime.now()  # noqa: DTZ005


def now_market() -> datetime:
    """Return the current time as an aware datetime in the US market timezone."""
    return datetime.now(MARKET_TZ)


def as_naive_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert aware timestamps to UTC before dropping timezone information."""
    ts = pd.Timestamp(value)
    return ts.tz_convert(UTC).tz_localize(None) if ts.tzinfo is not None else ts
