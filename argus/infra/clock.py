"""The one place ARGUS reads the wall clock and names its market timezone."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def now_local() -> datetime:
    """Returns the naive system-local time.

    Every `timestamp` and `session_timestamp` ARGUS stores is naive local
    time, so a timestamp written here compares directly with one read back.
    """
    return datetime.now()  # noqa: DTZ005


def now_market() -> datetime:
    """Returns the current time as an aware datetime in the US market timezone."""
    return datetime.now(MARKET_TZ)


def as_naive_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalizes a datetime/Timestamp to tz-naive so entry/exit comparisons never raise.

    A tz-aware value is converted to UTC before its tzinfo is dropped, rather
    than discarding whatever offset it carries, which would shift the value by
    that offset (LangGraph stamps checkpoint `ts` tz-aware, for example) and
    make a pruning decision early or late by the offset's magnitude.

    Args:
        value: A naive or aware datetime or pandas Timestamp.

    Returns:
        A naive pandas Timestamp.
    """
    ts = pd.Timestamp(value)
    return ts.tz_convert(UTC).tz_localize(None) if ts.tzinfo is not None else ts
