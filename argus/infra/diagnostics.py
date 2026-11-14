"""Per-run collector for the notes an agent emits when it degrades."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


class Diagnostics:
    """Logs each degradation note and keeps it for the caller."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def record(self, msg: str, level: Literal["warning", "error"] = "warning") -> None:
        """Log one note and append it to `messages`."""
        if level == "error":
            logger.error("%s", msg)
        else:
            logger.warning("%s", msg)
        self.messages.append(msg)
