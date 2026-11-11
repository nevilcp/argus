"""Per-run collector for the notes an agent emits when it degrades."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


class Diagnostics:
    """Logs each degradation note and keeps it for the caller.

    The notes end up in ``ARGUSState["errors"]``, so a caller that wants them
    reads ``messages`` and passes it on unchanged.

    Attributes:
        messages: Every recorded note, in the order recorded.
    """

    def __init__(self) -> None:
        """Starts with no notes."""
        self.messages: list[str] = []

    def record(self, msg: str, level: Literal["warning", "error"] = "warning") -> None:
        """Logs one note and appends it to ``messages``.

        Args:
            msg: Human-readable note, already prefixed with the agent stage.
            level: ``"warning"`` for a degraded-but-continuing path, ``"error"``
                for a failure the caller cannot recover from.
        """
        if level == "error":
            logger.error("%s", msg)
        else:
            logger.warning("%s", msg)
        self.messages.append(msg)
