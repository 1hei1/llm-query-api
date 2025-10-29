from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Emit structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        msg = record.msg
        if isinstance(msg, dict):
            payload.update(msg)
        else:
            payload["message"] = record.getMessage()

        if hasattr(record, "payload") and isinstance(record.payload, dict):
            payload.update(record.payload)

        if hasattr(record, "request_id") and record.request_id:
            payload["request_id"] = record.request_id

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging to emit JSON lines."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    logging.basicConfig(handlers=[handler], level=getattr(logging, level.upper(), logging.INFO), force=True)

    for noisy in ("httpx", "anyio", "tenacity"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
