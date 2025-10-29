from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

try:  # Import lazily to avoid heavy dependency at module import time during tests
    from mcp.server.fastmcp import Context
except ImportError:  # pragma: no cover - safety for environments without MCP installed
    Context = object  # type: ignore[assignment]


audit_logger = logging.getLogger("mcp_server.audit")


def log_tool_event(
    *,
    tool: str,
    request_id: str,
    status: str,
    duration_ms: float,
    arguments: dict[str, Any],
    context: Context | None = None,
    error: str | None = None,
) -> None:
    """Emit a structured audit log entry for a tool invocation."""

    payload: dict[str, Any] = {
        "event": "tool_invocation",
        "tool": tool,
        "status": status,
        "request_id": request_id,
        "duration_ms": round(duration_ms, 2),
        "arguments": arguments,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    client_id = None
    if context is not None and hasattr(context, "client_id"):
        client_id = getattr(context, "client_id")
    if client_id:
        payload["client_id"] = client_id

    if error:
        payload["error"] = error

    audit_logger.info({"audit": payload})
