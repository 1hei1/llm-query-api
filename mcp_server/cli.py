from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import get_settings
from .logging import configure_logging
from .server import MCPServerApplication

_TRANSPORT_CHOICES = ("stdio", "sse", "streamable-http")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-server",
        description="Read-only MCP server for llm-query-api glossaries",
    )
    parser.add_argument("command", nargs="?", default="run", choices=["run"], help="Command to execute")
    parser.add_argument(
        "--transport",
        choices=_TRANSPORT_CHOICES,
        default="stdio",
        help="Transport protocol used to serve MCP clients (default: stdio)",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Mount path for the SSE transport (only used when transport=sse)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 1

    settings = get_settings()
    configure_logging(settings.log_level)
    app = MCPServerApplication(settings=settings)
    server = app.create_server()

    try:
        server.run(transport=args.transport, mount_path=args.mount_path)
        return 0
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        return 0
    finally:
        _close_async(app)


def _close_async(app: MCPServerApplication) -> None:
    try:
        import anyio
    except ImportError:  # pragma: no cover - fallback path
        import asyncio

        asyncio.run(app.aclose())
    else:
        anyio.run(app.aclose)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
