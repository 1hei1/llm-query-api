from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from httpx import HTTPStatusError
from mcp.server.fastmcp import Context, FastMCP

from .audit import log_tool_event
from .client import ReadOnlyAPIClient
from .config import MCPServerSettings, get_settings
from .exceptions import RateLimitExceeded
from .rate_limiter import TokenBucket
from .utils import generate_request_id

_ALLOWED_TOOLS = ("search_glossary", "retrieve_docs")


class MCPServerApplication:
    """Encapsulates MCP server state and tool handlers."""

    def __init__(self, settings: MCPServerSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dataset_regex = re.compile(self.settings.dataset_id_pattern)
        self._rate_limiters = self._build_rate_limiters()
        api_key = self.settings.api_key.get_secret_value() if self.settings.api_key else None
        self._client = ReadOnlyAPIClient(
            base_url=str(self.settings.api_base_url),
            api_key=api_key,
            timeout=self.settings.http_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_wait=self.settings.retry_wait,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def create_server(self) -> FastMCP:
        """Instantiate and configure the MCP server."""

        instructions = (
            "Read-only glossary retrieval. Available tools: search_glossary, retrieve_docs. "
            "Requests are proxied directly to the upstream FastAPI service; no LLM answering is provided."
        )

        server = FastMCP(
            name="glossary-retrieval-mcp",
            instructions=instructions,
            log_level=self.settings.log_level.upper(),
        )

        @server.tool(
            name="search_glossary",
            description=(
                "Search glossary content for passages related to a term. "
                "Returns the upstream FastAPI response unchanged."
            ),
        )
        async def search_glossary_tool(
            dataset_id: str,
            term: str,
            top_k: int | None = None,
            ctx: Context | None = None,
        ) -> Any:
            return await self._handle_search_glossary(dataset_id=dataset_id, term=term, top_k=top_k, ctx=ctx)

        @server.tool(
            name="retrieve_docs",
            description=(
                "Retrieve glossary documents and chunk metadata for a query. "
                "Returns the upstream FastAPI response unchanged."
            ),
        )
        async def retrieve_docs_tool(
            dataset_id: str,
            query: str,
            top_k: int | None = None,
            keyword: bool = False,
            highlight: bool = False,
            ctx: Context | None = None,
        ) -> Any:
            return await self._handle_retrieve_docs(
                dataset_id=dataset_id,
                query=query,
                top_k=top_k,
                keyword=keyword,
                highlight=highlight,
                ctx=ctx,
            )

        return server

    async def _handle_search_glossary(
        self,
        *,
        dataset_id: str,
        term: str,
        top_k: int | None,
        ctx: Context | None,
    ) -> Any:
        validated_id = self._validate_dataset_id(dataset_id)
        validated_query = self._validate_query(term)
        use_top_k = self._normalize_top_k(top_k or self.settings.search_top_k)

        return await self._invoke_retrieval_tool(
            tool_name="search_glossary",
            dataset_id=validated_id,
            query=validated_query,
            top_k=use_top_k,
            keyword=False,
            highlight=True,
            ctx=ctx,
        )

    async def _handle_retrieve_docs(
        self,
        *,
        dataset_id: str,
        query: str,
        top_k: int | None,
        keyword: bool,
        highlight: bool,
        ctx: Context | None,
    ) -> Any:
        validated_id = self._validate_dataset_id(dataset_id)
        validated_query = self._validate_query(query)
        use_top_k = self._normalize_top_k(top_k or self.settings.search_top_k)

        return await self._invoke_retrieval_tool(
            tool_name="retrieve_docs",
            dataset_id=validated_id,
            query=validated_query,
            top_k=use_top_k,
            keyword=keyword,
            highlight=highlight,
            ctx=ctx,
        )

    async def _invoke_retrieval_tool(
        self,
        *,
        tool_name: str,
        dataset_id: str,
        query: str,
        top_k: int,
        keyword: bool,
        highlight: bool,
        ctx: Context | None,
    ) -> Any:
        request_id = generate_request_id()
        await self._enforce_rate_limit(tool_name)
        start = perf_counter()
        audit_args = {
            "dataset_id": dataset_id,
            "query_length": len(query),
            "top_k": top_k,
            "keyword": keyword,
            "highlight": highlight,
        }

        try:
            payload = await self._client.retrieve_glossary(
                request_id=request_id,
                dataset_id=dataset_id,
                question=query,
                top_k=top_k,
                similarity_threshold=self.settings.similarity_threshold,
                vector_similarity_weight=self.settings.vector_similarity_weight,
                keyword=keyword,
                highlight=highlight,
            )
        except Exception as exc:  # pragma: no cover - re-raised for MCP error handling
            duration_ms = (perf_counter() - start) * 1000
            error_message = self._format_http_error(exc) if isinstance(exc, HTTPStatusError) else str(exc)
            log_tool_event(
                tool=tool_name,
                request_id=request_id,
                status="error",
                duration_ms=duration_ms,
                arguments=audit_args,
                context=ctx,
                error=error_message,
            )
            if isinstance(exc, HTTPStatusError):
                raise ValueError(error_message) from exc
            raise
        else:
            duration_ms = (perf_counter() - start) * 1000
            log_tool_event(
                tool=tool_name,
                request_id=request_id,
                status="success",
                duration_ms=duration_ms,
                arguments=audit_args,
                context=ctx,
            )
            return payload

    @staticmethod
    def _format_http_error(exc: HTTPStatusError) -> str:
        status_code = exc.response.status_code if exc.response is not None else 0
        detail: str = ""
        if exc.response is not None:
            try:
                payload = exc.response.json()
            except json.JSONDecodeError:
                detail = exc.response.text or ""
            else:
                if isinstance(payload, dict):
                    raw_detail = payload.get("detail") or payload.get("message")
                    if raw_detail is None:
                        detail = json.dumps(payload)
                    elif isinstance(raw_detail, (str, int, float)):
                        detail = str(raw_detail)
                    else:
                        detail = json.dumps(raw_detail)
                else:
                    detail = str(payload)
        detail = detail.strip()
        if not detail:
            detail = exc.response.reason_phrase if exc.response is not None else exc.args[0]
        return f"Upstream request failed with status {status_code}: {detail}".strip()

    def _validate_dataset_id(self, dataset_id: str) -> str:
        value = (dataset_id or "").strip()
        if not value:
            raise ValueError("dataset_id is required")
        if not self._dataset_regex.fullmatch(value):
            raise ValueError("dataset_id contains unsupported characters")
        return value

    def _validate_query(self, query: str) -> str:
        value = (query or "").strip()
        if not value:
            raise ValueError("Query text is required.")
        if len(value) > self.settings.max_query_length:
            raise ValueError(
                f"Query exceeds maximum length of {self.settings.max_query_length} characters."
            )
        return value

    def _normalize_top_k(self, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be greater than zero")
        return min(value, 1024)

    async def _enforce_rate_limit(self, tool_name: str) -> None:
        limiter = self._rate_limiters.get(tool_name)
        if limiter is None:
            raise ValueError(f"Tool '{tool_name}' is not registered.")
        try:
            await limiter.consume()
        except RateLimitExceeded as exc:
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                raise ValueError(
                    f"Rate limit exceeded for {tool_name}. Try again in {retry_after:.1f} seconds."
                ) from exc
            raise ValueError(f"Rate limit exceeded for {tool_name}.") from exc

    def _build_rate_limiters(self) -> dict[str, TokenBucket]:
        limiters: dict[str, TokenBucket] = {}
        for tool in _ALLOWED_TOOLS:
            limit = self.settings.tool_rate_limits.get(tool, self.settings.rate_limit_capacity)
            limiters[tool] = TokenBucket(capacity=limit, refill_interval=self.settings.rate_limit_interval_seconds)
        return limiters


def create_server(settings: MCPServerSettings | None = None) -> FastMCP:
    """Convenience helper to instantiate the MCP server."""

    app = MCPServerApplication(settings=settings)
    return app.create_server()
