from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import ValidationError

from .audit import log_tool_event
from .client import LLMQueryAPIClient
from .config import MCPServerSettings, get_settings
from .exceptions import RateLimitExceeded
from .models import (
    GlossaryListResult,
    GlossarySummary,
    RetrievalChunkResult,
    RetrieveDefinitionsResult,
    SearchTermsResult,
)
from .rate_limiter import TokenBucket
from .utils import generate_request_id, sanitize_terms

_ALLOWED_TOOLS = ("list_glossaries", "get_glossary", "search_terms", "retrieve_definitions")


class MCPServerApplication:
    """Encapsulates MCP server state and tool handlers."""

    def __init__(self, settings: MCPServerSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dataset_regex = re.compile(self.settings.dataset_id_pattern)
        self._rate_limiters = self._build_rate_limiters()
        self._client = LLMQueryAPIClient(
            base_url=str(self.settings.llm_api_base_url),
            api_key=self.settings.llm_api_key.get_secret_value(),
            timeout=self.settings.http_timeout,
            retry_attempts=self.settings.retry_attempts,
            retry_wait=self.settings.retry_wait,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def create_server(self) -> FastMCP:
        """Instantiate and configure the MCP server."""

        instructions = (
            "Read-only glossary access. Available tools: list_glossaries, get_glossary, "
            "search_terms, retrieve_definitions. No dataset mutations or Q&A are permitted."
        )

        server = FastMCP(
            name="llm-query-api-readonly",
            instructions=instructions,
            log_level=self.settings.log_level.upper(),
        )

        @server.tool(
            name="list_glossaries",
            description="List available glossary datasets. Optionally filter by name substring.",
        )
        async def list_glossaries_tool(name: str | None = None, ctx: Context | None = None) -> dict[str, Any]:
            return await self._handle_list_glossaries(name=name, ctx=ctx)

        @server.tool(
            name="get_glossary",
            description="Fetch a glossary dataset by ID and return its metadata.",
        )
        async def get_glossary_tool(dataset_id: str, ctx: Context | None = None) -> dict[str, Any]:
            return await self._handle_get_glossary(dataset_id=dataset_id, ctx=ctx)

        @server.tool(
            name="search_terms",
            description=(
                "Search glossary content for passages related to a query. "
                "Returns matching chunks with similarity scores."
            ),
        )
        async def search_terms_tool(
            dataset_id: str,
            q: str,
            top_k: int | None = None,
            ctx: Context | None = None,
        ) -> dict[str, Any]:
            return await self._handle_search_terms(dataset_id=dataset_id, query=q, top_k=top_k, ctx=ctx)

        @server.tool(
            name="retrieve_definitions",
            description=(
                "Retrieve glossary definitions for one or more terms using semantic similarity."
            ),
        )
        async def retrieve_definitions_tool(
            dataset_id: str,
            terms: list[str],
            top_k: int | None = None,
            ctx: Context | None = None,
        ) -> dict[str, Any]:
            return await self._handle_retrieve_definitions(
                dataset_id=dataset_id,
                terms=terms,
                top_k=top_k,
                ctx=ctx,
            )

        return server

    async def _handle_list_glossaries(self, *, name: str | None, ctx: Context | None) -> dict[str, Any]:
        request_id = generate_request_id()
        await self._enforce_rate_limit("list_glossaries")
        start = perf_counter()

        filtered_name = name.strip() if name else None
        audit_args = {"name_filter": bool(filtered_name)}

        success = False
        try:
            payload = await self._client.list_glossaries(request_id=request_id, name=filtered_name)
            items = [GlossarySummary.from_payload(item) for item in payload.get("items", [])]
            total = int(payload.get("total", len(items)))
            result = GlossaryListResult(items=items, total=total)
            success = True
            return result.model_dump(mode="python", exclude_none=True)
        except Exception as exc:  # pragma: no cover - re-raised for MCP error handling
            log_tool_event(
                tool="list_glossaries",
                request_id=request_id,
                status="error",
                duration_ms=(perf_counter() - start) * 1000,
                arguments=audit_args,
                context=ctx,
                error=str(exc),
            )
            raise
        finally:
            if success:
                log_tool_event(
                    tool="list_glossaries",
                    request_id=request_id,
                    status="success",
                    duration_ms=(perf_counter() - start) * 1000,
                    arguments=audit_args,
                    context=ctx,
                )

    async def _handle_get_glossary(self, *, dataset_id: str, ctx: Context | None) -> dict[str, Any]:
        validated_id = self._validate_dataset_id(dataset_id)
        request_id = generate_request_id()
        await self._enforce_rate_limit("get_glossary")
        start = perf_counter()
        audit_args = {"dataset_id": validated_id}

        success = False
        try:
            data = await self._client.fetch_glossary(request_id=request_id, dataset_id=validated_id)
            if not data:
                raise ValueError(f"Glossary dataset '{validated_id}' was not found.")
            summary = GlossarySummary.from_payload(data)
            success = True
            return summary.model_dump(mode="python", exclude_none=True)
        except Exception as exc:  # pragma: no cover - re-raised for MCP error handling
            log_tool_event(
                tool="get_glossary",
                request_id=request_id,
                status="error",
                duration_ms=(perf_counter() - start) * 1000,
                arguments=audit_args,
                context=ctx,
                error=str(exc),
            )
            raise
        finally:
            if success:
                log_tool_event(
                    tool="get_glossary",
                    request_id=request_id,
                    status="success",
                    duration_ms=(perf_counter() - start) * 1000,
                    arguments=audit_args,
                    context=ctx,
                )

    async def _handle_search_terms(
        self,
        *,
        dataset_id: str,
        query: str,
        top_k: int | None,
        ctx: Context | None,
    ) -> dict[str, Any]:
        validated_id = self._validate_dataset_id(dataset_id)
        validated_query = self._validate_query(query)
        use_top_k = self._normalize_top_k(top_k or self.settings.search_top_k)

        request_id = generate_request_id()
        await self._enforce_rate_limit("search_terms")
        start = perf_counter()
        audit_args = {
            "dataset_id": validated_id,
            "query_length": len(validated_query),
            "top_k": use_top_k,
        }

        success = False
        try:
            payload = await self._client.search_glossary(
                request_id=request_id,
                dataset_id=validated_id,
                question=validated_query,
                top_k=use_top_k,
                similarity_threshold=self.settings.similarity_threshold,
                vector_similarity_weight=self.settings.vector_similarity_weight,
                keyword=False,
                highlight=True,
            )
            results, total = self._parse_retrieval(payload)
            response = SearchTermsResult(
                dataset_id=validated_id,
                query=validated_query,
                total=total,
                results=results,
            )
            success = True
            return response.model_dump(mode="python", exclude_none=True)
        except Exception as exc:  # pragma: no cover - re-raised for MCP error handling
            log_tool_event(
                tool="search_terms",
                request_id=request_id,
                status="error",
                duration_ms=(perf_counter() - start) * 1000,
                arguments=audit_args,
                context=ctx,
                error=str(exc),
            )
            raise
        finally:
            if success:
                log_tool_event(
                    tool="search_terms",
                    request_id=request_id,
                    status="success",
                    duration_ms=(perf_counter() - start) * 1000,
                    arguments=audit_args,
                    context=ctx,
                )

    async def _handle_retrieve_definitions(
        self,
        *,
        dataset_id: str,
        terms: list[str],
        top_k: int | None,
        ctx: Context | None,
    ) -> dict[str, Any]:
        validated_id = self._validate_dataset_id(dataset_id)
        normalized_terms = self._validate_terms(terms)
        use_top_k = self._normalize_top_k(top_k or max(self.settings.definition_top_k, len(normalized_terms)))
        question = self._build_definition_prompt(normalized_terms)

        request_id = generate_request_id()
        await self._enforce_rate_limit("retrieve_definitions")
        start = perf_counter()
        audit_args = {
            "dataset_id": validated_id,
            "term_count": len(normalized_terms),
            "top_k": use_top_k,
        }

        success = False
        try:
            payload = await self._client.search_glossary(
                request_id=request_id,
                dataset_id=validated_id,
                question=question,
                top_k=use_top_k,
                similarity_threshold=self.settings.similarity_threshold,
                vector_similarity_weight=self.settings.vector_similarity_weight,
                keyword=False,
                highlight=True,
            )
            results, total = self._parse_retrieval(payload)
            response = RetrieveDefinitionsResult(
                dataset_id=validated_id,
                terms=normalized_terms,
                total=total,
                results=results,
            )
            success = True
            return response.model_dump(mode="python", exclude_none=True)
        except Exception as exc:  # pragma: no cover - re-raised for MCP error handling
            log_tool_event(
                tool="retrieve_definitions",
                request_id=request_id,
                status="error",
                duration_ms=(perf_counter() - start) * 1000,
                arguments=audit_args,
                context=ctx,
                error=str(exc),
            )
            raise
        finally:
            if success:
                log_tool_event(
                    tool="retrieve_definitions",
                    request_id=request_id,
                    status="success",
                    duration_ms=(perf_counter() - start) * 1000,
                    arguments=audit_args,
                    context=ctx,
                )

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

    def _validate_terms(self, terms: list[str]) -> list[str]:
        normalized = sanitize_terms(terms)
        if not normalized:
            raise ValueError("At least one term is required.")
        if len(normalized) > self.settings.max_terms:
            raise ValueError(f"A maximum of {self.settings.max_terms} terms may be requested at once.")
        for term in normalized:
            if len(term) > self.settings.max_term_length:
                raise ValueError(
                    f"Term '{term}' exceeds maximum length of {self.settings.max_term_length} characters."
                )
        return normalized

    def _build_definition_prompt(self, terms: list[str]) -> str:
        header = "Provide glossary definitions for the following terms:"
        bullet_list = "\n".join(f"- {term}" for term in terms)
        return f"{header}\n{bullet_list}"

    def _normalize_top_k(self, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be greater than zero")
        return min(value, 1024)

    def _parse_retrieval(self, payload: dict[str, Any]) -> tuple[list[RetrievalChunkResult], int]:
        chunks_payload = payload.get("chunks") or payload.get("results") or []
        results: list[RetrievalChunkResult] = []
        for raw in chunks_payload:
            data = dict(raw)
            data.setdefault("id", data.get("chunk_id") or data.get("chunkId") or data.get("document_id", ""))
            try:
                results.append(RetrievalChunkResult.model_validate(data))
            except ValidationError:  # pragma: no cover - skip malformed chunks
                continue
        total = int(payload.get("total", len(results)))
        return results, total

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
