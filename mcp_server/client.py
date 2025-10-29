from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger("mcp_server.client")


class LLMQueryAPIClient:
    """Minimal async client for the upstream llm-query-api service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        retry_attempts: int,
        retry_wait: float,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for MCP server upstream calls")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_wait = retry_wait
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, request_id: str, *, include_json: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "llm-query-api-mcp/1.0",
            "X-Request-ID": request_id,
        }
        if include_json:
            headers["Content-Type"] = "application/json"
        return headers

    async def _request(
        self,
        method: str,
        url: str,
        *,
        request_id: str,
        params: dict[str, Any] | None = None,
        json_payload: Any | None = None,
    ) -> httpx.Response:
        headers = self._headers(request_id, include_json=json_payload is not None)
        retry = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_wait),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True,
        )

        async for attempt in retry:
            with attempt:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_payload,
                    headers=headers,
                )
                # For 5xx propagate HTTPStatusError to trigger retry
                if response.status_code >= 500:
                    response.raise_for_status()
                return response
        raise RuntimeError("Retry loop unexpectedly exhausted")

    async def list_glossaries(self, *, request_id: str, name: str | None = None) -> dict[str, Any]:
        params = {"name": name} if name else None
        response = await self._request("GET", "/glossaries", request_id=request_id, params=params)
        if response.status_code >= 400:
            response.raise_for_status()
        return self._json(response)

    async def fetch_glossary(self, *, request_id: str, dataset_id: str) -> dict[str, Any] | None:
        response = await self._request("GET", f"/glossaries/{dataset_id}", request_id=request_id)
        if response.status_code in {404, 405, 501}:
            logger.debug(
                "Direct glossary fetch returned status %s, falling back to list",
                response.status_code,
            )
        elif response.status_code >= 400:
            response.raise_for_status()
            return self._json(response)
        else:
            return self._json(response)

        listing = await self.list_glossaries(request_id=request_id)
        for item in listing.get("items", []):
            identifier = item.get("dataset_id") or item.get("id")
            if identifier == dataset_id:
                return item
        return None

    async def search_glossary(
        self,
        *,
        request_id: str,
        dataset_id: str,
        question: str,
        top_k: int,
        similarity_threshold: float,
        vector_similarity_weight: float,
        keyword: bool,
        highlight: bool,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "keyword": keyword,
            "highlight": highlight,
        }
        response = await self._request(
            "POST",
            f"/glossaries/{dataset_id}/retrieve",
            request_id=request_id,
            json_payload=payload,
        )
        if response.status_code >= 400:
            response.raise_for_status()
        return self._json(response)

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to decode JSON response", extra={"status": response.status_code})
            raise ValueError("Invalid JSON received from upstream service") from exc
