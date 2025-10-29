from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger("mcp_server.client")


class ReadOnlyAPIClient:
    """Minimal async client for the upstream read-only glossary service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float,
        retry_attempts: int,
        retry_wait: float,
    ) -> None:
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
            "Accept": "application/json",
            "User-Agent": "glossary-mcp/1.0",
            "X-Request-ID": request_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
                if response.status_code >= 500:
                    response.raise_for_status()
                return response
        raise RuntimeError("Retry loop unexpectedly exhausted")

    async def retrieve_glossary(
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
    ) -> Any:
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
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to decode JSON response", extra={"status": response.status_code})
            raise ValueError("Invalid JSON received from upstream service") from exc
