from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import httpx
from fastapi import HTTPException, status
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings


class RAGFlowClient:
    """Thin async wrapper around the RAGFlow HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        *,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_wait: float = 0.5,
    ) -> None:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAGFlow API key is not configured.",
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_wait = retry_wait

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        combined_headers = {**self._auth_headers, **(headers or {})}

        retry = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_wait),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True,
        )

        async for attempt in retry:
            with attempt:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                    response = await client.request(method, url, headers=combined_headers, **kwargs)

                if response.status_code >= 500:
                    response.raise_for_status()

                return response

        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Failed to communicate with RAGFlow.")

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._request(method, url, **kwargs)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail="Invalid JSON payload received from RAGFlow.",
            ) from exc

        if payload.get("code", 0) != 0:
            raise self._map_error(payload)

        return payload

    def _map_error(self, payload: dict[str, Any]) -> HTTPException:
        code = payload.get("code")
        message = payload.get("message", "RAGFlow request failed.")

        status_map = {
            1001: status.HTTP_400_BAD_REQUEST,
            1002: status.HTTP_400_BAD_REQUEST,
            400: status.HTTP_400_BAD_REQUEST,
            401: status.HTTP_401_UNAUTHORIZED,
            403: status.HTTP_403_FORBIDDEN,
            404: status.HTTP_404_NOT_FOUND,
            500: status.HTTP_502_BAD_GATEWAY,
            101: status.HTTP_409_CONFLICT,
        }
        status_code = status_map.get(code, status.HTTP_502_BAD_GATEWAY)

        return HTTPException(status_code=status_code, detail=message)

    async def ensure_dataset(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        chunk_method: Optional[str] = None,
    ) -> dict[str, Any]:
        listings = await self.list_datasets(name=name)
        for item in listings.get("data", []):
            if item.get("name", "").lower() == name.lower():
                return item

        payload = {
            "name": name,
        }
        if description:
            payload["description"] = description
        if chunk_method:
            payload["chunk_method"] = chunk_method

        response = await self._request_json(
            "POST",
            "/api/v1/datasets",
            json=payload,
        )
        return response.get("data", {})

    async def list_datasets(self, name: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if name:
            params["name"] = name

        response = await self._request_json(
            "GET",
            "/api/v1/datasets",
            params=params,
        )
        return response

    async def upload_documents(
        self,
        dataset_id: str,
        files: Sequence[tuple[str, tuple[str, bytes, str]]],
    ) -> list[dict[str, Any]]:
        response = await self._request_json(
            "POST",
            f"/api/v1/datasets/{dataset_id}/documents",
            files=list(files),
        )
        return response.get("data", [])

    async def parse_documents(self, dataset_id: str, document_ids: Sequence[str]) -> None:
        await self._request_json(
            "POST",
            f"/api/v1/datasets/{dataset_id}/chunks",
            json={"document_ids": list(document_ids)},
        )

    async def retrieval(
        self,
        *,
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
            "dataset_ids": [dataset_id],
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "keyword": keyword,
            "highlight": highlight,
        }

        response = await self._request_json(
            "POST",
            "/api/v1/retrieval",
            json=payload,
        )
        return response.get("data", {})


async def get_ragflow_client() -> RAGFlowClient:
    settings = get_settings()
    return RAGFlowClient(
        base_url=settings.ragflow_base_url,
        api_key=settings.ragflow_api_key,
        timeout=settings.http_timeout,
        retry_attempts=settings.ragflow_retry_attempts,
        retry_wait=settings.ragflow_retry_wait,
    )
