from __future__ import annotations

from typing import Any, List, Optional

from fastapi import HTTPException, status
from openai import AsyncOpenAI
from openai import OpenAIError

from app.config import get_settings


class LLMClient:
    """Thin async wrapper around an OpenAI-compatible chat API."""

    def __init__(self, *, api_key: Optional[str], base_url: Optional[str], timeout: float) -> None:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI-compatible API key is not configured.",
            )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

        self._client = AsyncOpenAI(**client_kwargs)

    async def chat(
        self,
        *,
        messages: List[dict[str, str]],
        model: str,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except OpenAIError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider returned an error.",
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Unexpected error communicating with LLM provider.",
            ) from exc

        if not response.choices:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider returned an empty response.",
            )

        message = response.choices[0].message
        content = (message.content or "").strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider returned an empty message.",
            )

        return content


async def get_llm_client() -> LLMClient:
    settings = get_settings()
    base_url = settings.openai_base_url
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
    else:
        base_url = "https://api.openai.com/v1"

    return LLMClient(
        api_key=settings.openai_api_key,
        base_url=base_url,
        timeout=settings.http_timeout,
    )
