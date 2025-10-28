from __future__ import annotations

import pytest
import respx
from httpx import Response

from app.services.llm_client import LLMClient


@pytest.mark.asyncio
async def test_llm_client_chat_returns_message() -> None:
    client = LLMClient(api_key="key", base_url="https://llm.example/v1", timeout=10)

    response_payload = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here is the answer.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    with respx.mock(assert_all_called=True) as router:
        router.post("https://llm.example/v1/chat/completions").mock(
            return_value=Response(200, json=response_payload)
        )

        answer = await client.chat(messages=[{"role": "user", "content": "Hi"}], model="test-model")

    assert answer == "Here is the answer."
