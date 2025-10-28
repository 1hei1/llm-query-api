from __future__ import annotations

import pytest
import respx
from httpx import Response

from app.services.ragflow_client import RAGFlowClient


@pytest.mark.asyncio
async def test_ensure_dataset_returns_existing() -> None:
    client = RAGFlowClient(
        base_url="https://ragflow.local",
        api_key="token",
        timeout=10,
        retry_attempts=1,
        retry_wait=0,
    )

    dataset_payload = {
        "id": "dataset-123",
        "name": "finance-glossary",
        "description": "Finance terms",
        "chunk_method": "naive",
    }

    with respx.mock(assert_all_called=True) as router:
        router.get("https://ragflow.local/api/v1/datasets").mock(
            return_value=Response(200, json={"code": 0, "data": [dataset_payload], "total": 1})
        )

        dataset = await client.ensure_dataset(name="finance-glossary")

    assert dataset["id"] == "dataset-123"


@pytest.mark.asyncio
async def test_ensure_dataset_creates_when_missing() -> None:
    client = RAGFlowClient(
        base_url="https://ragflow.local",
        api_key="token",
        timeout=10,
        retry_attempts=1,
        retry_wait=0,
    )

    created_payload = {
        "id": "dataset-456",
        "name": "new-glossary",
        "description": None,
    }

    with respx.mock(assert_all_called=True) as router:
        router.get("https://ragflow.local/api/v1/datasets").mock(
            return_value=Response(200, json={"code": 0, "data": [], "total": 0})
        )
        router.post("https://ragflow.local/api/v1/datasets").mock(
            return_value=Response(200, json={"code": 0, "data": created_payload})
        )

        dataset = await client.ensure_dataset(name="new-glossary")

    assert dataset == created_payload
