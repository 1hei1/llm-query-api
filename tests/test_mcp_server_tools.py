from __future__ import annotations

import json

import pytest
import pytest_asyncio
import respx
from httpx import Response

from mcp_server.config import MCPServerSettings
from mcp_server.server import MCPServerApplication


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> MCPServerSettings:
    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    return MCPServerSettings()


@pytest_asyncio.fixture
async def app(settings: MCPServerSettings) -> MCPServerApplication:
    application = MCPServerApplication(settings=settings)
    yield application
    await application.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_list_glossaries_returns_payload(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    respx_mock.get("https://api.example.com/glossaries").mock(
        return_value=Response(200, json={"items": [{"id": "dataset-1", "name": "Main"}], "total": 1})
    )

    result = await app._handle_list_glossaries(name=None, ctx=None)

    assert result["total"] == 1
    assert result["items"][0]["dataset_id"] == "dataset-1"
    assert respx_mock.calls.last.request.headers["Authorization"] == "Bearer test-key"


@respx.mock
@pytest.mark.asyncio
async def test_get_glossary_falls_back_to_listing(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    respx_mock.get("https://api.example.com/glossaries/dataset-1").respond(405)
    respx_mock.get("https://api.example.com/glossaries").mock(
        return_value=Response(200, json={"items": [{"id": "dataset-1", "name": "Glossary"}], "total": 1})
    )

    result = await app._handle_get_glossary(dataset_id="dataset-1", ctx=None)

    assert result["dataset_id"] == "dataset-1"


@respx.mock
@pytest.mark.asyncio
async def test_get_glossary_missing_raises(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    respx_mock.get("https://api.example.com/glossaries/dataset-404").respond(404)
    respx_mock.get("https://api.example.com/glossaries").mock(
        return_value=Response(200, json={"items": [], "total": 0})
    )

    with pytest.raises(ValueError):
        await app._handle_get_glossary(dataset_id="dataset-404", ctx=None)


@respx.mock
@pytest.mark.asyncio
async def test_search_terms_validates_and_calls_retrieve(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    route = respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(
            200,
            json={
                "chunks": [
                    {"id": "chunk-1", "content": "Definition", "similarity": 0.9, "document_name": "Doc"}
                ],
                "total": 1,
            },
        )
    )

    result = await app._handle_search_terms(dataset_id="dataset-1", query="term", top_k=5, ctx=None)

    assert result["total"] == 1
    assert result["results"][0]["chunk_id"] == "chunk-1"
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["question"] == "term"


@pytest.mark.asyncio
async def test_search_terms_rejects_long_query(app: MCPServerApplication) -> None:
    app.settings.max_query_length = 5
    with pytest.raises(ValueError):
        await app._handle_search_terms(dataset_id="dataset-1", query="too-long-query", top_k=None, ctx=None)


@respx.mock
@pytest.mark.asyncio
async def test_retrieve_definitions_sanitises_terms(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    route = respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(200, json={"chunks": [], "total": 0})
    )

    await app._handle_retrieve_definitions(dataset_id="dataset-1", terms=[" term a ", "", "term b"], top_k=None, ctx=None)

    payload = json.loads(route.calls.last.request.content.decode())
    assert "- term a" in payload["question"]
    assert "- term b" in payload["question"]
    assert payload["question"].strip().startswith("Provide glossary definitions")


@respx.mock
@pytest.mark.asyncio
async def test_rate_limit_enforced_per_tool(respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = MCPServerSettings(rate_limit_capacity=1, tool_rate_limits={"search_terms": 1})
    app = MCPServerApplication(settings=settings)
    try:
        respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
            return_value=Response(200, json={"chunks": [], "total": 0})
        )
        await app._handle_search_terms(dataset_id="dataset-1", query="term", top_k=None, ctx=None)

        with pytest.raises(ValueError, match="Rate limit"):
            await app._handle_search_terms(dataset_id="dataset-1", query="term", top_k=None, ctx=None)
    finally:
        await app.aclose()


@pytest.mark.asyncio
async def test_dataset_id_validation(app: MCPServerApplication) -> None:
    with pytest.raises(ValueError):
        await app._handle_get_glossary(dataset_id="bad id!", ctx=None)


@pytest.mark.asyncio
async def test_registered_tools_are_read_only(app: MCPServerApplication) -> None:
    server = app.create_server()
    tool_names = {tool.name for tool in server._tool_manager.list_tools()}  # type: ignore[attr-defined]
    assert tool_names == {"list_glossaries", "get_glossary", "search_terms", "retrieve_definitions"}
    assert all("rag" not in name for name in tool_names)
