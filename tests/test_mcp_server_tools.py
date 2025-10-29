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
    monkeypatch.setenv("MCP_API_BASE_URL", "https://api.example.com")
    return MCPServerSettings()


@pytest_asyncio.fixture
async def app(settings: MCPServerSettings) -> MCPServerApplication:
    application = MCPServerApplication(settings=settings)
    yield application
    await application.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_search_glossary_passthrough(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    expected_payload = {
        "chunks": [
            {"id": "chunk-1", "content": "Definition", "similarity": 0.9, "document_name": "Doc"}
        ],
        "doc_aggs": [],
        "total": 1,
    }
    route = respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(200, json=expected_payload)
    )

    result = await app._handle_search_glossary(dataset_id="dataset-1", term="term", top_k=5, ctx=None)

    assert result == expected_payload
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["question"] == "term"
    assert payload["top_k"] == 5
    assert payload["highlight"] is True
    assert payload["keyword"] is False


@respx.mock
@pytest.mark.asyncio
async def test_search_glossary_empty_results(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    expected_payload = {"chunks": [], "doc_aggs": [], "total": 0}
    respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(200, json=expected_payload)
    )

    result = await app._handle_search_glossary(dataset_id="dataset-1", term="missing", top_k=None, ctx=None)

    assert result == expected_payload
    assert result["chunks"] == []


@respx.mock
@pytest.mark.asyncio
async def test_retrieve_docs_passthrough(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    expected_payload = {
        "chunks": [],
        "doc_aggs": [
            {"doc_id": "doc-1", "doc_name": "Doc 1", "count": 2},
        ],
        "total": 2,
    }
    route = respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(200, json=expected_payload)
    )

    result = await app._handle_retrieve_docs(
        dataset_id="dataset-1",
        query="definition",
        top_k=None,
        keyword=True,
        highlight=False,
        ctx=None,
    )

    assert result == expected_payload
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["keyword"] is True
    assert payload["highlight"] is False


@pytest.mark.asyncio
async def test_search_glossary_rejects_long_query(app: MCPServerApplication) -> None:
    app.settings.max_query_length = 5
    with pytest.raises(ValueError):
        await app._handle_search_glossary(dataset_id="dataset-1", term="too-long-query", top_k=None, ctx=None)


@respx.mock
@pytest.mark.asyncio
async def test_rate_limit_enforced(respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_API_BASE_URL", "https://api.example.com")
    settings = MCPServerSettings(rate_limit_capacity=1, tool_rate_limits={"retrieve_docs": 1})
    app = MCPServerApplication(settings=settings)
    try:
        respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
            return_value=Response(200, json={"chunks": [], "doc_aggs": [], "total": 0})
        )
        await app._handle_retrieve_docs(
            dataset_id="dataset-1",
            query="term",
            top_k=None,
            keyword=False,
            highlight=False,
            ctx=None,
        )

        with pytest.raises(ValueError, match="Rate limit"):
            await app._handle_retrieve_docs(
                dataset_id="dataset-1",
                query="term",
                top_k=None,
                keyword=False,
                highlight=False,
                ctx=None,
            )
    finally:
        await app.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_upstream_error_mapped(app: MCPServerApplication, respx_mock: respx.Router) -> None:
    respx_mock.post("https://api.example.com/glossaries/dataset-1/retrieve").mock(
        return_value=Response(404, json={"detail": "Dataset not found."})
    )

    with pytest.raises(ValueError, match="status 404") as exc_info:
        await app._handle_search_glossary(dataset_id="dataset-1", term="term", top_k=None, ctx=None)

    assert "Dataset not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dataset_id_validation(app: MCPServerApplication) -> None:
    with pytest.raises(ValueError):
        await app._handle_search_glossary(dataset_id="bad id!", term="term", top_k=None, ctx=None)


@pytest.mark.asyncio
async def test_registered_tools_are_retrieval_only(app: MCPServerApplication) -> None:
    server = app.create_server()
    tool_names = {tool.name for tool in server._tool_manager.list_tools()}  # type: ignore[attr-defined]
    assert tool_names == {"search_glossary", "retrieve_docs"}
    assert all("answer" not in name and "qa" not in name for name in tool_names)
