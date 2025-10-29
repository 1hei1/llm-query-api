# MCP Read-Only Glossary Server

This package provides a Model Context Protocol (MCP) server that exposes the existing [`llm-query-api`](./README.md) glossary retrieval features to LLM agents in a read-only manner. The server strictly proxies glossary search operations and **does not** expose any dataset creation, ingestion, deletion, or RAG Q&A capabilities.

## Features

- ✅ Read-only access to glossary metadata and retrieval endpoints
- ✅ Token-bucket rate limiting with configurable defaults and per-tool overrides
- ✅ Structured JSON logging with audit records for every tool invocation
- ✅ Input validation for dataset identifiers, query length, and term limits
- ✅ Tenacity-based retries, HTTP timeouts, and upstream request correlation IDs
- ❌ No dataset writes, uploads, or deletions
- ❌ No RAG/Q&A or generative endpoints

## Available Tools

| Tool | Description |
| ---- | ----------- |
| `list_glossaries` | Lists available glossary datasets (optional name filter). |
| `get_glossary` | Returns metadata for a specific glossary dataset. |
| `search_terms` | Retrieves glossary chunks relevant to a free-text query. |
| `retrieve_definitions` | Retrieves glossary definitions for a list of terms. |

All tools require the upstream `llm-query-api` service to be reachable and authenticated with an API key.

## Configuration

The server reads configuration from environment variables (values in parentheses indicate defaults):

| Variable | Description |
| -------- | ----------- |
| `LLM_API_BASE_URL` (`http://127.0.0.1:8000`) | Base URL of the upstream FastAPI service. |
| `LLM_API_KEY` | **Required.** API key for authenticating with the upstream service. |
| `MCP_HTTP_TIMEOUT` (`30.0`) | Timeout for outbound HTTP requests (seconds). |
| `MCP_RETRY_ATTEMPTS` (`3`) | Retry attempts for recoverable upstream failures. |
| `MCP_RETRY_WAIT` (`0.5`) | Delay between retries (seconds). |
| `MCP_RATE_LIMIT_CAPACITY` (`10`) | Default number of requests allowed per tool per interval. |
| `MCP_RATE_LIMIT_INTERVAL_SECONDS` (`60.0`) | Interval length (seconds) for the token bucket. |
| `MCP_TOOL_RATE_LIMITS` | Optional JSON object with per-tool rate limits, e.g. `{"search_terms": 5}`. |
| `MCP_MAX_QUERY_LENGTH` (`256`) | Maximum character length for free-text queries. |
| `MCP_MAX_TERMS` (`10`) | Maximum number of terms for `retrieve_definitions`. |
| `MCP_MAX_TERM_LENGTH` (`128`) | Maximum character length for an individual term. |
| `MCP_LOG_LEVEL` (`INFO`) | Log level for the MCP server. |

Add these environment variables to your `.env` file or pass them directly when launching the server.

## Running the Server

Activate your virtual environment, install dependencies, and run either of the following commands:

```bash
pip install -r requirements.txt

# Option 1: module invocation
python -m mcp_server run

# Option 2: explicit transport selection
python -m mcp_server run --transport stdio
```

The default `stdio` transport targets CLI-based MCP clients. To offer SSE or streamable HTTP endpoints, switch the `--transport` flag accordingly (note that additional MCP configuration may be required on the client side).

### CLI Wrapper

A helper script named `mcp-server` is available inside the repository root and delegates to `python -m mcp_server`. Make the script executable to invoke:

```bash
chmod +x mcp-server
./mcp-server run
```

## Logging & Audit Trail

Logs are emitted as structured JSON and include a request correlation ID. Audit events are generated for every tool invocation with the following payload:

```json
{
  "event": "tool_invocation",
  "tool": "search_terms",
  "status": "success",
  "request_id": "...",
  "duration_ms": 12.4,
  "arguments": {
    "dataset_id": "dataset-1",
    "query_length": 24,
    "top_k": 5
  }
}
```

Authentication headers are never written to logs. Only sanitized arguments (lengths, counts, dataset IDs) appear in audit events.

## Example Anthropic MCP Client Configuration

```json
{
  "mcpServers": {
    "glossaries": {
      "command": "python",
      "args": [
        "-m",
        "mcp_server",
        "run"
      ],
      "env": {
        "LLM_API_BASE_URL": "https://llm-query-api.internal",
        "LLM_API_KEY": "${LLM_API_KEY}"
      }
    }
  }
}
```

## Docker Support

A dedicated Dockerfile is provided for the MCP server. Build and run it alongside the API using the supplied Compose profile:

```bash
# Build the MCP image
docker build -f Dockerfile.mcp -t llm-query-api-mcp .

# Launch via docker compose (requires docker-compose 1.29+)
docker compose --profile mcp up
```

The Compose profile expects the `LLM_API_KEY` environment variable to be defined locally. The MCP container forwards requests to the API service via the internal Docker network.

## Testing

Tests rely on `pytest`, `pytest-asyncio`, and `respx` to mock upstream HTTP interactions. Run the suite with:

```bash
pytest
```

## Security Notes

- An API key (`LLM_API_KEY`) is mandatory; the server refuses to start without it.
- Only read-only glossary endpoints are exposed. Dataset creation, ingestion, deletion, and `/rag` routes are intentionally blocked.
- Input validation, rate limiting, and structured audit logs are in place to discourage abuse and to simplify monitoring.
- Secrets (e.g., API keys, authorization headers) are never logged.
