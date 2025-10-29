# MCP Read-Only Glossary Server

This package provides a Model Context Protocol (MCP) server that exposes the existing [`llm-query-api`](./README.md) retrieval features to LLM agents in a read-only manner. The server strictly proxies glossary search operations and **does not** expose any dataset creation, ingestion, deletion, or generative Q&A capabilities.

## Features

- ✅ Direct passthrough to the FastAPI glossary retrieval endpoints (no response post-processing)
- ✅ Token-bucket rate limiting with configurable defaults and per-tool overrides
- ✅ Structured JSON logging with audit records for every tool invocation
- ✅ Input validation for dataset identifiers and query length
- ✅ Tenacity-based retries, HTTP timeouts, and upstream request correlation IDs
- ❌ No dataset writes, uploads, or deletions
- ❌ No RAG/Q&A or generative endpoints

## Available Tools

| Tool | Description |
| ---- | ----------- |
| `search_glossary` | Calls the FastAPI `/glossaries/{dataset_id}/retrieve` endpoint to search for chunks related to a term. The JSON payload from FastAPI is returned verbatim. |
| `retrieve_docs` | Calls the same retrieval endpoint to return chunk and document aggregation data for an arbitrary query. The upstream JSON payload is returned unchanged. |

Both tools require a dataset identifier and will surface an error if the upstream API returns a non-2xx status code.

## Configuration

The server reads configuration from environment variables (values in parentheses indicate defaults):

| Variable | Description |
| -------- | ----------- |
| `MCP_API_BASE_URL` (`http://127.0.0.1:8000`) | Base URL of the upstream FastAPI service. |
| `MCP_API_KEY` | Optional API key for authenticating with the upstream service. |
| `MCP_HTTP_TIMEOUT` (`30.0`) | Timeout for outbound HTTP requests (seconds). |
| `MCP_RETRY_ATTEMPTS` (`3`) | Retry attempts for recoverable upstream failures. |
| `MCP_RETRY_WAIT` (`0.5`) | Delay between retries (seconds). |
| `MCP_RATE_LIMIT_CAPACITY` (`10`) | Default number of requests allowed per tool per interval. |
| `MCP_RATE_LIMIT_INTERVAL_SECONDS` (`60.0`) | Interval length (seconds) for the token bucket. |
| `MCP_TOOL_RATE_LIMITS` | Optional JSON object with per-tool overrides, e.g. `{"search_glossary": 5}`. |
| `MCP_MAX_QUERY_LENGTH` (`256`) | Maximum character length for free-text queries. |
| `MCP_DATASET_ID_PATTERN` (`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`) | Regex used to validate dataset IDs. |
| `MCP_SEARCH_TOP_K` (`8`) | Default `top_k` value when none is supplied. |
| `MCP_SIMILARITY_THRESHOLD` (`0.2`) | Similarity threshold applied to retrieval requests. |
| `MCP_VECTOR_SIMILARITY_WEIGHT` (`0.3`) | Vector similarity weight applied to retrieval requests. |
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
  "tool": "search_glossary",
  "status": "success",
  "request_id": "...",
  "duration_ms": 12.4,
  "arguments": {
    "dataset_id": "dataset-1",
    "query_length": 24,
    "top_k": 5,
    "keyword": false,
    "highlight": true
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
        "MCP_API_BASE_URL": "https://llm-query-api.internal"
      }
    }
  }
}
```

Set `MCP_API_KEY` in the environment if your FastAPI deployment requires authentication.

## Docker Support

A dedicated Dockerfile is provided for the MCP server. Build and run it alongside the API using the supplied Compose profile:

```bash
# Build the MCP image
docker build -f Dockerfile.mcp -t llm-query-api-mcp .

# Launch via docker compose (requires docker-compose 1.29+)
docker compose --profile mcp up
```

The Compose profile can use `MCP_API_KEY` if authentication is enabled. The MCP container forwards requests to the API service via the internal Docker network.

## Testing

Tests rely on `pytest`, `pytest-asyncio`, and `respx` to mock upstream HTTP interactions. Run the suite with:

```bash
pytest
```

## Security Notes

- No writes or generative endpoints are exposed through MCP.
- Input validation, rate limiting, and structured audit logs are in place to discourage abuse and to simplify monitoring.
- Secrets (e.g., API keys, authorization headers) are never logged.
