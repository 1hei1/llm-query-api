# FastAPI RAG Glossary Service

This project provides a FastAPI-based service for managing industry glossaries on top of [RAGFlow](https://ragflow.io) and for generating retrieval-augmented answers with an OpenAI-compatible large language model (LLM).

## Features

- Create or list glossary datasets hosted on RAGFlow.
- Ingest glossary terms directly or through document uploads (TXT/PDF/DOCX/CSV – CSV files must include `term`, `definition`, and optional `synonyms`).
- Retrieve relevant glossary chunks for a user query.
- Generate LLM answers that cite retrieved glossary chunks.

## Requirements

- Python 3.11+
- RAGFlow instance and API key
- OpenAI-compatible API endpoint and key (only needed for `/rag/answer`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Load configuration from environment variables or a local `.env` file:

```
RAGFLOW_BASE_URL=http://192.168.148.107
RAGFLOW_API_KEY=your_ragflow_api_key
OPENAI_BASE_URL=https://api.openai.com
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

## Running the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/healthz
```

## Example Workflow

Create a glossary dataset:

```bash
curl -X POST http://localhost:8000/glossaries \
  -H "Content-Type: application/json" \
  -d '{
    "name": "finance-glossary",
    "description": "Finance terminology",
    "chunk_method": "naive"
  }'
```

Add glossary terms:

```bash
curl -X POST http://localhost:8000/glossaries/<dataset_id>/terms \
  -H "Content-Type: application/json" \
  -d '{
    "terms": [
      {"term": "EPS", "definition": "Earnings per share."},
      {"term": "ROI", "definition": "Return on investment."},
      {"term": "NPV", "definition": "Net present value."}
    ]
  }'
```

Retrieve relevant chunks:

```bash
curl -X POST http://localhost:8000/glossaries/<dataset_id>/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How is ROI defined?",
    "top_k": 5
  }'
```

Ask for an LLM answer with glossary context:

```bash
curl -X POST http://localhost:8000/rag/answer \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_id>",
    "question": "What does ROI mean in finance?",
    "top_n": 3
  }'
```

## Running Tests

```bash
pytest
```

The tests rely on `respx` to mock HTTP interactions with RAGFlow and the LLM provider.
