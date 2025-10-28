from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas import (
    DatasetResource,
    FileIngestionResponse,
    FileIngestionResult,
    GlossaryCreateRequest,
    GlossaryCreateResponse,
    GlossaryListResponse,
    GlossaryTermsRequest,
    GlossaryTermsResponse,
    RetrievalChunk,
    RetrievalDocAgg,
    RetrievalRequest,
    RetrievalResponse,
    TermEntry,
)
from app.services.ragflow_client import RAGFlowClient, get_ragflow_client

router = APIRouter(prefix="/glossaries", tags=["glossaries"])

_ALLOWED_SUFFIXES = {".txt", ".pdf", ".docx", ".csv"}


def _build_term_blocks(terms: Sequence[TermEntry]) -> str:
    return "".join(term.to_block() for term in terms)


def _decode_csv(file_bytes: bytes) -> List[TermEntry]:
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    required_columns = {"term", "definition"}
    lower_headers: dict[str, str] = {}
    for header in reader.fieldnames or []:
        if header is None:
            continue
        lower_headers[header.lower()] = header
    if not required_columns.issubset(lower_headers):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV must include 'term' and 'definition' columns.",
        )

    entries: List[TermEntry] = []
    for row in reader:
        term_value = row.get(lower_headers["term"], "").strip()
        definition_value = row.get(lower_headers["definition"], "").strip()
        if not term_value or not definition_value:
            continue
        synonyms_value = ""
        if "synonyms" in lower_headers:
            synonyms_value = row.get(lower_headers["synonyms"], "") or ""
        normalized_synonyms = synonyms_value.replace(";", ",")
        synonyms = [item.strip() for item in normalized_synonyms.split(",") if item.strip()]
        entries.append(TermEntry(term=term_value, definition=definition_value, synonyms=synonyms))

    if not entries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file does not contain any valid entries.",
        )

    return entries


@router.post("", response_model=GlossaryCreateResponse)
async def create_glossary(
    payload: GlossaryCreateRequest,
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
) -> GlossaryCreateResponse:
    dataset = await ragflow_client.ensure_dataset(
        name=payload.name,
        description=payload.description,
        chunk_method=payload.chunk_method,
    )
    return GlossaryCreateResponse.model_validate(dataset)


@router.get("", response_model=GlossaryListResponse)
async def list_glossaries(
    name: str | None = None,
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
) -> GlossaryListResponse:
    response = await ragflow_client.list_datasets(name=name)
    datasets = [DatasetResource.from_ragflow(item) for item in response.get("data", [])]
    total = response.get("total", len(datasets))
    return GlossaryListResponse(items=datasets, total=total)


@router.post("/{dataset_id}/terms", response_model=GlossaryTermsResponse)
async def ingest_terms(
    dataset_id: str,
    payload: GlossaryTermsRequest,
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
) -> GlossaryTermsResponse:
    if not payload.terms:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No terms provided.")

    glossary_text = _build_term_blocks(payload.terms)
    filename = f"glossary-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt"
    files = [
        (
            "file",
            (filename, glossary_text.encode("utf-8"), "text/plain"),
        )
    ]

    documents = await ragflow_client.upload_documents(dataset_id, files)
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAGFlow did not return document metadata.",
        )

    document = documents[0]
    document_id = document.get("id")
    if not document_id:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAGFlow response missing document ID.",
        )

    await ragflow_client.parse_documents(dataset_id, [document_id])

    return GlossaryTermsResponse(
        document_id=document_id,
        document_name=document.get("name", filename),
        term_count=len(payload.terms),
    )


@router.post("/{dataset_id}/files", response_model=FileIngestionResponse)
async def ingest_files(
    dataset_id: str,
    files: List[UploadFile] = File(...),
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
) -> FileIngestionResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    outbound_files: list[tuple[str, tuple[str, bytes, str]]] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Uploaded file is missing a filename.")

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {suffix}",
            )

        file_bytes = await upload.read()
        if not file_bytes:
            continue

        if suffix == ".csv":
            terms = _decode_csv(file_bytes)
            glossary_text = _build_term_blocks(terms)
            filename = Path(upload.filename).with_suffix(".txt").name
            outbound_files.append(("file", (filename, glossary_text.encode("utf-8"), "text/plain")))
        else:
            content_type = upload.content_type or "application/octet-stream"
            outbound_files.append(("file", (upload.filename, file_bytes, content_type)))

    if not outbound_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No valid files to ingest.")

    documents = await ragflow_client.upload_documents(dataset_id, outbound_files)
    document_ids = [doc.get("id") for doc in documents if doc.get("id")]
    if document_ids:
        await ragflow_client.parse_documents(dataset_id, document_ids)

    results = [FileIngestionResult.model_validate(doc) for doc in documents]
    return FileIngestionResponse(documents=results)


@router.post("/{dataset_id}/retrieve", response_model=RetrievalResponse)
async def retrieve_glossary(
    dataset_id: str,
    payload: RetrievalRequest,
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
) -> RetrievalResponse:
    data = await ragflow_client.retrieval(
        dataset_id=dataset_id,
        question=payload.question,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
        vector_similarity_weight=payload.vector_similarity_weight,
        keyword=payload.keyword,
        highlight=payload.highlight,
    )

    chunks: list[RetrievalChunk] = []
    for item in data.get("chunks", []):
        chunks.append(
            RetrievalChunk(
                id=item.get("id", ""),
                content=item.get("content", ""),
                similarity=item.get("similarity", 0.0),
                document_id=item.get("document_id"),
                document_name=(
                    item.get("document_name")
                    or item.get("document_keyword")
                    or item.get("docnm_kwd")
                ),
                highlight=item.get("highlight"),
                vector_similarity=item.get("vector_similarity"),
                term_similarity=item.get("term_similarity"),
            )
        )

    doc_aggs = [
        RetrievalDocAgg(
            doc_id=item.get("doc_id", ""),
            doc_name=item.get("doc_name", ""),
            count=item.get("count", 0),
        )
        for item in data.get("doc_aggs", [])
    ]

    total = data.get("total", len(chunks))
    return RetrievalResponse(chunks=chunks, doc_aggs=doc_aggs, total=total)
