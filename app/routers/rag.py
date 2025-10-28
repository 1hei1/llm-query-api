from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.schemas import RAGAnswerReference, RAGAnswerRequest, RAGAnswerResponse
from app.services.llm_client import LLMClient, get_llm_client
from app.services.ragflow_client import RAGFlowClient, get_ragflow_client
from app.utils.prompt import build_chat_messages

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/answer", response_model=RAGAnswerResponse)
async def rag_answer(
    payload: RAGAnswerRequest,
    ragflow_client: RAGFlowClient = Depends(get_ragflow_client),
    llm_client: LLMClient = Depends(get_llm_client),
) -> RAGAnswerResponse:
    settings = get_settings()

    retrieval_data = await ragflow_client.retrieval(
        dataset_id=payload.dataset_id,
        question=payload.question,
        top_k=1024,
        similarity_threshold=settings.rag_answer_similarity_threshold,
        vector_similarity_weight=settings.rag_answer_vector_similarity_weight,
        keyword=False,
        highlight=False,
    )

    chunks = retrieval_data.get("chunks", [])
    similarity_threshold = settings.rag_answer_similarity_threshold

    filtered = [
        chunk
        for chunk in chunks
        if chunk.get("similarity", 0.0) >= similarity_threshold
    ]
    filtered.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)

    selected = filtered[: payload.top_n]

    context_pairs: list[tuple[int, str]] = []
    references: List[RAGAnswerReference] = []
    remaining_chars = payload.max_context_chars

    for chunk in selected:
        if remaining_chars <= 0:
            break

        content = (chunk.get("content") or "").strip()
        if not content:
            continue

        if len(content) > remaining_chars:
            content = content[:remaining_chars]

        chunk_index = len(context_pairs) + 1
        context_pairs.append((chunk_index, content))
        references.append(
            RAGAnswerReference(
                chunk_index=chunk_index,
                chunk_id=chunk.get("id", ""),
                document_name=(
                    chunk.get("document_name")
                    or chunk.get("document_keyword")
                    or chunk.get("docnm_kwd")
                ),
                similarity=chunk.get("similarity"),
            )
        )
        remaining_chars -= len(content)

    messages = build_chat_messages(payload.question, context_pairs)

    model_name = payload.model or settings.openai_model
    answer = await llm_client.chat(messages=messages, model=model_name)

    return RAGAnswerResponse(answer=answer, references=references)
