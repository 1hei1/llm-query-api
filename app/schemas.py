from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GlossaryCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    chunk_method: Optional[Literal["naive", "manual"]] = Field(default=None)


class DatasetResource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_id: str = Field(alias="id")
    name: str
    description: Optional[str] = None
    chunk_method: Optional[str] = None
    document_count: Optional[int] = Field(default=None)
    chunk_count: Optional[int] = Field(default=None)

    @classmethod
    def from_ragflow(cls, payload: dict[str, Any]) -> "DatasetResource":
        return cls.model_validate(payload)


class GlossaryCreateResponse(DatasetResource):
    ...


class GlossaryListResponse(BaseModel):
    items: List[DatasetResource]
    total: int


class TermEntry(BaseModel):
    term: str = Field(..., min_length=1)
    definition: str = Field(..., min_length=1)
    synonyms: Optional[List[str]] = None

    def to_block(self) -> str:
        synonyms = ", ".join(self.synonyms) if self.synonyms else ""
        lines = [
            f"Term: {self.term.strip()}",
            f"Definition: {self.definition.strip()}",
            f"Synonyms: {synonyms}",
        ]
        return "\n".join(lines) + "\n---\n"


class GlossaryTermsRequest(BaseModel):
    terms: List[TermEntry] = Field(..., min_length=1)
    upsert: bool = False


class GlossaryTermsResponse(BaseModel):
    document_id: str
    document_name: str
    term_count: int


class FileIngestionResult(BaseModel):
    document_id: str = Field(alias="id")
    document_name: str = Field(alias="name")
    status: str = Field(alias="run")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class FileIngestionResponse(BaseModel):
    documents: List[FileIngestionResult]


class RetrievalRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=1024)
    similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    vector_similarity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    keyword: bool = False
    highlight: bool = True


class RetrievalChunk(BaseModel):
    id: str
    content: str
    similarity: float
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    highlight: Optional[str] = None
    vector_similarity: Optional[float] = None
    term_similarity: Optional[float] = None


class RetrievalDocAgg(BaseModel):
    doc_id: str
    doc_name: str
    count: int


class RetrievalResponse(BaseModel):
    chunks: List[RetrievalChunk]
    doc_aggs: List[RetrievalDocAgg]
    total: int


class RAGAnswerRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_n: int = Field(default=6, ge=1, le=1024)
    max_context_chars: int = Field(default=6000, ge=500)
    model: Optional[str] = None


class RAGAnswerReference(BaseModel):
    chunk_index: int = Field(..., ge=1)
    chunk_id: str
    document_name: Optional[str] = None
    similarity: Optional[float] = None


class RAGAnswerResponse(BaseModel):
    answer: str
    references: List[RAGAnswerReference]
