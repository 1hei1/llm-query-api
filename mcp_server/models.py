from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GlossarySummary(BaseModel):
    dataset_id: str = Field(alias="dataset_id")
    name: str
    description: Optional[str] = None
    chunk_method: Optional[str] = None
    document_count: Optional[int] = Field(default=None, alias="document_count")
    chunk_count: Optional[int] = Field(default=None, alias="chunk_count")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GlossarySummary":
        dataset_id = payload.get("dataset_id") or payload.get("id")
        data = dict(payload)
        if dataset_id is not None:
            data["dataset_id"] = dataset_id
        return cls.model_validate(data)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class GlossaryListResult(BaseModel):
    items: list[GlossarySummary]
    total: int


class RetrievalChunkResult(BaseModel):
    chunk_id: str = Field(alias="id")
    content: str
    similarity: float | None = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    highlight: Optional[str] = None
    vector_similarity: Optional[float] = None
    term_similarity: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SearchTermsResult(BaseModel):
    dataset_id: str
    query: str
    total: int
    results: list[RetrievalChunkResult]


class RetrieveDefinitionsResult(BaseModel):
    dataset_id: str
    terms: list[str]
    total: int
    results: list[RetrievalChunkResult]
