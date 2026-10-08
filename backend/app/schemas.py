"""Data types shared by the index, the retrievers and the API."""

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["vector", "hybrid", "hybrid_rerank"]
Status = Literal["answered", "refused", "conflict", "error"]


class Chunk(BaseModel):
    """One searchable piece of a document. Stored as the Qdrant payload."""

    chunk_id: str  # clause id such as "TEP-4.2.2", or "<doc>-c001" for fixed-size pieces
    doc_id: str
    doc_title: str
    source: str  # file name
    section: str = ""
    text: str
    version: str = ""
    effective_date: str = ""

    def search_text(self) -> str:
        """Text used for embedding and BM25: title, section, id and clause text."""
        return f"{self.doc_title} | {self.section}\n{self.chunk_id} {self.text}"


class Hit(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    strategy: Strategy = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    strategy: Strategy
    hits: list[Hit]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    strategy: Strategy = "hybrid_rerank"


class AskResponse(BaseModel):
    query: str
    strategy: Strategy
    status: Status
    answer: str
    citations: list[str]
    hits: list[Hit]
    timings_ms: dict[str, float]
    tokens: dict[str, int]
    guardrail_notes: list[str] = []  # what the checks changed in the model reply
