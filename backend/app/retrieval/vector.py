"""Tier 1: semantic search with embeddings only."""

from app.index.embedder import Embedder
from app.index.vector_store import VectorStore
from app.retrieval.base import Retriever
from app.schemas import Hit
from app.tracing import step


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int) -> list[Hit]:
        with step("embed"):
            vector = self.embedder.embed_query(query)
        with step("dense"):
            results = self.store.search(vector, k)
        return [
            Hit(chunk=chunk, score=score, dense_rank=rank)
            for rank, (chunk, score) in enumerate(results, start=1)
        ]
