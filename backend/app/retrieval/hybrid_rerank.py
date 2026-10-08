"""Tier 3: hybrid candidates, reordered by the cross-encoder."""

from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.schemas import Hit
from app.tracing import step


class HybridRerankRetriever(Retriever):
    name = "hybrid_rerank"

    def __init__(self, hybrid: HybridRetriever, reranker: Reranker, fetch_k: int):
        self.hybrid = hybrid
        self.reranker = reranker
        self.fetch_k = fetch_k

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.hybrid.search(query, self.fetch_k)
        with step("rerank"):
            scores = self.reranker.score(query, [hit.chunk.search_text() for hit in candidates])
        for hit, score in zip(candidates, scores, strict=True):
            hit.rerank_score = score
            hit.score = score
        candidates.sort(key=lambda hit: hit.score, reverse=True)
        return candidates[:k]
