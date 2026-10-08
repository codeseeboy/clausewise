"""Tier 2: dense search and BM25, merged with Reciprocal Rank Fusion."""

from app.index.bm25_index import BM25Index
from app.retrieval.base import Retriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector import VectorRetriever
from app.schemas import Hit
from app.tracing import step


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, vector: VectorRetriever, bm25: BM25Index, fetch_k: int, rrf_k: int):
        self.vector = vector
        self.bm25 = bm25
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        dense = self.vector.search(query, self.fetch_k)
        with step("bm25"):
            sparse = self.bm25.search(query, self.fetch_k)

        dense_ids = [hit.chunk.chunk_id for hit in dense]
        sparse_ids = [chunk.chunk_id for chunk, _ in sparse]
        with step("rrf"):
            fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)

        chunks = {hit.chunk.chunk_id: hit.chunk for hit in dense}
        chunks.update({chunk.chunk_id: chunk for chunk, _ in sparse})
        dense_rank = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}
        bm25_rank = {cid: rank for rank, cid in enumerate(sparse_ids, start=1)}
        return [
            Hit(chunk=chunks[cid], score=score, dense_rank=dense_rank.get(cid), bm25_rank=bm25_rank.get(cid))
            for cid, score in fused[:k]
        ]
