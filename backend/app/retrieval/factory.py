"""Build the three retrievers from the shared components."""

from app.config import Settings
from app.index.bm25_index import BM25Index
from app.index.embedder import Embedder
from app.index.vector_store import VectorStore
from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.hybrid_rerank import HybridRerankRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector import VectorRetriever


def build_retrievers(
    embedder: Embedder, store: VectorStore, bm25: BM25Index, reranker: Reranker, settings: Settings
) -> dict[str, Retriever]:
    vector = VectorRetriever(embedder, store)
    hybrid = HybridRetriever(vector, bm25, settings.fetch_k, settings.rrf_k)
    rerank = HybridRerankRetriever(hybrid, reranker, settings.fetch_k)
    return {vector.name: vector, hybrid.name: hybrid, rerank.name: rerank}
