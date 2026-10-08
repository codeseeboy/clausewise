from app.config import Settings
from app.index.bm25_index import BM25Index
from app.retrieval.factory import build_retrievers
from tests.conftest import FakeReranker


def test_three_strategies_are_built(retrievers):
    assert set(retrievers) == {"vector", "hybrid", "hybrid_rerank"}


def test_vector_hits_carry_dense_rank(retrievers):
    hits = retrievers["vector"].search("hot work in the tank farm", 5)
    assert len(hits) == 5
    assert [h.dense_rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.bm25_rank is None for h in hits)


def test_hybrid_keeps_the_exact_clause_id_match(retrievers):
    # BM25 ranks the clause first; fusion must keep it in the results even if dense search does not.
    hits = {h.chunk.chunk_id: h for h in retrievers["hybrid"].search("What does clause TEP-4.2.2 say?", 5)}
    assert "TEP-4.2.2" in hits
    assert hits["TEP-4.2.2"].bm25_rank == 1


def test_hybrid_scores_are_sorted(retrievers):
    hits = retrievers["hybrid"].search("Who approves an expense of INR 20,000?", 5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_rerank_orders_by_reranker_score(retrievers):
    hits = retrievers["hybrid_rerank"].search("lost badge reported to security", 5)
    assert all(h.rerank_score is not None for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].chunk.chunk_id == "ACP-5.2"


def test_empty_index_returns_no_hits(fake_embedder, memory_store):
    settings = Settings(qdrant_url="memory")
    empty = build_retrievers(fake_embedder, memory_store, BM25Index([]), FakeReranker(), settings)
    for retriever in empty.values():
        assert retriever.search("anything", 5) == []
