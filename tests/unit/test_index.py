from app.index.bm25_index import BM25Index
from app.index.tokenizer import tokenize
from app.index.vector_store import VectorStore, collection_name
from app.schemas import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("-")[0], doc_title="Policy", source="x.md", text=text)


def test_tokenizer_keeps_clause_id_whole():
    tokens = tokenize("See TEP-4.2.1 and section 4.2.11")
    assert "tep-4.2.1" in tokens
    assert "4.2.11" in tokens
    assert "4" not in tokens


def test_tokenizer_joins_thousand_separators():
    assert tokenize("INR 20,000") == ["inr", "20000"]


def test_bm25_exact_clause_id_beats_same_number_in_other_document():
    # Hard negative: both clauses are numbered 3.2, only the prefix differs.
    index = BM25Index(
        [
            _chunk("TEP-3.2", "Business class is allowed only for international flights."),
            _chunk("HWP-3.2", "Hot work is not permitted in Zone A."),
            _chunk("ACP-3.2", "Visitors are not allowed in the control room."),
        ]
    )
    results = index.search("What does HWP-3.2 say?", 3)
    assert results[0][0].chunk_id == "HWP-3.2"


def test_bm25_empty_index_and_unrelated_query():
    assert BM25Index([]).search("anything", 5) == []
    index = BM25Index([_chunk("TEP-5.1", "Alcohol is not reimbursed.")])
    assert index.search("helmet", 5) == []


def test_reindex_overwrites_instead_of_duplicating(fake_embedder, memory_store):
    chunks = [_chunk("TEP-5.1", "Alcohol is not reimbursed."), _chunk("TEP-5.2", "Meals up to INR 1,500.")]
    vectors = fake_embedder.embed_documents([c.search_text() for c in chunks])
    memory_store.upsert(chunks, vectors)
    memory_store.upsert(chunks, vectors)
    assert memory_store.count() == 2


def test_collection_name_carries_the_model():
    assert collection_name("BAAI/bge-small-en-v1.5") == "chunks__bge-small-en-v1.5"
    store = VectorStore("memory", "BAAI/bge-small-en-v1.5", 8)
    assert store.name == "chunks__bge-small-en-v1.5"
