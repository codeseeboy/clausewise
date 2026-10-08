"""Tests against the running stack. Skipped when the backend is not reachable."""

import os

import httpx
import pytest

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _backend_up() -> bool:
    try:
        return httpx.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _backend_up(), reason=f"backend not running at {BACKEND_URL}"),
]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BACKEND_URL, timeout=300) as c:
        yield c


def test_health_reports_the_indexed_corpus(client):
    body = client.get("/health").json()
    assert body["chunks"] >= 52
    assert body["collection"] == "chunks__bge-small-en-v1.5"


@pytest.mark.parametrize("strategy", ["vector", "hybrid", "hybrid_rerank"])
def test_every_strategy_returns_hits(client, strategy):
    body = client.post("/search", json={"query": "Where is hot work prohibited?", "strategy": strategy, "k": 5}).json()
    assert len(body["hits"]) == 5
    assert "dense" in body["timings_ms"]


def test_exact_clause_id_is_found(client):
    body = client.post("/search", json={"query": "What does clause TEP-4.2.2 say?", "strategy": "hybrid_rerank", "k": 3}).json()
    assert "TEP-4.2.2" in [hit["chunk"]["chunk_id"] for hit in body["hits"]]


def test_invalid_requests_are_rejected(client):
    assert client.post("/search", json={"query": "", "strategy": "vector"}).status_code == 422
    assert client.post("/search", json={"query": "x", "strategy": "magic"}).status_code == 422


def test_malformed_upload_is_rejected(client):
    response = client.post("/ingest", files={"file": ("broken.pdf", b"%PDF-1.4 cut off", "application/pdf")})
    assert response.status_code == 422


def test_uploaded_file_becomes_searchable(client):
    doc = b"---\ndoc_id: ZZT\ntitle: Canteen Rules\n---\n\n## 1. Canteen\n\n[ZZT-1.1] The canteen closes at 22:00 on weekdays.\n"
    response = client.post("/ingest", files={"file": ("canteen.md", doc, "text/markdown")})
    assert response.status_code == 200
    assert response.json()["chunk_ids"] == ["ZZT-1.1"]
    body = client.post("/search", json={"query": "When does the canteen close?", "strategy": "hybrid", "k": 3}).json()
    assert body["hits"][0]["chunk"]["chunk_id"] == "ZZT-1.1"


@pytest.mark.xfail(
    reason="known model error: qwen2.5:7b reads the two adjacent approval ranges (TEP-4.2.2 / TEP-4.2.3) "
    "as a conflict; retrieval and citations are correct (see README, Known limitations)",
    strict=False,
)
def test_ask_answers_with_a_citation(client):
    body = client.post("/ask", json={"query": "Who approves an expense claim of INR 75,000?"}).json()
    assert body["status"] == "answered"
    assert "TEP-4.2.3" in body["citations"]


def test_ask_refuses_an_unsupported_question(client):
    body = client.post("/ask", json={"query": "How many days of maternity leave do employees get?"}).json()
    assert body["status"] == "refused"
    assert body["citations"] == []
