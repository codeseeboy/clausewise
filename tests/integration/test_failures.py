"""API behaviour when a dependency fails. Runs in-process with the small fake models."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app import main
from app.config import Settings
from app.index.vector_store import VectorStore
from app.ingest.pipeline import index_folder
from app.rag.llm import FakeLLM, OllamaLLM
from tests.conftest import CORPUS, FakeEmbedder, FakeReranker


@pytest.fixture
def api(tmp_path, monkeypatch):
    embedder = FakeEmbedder()
    store = VectorStore("memory", embedder.model_name, embedder.dim)
    report = index_folder(CORPUS, embedder, store)
    monkeypatch.setattr(main, "state", {})
    main.state.update(
        settings=Settings(qdrant_url="memory", trace_file=tmp_path / "traces.jsonl"),
        embedder=embedder,
        store=store,
        reranker=FakeReranker(),
        llm=FakeLLM(),
        index_report=report,
    )
    main.rebuild_retrievers()
    return TestClient(main.app)  # used without "with", so the startup indexing does not run


def test_search_and_ask_work(api):
    assert api.post("/search", json={"query": "lost badge", "strategy": "hybrid"}).status_code == 200
    body = api.post("/ask", json={"query": "lost badge"}).json()
    assert body["status"] == "answered"
    assert body["citations"]


def test_vector_db_down_returns_503_and_is_traced(api):
    main.state["store"].client = QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    response = api.post("/search", json={"query": "lost badge", "strategy": "vector"})
    assert response.status_code == 503
    trace = json.loads(main.state["settings"].trace_file.read_text().splitlines()[-1])
    assert trace["error"].startswith("dense")


def test_llm_down_returns_503(api):
    main.state["llm"] = OllamaLLM("http://127.0.0.1:9", "test-model", timeout_s=1, max_retries=1, backoff_s=0)
    response = api.post("/ask", json={"query": "lost badge"})
    assert response.status_code == 503
    assert "language model unavailable" in response.json()["detail"]


def test_llm_rate_limit_is_retried(api):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        reply = {"status": "refused", "answer": "Not covered.", "citations": []}
        return httpx.Response(200, json={"message": {"content": json.dumps(reply)}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    main.state["llm"] = OllamaLLM("http://ollama", "test-model", backoff_s=0, client=client)
    response = api.post("/ask", json={"query": "maternity leave"})
    assert response.status_code == 200
    assert response.json()["status"] == "refused"
    assert len(calls) == 2


def test_upload_over_the_limit_is_rejected(api):
    big = b"x" * (main.MAX_UPLOAD_BYTES + 1)
    response = api.post("/ingest", files={"file": ("big.txt", big, "text/plain")})
    assert response.status_code == 413
