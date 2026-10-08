"""Shared fixtures: small fake models so unit tests run fast and offline."""

import hashlib
import math
from pathlib import Path

import pytest

from app.config import Settings
from app.index.bm25_index import BM25Index
from app.index.tokenizer import tokenize
from app.index.vector_store import VectorStore
from app.ingest.pipeline import index_folder
from app.retrieval.factory import build_retrievers

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Bag-of-words vectors from hashed tokens. Similar wording gives similar vectors."""

    model_name = "fake-embedder"
    num_parameters = 0
    dim = 64

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    """Scores a text by how many query tokens it contains."""

    num_parameters = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(t)))) for t in texts]


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with the given text (no parentheses in the text)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store(fake_embedder) -> VectorStore:
    return VectorStore("memory", fake_embedder.model_name, fake_embedder.dim)


@pytest.fixture
def indexed_store(fake_embedder, memory_store) -> VectorStore:
    index_folder(CORPUS, fake_embedder, memory_store)
    return memory_store


@pytest.fixture
def retrievers(fake_embedder, indexed_store):
    bm25 = BM25Index(indexed_store.all_chunks())
    settings = Settings(qdrant_url="memory", fetch_k=20, rrf_k=60)
    return build_retrievers(fake_embedder, indexed_store, bm25, FakeReranker(), settings)
