"""Indexing flow: load a file, cut it into chunks, embed the chunks and store them."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.index.embedder import Embedder
from app.index.vector_store import VectorStore
from app.ingest.chunker import chunk_document
from app.ingest.loaders import LoaderError, load_bytes
from app.schemas import Chunk
from app.tracing import step


@dataclass
class IndexReport:
    files: int = 0
    chunks: int = 0
    failed: dict[str, str] = field(default_factory=dict)  # file name -> reason
    embed_seconds: float = 0.0
    total_seconds: float = 0.0


def index_bytes(filename: str, data: bytes, embedder: Embedder, store: VectorStore) -> list[Chunk]:
    """Index one uploaded file. Raises LoaderError if the file cannot be read."""
    with step("load"):
        doc = load_bytes(filename, data)
    with step("chunk"):
        chunks = chunk_document(doc)
    with step("embed"):
        vectors = embedder.embed_documents([c.search_text() for c in chunks])
    with step("upsert"):
        store.upsert(chunks, vectors)
    return chunks


def index_folder(folder: Path, embedder: Embedder, store: VectorStore) -> IndexReport:
    """Index every file in a folder. A bad file is recorded in the report and skipped."""
    report = IndexReport()
    started = time.perf_counter()
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        try:
            chunks = chunk_document(load_bytes(path.name, path.read_bytes()))
        except LoaderError as exc:
            report.failed[path.name] = str(exc)
            continue
        embed_start = time.perf_counter()
        vectors = embedder.embed_documents([c.search_text() for c in chunks])
        report.embed_seconds += time.perf_counter() - embed_start
        store.upsert(chunks, vectors)
        report.files += 1
        report.chunks += len(chunks)
    report.total_seconds = time.perf_counter() - started
    return report
