"""HTTP API. Indexes the corpus at startup, then serves search, ask and file upload."""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import psutil
from fastapi import FastAPI, HTTPException, UploadFile

from app.config import get_settings
from app.index.bm25_index import BM25Index
from app.index.embedder import Embedder
from app.index.vector_store import VectorStore
from app.ingest.loaders import LoaderError
from app.ingest.pipeline import index_bytes, index_folder
from app.rag.answer import answer_question
from app.rag.llm import LLMError, build_llm
from app.retrieval.factory import build_retrievers
from app.retrieval.reranker import Reranker
from app.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from app.tracing import finish_trace, start_trace

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("backend")
state: dict = {}


def rebuild_retrievers() -> None:
    """Rebuild BM25 from everything in Qdrant and wire up the three retrievers."""
    started = time.perf_counter()
    bm25 = BM25Index(state["store"].all_chunks())
    state["bm25_build_seconds"] = time.perf_counter() - started
    state["retrievers"] = build_retrievers(
        state["embedder"], state["store"], bm25, state["reranker"], state["settings"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    embedder = Embedder(settings.embed_model)
    store = VectorStore(settings.qdrant_url, settings.embed_model, embedder.dim)
    state.update(
        settings=settings,
        embedder=embedder,
        store=store,
        reranker=Reranker(settings.rerank_model),
        llm=build_llm(settings),
    )
    report = index_folder(settings.corpus_dir, embedder, store)
    state["index_report"] = report
    logger.info("indexed %d files, %d chunks, failed: %s", report.files, report.chunks, report.failed)
    rebuild_retrievers()
    yield


app = FastAPI(title="Clausewise", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    settings = state["settings"]
    return {
        "status": "ok",
        "collection": state["store"].name,
        "chunks": state["store"].count(),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "llm": f"{settings.llm_provider}:{settings.llm_model}",
    }


@app.get("/index/report")
def index_report() -> dict:
    """Startup indexing timings and model sizes, for the index-cost and resource numbers."""
    return {
        **asdict(state["index_report"]),
        "bm25_build_seconds": state["bm25_build_seconds"],
        "embedder_parameters": state["embedder"].num_parameters,
        "reranker_parameters": state["reranker"].num_parameters,
        "backend_memory_mb": round(psutil.Process().memory_info().rss / 2**20),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    settings = state["settings"]
    start_trace("search", strategy=request.strategy)
    try:
        hits = state["retrievers"][request.strategy].search(request.query, request.k)
    except Exception as exc:  # vector store or model failure
        finish_trace(settings.trace_file, error=exc)
        raise HTTPException(503, f"search failed: {type(exc).__name__}") from exc
    trace = finish_trace(settings.trace_file)
    return SearchResponse(query=request.query, strategy=request.strategy, hits=hits, timings_ms=trace["steps"])


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    settings = state["settings"]
    start_trace("ask", strategy=request.strategy)
    try:
        checked, hits = answer_question(
            request.query, state["retrievers"][request.strategy], state["llm"], settings.context_chunks
        )
    except LLMError as exc:
        finish_trace(settings.trace_file, error=exc)
        raise HTTPException(503, f"language model unavailable: {exc}") from exc
    except Exception as exc:  # vector store or model failure
        finish_trace(settings.trace_file, error=exc)
        raise HTTPException(503, f"search failed: {type(exc).__name__}") from exc
    trace = finish_trace(settings.trace_file)
    return AskResponse(
        query=request.query,
        strategy=request.strategy,
        status=checked.status,
        answer=checked.answer,
        citations=checked.citations,
        hits=hits,
        timings_ms=trace["steps"],
        tokens=trace["tokens"],
        guardrail_notes=checked.problems,
    )


@app.post("/ingest")
def ingest(file: UploadFile) -> dict:
    """Index an uploaded file (md, txt or pdf) and make it searchable right away."""
    settings = state["settings"]
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 10 MB")
    start_trace("ingest")
    try:
        chunks = index_bytes(file.filename or "upload", data, state["embedder"], state["store"])
    except LoaderError as exc:
        finish_trace(settings.trace_file, error=exc)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # vector store or model failure
        finish_trace(settings.trace_file, error=exc)
        raise HTTPException(503, f"indexing failed: {type(exc).__name__}") from exc
    rebuild_retrievers()
    trace = finish_trace(settings.trace_file)
    return {"file": file.filename, "chunks": len(chunks), "chunk_ids": [c.chunk_id for c in chunks], "timings_ms": trace["steps"]}
