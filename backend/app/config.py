"""Settings, read from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    corpus_dir: Path = Path("data/corpus")
    trace_file: Path = Path("results/traces.jsonl")

    # "memory" runs Qdrant inside the process (tests and runs without Docker).
    qdrant_url: str = "http://localhost:6333"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    fetch_k: int = 20  # candidates taken from each list before fusion and reranking
    rrf_k: int = 60

    llm_provider: str = "ollama"  # "ollama" or "fake"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    context_chunks: int = 5  # chunks placed in the prompt


@lru_cache
def get_settings() -> Settings:
    return Settings()
