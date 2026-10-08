"""Keyword search over the chunks with BM25."""

from rank_bm25 import BM25Okapi

from app.index.tokenizer import tokenize
from app.schemas import Chunk


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.search_text()) for c in chunks]) if chunks else None

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        """Best matching chunks, best first. Chunks sharing no word with the query are left out."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in order[:limit] if scores[i] > 0]
