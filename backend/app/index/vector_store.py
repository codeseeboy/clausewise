"""Qdrant collection holding one vector per chunk; the chunk itself is the payload."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from app.schemas import Chunk

# Payload fields Qdrant indexes for filtering. The full payload is Chunk plus "embedding_model".
PAYLOAD_INDEXES = {
    "chunk_id": PayloadSchemaType.KEYWORD,
    "doc_id": PayloadSchemaType.KEYWORD,
    "version": PayloadSchemaType.KEYWORD,
    "effective_date": PayloadSchemaType.KEYWORD,
    "embedding_model": PayloadSchemaType.KEYWORD,
}


def collection_name(model_name: str) -> str:
    """One collection per embedding model, so vectors from two models never mix."""
    return "chunks__" + model_name.split("/")[-1].lower()


def point_id(chunk_id: str) -> str:
    """Same chunk id gives the same point id, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, url: str, model_name: str, dim: int):
        in_memory = url == "memory"
        self.client = QdrantClient(":memory:") if in_memory else QdrantClient(url=url, timeout=10)
        self.model_name = model_name
        self.name = collection_name(model_name)
        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            if not in_memory:  # the in-process mode has no payload indexes
                for field_name, schema in PAYLOAD_INDEXES.items():
                    self.client.create_payload_index(self.name, field_name, field_schema=schema)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector,
                payload={**chunk.model_dump(), "embedding_model": self.model_name},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.name, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        """Nearest chunks by cosine similarity, best first."""
        result = self.client.query_points(self.name, query=vector, limit=limit, with_payload=True)
        return [(Chunk.model_validate(point.payload), point.score) for point in result.points]

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.name, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(Chunk.model_validate(point.payload) for point in points)
            if offset is None:
                return chunks

    def count(self) -> int:
        return self.client.count(self.name).count
