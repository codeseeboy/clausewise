"""Turn text into vectors with a sentence-transformers model."""

from sentence_transformers import SentenceTransformer

# bge v1.5 models expect this prefix on short search queries (not on the documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = self._model.get_sentence_embedding_dimension()
        self.num_parameters = sum(p.numel() for p in self._model.parameters())
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self._query_prefix + text, normalize_embeddings=True).tolist()
