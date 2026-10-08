"""Cross-encoder reranker: reads the query and a chunk together and gives one relevance score."""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=512, device="cpu")
        self.num_parameters = sum(p.numel() for p in self._model.model.parameters())

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, text) for text in texts])]
