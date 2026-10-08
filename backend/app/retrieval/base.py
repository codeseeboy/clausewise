"""Common interface of the three retrieval strategies."""

from abc import ABC, abstractmethod

from app.schemas import Hit


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, k: int) -> list[Hit]:
        """Return the k best hits for the query, best first."""
