"""Shared primitives for embedding and vector retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence

from src.models.chunk import Chunk


class EmbeddingError(RuntimeError):
    """Raised when embedding or retrieval operations fail."""


class EmbeddingDependencyError(EmbeddingError):
    """Raised when an optional embedding dependency is unavailable."""


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned from semantic search with similarity metadata."""

    chunk: Chunk
    score: float
    distance: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for transport or logging."""

        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "distance": self.distance,
            "metadata": dict(self.metadata),
        }


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return a unit-length copy of a vector."""

    values = [float(value) for value in vector]
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        return values
    return [value / magnitude for value in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two vectors."""

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left_values, right_values)
    )
    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class BaseEmbedder(ABC):
    """Abstract text embedder."""

    dimension: int

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed a batch of chunks using their text content."""

        return self.embed_texts([chunk.text for chunk in chunks])


@dataclass(frozen=True)
class StoredChunkRecord:
    """Serialized chunk payload kept in a vector store."""

    chunk: Chunk
    vector: list[float]
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """Abstract vector store contract used by retrieval."""

    backend_name: str

    @abstractmethod
    def add(
        self,
        chunk: Chunk,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store a single chunk embedding."""

    @abstractmethod
    def add_many(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        """Store a batch of chunk embeddings."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the nearest chunks for a query vector."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseVectorStore":
        """Load a vector store from disk."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all indexed chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed chunks."""