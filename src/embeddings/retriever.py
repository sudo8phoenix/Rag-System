"""Semantic retrieval built on top of an embedder and vector store."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.models.chunk import Chunk

from .base import BaseEmbedder, BaseVectorStore, SearchResult


class SemanticRetriever:
    """Dense retrieval helper for chunk indexing and semantic search."""

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        if not chunks:
            return

        vectors = self.embedder.embed_chunks(chunks)
        self.vector_store.add_many(chunks, vectors, metadata)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedder.embed_text(query)
        return self.vector_store.search(query_vector, top_k=top_k, filters=filters)