"""High-level embedding and retrieval orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.settings import AppConfig, EmbeddingConfig, RetrievalConfig
from src.models.chunk import Chunk

from .base import BaseEmbedder, BaseVectorStore, SearchResult
from .embedder import create_embedder
from .retriever import SemanticRetriever
from .vectorstore import ChromaVectorStore, FaissVectorStore, LocalVectorStore


def create_vector_store(config: EmbeddingConfig | None = None) -> BaseVectorStore:
    """Create the configured vector store wrapper."""

    embedding_config = config or EmbeddingConfig()
    if embedding_config.vector_store == "faiss":
        return FaissVectorStore()
    if embedding_config.vector_store == "chroma":
        return ChromaVectorStore()
    return LocalVectorStore()


class EmbeddingOrchestrator:
    """Owns the embedder, vector store, and semantic retriever."""

    def __init__(
        self,
        *,
        embedder: BaseEmbedder | None = None,
        vector_store: BaseVectorStore | None = None,
        embedding_config: EmbeddingConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.embedder = embedder or create_embedder(self.embedding_config)
        self.vector_store = vector_store or create_vector_store(self.embedding_config)
        self.retriever = SemanticRetriever(self.embedder, self.vector_store)

    @classmethod
    def from_config(cls, config: AppConfig) -> "EmbeddingOrchestrator":
        """Build the orchestrator from the application config."""

        return cls(
            embedding_config=config.embedding,
            retrieval_config=config.retrieval,
        )

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.retriever.index_chunks(chunks, metadata=metadata)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        return self.retriever.search(
            query,
            top_k=top_k or self.retrieval_config.top_k,
            filters=filters,
        )

    def save(self, path: str | Path) -> None:
        self.vector_store.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedding_config: EmbeddingConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> "EmbeddingOrchestrator":
        """Load an orchestrator from a persisted vector store."""

        embedding_config = embedding_config or EmbeddingConfig()
        vector_store = create_vector_store(embedding_config)
        vector_store = vector_store.__class__.load(str(path))
        return cls(
            vector_store=vector_store,
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
        )