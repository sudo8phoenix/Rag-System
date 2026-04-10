"""High-level embedding and retrieval orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.settings import AppConfig, EmbeddingConfig, RetrievalConfig
from src.models.chunk import Chunk

from .base import BaseEmbedder, BaseVectorStore, EmbeddingDependencyError, SearchResult
from .embedder import create_embedder
from .retriever import (
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
    SemanticRetriever,
)
from .vectorstore import (
    ChromaVectorStore,
    FaissVectorStore,
    LocalVectorStore,
    QdrantVectorStore,
)


def create_vector_store(config: EmbeddingConfig | None = None) -> BaseVectorStore:
    """Create the configured vector store wrapper."""

    embedding_config = config or EmbeddingConfig()
    if embedding_config.vector_store == "faiss":
        return FaissVectorStore()
    if embedding_config.vector_store == "chroma":
        return ChromaVectorStore.load(embedding_config.chroma_path)
    if embedding_config.vector_store == "qdrant":
        return QdrantVectorStore.load("./data/qdrant_db")
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
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.embedder = embedder or create_embedder(self.embedding_config)
        self.vector_store = vector_store or create_vector_store(self.embedding_config)
        self.retriever = SemanticRetriever(self.embedder, self.vector_store)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            self.retriever,
            self.bm25_retriever,
            bm25_weight=self.retrieval_config.bm25_weight,
        )
        self._reranker = reranker
        self._reranker_unavailable = False

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
        self.bm25_retriever.add_chunks(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        effective_top_k = top_k or self.retrieval_config.top_k
        candidate_top_k = (
            max(effective_top_k, self.retrieval_config.rerank_candidate_pool)
            if self.retrieval_config.rerank
            else effective_top_k
        )

        if self.retrieval_config.hybrid_search and self.bm25_retriever.count() > 0:
            self.hybrid_retriever.bm25_weight = self.retrieval_config.bm25_weight
            initial_results = self.hybrid_retriever.search(
                query,
                top_k=candidate_top_k,
                filters=filters,
            )
        else:
            initial_results = self.retriever.search(
                query,
                top_k=candidate_top_k,
                filters=filters,
            )

        if not self.retrieval_config.rerank:
            return initial_results[:effective_top_k]

        reranker = self._get_reranker()
        if reranker is None:
            return initial_results[:effective_top_k]

        return reranker.rerank(
            query,
            initial_results,
            top_k=effective_top_k,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """Summarize indexed source documents from the persisted vector store."""

        records = getattr(self.vector_store, "_records", {})
        if not isinstance(records, dict) or not records:
            return []

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records.values():
            chunk = getattr(record, "chunk", None)
            if chunk is None:
                continue

            metadata = dict(getattr(record, "metadata", {}) or {})
            source_path = str(
                metadata.get("source_path")
                or chunk.source_doc.original_metadata.get("path")
                or chunk.source_doc.filename
            )
            key = (source_path, chunk.source_doc.source_type)
            summary = grouped.setdefault(
                key,
                {
                    "source_path": source_path,
                    "filename": chunk.source_doc.filename,
                    "source_type": chunk.source_doc.source_type,
                    "chunk_count": 0,
                    "latest_ingested_at": None,
                },
            )
            summary["chunk_count"] += 1

            ingested_at = metadata.get("ingested_at")
            if isinstance(ingested_at, str) and ingested_at:
                current_latest = summary["latest_ingested_at"]
                if current_latest is None or ingested_at > current_latest:
                    summary["latest_ingested_at"] = ingested_at

        return sorted(
            grouped.values(),
            key=lambda item: (item["filename"].lower(), item["source_path"].lower()),
        )

    def _get_reranker(self) -> CrossEncoderReranker | None:
        if self._reranker is not None:
            return self._reranker
        if self._reranker_unavailable:
            return None

        try:
            self._reranker = CrossEncoderReranker(
                self.retrieval_config.rerank_model,
                min_score=self.retrieval_config.rerank_min_score,
            )
        except EmbeddingDependencyError:
            self._reranker_unavailable = True
            return None
        return self._reranker

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
