"""Semantic retrieval built on top of an embedder and vector store."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .base import BaseEmbedder, BaseVectorStore, EmbeddingDependencyError, SearchResult

try:  # pragma: no cover - dependency is optional in constrained environments
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - fallback path covered by tests
    BM25Okapi = None

from src.models.chunk import Chunk

try:  # pragma: no cover - dependency is optional in constrained environments
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - fallback path covered by tests
    CrossEncoder = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _min_max_normalize(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 if maximum > 0.0 else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


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


class BM25Retriever:
    """Sparse retriever based on keyword matching with BM25 scoring."""

    def __init__(self, chunks: Sequence[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25_index: BM25Okapi | None = None
        if chunks:
            self.index_chunks(chunks)

    def count(self) -> int:
        return len(self._chunks)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._rebuild_index()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._rebuild_index()

    def clear(self) -> None:
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._bm25_index = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not self._chunks:
            return []

        candidate_indices = [
            index
            for index, chunk in enumerate(self._chunks)
            if self._matches_filters(chunk, filters)
        ]
        if not candidate_indices:
            return []

        query_tokens = _tokenize(query)
        all_scores = self._compute_scores(query_tokens)
        candidate_scores = [(all_scores[index], index) for index in candidate_indices]
        candidate_scores.sort(key=lambda item: item[0], reverse=True)

        max_score = max((score for score, _ in candidate_scores), default=0.0)
        results: list[SearchResult] = []
        for score, index in candidate_scores[:top_k]:
            chunk = self._chunks[index]
            normalized_score = score / max_score if max_score > 0.0 else 0.0
            metadata = {
                **dict(chunk.source_doc.original_metadata),
                **dict(chunk.metadata),
                "retrieval_backend": "bm25",
                "bm25_score": float(score),
            }
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(normalized_score),
                    distance=1.0 - float(normalized_score),
                    metadata=metadata,
                )
            )
        return results

    def _rebuild_index(self) -> None:
        self._tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]
        if BM25Okapi is None:
            self._bm25_index = None
            return
        self._bm25_index = BM25Okapi(self._tokenized_corpus)

    def _compute_scores(self, query_tokens: Sequence[str]) -> list[float]:
        if not self._chunks:
            return []

        if self._bm25_index is not None and query_tokens:
            return [float(score) for score in self._bm25_index.get_scores(list(query_tokens))]

        return [
            self._fallback_score(query_tokens, document_tokens)
            for document_tokens in self._tokenized_corpus
        ]

    def _fallback_score(
        self,
        query_tokens: Sequence[str],
        document_tokens: Sequence[str],
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0

        document_counts: dict[str, int] = {}
        for token in document_tokens:
            document_counts[token] = document_counts.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            score += float(document_counts.get(token, 0))
        return score

    def _matches_filters(
        self,
        chunk: Chunk,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        combined_metadata = {
            **dict(chunk.source_doc.original_metadata),
            **dict(chunk.metadata),
            "filename": chunk.source_doc.filename,
            "source_type": chunk.source_doc.source_type,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
        }
        return all(combined_metadata.get(key) == value for key, value in filters.items())


class HybridRetriever:
    """Combines dense and BM25 retrieval scores into a hybrid ranking."""

    def __init__(
        self,
        dense_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        *,
        bm25_weight: float = 0.3,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        candidate_top_k = max(top_k * 3, top_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_top_k, filters=filters)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_top_k, filters=filters)

        if not dense_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return dense_results[:top_k]

        dense_norm = _min_max_normalize([result.score for result in dense_results])
        bm25_norm = _min_max_normalize([result.score for result in bm25_results])

        aggregate: dict[str, dict[str, Any]] = {}

        for result, normalized_score in zip(dense_results, dense_norm):
            item = aggregate.setdefault(
                result.chunk.chunk_id,
                {
                    "chunk": result.chunk,
                    "metadata": dict(result.metadata),
                    "dense": 0.0,
                    "bm25": 0.0,
                },
            )
            item["dense"] = float(normalized_score)

        for result, normalized_score in zip(bm25_results, bm25_norm):
            item = aggregate.setdefault(
                result.chunk.chunk_id,
                {
                    "chunk": result.chunk,
                    "metadata": dict(result.metadata),
                    "dense": 0.0,
                    "bm25": 0.0,
                },
            )
            item["metadata"].update(result.metadata)
            item["bm25"] = float(normalized_score)

        dense_weight = 1.0 - self.bm25_weight
        merged: list[SearchResult] = []
        for item in aggregate.values():
            combined_score = (dense_weight * item["dense"]) + (self.bm25_weight * item["bm25"])
            metadata = dict(item["metadata"])
            metadata.update(
                {
                    "retrieval_backend": "hybrid",
                    "dense_score": item["dense"],
                    "bm25_score": item["bm25"],
                    "hybrid_score": combined_score,
                }
            )
            merged.append(
                SearchResult(
                    chunk=item["chunk"],
                    score=combined_score,
                    distance=1.0 - combined_score,
                    metadata=metadata,
                )
            )

        merged.sort(key=lambda result: result.score, reverse=True)
        return merged[:top_k]


class CrossEncoderReranker:
    """Rerank candidate chunks with a cross-encoder relevance model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        min_score: float | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.min_score = min_score

        if model is not None:
            self._model = model
            return

        if CrossEncoder is None:
            raise EmbeddingDependencyError(
                "sentence-transformers is not installed; cross-encoder reranking is unavailable"
            )

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0 or not candidates:
            return []

        pairs = [(query, result.chunk.text) for result in candidates]
        raw_scores = self._model.predict(pairs)
        scores = [float(score) for score in raw_scores]

        reranked_pairs = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked: list[SearchResult] = []
        for result, score in reranked_pairs:
            if self.min_score is not None and score < self.min_score:
                continue

            metadata = dict(result.metadata)
            metadata.update(
                {
                    "reranked": True,
                    "reranker_model": self.model_name,
                    "rerank_score": score,
                }
            )
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    distance=1.0 - score,
                    metadata=metadata,
                )
            )
            if len(reranked) >= top_k:
                break

        return reranked