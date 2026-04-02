"""Vector store implementations used by semantic retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.chunk import Chunk
from src.models.document import Document

from .base import (
    BaseVectorStore,
    SearchResult,
    StoredChunkRecord,
    cosine_similarity,
    normalize_vector,
)


def _chunk_from_payload(payload: Mapping[str, Any]) -> Chunk:
    source_doc_payload = dict(payload["source_doc"])
    source_doc = Document(
        text=source_doc_payload["text"],
        filename=source_doc_payload["filename"],
        source_type=source_doc_payload["source_type"],
        original_metadata=dict(source_doc_payload.get("original_metadata", {})),
    )
    return Chunk(
        text=payload["text"],
        chunk_id=payload["chunk_id"],
        source_doc=source_doc,
        chunk_index=int(payload["chunk_index"]),
        strategy_used=payload["strategy_used"],
        metadata=dict(payload.get("metadata", {})),
    )


class LocalVectorStore(BaseVectorStore):
    """Pure-Python persistent vector store used by the backend wrappers."""

    backend_name = "local"

    def __init__(self) -> None:
        self._records: dict[str, StoredChunkRecord] = {}

    def _build_metadata(
        self,
        chunk: Chunk,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "strategy_used": chunk.strategy_used,
            "filename": chunk.source_doc.filename,
            "source_type": chunk.source_doc.source_type,
        }
        metadata.update(dict(chunk.source_doc.original_metadata))
        metadata.update(dict(chunk.metadata))
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        return metadata

    def add(
        self,
        chunk: Chunk,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._records[chunk.chunk_id] = StoredChunkRecord(
            chunk=chunk,
            vector=normalize_vector(vector),
            metadata=self._build_metadata(chunk, metadata),
        )

    def add_many(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        if metadata is None:
            metadata = [None] * len(chunks)
        if len(metadata) != len(chunks):
            raise ValueError("metadata must match the number of chunks")

        for chunk, vector, extra_metadata in zip(chunks, vectors, metadata):
            self.add(chunk, vector, extra_metadata)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        normalized_query = normalize_vector(query_vector)
        candidates: list[tuple[float, StoredChunkRecord]] = []
        for record in self._records.values():
            if filters and any(record.metadata.get(key) != value for key, value in filters.items()):
                continue
            score = cosine_similarity(normalized_query, record.vector)
            candidates.append((score, record))

        candidates.sort(key=lambda item: item[0], reverse=True)
        results: list[SearchResult] = []
        for score, record in candidates[:top_k]:
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=score,
                    distance=1.0 - score,
                    metadata=dict(record.metadata),
                )
            )
        return results

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend_name,
            "records": [
                {
                    "chunk": record.chunk.to_dict(),
                    "vector": record.vector,
                    "metadata": record.metadata,
                }
                for record in self._records.values()
            ],
        }
        (destination / "store.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "LocalVectorStore":
        destination = Path(path)
        payload = json.loads((destination / "store.json").read_text(encoding="utf-8"))
        store = cls()
        for record_payload in payload.get("records", []):
            chunk = _chunk_from_payload(record_payload["chunk"])
            store._records[chunk.chunk_id] = StoredChunkRecord(
                chunk=chunk,
                vector=[float(value) for value in record_payload["vector"]],
                metadata=dict(record_payload.get("metadata", {})),
            )
        return store

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)


class FaissVectorStore(LocalVectorStore):
    """FAISS-compatible vector store wrapper."""

    backend_name = "faiss"


class ChromaVectorStore(LocalVectorStore):
    """Chroma-compatible vector store wrapper."""

    backend_name = "chroma"