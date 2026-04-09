"""Vector store implementations used by semantic retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import chromadb
except ImportError:  # pragma: no cover - fallback path in this environment
    chromadb = None

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - fallback path in this environment
    QdrantClient = None
    qdrant_models = None

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


def _record_from_payload(payload: Mapping[str, Any]) -> StoredChunkRecord:
    chunk = _chunk_from_payload(payload["chunk"])
    return StoredChunkRecord(
        chunk=chunk,
        vector=[float(value) for value in payload["vector"]],
        metadata=dict(payload.get("metadata", {})),
    )


def _merge_metadata(
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


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    safe_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_metadata[key] = value
        else:
            safe_metadata[f"{key}_json"] = json.dumps(value, sort_keys=True, default=str)
    return safe_metadata


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
        return _merge_metadata(chunk, extra_metadata)

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
            record = _record_from_payload(record_payload)
            store._records[record.chunk.chunk_id] = record
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

    def __init__(
        self,
        persist_directory: str | Path = "./data/chroma_db",
        collection_name: str = "rag_chunks",
    ) -> None:
        super().__init__()
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._chroma_client: Any | None = None
        self._chroma_collection: Any | None = None
        self._using_chromadb = False
        self._initialize_chromadb()

    def _initialize_chromadb(self) -> None:
        if chromadb is None:
            return

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "backend": self.backend_name},
            )
            self._using_chromadb = True
        except Exception:  # pragma: no cover - optional backend must not break fallback
            self._chroma_client = None
            self._chroma_collection = None
            self._using_chromadb = False

    def _chroma_metadata(
        self,
        chunk: Chunk,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = _merge_metadata(chunk, extra_metadata)
        chroma_metadata = _json_safe_metadata(metadata)
        chroma_metadata["source_doc_text"] = chunk.source_doc.text
        chroma_metadata["source_doc_metadata_json"] = json.dumps(
            chunk.source_doc.original_metadata,
            sort_keys=True,
            default=str,
        )
        chroma_metadata["chunk_metadata_json"] = json.dumps(
            chunk.metadata,
            sort_keys=True,
            default=str,
        )
        return chroma_metadata

    def _rebuild_from_local_cache(self) -> None:
        cache_path = self.persist_directory / "store.json"
        if not cache_path.exists():
            return

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        self._records.clear()
        for record_payload in payload.get("records", []):
            record = _record_from_payload(record_payload)
            self._records[record.chunk.chunk_id] = record

    def _sync_chromadb_collection(self) -> None:
        if self._chroma_collection is None or not self._records:
            return

        self._chroma_collection.upsert(
            ids=list(self._records.keys()),
            documents=[record.chunk.text for record in self._records.values()],
            embeddings=[record.vector for record in self._records.values()],
            metadatas=[
                self._chroma_metadata(record.chunk, record.metadata)
                for record in self._records.values()
            ],
        )

    def add(
        self,
        chunk: Chunk,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().add(chunk, vector, metadata)
        if self._chroma_collection is not None:
            self._chroma_collection.upsert(
                ids=[chunk.chunk_id],
                documents=[chunk.text],
                embeddings=[normalize_vector(vector)],
                metadatas=[self._chroma_metadata(chunk, metadata)],
            )

    def add_many(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        super().add_many(chunks, vectors, metadata)
        if self._chroma_collection is not None:
            metadata = metadata or [None] * len(chunks)
            self._chroma_collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=[normalize_vector(vector) for vector in vectors],
                metadatas=[
                    self._chroma_metadata(chunk, extra_metadata)
                    for chunk, extra_metadata in zip(chunks, metadata)
                ],
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        normalized_query = normalize_vector(query_vector)

        if self._chroma_collection is None:
            return super().search(query_vector, top_k=top_k, filters=filters)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [normalized_query],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = dict(filters)

        query_result = self._chroma_collection.query(**query_kwargs)
        ids = query_result.get("ids", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for index, chunk_id in enumerate(ids):
            record = self._records.get(chunk_id)
            if record is None:
                continue

            distance = float(distances[index]) if index < len(distances) else 1.0
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=1.0 - distance,
                    distance=distance,
                    metadata=dict(metadatas[index] if index < len(metadatas) else record.metadata),
                )
            )

        return results

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        super().save(str(destination))

    @classmethod
    def load(cls, path: str) -> "ChromaVectorStore":
        destination = Path(path)
        store = cls(persist_directory=destination)
        store._rebuild_from_local_cache()
        store._sync_chromadb_collection()

        if not store._records and store._chroma_collection is not None:
            collection_payload = store._chroma_collection.get(include=["documents", "metadatas", "embeddings"])
            ids = collection_payload.get("ids", [])
            documents = collection_payload.get("documents", [])
            metadatas = collection_payload.get("metadatas", [])
            embeddings = collection_payload.get("embeddings", [])

            for index, chunk_id in enumerate(ids):
                metadata = dict(metadatas[index] or {})
                source_doc = Document(
                    text=str(metadata.get("source_doc_text", documents[index] or "")),
                    filename=str(metadata.get("filename", chunk_id)),
                    source_type=str(metadata.get("source_type", "unknown")),
                    original_metadata=json.loads(metadata.get("source_doc_metadata_json", "{}")),
                )
                chunk_metadata = json.loads(metadata.get("chunk_metadata_json", "{}"))
                chunk = Chunk(
                    text=str(documents[index] or ""),
                    chunk_id=str(chunk_id),
                    source_doc=source_doc,
                    chunk_index=int(metadata.get("chunk_index", index)),
                    strategy_used=str(metadata.get("strategy_used", "chroma")),
                    metadata=chunk_metadata,
                )
                store._records[chunk.chunk_id] = StoredChunkRecord(
                    chunk=chunk,
                    vector=[float(value) for value in (embeddings[index] or [])],
                    metadata=_merge_metadata(chunk, chunk_metadata),
                )

        return store

    def clear(self) -> None:
        super().clear()
        if self._chroma_collection is not None:
            self._chroma_collection.delete(where={})


class QdrantVectorStore(LocalVectorStore):
    """Qdrant-compatible vector store wrapper."""

    backend_name = "qdrant"

    def __init__(
        self,
        *,
        persist_directory: str | Path = "./data/qdrant_db",
        collection_name: str = "rag_chunks",
        url: str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self._qdrant_client: Any | None = client
        self._using_qdrant = client is not None
        self._initialize_qdrant()

    def _initialize_qdrant(self) -> None:
        if self._qdrant_client is not None:
            return

        if QdrantClient is None:
            return

        try:
            if self.url:
                self._qdrant_client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc,
                )
            else:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self._qdrant_client = QdrantClient(path=str(self.persist_directory))
            self._using_qdrant = True
        except Exception:  # pragma: no cover - optional backend must not break fallback
            self._qdrant_client = None
            self._using_qdrant = False

    def _qdrant_metadata(
        self,
        chunk: Chunk,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = _merge_metadata(chunk, extra_metadata)
        payload = _json_safe_metadata(metadata)
        payload["chunk_json"] = json.dumps(chunk.to_dict(), sort_keys=True, default=str)
        payload["metadata_json"] = json.dumps(metadata, sort_keys=True, default=str)
        return payload

    def _ensure_collection(self, vector_size: int) -> None:
        if self._qdrant_client is None:
            return

        collection_exists = getattr(self._qdrant_client, "collection_exists", None)
        if callable(collection_exists) and collection_exists(self.collection_name):
            return

        if qdrant_models is not None:
            vectors_config = qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            )
        else:  # pragma: no cover - executed only when qdrant dependency is absent
            vectors_config = {"size": vector_size, "distance": "Cosine"}

        create_collection = getattr(self._qdrant_client, "create_collection", None)
        if callable(create_collection):
            create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )

    def _upsert_points(self, points: Sequence[dict[str, Any]]) -> None:
        if self._qdrant_client is None or not points:
            return

        upsert = getattr(self._qdrant_client, "upsert", None)
        if not callable(upsert):
            return

        upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _build_qdrant_filter(
        self,
        filters: Mapping[str, Any] | None,
    ) -> Any:
        if not filters:
            return None

        if qdrant_models is None:
            return dict(filters)

        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and {"gte", "lte"} & set(value):
                range_kwargs: dict[str, Any] = {}
                if "gte" in value:
                    range_kwargs["gte"] = value["gte"]
                if "lte" in value:
                    range_kwargs["lte"] = value["lte"]
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        range=qdrant_models.Range(**range_kwargs),
                    )
                )
            else:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )

        return qdrant_models.Filter(must=conditions)

    def _search_collection(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        if self._qdrant_client is None:
            return []

        query_filter = self._build_qdrant_filter(filters)

        search = getattr(self._qdrant_client, "search", None)
        if callable(search):
            return list(
                search(
                    collection_name=self.collection_name,
                    query_vector=list(query_vector),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
            )

        query_points = getattr(self._qdrant_client, "query_points", None)
        if callable(query_points):
            response = query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return list(getattr(response, "points", response) or [])

        return []

    def _record_from_point(self, point: Any) -> StoredChunkRecord | None:
        payload = dict(getattr(point, "payload", {}) or {})
        chunk_payload = payload.get("chunk_json")
        if chunk_payload:
            try:
                chunk = _chunk_from_payload(json.loads(chunk_payload))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                chunk = None
        else:
            chunk = None

        if chunk is None:
            chunk_id = str(getattr(point, "id", payload.get("chunk_id", "")))
            if not chunk_id:
                return None
            source_doc = Document(
                text=str(payload.get("source_doc_text", payload.get("text", chunk_id))),
                filename=str(payload.get("filename", chunk_id)),
                source_type=str(payload.get("source_type", "unknown")),
                original_metadata={},
            )
            chunk = Chunk(
                text=str(payload.get("text", chunk_id)),
                chunk_id=chunk_id,
                source_doc=source_doc,
                chunk_index=int(payload.get("chunk_index", 0)),
                strategy_used=str(payload.get("strategy_used", self.backend_name)),
                metadata={},
            )

        vector = list(getattr(point, "vector", []) or [])
        score = float(getattr(point, "score", 0.0))
        metadata = dict(payload)
        metadata.setdefault("score", score)
        return StoredChunkRecord(chunk=chunk, vector=[float(value) for value in vector], metadata=metadata)

    def add(
        self,
        chunk: Chunk,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_vector = normalize_vector(vector)
        super().add(chunk, normalized_vector, metadata)
        if self._qdrant_client is None:
            return

        self._ensure_collection(len(normalized_vector))
        self._upsert_points(
            [
                {
                    "id": chunk.chunk_id,
                    "vector": normalized_vector,
                    "payload": self._qdrant_metadata(chunk, metadata),
                }
            ]
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

        normalized_vectors = [normalize_vector(vector) for vector in vectors]
        super().add_many(chunks, normalized_vectors, metadata)

        if self._qdrant_client is None:
            return

        if normalized_vectors:
            self._ensure_collection(len(normalized_vectors[0]))
        self._upsert_points(
            [
                {
                    "id": chunk.chunk_id,
                    "vector": vector,
                    "payload": self._qdrant_metadata(chunk, extra_metadata),
                }
                for chunk, vector, extra_metadata in zip(chunks, normalized_vectors, metadata)
            ]
        )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        normalized_query = normalize_vector(query_vector)

        if self._qdrant_client is None:
            return super().search(query_vector, top_k=top_k, filters=filters)

        qdrant_points = self._search_collection(
            normalized_query,
            top_k=top_k,
            filters=filters,
        )
        results: list[SearchResult] = []
        for point in qdrant_points:
            record = self._records.get(str(getattr(point, "id", "")))
            if record is None:
                record = self._record_from_point(point)
                if record is not None:
                    self._records[record.chunk.chunk_id] = record

            if record is None:
                continue

            score = float(getattr(point, "score", 1.0 - float(getattr(point, "distance", 1.0))))
            distance = float(getattr(point, "distance", 1.0 - score))
            payload = dict(getattr(point, "payload", {}) or record.metadata)
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=score,
                    distance=distance,
                    metadata=payload,
                )
            )

        return results

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        super().save(str(destination))

    @classmethod
    def load(cls, path: str) -> "QdrantVectorStore":
        destination = Path(path)
        store = cls(persist_directory=destination)
        cache_path = destination / "store.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            for record_payload in payload.get("records", []):
                record = _record_from_payload(record_payload)
                store._records[record.chunk.chunk_id] = record

        if store._qdrant_client is not None and store._records:
            first_vector = next(iter(store._records.values())).vector
            store._ensure_collection(len(first_vector))
            store._upsert_points(
                [
                    {
                        "id": record.chunk.chunk_id,
                        "vector": record.vector,
                        "payload": store._qdrant_metadata(record.chunk, record.metadata),
                    }
                    for record in store._records.values()
                ]
            )

        return store

    def clear(self) -> None:
        super().clear()
        if self._qdrant_client is None:
            return

        delete_collection = getattr(self._qdrant_client, "delete_collection", None)
        if callable(delete_collection):
            try:
                delete_collection(collection_name=self.collection_name)
            except TypeError:
                delete_collection(self.collection_name)

    def count(self) -> int:
        return len(self._records)