from __future__ import annotations

from pathlib import Path

from src.embeddings.base import SearchResult
from src.embeddings.embedder import DeterministicTextEmbedder
from src.embeddings.vectorstore import ChromaVectorStore, FaissVectorStore
from src.models.chunk import Chunk
from src.models.document import Document


def _make_chunk(text: str, chunk_id: str, *, topic: str, index: int) -> Chunk:
    source_doc = Document(
        text=text,
        filename=f"{topic}.txt",
        source_type="txt",
        original_metadata={"topic": topic},
    )
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source_doc=source_doc,
        chunk_index=index,
        strategy_used="paragraph",
        metadata={"topic": topic},
    )


def _index_store(store: FaissVectorStore | ChromaVectorStore) -> None:
    embedder = DeterministicTextEmbedder(dimension=64)
    chunks = [
        _make_chunk("Quarterly sales summary with revenue growth", "chunk-1", topic="sales", index=0),
        _make_chunk("Chocolate cake recipe and ingredients", "chunk-2", topic="cooking", index=1),
        _make_chunk("Sales forecast for next quarter and pipeline", "chunk-3", topic="sales", index=2),
    ]
    store.add_many(chunks, embedder.embed_chunks(chunks))


def test_vector_store_returns_most_similar_chunks() -> None:
    store = FaissVectorStore()
    embedder = DeterministicTextEmbedder(dimension=64)
    _index_store(store)

    results = store.search(embedder.embed_text("sales forecast"), top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["chunk-3", "chunk-1"]
    assert all(isinstance(result, SearchResult) for result in results)
    assert results[0].score >= results[1].score


def test_vector_store_applies_metadata_filters() -> None:
    store = ChromaVectorStore()
    embedder = DeterministicTextEmbedder(dimension=64)
    _index_store(store)

    results = store.search(
        embedder.embed_text("sales"),
        top_k=5,
        filters={"topic": "cooking"},
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-2"


def test_vector_store_persists_and_restores_round_trip(tmp_path: Path) -> None:
    store = FaissVectorStore()
    _index_store(store)

    store.save(str(tmp_path / "index"))

    loaded = FaissVectorStore.load(str(tmp_path / "index"))
    assert loaded.count() == store.count()
    assert (
        loaded.search(DeterministicTextEmbedder(dimension=64).embed_text("sales"), top_k=1)[0].chunk.chunk_id
        == "chunk-3"
    )