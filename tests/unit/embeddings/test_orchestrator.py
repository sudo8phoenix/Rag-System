from __future__ import annotations

from src.config.settings import AppConfig, EmbeddingConfig, RetrievalConfig
from src.embeddings.embedder import DeterministicTextEmbedder
from src.embeddings.orchestrator import EmbeddingOrchestrator
from src.embeddings.vectorstore import LocalVectorStore
from src.models.chunk import Chunk
from src.models.document import Document


def _make_chunk(text: str, chunk_id: str, chunk_index: int, topic: str) -> Chunk:
    document = Document(
        text=text,
        filename=f"{topic}.txt",
        source_type="txt",
        original_metadata={"topic": topic},
    )
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source_doc=document,
        chunk_index=chunk_index,
        strategy_used="paragraph",
        metadata={"topic": topic},
    )


def test_orchestrator_indexes_and_searches_chunks() -> None:
    orchestrator = EmbeddingOrchestrator(
        embedder=DeterministicTextEmbedder(dimension=64),
        vector_store=LocalVectorStore(),
        embedding_config=EmbeddingConfig(vector_store="faiss"),
        retrieval_config=RetrievalConfig(top_k=2),
    )

    chunks = [
        _make_chunk("Revenue and sales grew this quarter", "chunk-1", 0, "sales"),
        _make_chunk("Meeting agenda and notes", "chunk-2", 1, "general"),
    ]

    orchestrator.index_chunks(chunks)
    results = orchestrator.search("sales report")

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"


def test_orchestrator_can_be_constructed_from_app_config() -> None:
    config = AppConfig(
        embedding=EmbeddingConfig(vector_store="chroma"),
        retrieval=RetrievalConfig(top_k=3),
    )

    orchestrator = EmbeddingOrchestrator.from_config(config)

    assert orchestrator.retrieval_config.top_k == 3
    assert orchestrator.embedding_config.vector_store == "chroma"