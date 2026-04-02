from __future__ import annotations

from src.config.settings import EmbeddingConfig
from src.embeddings.embedder import DeterministicTextEmbedder, create_embedder


def test_deterministic_embedder_is_stable_and_normalized() -> None:
    embedder = DeterministicTextEmbedder(dimension=64)

    first = embedder.embed_text("Quarterly sales report")
    second = embedder.embed_text("Quarterly sales report")

    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_create_embedder_falls_back_when_sentence_transformers_is_missing() -> None:
    embedder = create_embedder(EmbeddingConfig(model="BAAI/bge-m3"))

    assert isinstance(embedder, DeterministicTextEmbedder)
    assert len(embedder.embed_text("hello world")) == 1024