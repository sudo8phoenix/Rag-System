"""Embedding model factories and implementations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from src.config.settings import EmbeddingConfig

from .base import BaseEmbedder, EmbeddingDependencyError, normalize_vector


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class DeterministicTextEmbedder(BaseEmbedder):
    """Pure-Python embedder used when sentence-transformers is unavailable."""

    dimension: int = 1024

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            tokens = [text.lower().strip()]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            direction = 1.0 if digest[4] % 3 else -1.0
            weight = 1.0 + min(len(token), 12) / 12.0
            vector[index] += direction * weight

        for left_token, right_token in zip(tokens, tokens[1:]):
            bigram = f"{left_token}::{right_token}"
            bigram_digest = hashlib.sha256(bigram.encode("utf-8")).digest()
            bigram_index = int.from_bytes(bigram_digest[:4], "big") % self.dimension
            vector[bigram_index] += 0.5

        return normalize_vector(vector)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformers backed embedder."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised when dependency exists
            raise EmbeddingDependencyError("sentence-transformers is not installed") from exc

        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(texts),
            batch_size=max(1, min(32, len(texts) or 1)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [normalize_vector(vector.tolist()) for vector in embeddings]


def create_embedder(
    config: EmbeddingConfig | None = None,
    *,
    allow_fallback: bool = True,
) -> BaseEmbedder:
    """Create the configured embedder, falling back to a deterministic model."""

    embedding_config = config or EmbeddingConfig()
    try:
        return SentenceTransformerEmbedder(
            embedding_config.model,
            device=embedding_config.device,
        )
    except EmbeddingDependencyError:
        if not allow_fallback:
            raise
        return DeterministicTextEmbedder()