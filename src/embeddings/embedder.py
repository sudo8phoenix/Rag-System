"""Embedding model factories and implementations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Sequence
from urllib import error, request

from src.config.settings import EmbeddingConfig

from .base import BaseEmbedder, EmbeddingDependencyError, normalize_vector


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

OLLAMA_EMBED_MODELS = {
    "all-minilm",
    "mxbai-embed-large",
    "nomic-embed-text",
    "snowflake-arctic-embed",
}


def _http_post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - trusted endpoint
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


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


class OllamaEmbedder(BaseEmbedder):
    """Ollama-backed embedder using the local HTTP API."""

    DEFAULT_TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_post_json=None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_post_json = http_post_json or _http_post_json
        self.dimension = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._embed_batch(list(texts))
        if vectors is None:
            vectors = [self._embed_single(text) for text in texts]

        self._set_dimension(vectors)
        return [normalize_vector(vector) for vector in vectors]

    def _post(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._http_post_json(url, payload, self.timeout_seconds)
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise EmbeddingDependencyError(
                "Unable to connect to Ollama embeddings API. Ensure `ollama serve` is running."
            ) from exc

        if not isinstance(response, dict):
            raise EmbeddingDependencyError(f"Unexpected Ollama response type for {endpoint}")
        return response

    def _embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        response = self._post(
            "/api/embed",
            {
                "model": self.model_name,
                "input": texts,
            },
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            return None

        vectors: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list):
                raise EmbeddingDependencyError("Invalid vector payload returned by Ollama /api/embed")
            vectors.append([float(value) for value in vector])
        return vectors

    def _embed_single(self, text: str) -> list[float]:
        response = self._post(
            "/api/embeddings",
            {
                "model": self.model_name,
                "prompt": text,
            },
        )
        vector = response.get("embedding")
        if not isinstance(vector, list):
            raise EmbeddingDependencyError("Invalid vector payload returned by Ollama /api/embeddings")
        return [float(value) for value in vector]

    def _set_dimension(self, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return

        inferred = len(vectors[0])
        if inferred == 0:
            raise EmbeddingDependencyError("Ollama returned an empty embedding vector")

        if self.dimension == 0:
            self.dimension = inferred
            return

        if inferred != self.dimension:
            raise EmbeddingDependencyError(
                f"Embedding dimension changed from {self.dimension} to {inferred}"
            )


def _should_use_ollama_backend(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    if normalized.startswith("ollama:") or normalized.startswith("ollama/"):
        return True

    base_model = normalized.split(":", 1)[0]
    return base_model in OLLAMA_EMBED_MODELS


def _resolve_ollama_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    if normalized.lower().startswith("ollama:"):
        return normalized.split(":", 1)[1]
    if normalized.lower().startswith("ollama/"):
        return normalized.split("/", 1)[1]
    return normalized


def create_embedder(
    config: EmbeddingConfig | None = None,
    *,
    allow_fallback: bool = True,
) -> BaseEmbedder:
    """Create the configured embedder, falling back to a deterministic model."""

    embedding_config = config or EmbeddingConfig()

    if _should_use_ollama_backend(embedding_config.model):
        try:
            return OllamaEmbedder(
                _resolve_ollama_model_name(embedding_config.model),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            )
        except EmbeddingDependencyError:
            if not allow_fallback:
                raise
            return DeterministicTextEmbedder()

    try:
        return SentenceTransformerEmbedder(
            embedding_config.model,
            device=embedding_config.device,
        )
    except EmbeddingDependencyError:
        if not allow_fallback:
            raise
        return DeterministicTextEmbedder()