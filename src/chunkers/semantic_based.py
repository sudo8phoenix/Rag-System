"""Semantic similarity-based chunking strategy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker

if TYPE_CHECKING:
    from src.embeddings.base import BaseEmbedder


class SemanticBasedChunker(BaseChunker):
    """Split documents on semantic breakpoints using embedding similarity."""

    strategy_name = "semantic"

    def __init__(self, embedder: BaseEmbedder | None = None) -> None:
        """Initialize semantic chunker with optional embedder dependency.

        Args:
            embedder: Embedder to compute similarity scores. If None, embedder
                     must be provided at chunk time via config injection.
        """
        self.embedder = embedder

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        """Split document at semantic breakpoints using embedding similarity.

        Args:
            document: Document to chunk.
            config: Chunking configuration including semantic_similarity_threshold.

        Returns:
            List of Chunk objects split at semantic boundaries.

        Raises:
            ValueError: If embedder is required but not available.
        """
        if not document.text.strip():
            return []

        # Split into sentences to create semantic units
        sentences = self._split_sentences(document.text)
        if not sentences:
            return []

        # Get embedder from config or instance
        embedder = self._get_embedder(config)
        if embedder is None:
            # Fallback to paragraph-based if no embedder available
            return self._fallback_paragraph_chunking(document, config, sentences)

        # Embed all sentences
        try:
            embeddings = embedder.embed_texts(sentences)
        except Exception as exc:
            # Fallback if embedding fails
            return self._fallback_paragraph_chunking(document, config, sentences)

        # Detect semantic breakpoints
        breakpoints = self._find_semantic_breakpoints(
            embeddings, config.semantic_similarity_threshold
        )

        # Build chunks respecting breakpoints and size constraints
        chunks = self._build_chunks(sentences, breakpoints, document, config)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using basic heuristics.

        Args:
            text: Text to split.

        Returns:
            List of sentences with whitespace normalized.
        """
        # Split on sentence boundaries (., !, ?)
        # Try to preserve formatting while splitting
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        # Filter and clean
        cleaned = []
        for sentence in sentences:
            cleaned_sent = sentence.strip()
            if len(cleaned_sent) > 1:  # Avoid single-char "sentences"
                cleaned.append(cleaned_sent)

        return cleaned

    @staticmethod
    def _find_semantic_breakpoints(
        embeddings: list[list[float]],
        threshold: float,
    ) -> list[int]:
        """Find breakpoints where semantic similarity drops below threshold.

        Args:
            embeddings: List of embedding vectors (normalized).
            threshold: Similarity threshold for breakpoints.

        Returns:
            List of sentence indices where breakpoints occur.
        """
        if len(embeddings) == 0:
            return []

        if len(embeddings) == 1:
            return [0]  # Always start with first sentence

        breakpoints = [0]  # Always start with first sentence

        for i in range(len(embeddings) - 1):
            # Compute cosine similarity
            similarity = SemanticBasedChunker._cosine_similarity(
                embeddings[i], embeddings[i + 1]
            )

            # If similarity drops below threshold, mark as breakpoint
            if similarity < threshold:
                breakpoints.append(i + 1)

        return breakpoints

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score (0 to 1).
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return min(1.0, max(0.0, dot_product / (mag_a * mag_b)))

    def _build_chunks(
        self,
        sentences: list[str],
        breakpoints: list[int],
        document: Document,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Build chunks from sentences and breakpoints.

        Respects chunk_size and chunk_overlap constraints.

        Args:
            sentences: List of sentences.
            breakpoints: Indices of semantic breakpoints.
            document: Source document.
            config: Chunking configuration.

        Returns:
            List of Chunk objects.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for i in range(len(breakpoints)):
            start_idx = breakpoints[i]

            # Find end of this chunk (next breakpoint or end)
            end_idx = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(sentences)

            # Collect sentences for this chunk
            chunk_sentences: list[str] = []
            current_size = 0

            for j in range(start_idx, end_idx):
                sentence = sentences[j]
                prospective_text = " ".join(chunk_sentences + [sentence])

                # Check size constraints
                if (
                    current_size > 0
                    and len(prospective_text) > config.chunk_size
                    and len(" ".join(chunk_sentences)) >= config.min_chunk_size
                ):
                    # Current chunk is large enough, save it
                    if chunk_sentences:
                        chunk_text = " ".join(chunk_sentences)
                        metadata = self._base_metadata(
                            document, config, self.strategy_name, chunk_index
                        )
                        metadata.update(
                            {
                                "sentence_start": start_idx,
                                "sentence_end": j,
                                "sentence_count": len(chunk_sentences),
                            }
                        )
                        chunks.append(
                            self._build_chunk(
                                text=chunk_text,
                                document=document,
                                chunk_index=chunk_index,
                                strategy_used=self.strategy_name,
                                metadata=metadata,
                            )
                        )
                        chunk_index += 1
                    chunk_sentences = [sentence]
                    current_size = len(sentence)
                else:
                    chunk_sentences.append(sentence)
                    current_size = len(prospective_text)

            # Save remaining chunk
            if (
                chunk_sentences
                and len(" ".join(chunk_sentences)) >= config.min_chunk_size
            ):
                chunk_text = " ".join(chunk_sentences)
                metadata = self._base_metadata(
                    document, config, self.strategy_name, chunk_index
                )
                metadata.update(
                    {
                        "sentence_start": start_idx,
                        "sentence_end": end_idx,
                        "sentence_count": len(chunk_sentences),
                    }
                )
                chunks.append(
                    self._build_chunk(
                        text=chunk_text,
                        document=document,
                        chunk_index=chunk_index,
                        strategy_used=self.strategy_name,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        return chunks

    def _fallback_paragraph_chunking(
        self,
        document: Document,
        config: ChunkingConfig,
        sentences: list[str],
    ) -> list[Chunk]:
        """Fallback to paragraph-like chunking when embedding unavailable.

        Args:
            document: Source document.
            config: Chunking configuration.
            sentences: Pre-split sentences.

        Returns:
            List of Chunk objects.
        """
        chunks: list[Chunk] = []
        chunk_index = 0
        chunk_sentences: list[str] = []
        current_size = 0

        for sentence in sentences:
            prospective = " ".join(chunk_sentences + [sentence])

            if (
                chunk_sentences
                and len(prospective) > config.chunk_size
                and current_size >= config.min_chunk_size
            ):
                # Save current chunk
                chunk_text = " ".join(chunk_sentences)
                metadata = self._base_metadata(
                    document, config, self.strategy_name, chunk_index
                )
                chunks.append(
                    self._build_chunk(
                        text=chunk_text,
                        document=document,
                        chunk_index=chunk_index,
                        strategy_used=self.strategy_name,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
                chunk_sentences = [sentence]
                current_size = len(sentence)
            else:
                chunk_sentences.append(sentence)
                current_size = len(prospective)

        # Save remaining
        if chunk_sentences and current_size >= config.min_chunk_size:
            chunk_text = " ".join(chunk_sentences)
            metadata = self._base_metadata(
                document, config, self.strategy_name, chunk_index
            )
            chunks.append(
                self._build_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                    strategy_used=self.strategy_name,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _get_embedder(config: ChunkingConfig) -> BaseEmbedder | None:
        """Retrieve embedder from config if available.

        Args:
            config: Chunking configuration.

        Returns:
            Embedder instance from config extras, or None.
        """
        # Try to get embedder from config extras
        if hasattr(config, "__pydantic_extra__"):
            return config.__pydantic_extra__.get("embedder")
        return None
