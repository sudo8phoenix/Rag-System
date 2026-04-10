"""Shared chunking primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document


class ChunkingError(RuntimeError):
    """Raised when a chunking strategy cannot process a document."""


class BaseChunker(ABC):
    """Base class for deterministic chunking strategies."""

    strategy_name: str = ""

    @abstractmethod
    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        """Split a document into chunks using the provided configuration."""

    @staticmethod
    def _build_chunk(
        *,
        text: str,
        document: Document,
        chunk_index: int,
        strategy_used: str,
        metadata: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            text=text,
            chunk_id=f"{document.filename}:{strategy_used}:{chunk_index}",
            source_doc=document,
            chunk_index=chunk_index,
            strategy_used=strategy_used,
            metadata=metadata,
        )

    @staticmethod
    def _base_metadata(
        document: Document,
        config: ChunkingConfig,
        strategy_used: str,
        chunk_index: int,
    ) -> dict[str, Any]:
        metadata = dict(document.original_metadata)
        metadata.update(
            {
                "filename": document.filename,
                "source_type": document.source_type,
                "strategy": strategy_used,
                "chunk_index": chunk_index,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunk_unit": config.chunk_unit,
            }
        )
        return metadata
