"""Chunking registry and config-driven dispatcher."""

from __future__ import annotations

from typing import Any

from src.config.settings import AppConfig, ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker
from .character_based import CharacterBasedChunker
from .line_based import LineBasedChunker
from .paragraph_based import ParagraphBasedChunker


def _normalize_strategy(strategy: str | None) -> str:
    if not strategy:
        return "paragraph"

    normalized = strategy.strip().lower().replace("-", "_")
    alias_map = {
        "line_based": "line",
        "line": "line",
        "character_based": "character",
        "char": "character",
        "character": "character",
        "paragraph_based": "paragraph",
        "paragraph": "paragraph",
    }
    return alias_map.get(normalized, normalized)


class ChunkingRegistry:
    """Select and apply chunkers using global and per-format config."""

    def __init__(self, config: ChunkingConfig | AppConfig | None = None) -> None:
        if isinstance(config, AppConfig):
            self.config = config.chunking
        else:
            self.config = config or ChunkingConfig()

        self._chunkers: dict[str, BaseChunker] = {
            "line": LineBasedChunker(),
            "character": CharacterBasedChunker(),
            "paragraph": ParagraphBasedChunker(),
        }

    @property
    def supported_strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._chunkers))

    def get_chunker(self, document: Document) -> BaseChunker:
        strategy = self._selected_strategy(document)
        return self._chunkers.get(strategy, self._chunkers[_normalize_strategy(self.config.strategy)])

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunker = self.get_chunker(document)
        config = self._effective_config(document)
        return chunker.chunk(document, config)

    def _selected_strategy(self, document: Document) -> str:
        format_config = self.config.per_format.get(document.source_type)
        if format_config is not None and format_config.strategy:
            normalized = _normalize_strategy(format_config.strategy)
            if normalized in self._chunkers:
                return normalized

        return _normalize_strategy(self.config.strategy)

    def _effective_config(self, document: Document) -> ChunkingConfig:
        override = self.config.per_format.get(document.source_type)
        if override is None:
            return self.config

        payload: dict[str, Any] = self.config.model_dump()
        override_payload = override.model_dump(exclude_none=True)
        for key in ("strategy", "chunk_size", "chunk_overlap", "chunk_unit"):
            if key in override_payload:
                payload[key] = override_payload[key]

        for key, value in override_payload.items():
            if key not in {"strategy", "chunk_size", "chunk_overlap", "chunk_unit"}:
                payload[key] = value

        payload["per_format"] = self.config.per_format
        return ChunkingConfig.model_validate(payload)


def get_chunker_for_document(
    document: Document,
    config: ChunkingConfig | AppConfig | None = None,
) -> BaseChunker:
    """Return the chunker selected for a document."""

    return ChunkingRegistry(config).get_chunker(document)


def chunk_document(
    document: Document,
    config: ChunkingConfig | AppConfig | None = None,
) -> list[Chunk]:
    """Chunk a document using the configured strategy registry."""

    return ChunkingRegistry(config).chunk_document(document)