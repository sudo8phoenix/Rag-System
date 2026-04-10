"""Chunking registry and config-driven dispatcher."""

from __future__ import annotations

from src.config.settings import AppConfig, ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker
from .character_based import CharacterBasedChunker
from .chapter_based import ChapterBasedChunker
from .heading_hierarchy import HeadingHierarchyChunker
from .line_based import LineBasedChunker
from .array_item import ArrayItemChunker
from .paragraph_based import ParagraphBasedChunker
from .row_based import RowBasedChunker
from .slide_based import SlideBasedChunker
from .tag_based import TagBasedChunker
from .semantic_based import SemanticBasedChunker
from .token_based import TokenBasedChunker


def _normalize_strategy(strategy: str | None) -> str:
    if not strategy:
        return "paragraph"

    normalized = strategy.strip().lower().replace("-", "_")
    alias_map = {
        "line_based": "line",
        "line_per_chunk": "line",
        "line": "line",
        "character_based": "character",
        "char": "character",
        "character": "character",
        "paragraph_based": "paragraph",
        "main_content": "paragraph",
        "paragraph": "paragraph",
        "heading_hierarchy": "heading_hierarchy",
        "header_based": "heading_hierarchy",
        "heading": "heading_hierarchy",
        "row_based": "row_based",
        "rows": "row_based",
        "array_item": "array_item",
        "array": "array_item",
        "slide_based": "slide_based",
        "slide": "slide_based",
        "tag_based": "tag_based",
        "tag": "tag_based",
        "chapter_based": "chapter_based",
        "chapter": "chapter_based",
        "semantic_based": "semantic",
        "semantic": "semantic",
        "token_based": "token",
        "token": "token",
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
            "heading_hierarchy": HeadingHierarchyChunker(),
            "row_based": RowBasedChunker(),
            "array_item": ArrayItemChunker(),
            "slide_based": SlideBasedChunker(),
            "tag_based": TagBasedChunker(),
            "chapter_based": ChapterBasedChunker(),
            "semantic": SemanticBasedChunker(),
            "token": TokenBasedChunker(),
        }

    @property
    def supported_strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._chunkers))

    def get_chunker(self, document: Document) -> BaseChunker:
        strategy = self._selected_strategy(document)
        return self._chunkers.get(
            strategy, self._chunkers[_normalize_strategy(self.config.strategy)]
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunker = self.get_chunker(document)
        config = self.config.effective_for_format(document.source_type)
        return chunker.chunk(document, config)

    def _selected_strategy(self, document: Document) -> str:
        format_config = self.config.effective_for_format(document.source_type)
        normalized = _normalize_strategy(format_config.strategy)
        if normalized in self._chunkers:
            return normalized

        return _normalize_strategy(self.config.strategy)


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
