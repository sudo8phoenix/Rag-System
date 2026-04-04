"""Chunking strategies and registry."""

from .base import BaseChunker, ChunkingError
from .array_item import ArrayItemChunker
from .chapter_based import ChapterBasedChunker
from .character_based import CharacterBasedChunker
from .heading_hierarchy import HeadingHierarchyChunker
from .line_based import LineBasedChunker
from .paragraph_based import ParagraphBasedChunker
from .row_based import RowBasedChunker
from .slide_based import SlideBasedChunker
from .tag_based import TagBasedChunker
from .registry import ChunkingRegistry, chunk_document, get_chunker_for_document

__all__ = [
    "BaseChunker",
    "ChunkingError",
    "LineBasedChunker",
    "CharacterBasedChunker",
    "ParagraphBasedChunker",
    "HeadingHierarchyChunker",
    "RowBasedChunker",
    "ArrayItemChunker",
    "SlideBasedChunker",
    "TagBasedChunker",
    "ChapterBasedChunker",
    "ChunkingRegistry",
    "chunk_document",
    "get_chunker_for_document",
]