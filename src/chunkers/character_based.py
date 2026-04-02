"""Character-based chunking strategy."""

from __future__ import annotations

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class CharacterBasedChunker(BaseChunker):
    """Split documents by character counts while avoiding mid-word cuts."""

    strategy_name = "character"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0

        while start < len(text):
            target_end = min(start + config.chunk_size, len(text))
            end = self._find_chunk_end(text, start, target_end)
            if end <= start:
                end = min(start + config.chunk_size, len(text))
            if end <= start:
                break

            raw_slice = text[start:end]
            chunk_text = raw_slice.strip()
            if not chunk_text:
                start = end if end > start else start + 1
                continue

            metadata = self._base_metadata(document, config, self.strategy_name, chunk_index)
            metadata.update(
                {
                    "char_start": start + 1,
                    "char_end": end,
                    "char_length": len(raw_slice),
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

            next_start = end - config.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            start = next_start

        return chunks

    @staticmethod
    def _find_chunk_end(text: str, start: int, target_end: int) -> int:
        if target_end >= len(text):
            return len(text)

        if text[target_end].isspace():
            return target_end

        for position in range(target_end, start, -1):
            if text[position - 1].isspace():
                return position

        return target_end