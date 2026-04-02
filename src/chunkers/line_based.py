"""Line-based chunking strategy."""

from __future__ import annotations

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class LineBasedChunker(BaseChunker):
    """Split documents on newline boundaries."""

    strategy_name = "line"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        lines = document.text.splitlines()
        if not lines:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0

        while start < len(lines):
            end = min(start + config.chunk_size, len(lines))
            if end <= start:
                end = start + 1

            while end > start and not lines[end - 1].strip():
                end -= 1

            if end <= start:
                start += 1
                continue

            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines).strip()
            if not chunk_text:
                start = end
                continue

            metadata = self._base_metadata(document, config, self.strategy_name, chunk_index)
            metadata.update(
                {
                    "line_start": start + 1,
                    "line_end": end,
                    "line_count": len(chunk_lines),
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

            if end >= len(lines):
                break

            next_start = end - config.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks