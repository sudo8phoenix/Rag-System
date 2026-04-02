"""Paragraph-based chunking strategy."""

from __future__ import annotations

import re

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class ParagraphBasedChunker(BaseChunker):
    """Split documents on blank-line paragraph boundaries."""

    strategy_name = "paragraph"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", document.text) if part.strip()]
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        start_index = 0

        while start_index < len(paragraphs):
            chunk_paragraphs: list[str] = []
            overlap_count = 1 if start_index > 0 and config.chunk_overlap > 0 else 0

            cursor = start_index
            new_paragraphs_consumed = 0

            while cursor < len(paragraphs):
                candidate = paragraphs[cursor]
                prospective = chunk_paragraphs + [candidate]
                prospective_text = "\n\n".join(prospective)
                current_text = "\n\n".join(chunk_paragraphs)

                if chunk_paragraphs and len(prospective_text) > config.chunk_size:
                    if new_paragraphs_consumed == 0 and overlap_count > 0:
                        overlap_count = 0
                        continue
                    if len(current_text) >= config.min_chunk_size or new_paragraphs_consumed > 0:
                        break

                chunk_paragraphs.append(candidate)
                cursor += 1
                new_paragraphs_consumed += 1

                if len("\n\n".join(chunk_paragraphs)) >= config.chunk_size:
                    break

            if new_paragraphs_consumed == 0:
                chunk_paragraphs = [paragraphs[cursor]]
                cursor += 1
                new_paragraphs_consumed = 1

            chunk_text = "\n\n".join(chunk_paragraphs).strip()
            if not chunk_text:
                start_index = cursor
                continue

            paragraph_start = start_index + 1
            paragraph_end = cursor

            metadata = self._base_metadata(document, config, self.strategy_name, chunk_index)
            metadata.update(
                {
                    "paragraph_start": paragraph_start,
                    "paragraph_end": paragraph_end,
                    "paragraph_count": new_paragraphs_consumed,
                    "overlap_paragraphs": overlap_count,
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
            start_index = cursor

        return chunks