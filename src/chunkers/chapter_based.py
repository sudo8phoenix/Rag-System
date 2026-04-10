"""Chapter-based chunker for EPUB-style content."""

from __future__ import annotations

import re

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class ChapterBasedChunker(BaseChunker):
    """Split books by chapter boundaries, one chunk per chapter."""

    strategy_name = "chapter_based"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        chapters = self._resolve_chapters(document)
        chunks: list[Chunk] = []

        for index, chapter in enumerate(chapters):
            title = str(chapter.get("title", f"Chapter {index + 1}")).strip()
            number = chapter.get("number", index + 1)
            content = str(chapter.get("content", "")).strip()
            if not content:
                continue

            text = f"{title}\n\n{content}".strip()
            metadata = self._base_metadata(
                document, config, self.strategy_name, len(chunks)
            )
            metadata.update(
                {
                    "chapter_title": title,
                    "chapter_number": number,
                }
            )
            chunks.append(
                self._build_chunk(
                    text=text,
                    document=document,
                    chunk_index=len(chunks),
                    strategy_used=self.strategy_name,
                    metadata=metadata,
                )
            )

        return chunks

    def _resolve_chapters(self, document: Document) -> list[dict[str, object]]:
        raw = document.original_metadata.get("chapters")
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]

        pattern = re.compile(r"(?im)^chapter\s+(\d+)\s*[:\.-]?\s*(.+)$")
        matches = list(pattern.finditer(document.text))
        if not matches:
            return [{"number": 1, "title": "Chapter 1", "content": document.text}]

        chapters: list[dict[str, object]] = []
        for idx, match in enumerate(matches):
            number = int(match.group(1))
            title = f"Chapter {number}: {match.group(2).strip()}"
            start = match.end()
            end = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else len(document.text)
            )
            content = document.text[start:end].strip()
            chapters.append({"number": number, "title": title, "content": content})

        return chapters
