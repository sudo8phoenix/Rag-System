"""Slide-based chunker for PPTX-like content."""

from __future__ import annotations

import re

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class SlideBasedChunker(BaseChunker):
    """Emit one chunk per slide, optionally including notes."""

    strategy_name = "slide_based"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        slides = self._resolve_slides(document)
        chunks: list[Chunk] = []

        for index, slide in enumerate(slides):
            number = int(slide.get("number", index + 1))
            title = str(slide.get("title", "")).strip()
            content = str(slide.get("content", "")).strip()
            notes = str(slide.get("notes", "")).strip()

            parts = [part for part in (title, content) if part]
            if config.include_notes and notes:
                parts.append(f"Speaker notes: {notes}")

            text = "\n".join(parts).strip()
            if not text:
                continue

            metadata = self._base_metadata(
                document, config, self.strategy_name, len(chunks)
            )
            metadata.update(
                {
                    "slide_number": number,
                    "slide_title": title,
                    "has_notes": bool(notes),
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

    def _resolve_slides(self, document: Document) -> list[dict[str, object]]:
        raw = document.original_metadata.get("slides")
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]

        segments = re.split(
            r"\n\s*---\s*slide\s*---\s*\n", document.text, flags=re.IGNORECASE
        )
        slides: list[dict[str, object]] = []
        for index, segment in enumerate(segments, start=1):
            text = segment.strip()
            if not text:
                continue

            notes_match = re.split(
                r"\n\s*notes\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE
            )
            body = notes_match[0].strip()
            notes = notes_match[1].strip() if len(notes_match) > 1 else ""

            lines = [line.strip() for line in body.splitlines() if line.strip()]
            title = lines[0] if lines else f"Slide {index}"
            content = "\n".join(lines[1:]) if len(lines) > 1 else ""

            slides.append(
                {"number": index, "title": title, "content": content, "notes": notes}
            )

        return slides
