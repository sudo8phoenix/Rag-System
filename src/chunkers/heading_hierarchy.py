"""Heading hierarchy chunker for structured documents like DOCX/PDF."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


@dataclass(frozen=True)
class _Heading:
    line_index: int
    level: int
    text: str


class HeadingHierarchyChunker(BaseChunker):
    """Split text at heading boundaries while preserving hierarchy context."""

    strategy_name = "heading_hierarchy"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        lines = document.text.splitlines()
        if not lines:
            return []

        headings = self._detect_headings(lines, document, config)
        if not headings:
            metadata = self._base_metadata(document, config, self.strategy_name, 0)
            metadata.update({"heading_path": [], "heading_level": 0})
            return [
                self._build_chunk(
                    text=document.text.strip(),
                    document=document,
                    chunk_index=0,
                    strategy_used=self.strategy_name,
                    metadata=metadata,
                )
            ]

        allowed_levels = set(config.heading_levels or [1, 2, 3])
        contexts = self._build_contexts(headings)

        chunks: list[Chunk] = []
        for index, heading in enumerate(headings):
            if heading.level not in allowed_levels:
                continue

            start = heading.line_index
            end = self._section_end(headings, index, len(lines))
            section_lines = lines[start:end]
            text = "\n".join(section_lines).strip()
            if not text:
                continue

            path = contexts[index]
            metadata = self._base_metadata(
                document, config, self.strategy_name, len(chunks)
            )
            metadata.update(
                {
                    "heading": heading.text,
                    "heading_level": heading.level,
                    "heading_path": path,
                    "section_line_start": start + 1,
                    "section_line_end": end,
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

    def _detect_headings(
        self,
        lines: list[str],
        document: Document,
        config: ChunkingConfig,
    ) -> list[_Heading]:
        headings: list[_Heading] = []
        expected = document.original_metadata.get("headings", [])
        expected_queue: list[tuple[str, int]] = []
        for item in expected:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            level = item.get("level")
            if text and isinstance(level, int):
                expected_queue.append((text, level))

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            if expected_queue and line == expected_queue[0][0]:
                heading_text, heading_level = expected_queue.pop(0)
                headings.append(
                    _Heading(line_index=index, level=heading_level, text=heading_text)
                )
                continue

            parsed = self._parse_heading_line(line)
            if parsed is None:
                continue

            level, text = parsed
            if level in set(config.heading_levels or [1, 2, 3]):
                headings.append(_Heading(line_index=index, level=level, text=text))

        deduped: list[_Heading] = []
        for heading in headings:
            if deduped and deduped[-1].line_index == heading.line_index:
                continue
            deduped.append(heading)
        return deduped

    def _parse_heading_line(self, line: str) -> tuple[int, str] | None:
        md_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if md_match:
            return len(md_match.group(1)), md_match.group(2).strip()

        numbered = re.match(r"^(\d+(?:\.\d+){0,2})\s+(.+)$", line)
        if numbered:
            level = numbered.group(1).count(".") + 1
            return min(level, 3), numbered.group(2).strip()

        title = re.match(
            r"^(chapter|section)\s+\d+[:\.-]?\s+(.+)$", line, re.IGNORECASE
        )
        if title:
            keyword = title.group(1).lower()
            return (1 if keyword == "chapter" else 2), title.group(2).strip()

        return None

    def _build_contexts(self, headings: list[_Heading]) -> list[list[str]]:
        contexts: list[list[str]] = []
        stack: list[_Heading] = []

        for heading in headings:
            while stack and stack[-1].level >= heading.level:
                stack.pop()
            stack.append(heading)
            contexts.append([item.text for item in stack])

        return contexts

    def _section_end(self, headings: list[_Heading], index: int, max_lines: int) -> int:
        current = headings[index]
        for candidate in headings[index + 1 :]:
            if candidate.level <= current.level:
                return candidate.line_index
        return max_lines
