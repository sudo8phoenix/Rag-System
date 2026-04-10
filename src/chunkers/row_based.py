"""Row-based chunker for tabular formats (CSV/XLSX/XLS)."""

from __future__ import annotations

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class RowBasedChunker(BaseChunker):
    """Group rows into chunks and prepend headers in every chunk."""

    strategy_name = "row_based"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        lines = [line.strip() for line in document.text.splitlines() if line.strip()]
        if not lines:
            return []

        headers = self._resolve_headers(document, lines)
        if not headers:
            return []

        delimiter = self._resolve_delimiter(document, lines[0])
        first_row = self._normalize_row(
            self._split_row(lines[0], delimiter), len(headers)
        )
        header_row = self._normalize_row(headers, len(headers))
        data_lines = lines[1:] if first_row == header_row else lines
        rows = [
            self._normalize_row(self._split_row(line, delimiter), len(headers))
            for line in data_lines
        ]

        rows_per_chunk = max(1, config.rows_per_chunk)
        chunks: list[Chunk] = []

        for start in range(0, len(rows), rows_per_chunk):
            end = min(start + rows_per_chunk, len(rows))
            row_block = rows[start:end]
            if not row_block:
                continue

            rendered_rows = [" | ".join(headers)] + [
                " | ".join(row) for row in row_block
            ]
            text = "\n".join(rendered_rows).strip()

            metadata = self._base_metadata(
                document, config, self.strategy_name, len(chunks)
            )
            metadata.update(
                {
                    "headers": headers,
                    "row_start": start + 1,
                    "row_end": end,
                    "row_count": len(row_block),
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

    def _resolve_headers(self, document: Document, lines: list[str]) -> list[str]:
        configured = document.original_metadata.get("column_names")
        if isinstance(configured, list) and configured:
            return [str(item).strip() for item in configured]

        if not lines:
            return []

        delimiter = self._resolve_delimiter(document, lines[0])
        return self._split_row(lines[0], delimiter)

    def _resolve_delimiter(self, document: Document, sample_line: str) -> str:
        configured = document.original_metadata.get("delimiter")
        if isinstance(configured, str) and configured:
            return configured

        for delimiter in ("|", ",", "\t", ";"):
            if delimiter in sample_line:
                return delimiter
        return "|"

    def _split_row(self, line: str, delimiter: str) -> list[str]:
        if delimiter == "|":
            return [part.strip() for part in line.split("|")]
        return [part.strip() for part in line.split(delimiter)]

    def _normalize_row(self, row: list[str], width: int) -> list[str]:
        normalized = [cell if cell else "" for cell in row[:width]]
        if len(normalized) < width:
            normalized.extend([""] * (width - len(normalized)))
        return normalized
