"""Array-item chunker for JSON array payloads."""

from __future__ import annotations

import json
from typing import Any

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class ArrayItemChunker(BaseChunker):
    """Create one chunk per top-level array item with flattened content."""

    strategy_name = "array_item"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        payload = self._resolve_payload(document)
        if isinstance(payload, list):
            items = payload
        else:
            items = [payload]

        chunks: list[Chunk] = []
        for index, item in enumerate(items):
            lines: list[str] = []
            self._flatten(item, "item", lines)
            text = "\n".join(lines).strip() or json.dumps(item, ensure_ascii=False)

            metadata = self._base_metadata(document, config, self.strategy_name, index)
            metadata.update(
                {
                    "array_index": index,
                    "item_type": type(item).__name__,
                }
            )
            chunks.append(
                self._build_chunk(
                    text=text,
                    document=document,
                    chunk_index=index,
                    strategy_used=self.strategy_name,
                    metadata=metadata,
                )
            )

        return chunks

    def _resolve_payload(self, document: Document) -> Any:
        if "parsed_json" in document.original_metadata:
            return document.original_metadata["parsed_json"]

        raw = document.original_metadata.get("raw_json")
        if isinstance(raw, (list, dict)):
            return raw

        try:
            return json.loads(document.text)
        except json.JSONDecodeError:
            return document.text

    def _flatten(self, value: Any, path: str, lines: list[str]) -> None:
        if isinstance(value, dict):
            if not value:
                lines.append(f"{path}: {{}}")
                return
            for key, child in value.items():
                self._flatten(child, f"{path}.{key}", lines)
            return

        if isinstance(value, list):
            if not value:
                lines.append(f"{path}: []")
                return
            for index, child in enumerate(value):
                self._flatten(child, f"{path}[{index}]", lines)
            return

        lines.append(f"{path}: {json.dumps(value, ensure_ascii=False)}")
