"""Token-based chunking strategy backed by tiktoken."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker, ChunkingError


DEFAULT_TOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> Any:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised in runtime when optional dep is missing
        raise ChunkingError(
            "tiktoken is required for token-based chunking. Install it with `pip install tiktoken`."
        ) from exc

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # pragma: no cover - protects against unknown encoding names
        raise ChunkingError(f"Unsupported token encoding: {encoding_name}") from exc


class TokenBasedChunker(BaseChunker):
    """Split documents by token counts using tiktoken."""

    strategy_name = "token"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        encoding_name = self._resolve_encoding_name(config)
        encoder = _get_encoding(encoding_name)
        token_ids: list[int] = encoder.encode(text)
        if not token_ids:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0
        overlap = max(config.chunk_overlap, 0)

        while start < len(token_ids):
            end = min(start + config.chunk_size, len(token_ids))
            if end <= start:
                end = start + 1

            end = self._snap_to_word_boundary(encoder, token_ids, start, end)
            if end <= start:
                end = min(start + config.chunk_size, len(token_ids))
            if end <= start:
                break

            chunk_tokens = token_ids[start:end]
            raw_text = encoder.decode(chunk_tokens)
            chunk_text = raw_text.strip()

            if not chunk_text:
                start = end if end > start else start + 1
                continue

            metadata = self._base_metadata(document, config, self.strategy_name, chunk_index)
            metadata.update(
                {
                    "token_start": start + 1,
                    "token_end": end,
                    "token_count": len(chunk_tokens),
                    "token_encoding": encoding_name,
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

            if end >= len(token_ids):
                break

            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    @staticmethod
    def count_tokens(text: str, *, encoding_name: str = DEFAULT_TOKEN_ENCODING) -> int:
        """Return token count for text using the configured encoding."""

        if not text:
            return 0
        encoder = _get_encoding(encoding_name)
        return len(encoder.encode(text))

    @staticmethod
    def _resolve_encoding_name(config: ChunkingConfig) -> str:
        encoding_name = getattr(config, "token_encoding", DEFAULT_TOKEN_ENCODING)
        if not isinstance(encoding_name, str) or not encoding_name.strip():
            return DEFAULT_TOKEN_ENCODING
        return encoding_name.strip()

    @staticmethod
    def _snap_to_word_boundary(encoder: Any, token_ids: list[int], start: int, end: int) -> int:
        if end >= len(token_ids):
            return end

        safe_end = end
        while safe_end > start + 1:
            left = encoder.decode(token_ids[start:safe_end])
            right = encoder.decode([token_ids[safe_end]])
            if TokenBasedChunker._is_word_boundary(left, right):
                break
            safe_end -= 1

        return safe_end

    @staticmethod
    def _is_word_boundary(left: str, right: str) -> bool:
        if not left:
            return False
        if not right:
            return True

        left_tail = left[-1]
        right_head = right[0]
        if left_tail.isspace() or right_head.isspace():
            return True
        if re.match(r"\W", left_tail) or re.match(r"\W", right_head):
            return True
        return False
