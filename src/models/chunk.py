"""Chunk data model representing split document segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .document import Document


@dataclass(frozen=True)
class Chunk:
    """Represents one chunk derived from a source document."""

    text: str
    chunk_id: str
    source_doc: Document
    chunk_index: int
    strategy_used: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not self.chunk_id or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be greater than or equal to 0")
        if not self.strategy_used or not self.strategy_used.strip():
            raise ValueError("strategy_used must be a non-empty string")
        if not isinstance(self.source_doc, Document):
            raise TypeError("source_doc must be an instance of Document")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a plain dictionary including source context."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc.to_dict(),
            "chunk_index": self.chunk_index,
            "strategy_used": self.strategy_used,
            "metadata": dict(self.metadata),
        }
