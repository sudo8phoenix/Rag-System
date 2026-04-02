"""Document data model used by parsers and downstream processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents normalized document content plus source metadata."""

    text: str
    filename: str
    source_type: str
    original_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not self.filename or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not self.source_type or not self.source_type.strip():
            raise ValueError("source_type must be a non-empty string")
        if not isinstance(self.original_metadata, dict):
            raise TypeError("original_metadata must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        """Serialize document to a plain dictionary for persistence or transport."""
        return {
            "text": self.text,
            "filename": self.filename,
            "source_type": self.source_type,
            "original_metadata": dict(self.original_metadata),
        }
