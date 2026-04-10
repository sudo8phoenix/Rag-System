"""Shared parsing primitives used by the document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.document import Document


class ParserError(RuntimeError):
    """Raised when a parser cannot extract content from a file."""


class BaseParser(ABC):
    """Base class for document parsers."""

    source_type: str = ""
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, file_path: str | Path) -> Document:
        """Parse a file into a normalized Document."""

    @staticmethod
    def _build_document(
        *,
        text: str,
        file_path: str | Path,
        source_type: str,
        metadata: dict[str, Any],
    ) -> Document:
        path = Path(file_path)
        return Document(
            text=text,
            filename=path.name,
            source_type=source_type,
            original_metadata=metadata,
        )

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _strip_outer_whitespace(lines: list[str]) -> list[str]:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return lines
