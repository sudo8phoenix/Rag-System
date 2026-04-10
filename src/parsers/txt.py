"""Plain text parser with encoding detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chardet
except ImportError:  # pragma: no cover - optional dependency
    chardet = None

from .base import BaseParser, ParserError


class TxtParser(BaseParser):
    """Parse plain text files into Document objects."""

    source_type = "txt"
    supported_extensions = (".txt", ".text", ".log")

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"Text file not found: {path}")

        raw_bytes = path.read_bytes()
        text, encoding, metadata = self._decode_text(raw_bytes)
        normalized = self._normalize_newlines(text).strip()
        if not normalized:
            raise ParserError(f"Text file is empty: {path}")

        metadata.update(
            {
                "format": self.source_type,
                "encoding": encoding,
                "line_count": normalized.count("\n") + 1,
                "byte_count": len(raw_bytes),
            }
        )
        return self._build_document(
            text=normalized,
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _decode_text(self, raw_bytes: bytes) -> tuple[str, str, dict[str, Any]]:
        detected_encoding = self._detect_encoding(raw_bytes)
        candidates = [detected_encoding, "utf-8-sig", "utf-8", "cp1252", "latin-1"]
        tried: list[str] = []

        for encoding in candidates:
            if not encoding or encoding in tried:
                continue
            tried.append(encoding)
            try:
                return (
                    raw_bytes.decode(encoding),
                    encoding,
                    {"encoding_candidates": tried},
                )
            except UnicodeDecodeError:
                continue

        raise ParserError("Unable to decode text file")

    def _detect_encoding(self, raw_bytes: bytes) -> str | None:
        if chardet is None:
            return None

        result = chardet.detect(raw_bytes)
        encoding = result.get("encoding") if isinstance(result, dict) else None
        if not encoding:
            return None
        return encoding.lower()
