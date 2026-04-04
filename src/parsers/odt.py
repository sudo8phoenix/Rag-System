"""ODT parser with DOCX conversion primary path and odfpy fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .base import BaseParser, ParserError
from .conversion import convert_file_with_libreoffice
from .docx import DocxParser


class OdtParser(BaseParser):
    """Parse .odt by converting to .docx, with odfpy fallback when needed."""

    source_type = "odt"
    supported_extensions = (".odt",)

    def __init__(self) -> None:
        self._docx_parser = DocxParser()

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"ODT file not found: {path}")

        try:
            text, metadata = self._parse_via_conversion(path)
            metadata.update(
                {
                    "format": self.source_type,
                    "extraction_method": "libreoffice-to-docx",
                    "converted_from": ".odt",
                }
            )
            return self._build_document(
                text=text,
                file_path=path,
                source_type=self.source_type,
                metadata=metadata,
            )
        except Exception as conversion_error:
            fallback_text, fallback_meta = self._parse_with_odfpy(path)
            if fallback_text:
                fallback_meta.update(
                    {
                        "format": self.source_type,
                        "extraction_method": "odfpy",
                        "conversion_error": str(conversion_error),
                    }
                )
                return self._build_document(
                    text=fallback_text,
                    file_path=path,
                    source_type=self.source_type,
                    metadata=fallback_meta,
                )
            raise ParserError(
                f"Unable to parse ODT file {path}: {conversion_error}"
            ) from conversion_error

    def _parse_via_conversion(self, path: Path) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="rag-odt-") as temp_dir:
            converted = convert_file_with_libreoffice(
                path,
                output_dir=Path(temp_dir),
                target_extension="docx",
            )
            converted_document = self._docx_parser.parse(converted)
        return converted_document.text, dict(converted_document.original_metadata)

    def _parse_with_odfpy(self, path: Path) -> tuple[str, dict[str, object]]:
        try:
            from odf.opendocument import load
            from odf import text as odf_text
            from odf.teletype import extractText
        except ImportError:  # pragma: no cover - optional dependency
            return "", {}

        try:
            document = load(str(path))
        except Exception:
            return "", {}

        lines: list[str] = []
        heading_count = 0
        paragraph_count = 0

        for heading in document.getElementsByType(odf_text.H):
            value = extractText(heading).strip()
            if value:
                heading_count += 1
                lines.append(value)

        for paragraph in document.getElementsByType(odf_text.P):
            value = extractText(paragraph).strip()
            if value:
                paragraph_count += 1
                lines.append(value)

        lines = self._strip_outer_whitespace(lines)
        if not lines:
            return "", {}

        metadata = {
            "heading_count": heading_count,
            "paragraph_count": paragraph_count,
        }
        return "\n".join(lines), metadata
