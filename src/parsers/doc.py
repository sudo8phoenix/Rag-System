"""Legacy DOC parser using DOCX conversion and textract fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .base import BaseParser, ParserError
from .conversion import convert_file_with_libreoffice, try_textract_extract
from .docx import DocxParser


class DocParser(BaseParser):
    """Parse .doc by converting to .docx with fallback extraction."""

    source_type = "doc"
    supported_extensions = (".doc",)

    def __init__(self) -> None:
        self._docx_parser = DocxParser()

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"DOC file not found: {path}")

        try:
            text, metadata = self._parse_via_conversion(path)
            metadata.update(
                {
                    "format": self.source_type,
                    "extraction_method": "libreoffice-to-docx",
                    "converted_from": ".doc",
                }
            )
            return self._build_document(
                text=text,
                file_path=path,
                source_type=self.source_type,
                metadata=metadata,
            )
        except Exception as conversion_error:
            fallback_text = try_textract_extract(path)
            if fallback_text:
                return self._build_document(
                    text=fallback_text,
                    file_path=path,
                    source_type=self.source_type,
                    metadata={
                        "format": self.source_type,
                        "extraction_method": "textract",
                        "conversion_error": str(conversion_error),
                    },
                )
            raise ParserError(
                f"Unable to parse DOC file {path}: {conversion_error}"
            ) from conversion_error

    def _parse_via_conversion(self, path: Path) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="rag-doc-") as temp_dir:
            converted = convert_file_with_libreoffice(
                path,
                output_dir=Path(temp_dir),
                target_extension="docx",
            )
            converted_document = self._docx_parser.parse(converted)
        return converted_document.text, dict(converted_document.original_metadata)
