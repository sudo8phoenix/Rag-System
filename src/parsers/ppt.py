"""Legacy PPT parser using LibreOffice conversion with textract fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .base import BaseParser, ParserError
from .conversion import convert_file_with_libreoffice, try_textract_extract
from .pptx import PptxParser


class PptParser(BaseParser):
    """Parse .ppt by converting to .pptx first, then using the PPTX parser."""

    source_type = "ppt"
    supported_extensions = (".ppt",)

    def __init__(self) -> None:
        self._pptx_parser = PptxParser()

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"PPT file not found: {path}")

        try:
            text, metadata = self._parse_via_conversion(path)
            metadata.update(
                {
                    "format": self.source_type,
                    "extraction_method": "libreoffice-to-pptx",
                    "converted_from": ".ppt",
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
                f"Unable to parse PPT file {path}: {conversion_error}"
            ) from conversion_error

    def _parse_via_conversion(self, path: Path) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="rag-ppt-") as temp_dir:
            converted = convert_file_with_libreoffice(
                path,
                output_dir=Path(temp_dir),
                target_extension="pptx",
            )
            converted_document = self._pptx_parser.parse(converted)
        return converted_document.text, dict(converted_document.original_metadata)
