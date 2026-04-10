"""Parser registry and dispatcher."""

from __future__ import annotations

from pathlib import Path

from src.models.document import Document

from .base import BaseParser, ParserError
from .conversion import get_legacy_conversion_target, temporary_converted_file
from .csv import CsvParser
from .doc import DocParser
from .docx import DocxParser
from .epub import EpubParser
from .html import HtmlParser
from .json import JsonParser
from .jsonl import JsonlParser
from .md import MarkdownParser
from .odt import OdtParser
from .pdf import PdfParser
from .ppt import PptParser
from .pptx import PptxParser
from .txt import TxtParser
from .xls import XlsParser
from .xlsx import XlsxParser
from .xml import XmlParser


class ParserRegistry:
    """Dispatch files to the appropriate parser based on extension."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {
            ".txt": TxtParser(),
            ".text": TxtParser(),
            ".log": TxtParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".mdown": MarkdownParser(),
            ".docx": DocxParser(),
            ".doc": DocParser(),
            ".pdf": PdfParser(),
            ".csv": CsvParser(),
            ".tsv": CsvParser(),
            ".json": JsonParser(),
            ".jsonl": JsonlParser(),
            ".html": HtmlParser(),
            ".htm": HtmlParser(),
            ".ndjson": JsonlParser(),
            ".xlsx": XlsxParser(),
            ".xls": XlsParser(),
            ".xml": XmlParser(),
            ".epub": EpubParser(),
            ".pptx": PptxParser(),
            ".ppt": PptParser(),
            ".odt": OdtParser(),
        }
        self._fallback_parser = TxtParser()

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._parsers))

    def get_parser(self, file_path: str | Path) -> BaseParser:
        extension = Path(file_path).suffix.lower()
        parser = self._parsers.get(extension)
        if parser is not None:
            return parser
        return self._fallback_parser

    def parse_file(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        conversion_error: str | None = None

        legacy_target = get_legacy_conversion_target(path.suffix.lower())
        if legacy_target:
            try:
                converted_document = self._parse_via_legacy_conversion(
                    path, legacy_target
                )
                converted_metadata = dict(converted_document.original_metadata)
                converted_metadata.update(
                    {
                        "parser_used": converted_document.source_type,
                        "requested_extension": path.suffix.lower() or None,
                        "conversion_attempted": True,
                        "conversion_applied": True,
                        "conversion_method": "libreoffice",
                        "conversion_target_extension": f".{legacy_target}",
                    }
                )
                return Document(
                    text=converted_document.text,
                    filename=path.name,
                    source_type=path.suffix.lstrip(".").lower()
                    or converted_document.source_type,
                    original_metadata=converted_metadata,
                )
            except ParserError as exc:
                conversion_error = str(exc)

        parser = self.get_parser(path)
        try:
            document = parser.parse(path)
        except ParserError as exc:
            if isinstance(parser, TxtParser):
                raise

            fallback_document = self._fallback_parser.parse(path)
            metadata = dict(fallback_document.original_metadata)
            metadata.update(
                {
                    "fallback_reason": str(exc),
                    "parser_used": self._fallback_parser.source_type,
                    "requested_extension": path.suffix.lower() or None,
                }
            )
            return Document(
                text=fallback_document.text,
                filename=fallback_document.filename,
                source_type=fallback_document.source_type,
                original_metadata=metadata,
            )

        metadata = dict(document.original_metadata)
        metadata.setdefault("parser_used", parser.source_type)
        metadata.setdefault("requested_extension", path.suffix.lower() or None)
        if legacy_target:
            metadata.setdefault("conversion_attempted", True)
            metadata.setdefault("conversion_applied", False)
            if conversion_error:
                metadata.setdefault("conversion_error", conversion_error)
        return Document(
            text=document.text,
            filename=document.filename,
            source_type=document.source_type,
            original_metadata=metadata,
        )

    def _parse_via_legacy_conversion(
        self, source_path: Path, target_extension: str
    ) -> Document:
        """Convert a legacy file to a modern format and parse it with the converted parser."""

        with temporary_converted_file(
            source_path,
            target_extension=target_extension,
            temp_dir_prefix="rag-registry-convert-",
        ) as converted_path:
            converted_parser = self.get_parser(converted_path)
            return converted_parser.parse(converted_path)


def get_parser_for_path(file_path: str | Path) -> BaseParser:
    """Return the parser selected for the given path."""

    return ParserRegistry().get_parser(file_path)


def parse_file(file_path: str | Path) -> Document:
    """Parse a file path using the default parser registry."""

    return ParserRegistry().parse_file(file_path)
