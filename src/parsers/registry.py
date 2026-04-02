"""Parser registry and dispatcher."""

from __future__ import annotations

from pathlib import Path

from src.models.document import Document

from .base import BaseParser, ParserError
from .csv import CsvParser
from .docx import DocxParser
from .json import JsonParser
from .md import MarkdownParser
from .pdf import PdfParser
from .txt import TxtParser


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
            ".pdf": PdfParser(),
            ".csv": CsvParser(),
            ".tsv": CsvParser(),
            ".json": JsonParser(),
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
        return Document(
            text=document.text,
            filename=document.filename,
            source_type=document.source_type,
            original_metadata=metadata,
        )


def get_parser_for_path(file_path: str | Path) -> BaseParser:
    """Return the parser selected for the given path."""

    return ParserRegistry().get_parser(file_path)


def parse_file(file_path: str | Path) -> Document:
    """Parse a file path using the default parser registry."""

    return ParserRegistry().parse_file(file_path)