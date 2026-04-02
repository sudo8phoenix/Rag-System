"""Document parser package and registry."""

from .base import BaseParser, ParserError
from .csv import CsvParser
from .docx import DocxParser
from .json import JsonParser
from .md import MarkdownParser
from .pdf import PdfParser
from .registry import ParserRegistry, get_parser_for_path, parse_file
from .txt import TxtParser

__all__ = [
    "BaseParser",
    "ParserError",
    "ParserRegistry",
    "TxtParser",
    "MarkdownParser",
    "DocxParser",
    "PdfParser",
    "CsvParser",
    "JsonParser",
    "get_parser_for_path",
    "parse_file",
]