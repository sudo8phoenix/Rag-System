"""Document parser package and registry."""

from .base import BaseParser, ParserError
from .csv import CsvParser
from .doc import DocParser
from .docx import DocxParser
from .epub import EpubParser
from .html import HtmlParser
from .json import JsonParser
from .jsonl import JsonlParser
from .md import MarkdownParser
from .pdf import PdfParser
from .ppt import PptParser
from .pptx import PptxParser
from .registry import ParserRegistry, get_parser_for_path, parse_file
from .txt import TxtParser
from .xls import XlsParser
from .xlsx import XlsxParser
from .xml import XmlParser
from .odt import OdtParser

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
    "JsonlParser",
    "XlsxParser",
    "XlsParser",
    "XmlParser",
    "HtmlParser",
    "EpubParser",
    "PptxParser",
    "PptParser",
    "OdtParser",
    "DocParser",
    "get_parser_for_path",
    "parse_file",
]
