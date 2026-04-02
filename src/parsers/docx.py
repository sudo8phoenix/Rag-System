"""DOCX parser using the OpenXML package structure."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .base import BaseParser, ParserError


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxParser(BaseParser):
    """Parse DOCX files without requiring python-docx at runtime."""

    source_type = "docx"
    supported_extensions = (".docx",)

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"DOCX file not found: {path}")

        try:
            with zipfile.ZipFile(path) as archive:
                document_xml = archive.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile, FileNotFoundError) as exc:
            raise ParserError(f"Unable to read DOCX document: {path}") from exc

        root = ET.fromstring(document_xml)
        body = root.find(f"{W_NS}body")
        if body is None:
            raise ParserError(f"DOCX document body missing: {path}")

        output_lines: list[str] = []
        headings: list[dict[str, object]] = []
        tables: list[dict[str, object]] = []
        paragraph_count = 0

        for child in body:
            tag = child.tag
            if tag == f"{W_NS}p":
                paragraph_count += 1
                paragraph_text = self._extract_paragraph_text(child)
                if not paragraph_text:
                    continue
                style = self._extract_paragraph_style(child)
                if style and style.startswith("Heading"):
                    level = self._extract_heading_level(style)
                    if level is not None:
                        headings.append({"level": level, "text": paragraph_text})
                    output_lines.append(paragraph_text)
                else:
                    output_lines.append(paragraph_text)
            elif tag == f"{W_NS}tbl":
                table_lines = self._extract_table_lines(child)
                if table_lines:
                    tables.append({"rows": table_lines, "row_count": len(table_lines)})
                    output_lines.append(f"Table {len(tables)}:")
                    output_lines.extend(table_lines)

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"DOCX file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "paragraph_count": paragraph_count,
            "table_count": len(tables),
            "headings": headings,
            "tables": tables,
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _extract_paragraph_text(self, paragraph: ET.Element) -> str:
        pieces = [node.text or "" for node in paragraph.findall(f".//{W_NS}t")]
        return "".join(pieces).strip()

    def _extract_paragraph_style(self, paragraph: ET.Element) -> str | None:
        style = paragraph.find(f"./{W_NS}pPr/{W_NS}pStyle")
        if style is None:
            return None
        return style.attrib.get(f"{W_NS}val") or style.attrib.get("val")

    def _extract_heading_level(self, style: str) -> int | None:
        match = re.search(r"Heading\s*(\d+)", style)
        if not match:
            return None
        return int(match.group(1))

    def _extract_table_lines(self, table: ET.Element) -> list[str]:
        table_lines: list[str] = []
        for row in table.findall(f"./{W_NS}tr"):
            cell_texts = []
            for cell in row.findall(f"./{W_NS}tc"):
                pieces = [node.text or "" for node in cell.findall(f".//{W_NS}t")]
                cell_text = " ".join(piece.strip() for piece in pieces if piece.strip())
                cell_texts.append(cell_text)
            row_text = " | ".join(cell_texts).strip()
            if row_text:
                table_lines.append(row_text)
        return table_lines