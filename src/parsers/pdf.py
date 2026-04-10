"""PDF parser with a PyMuPDF fast path and a minimal text fallback."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

from .base import BaseParser, ParserError

OBJECT_RE = re.compile(r"(?ms)^(\d+)\s+(\d+)\s+obj\s*(.*?)\s*endobj\s*$")
STREAM_RE = re.compile(r"(?ms)stream\s*(.*?)\s*endstream")
LITERAL_STRING_RE = re.compile(r"\((?:\\.|[^\\)])*\)")


class PdfParser(BaseParser):
    """Parse text-based PDF documents into normalized text."""

    source_type = "pdf"
    supported_extensions = (".pdf",)

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"PDF file not found: {path}")

        parsed_document = None
        if fitz is not None:
            try:
                parsed_document = self._parse_with_pymupdf(path)
            except Exception:
                parsed_document = None

        if parsed_document is None:
            raw_bytes = path.read_bytes()
            text, metadata = self._parse_with_fallback(raw_bytes)
            normalized = self._normalize_newlines(text).strip()
            if not normalized:
                raise ParserError(f"Unable to extract text from PDF: {path}")

            metadata.update({"format": self.source_type})
            parsed_document = self._build_document(
                text=normalized,
                file_path=path,
                source_type=self.source_type,
                metadata=metadata,
            )

        return self._enrich_with_tables(parsed_document, path)

    def _enrich_with_tables(self, document, path: Path):
        extracted_tables = self._extract_tables(path)
        if not extracted_tables:
            return document

        table_text = self._render_tables(extracted_tables)
        metadata = dict(document.original_metadata)
        metadata["table_count"] = len(extracted_tables)
        metadata["table_extraction_method"] = "pdfplumber"
        metadata["tables"] = [
            {
                "page": table["page"],
                "table_index": table["table_index"],
                "row_count": table["row_count"],
                "column_count": table["column_count"],
                "bbox": table["bbox"],
            }
            for table in extracted_tables
        ]

        full_text = document.text
        if table_text:
            full_text = f"{document.text}\n\n{table_text}".strip()

        return self._build_document(
            text=full_text,
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _parse_with_pymupdf(self, path: Path):
        document = fitz.open(path)  # type: ignore[union-attr]
        pages: list[str] = []
        page_details: list[dict[str, object]] = []

        for index, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append(page_text)
                page_details.append({"page": index, "text_length": len(page_text)})

        if not pages:
            raise ParserError(f"Unable to extract text from PDF: {path}")

        metadata = {
            "format": self.source_type,
            "page_count": len(pages),
            "pages": page_details,
            "extraction_method": "pymupdf",
        }
        return self._build_document(
            text="\n\n".join(pages),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _extract_tables(self, path: Path) -> list[dict[str, Any]]:
        if pdfplumber is None:
            return []

        tables: list[dict[str, Any]] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    for table_index, table in enumerate(page.find_tables(), start=1):
                        rows = table.extract() or []
                        normalized_rows = self._normalize_table_rows(rows)
                        if not normalized_rows:
                            continue

                        bbox = table.bbox if hasattr(table, "bbox") else None
                        tables.append(
                            {
                                "page": page_number,
                                "table_index": table_index,
                                "rows": normalized_rows,
                                "csv_lines": self._to_csv_lines(normalized_rows),
                                "row_count": len(normalized_rows),
                                "column_count": max(
                                    (len(row) for row in normalized_rows), default=0
                                ),
                                "bbox": self._format_bbox(bbox),
                            }
                        )
        except Exception:
            return []

        return tables

    def _normalize_table_rows(self, rows: list[list[object]]) -> list[list[str]]:
        normalized_rows: list[list[str]] = []
        for row in rows:
            cleaned = [("" if cell is None else str(cell).strip()) for cell in row]
            if any(cleaned):
                normalized_rows.append(cleaned)

        if not normalized_rows:
            return []

        width = max(len(row) for row in normalized_rows)
        for row in normalized_rows:
            if len(row) < width:
                row.extend([""] * (width - len(row)))
        return normalized_rows

    def _to_csv_lines(self, rows: list[list[str]]) -> list[str]:
        stream = io.StringIO()
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow(row)
        return [line for line in stream.getvalue().splitlines() if line.strip()]

    def _format_bbox(self, bbox: object) -> dict[str, float] | None:
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            return None
        x0, top, x1, bottom = bbox
        return {
            "x0": float(x0),
            "top": float(top),
            "x1": float(x1),
            "bottom": float(bottom),
        }

    def _render_tables(self, tables: list[dict[str, Any]]) -> str:
        output_lines: list[str] = []
        for table in tables:
            page_number = table["page"]
            table_index = table["table_index"]
            bbox = table.get("bbox")
            if bbox:
                output_lines.append(
                    "Table "
                    f"{table_index} (page {page_number}, bbox={bbox['x0']:.1f},{bbox['top']:.1f},{bbox['x1']:.1f},{bbox['bottom']:.1f}):"
                )
            else:
                output_lines.append(f"Table {table_index} (page {page_number}):")
            output_lines.extend(table["csv_lines"])
            output_lines.append("")

        return "\n".join(self._strip_outer_whitespace(output_lines))

    def _parse_with_fallback(self, raw_bytes: bytes) -> tuple[str, dict[str, object]]:
        text = raw_bytes.decode("latin-1", errors="ignore")
        object_map = self._collect_objects(text)
        page_objects = [
            (int(number), body)
            for number, body in object_map.items()
            if "/Type /Page" in body
        ]
        page_objects.sort(key=lambda item: item[0])

        page_texts: list[str] = []
        page_details: list[dict[str, object]] = []

        for _, body in page_objects:
            content_refs = self._extract_content_refs(body)
            extracted_parts: list[str] = []
            for ref in content_refs:
                stream_body = object_map.get(ref)
                if not stream_body:
                    continue
                stream_text = self._extract_stream_text(stream_body)
                extracted_parts.extend(self._extract_pdf_strings(stream_text))

            if extracted_parts:
                page_text = "\n".join(
                    part.strip() for part in extracted_parts if part.strip()
                )
                if page_text:
                    page_texts.append(page_text)
                    page_details.append(
                        {"page": len(page_details) + 1, "text_length": len(page_text)}
                    )

        if not page_texts:
            fallback_strings = self._extract_pdf_strings(text)
            if fallback_strings:
                page_texts = ["\n".join(fallback_strings)]
                page_details = [{"page": 1, "text_length": len(page_texts[0])}]

        metadata = {
            "page_count": len(page_texts),
            "pages": page_details,
            "extraction_method": "fallback",
        }
        return "\n\n".join(page_texts), metadata

    def _collect_objects(self, text: str) -> dict[int, str]:
        objects: dict[int, str] = {}
        for match in OBJECT_RE.finditer(text):
            number = int(match.group(1))
            body = match.group(3)
            objects[number] = body
        return objects

    def _extract_content_refs(self, body: str) -> list[int]:
        refs = [int(ref) for ref in re.findall(r"/Contents\s+(\d+)\s+\d+\s+R", body)]
        if refs:
            return refs

        array_match = re.search(r"/Contents\s*\[(.*?)\]", body, re.S)
        if not array_match:
            return []

        return [
            int(ref) for ref in re.findall(r"(\d+)\s+\d+\s+R", array_match.group(1))
        ]

    def _extract_stream_text(self, body: str) -> str:
        match = STREAM_RE.search(body)
        if not match:
            return body
        return match.group(1)

    def _extract_pdf_strings(self, text: str) -> list[str]:
        strings: list[str] = []

        for match in re.finditer(r"\[(.*?)\]\s*TJ", text, re.S):
            array_text = match.group(1)
            for literal in LITERAL_STRING_RE.findall(array_text):
                strings.append(self._unescape_pdf_string(literal[1:-1]))

        for match in re.finditer(r"\((?:\\.|[^\\)])*\)\s*Tj", text):
            literal = match.group(0).split("Tj", 1)[0].strip()
            strings.append(self._unescape_pdf_string(literal[1:-1]))

        if strings:
            return strings

        return [
            self._unescape_pdf_string(literal[1:-1])
            for literal in LITERAL_STRING_RE.findall(text)
        ]

    def _unescape_pdf_string(self, value: str) -> str:
        replacements = {
            r"\\": "\\",
            r"\(": "(",
            r"\)": ")",
            r"\n": "\n",
            r"\r": "\r",
            r"\t": "\t",
            r"\b": "\b",
            r"\f": "\f",
        }
        result = value
        for source, target in replacements.items():
            result = result.replace(source, target)
        return result
