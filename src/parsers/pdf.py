"""PDF parser with a PyMuPDF fast path and a minimal text fallback."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

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

        if fitz is not None:
            try:
                return self._parse_with_pymupdf(path)
            except Exception:
                pass

        raw_bytes = path.read_bytes()
        text, metadata = self._parse_with_fallback(raw_bytes)
        normalized = self._normalize_newlines(text).strip()
        if not normalized:
            raise ParserError(f"Unable to extract text from PDF: {path}")

        metadata.update({"format": self.source_type})
        return self._build_document(
            text=normalized,
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
                page_text = "\n".join(part.strip() for part in extracted_parts if part.strip())
                if page_text:
                    page_texts.append(page_text)
                    page_details.append({"page": len(page_details) + 1, "text_length": len(page_text)})

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

        return [int(ref) for ref in re.findall(r"(\d+)\s+\d+\s+R", array_match.group(1))]

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

        return [self._unescape_pdf_string(literal[1:-1]) for literal in LITERAL_STRING_RE.findall(text)]

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