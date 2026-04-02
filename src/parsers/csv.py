"""CSV parser that preserves headers and row metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from .base import BaseParser, ParserError


class CsvParser(BaseParser):
    """Parse CSV/TSV-like tabular text into readable documents."""

    source_type = "csv"
    supported_extensions = (".csv", ".tsv")

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"CSV file not found: {path}")

        raw_text = self._read_text(path)
        normalized = self._normalize_newlines(raw_text).strip()
        if not normalized:
            raise ParserError(f"CSV file is empty: {path}")

        try:
            dialect = csv.Sniffer().sniff(normalized[:4096], delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        rows = self._parse_rows(normalized, delimiter)
        if not rows:
            raise ParserError(f"CSV file has no rows: {path}")

        header = rows[0]
        data_rows = rows[1:]
        output_lines = [" | ".join(header)]
        for row in data_rows:
            output_lines.append(" | ".join(row))

        metadata = {
            "format": self.source_type,
            "delimiter": delimiter,
            "column_names": header,
            "row_count": len(data_rows),
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _read_text(self, path: Path) -> str:
        raw_bytes = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ParserError(f"Unable to decode CSV file: {path}")

    def _parse_rows(self, text: str, delimiter: str) -> list[list[str]]:
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        rows = []
        for row in reader:
            cleaned = [cell.strip() for cell in row]
            if any(cleaned):
                rows.append(cleaned)
        return rows