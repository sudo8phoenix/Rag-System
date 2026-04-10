"""HTML parser with table-aware extraction and merged-cell support."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None
    Tag = Any  # type: ignore[assignment]

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional dependency
    trafilatura = None

from .base import BaseParser, ParserError


class HtmlParser(BaseParser):
    """Parse HTML documents while preserving table structure and order."""

    source_type = "html"
    supported_extensions = (".html", ".htm")

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"HTML file not found: {path}")
        if BeautifulSoup is None:
            raise ParserError("beautifulsoup4 is required for HTML parsing")

        raw_html = self._read_text(path)
        soup = BeautifulSoup(raw_html, "html.parser")

        if callable(getattr(soup, "__call__", None)):
            for node in soup(["script", "style", "noscript"]):
                node.decompose()

        used_trafilatura = False
        trafilatura_text = ""
        if trafilatura is not None:
            extracted = trafilatura.extract(
                raw_html, include_links=False, include_formatting=False
            )
            if extracted and extracted.strip():
                used_trafilatura = True
                trafilatura_text = extracted.strip()

        tables = (
            self._extract_tables(soup, raw_html) if hasattr(soup, "find_all") else []
        )

        body = getattr(soup, "body", None) or soup
        text_lines: list[str] = []
        if hasattr(body, "stripped_strings"):
            text_lines = [
                line.strip() for line in body.stripped_strings if line.strip()
            ]
        elif hasattr(soup, "select"):
            for selector in ("h1, h2, h3, h4, h5, h6", "p", "li"):
                for node in soup.select(selector):
                    text = node.get_text(" ", strip=True)
                    if text:
                        text_lines.append(text)

        if used_trafilatura and trafilatura_text:
            content_text = trafilatura_text
        else:
            content_text = "\n".join(text_lines).strip()

        title = (
            soup.title.string.strip()
            if getattr(soup, "title", None) and soup.title.string
            else None
        )
        table_text = self._render_tables(tables)
        full_text = content_text
        if title:
            full_text = f"{title}\n{full_text}".strip()
        if table_text:
            full_text = f"{content_text}\n\n{table_text}".strip()
            if title:
                full_text = f"{title}\n{full_text}".strip()

        if not full_text:
            raise ParserError(f"HTML file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "title": title,
            "used_trafilatura": used_trafilatura,
            "table_count": len(tables),
            "tables": [
                {
                    "table_index": table["table_index"],
                    "dom_index": table["dom_index"],
                    "start_offset": table["start_offset"],
                    "row_count": table["row_count"],
                    "column_count": table["column_count"],
                    "headers": table["headers"],
                }
                for table in tables
            ],
        }
        return self._build_document(
            text=full_text,
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _extract_tables(self, soup: Any, raw_html: str) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        for table_index, table_tag in enumerate(soup.find_all("table"), start=1):
            rows, headers = self._extract_table_grid(table_tag)
            if not rows:
                table_tag.decompose()
                continue

            table_markup = str(table_tag)
            start_offset = raw_html.find(table_markup)
            dom_index = sum(1 for _ in table_tag.find_all_previous())

            width = max((len(row) for row in rows), default=0)
            tables.append(
                {
                    "table_index": table_index,
                    "dom_index": dom_index,
                    "start_offset": None if start_offset < 0 else start_offset,
                    "rows": rows,
                    "headers": headers,
                    "row_count": len(rows),
                    "column_count": width,
                    "csv_lines": self._to_csv_lines(rows),
                }
            )
            table_tag.decompose()

        return tables

    def _extract_table_grid(self, table_tag: Tag) -> tuple[list[list[str]], list[str]]:
        grid: list[list[str]] = []
        span_map: dict[tuple[int, int], str] = {}
        header_row_index: int | None = None

        row_tags = table_tag.find_all("tr")
        for row_index, row_tag in enumerate(row_tags):
            row: list[str] = []
            col_index = 0

            while (row_index, col_index) in span_map:
                row.append(span_map[(row_index, col_index)])
                col_index += 1

            cell_tags = row_tag.find_all(["th", "td"], recursive=False)
            if not cell_tags:
                continue

            contains_header = any(cell.name == "th" for cell in cell_tags)
            if contains_header and header_row_index is None:
                header_row_index = len(grid)

            for cell_tag in cell_tags:
                while (row_index, col_index) in span_map:
                    row.append(span_map[(row_index, col_index)])
                    col_index += 1

                value = " ".join(cell_tag.stripped_strings).strip()
                rowspan = int(cell_tag.get("rowspan") or 1)
                colspan = int(cell_tag.get("colspan") or 1)

                for col_offset in range(colspan):
                    row.append(value)
                    for row_offset in range(1, rowspan):
                        span_map[(row_index + row_offset, col_index + col_offset)] = (
                            value
                        )
                col_index += colspan

            while (row_index, col_index) in span_map:
                row.append(span_map[(row_index, col_index)])
                col_index += 1

            if any(cell.strip() for cell in row):
                grid.append(row)

        if not grid:
            return [], []

        width = max(len(row) for row in grid)
        for row in grid:
            if len(row) < width:
                row.extend([""] * (width - len(row)))

        if header_row_index is not None and header_row_index < len(grid):
            headers = grid[header_row_index]
        else:
            headers = grid[0]

        return grid, headers

    def _to_csv_lines(self, rows: list[list[str]]) -> list[str]:
        stream = io.StringIO()
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow(row)
        return [line for line in stream.getvalue().splitlines() if line.strip()]

    def _render_tables(self, tables: list[dict[str, Any]]) -> str:
        output_lines: list[str] = []
        for table in tables:
            output_lines.append(
                "Table "
                f"{table['table_index']} (dom_index={table['dom_index']}, start_offset={table['start_offset']}):"
            )
            output_lines.extend(table["csv_lines"])
            output_lines.append("")
        return "\n".join(self._strip_outer_whitespace(output_lines))

    def _read_text(self, path: Path) -> str:
        payload = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ParserError(f"Unable to decode HTML file: {path}")
