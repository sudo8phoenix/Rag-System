"""Legacy XLS parser with xlrd primary path and LibreOffice fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    import xlrd
except ImportError:  # pragma: no cover - optional dependency
    xlrd = None

from .base import BaseParser, ParserError
from .conversion import convert_file_with_libreoffice
from .xlsx import XlsxParser


class XlsParser(BaseParser):
    """Parse .xls workbooks via xlrd with LibreOffice conversion fallback."""

    source_type = "xls"
    supported_extensions = (".xls",)

    def __init__(self) -> None:
        self._xlsx_parser = XlsxParser()

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"XLS file not found: {path}")

        try:
            text, metadata = self._parse_with_xlrd(path)
            extraction_method = "xlrd"
        except Exception as exc:
            text, metadata = self._parse_via_conversion(path)
            extraction_method = "libreoffice-to-xlsx"
            metadata["xlrd_error"] = str(exc)

        metadata.update({"format": self.source_type, "extraction_method": extraction_method})
        return self._build_document(
            text=text,
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _parse_with_xlrd(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        if xlrd is None:
            raise ParserError("xlrd is required to parse .xls files")

        try:
            workbook = xlrd.open_workbook(str(file_path), on_demand=True)
        except Exception as exc:  # pragma: no cover - backend exceptions vary
            raise ParserError(f"Unable to open XLS workbook {file_path}: {exc}") from exc

        output_lines: list[str] = []
        sheet_metadata: list[dict[str, Any]] = []
        for sheet in workbook.sheets():
            rows: list[list[str]] = []
            for row_index in range(sheet.nrows):
                row_values = [self._format_cell(sheet.cell_value(row_index, col)) for col in range(sheet.ncols)]
                if any(value.strip() for value in row_values):
                    rows.append(row_values)

            if not rows:
                continue

            header = rows[0]
            data_rows = rows[1:]
            output_lines.append(f"Sheet: {sheet.name}")
            output_lines.append(" | ".join(header))
            output_lines.extend(" | ".join(row) for row in data_rows)
            output_lines.append("")

            sheet_metadata.append(
                {
                    "name": sheet.name,
                    "range": f"R1C1:R{max(sheet.nrows, 1)}C{max(sheet.ncols, 1)}",
                    "column_count": len(header),
                    "row_count": len(data_rows),
                }
            )

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"XLS file is empty: {file_path}")

        metadata = {
            "sheet_count": len(sheet_metadata),
            "sheet_names": [sheet["name"] for sheet in sheet_metadata],
            "sheets": sheet_metadata,
        }
        return "\n".join(output_lines), metadata

    def _parse_via_conversion(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="rag-xls-") as temp_dir:
            converted = convert_file_with_libreoffice(
                file_path,
                output_dir=Path(temp_dir),
                target_extension="xlsx",
            )
            converted_document = self._xlsx_parser.parse(converted)

        metadata = dict(converted_document.original_metadata)
        metadata["converted_from"] = ".xls"
        return converted_document.text, metadata

    def _format_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    return value.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            return value.decode("utf-8", errors="ignore").strip()
        return str(value).strip()
