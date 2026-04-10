"""JSONL/NDJSON parser that preserves source line numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseParser, ParserError


class JsonlParser(BaseParser):
    """Parse .jsonl/.ndjson files into flattened record text."""

    source_type = "jsonl"
    supported_extensions = (".jsonl", ".ndjson")

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"JSONL file not found: {path}")

        lines = self._read_lines(path)
        output_lines: list[str] = []
        records: list[dict[str, Any]] = []

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParserError(
                    f"Invalid JSONL at line {line_number} in {path}: {exc}"
                ) from exc

            flattened: list[str] = []
            self._flatten(payload, "root", flattened)
            if not flattened:
                flattened = ["root: null"]

            output_lines.append(f"Record {len(records) + 1} (line {line_number}):")
            output_lines.extend(f"  {item}" for item in flattened)
            output_lines.append("")

            records.append(
                {
                    "line_number": line_number,
                    "root_type": type(payload).__name__,
                    "leaf_count": len(flattened),
                }
            )

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"JSONL file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "record_count": len(records),
            "records": records,
            "line_numbers": [record["line_number"] for record in records],
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _read_lines(self, path: Path) -> list[str]:
        payload = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return payload.decode(encoding).splitlines()
            except UnicodeDecodeError:
                continue
        raise ParserError(f"Unable to decode JSONL file: {path}")

    def _flatten(self, value: Any, path: str, output: list[str]) -> None:
        if isinstance(value, dict):
            if not value:
                output.append(f"{path}: {{}}")
                return
            for key, item in value.items():
                self._flatten(item, f"{path}.{key}", output)
            return

        if isinstance(value, list):
            if not value:
                output.append(f"{path}: []")
                return
            for index, item in enumerate(value):
                self._flatten(item, f"{path}[{index}]", output)
            return

        output.append(f"{path}: {json.dumps(value, ensure_ascii=False)}")
