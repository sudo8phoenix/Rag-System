"""JSON parser that flattens nested structures into readable text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseParser, ParserError


class JsonParser(BaseParser):
    """Parse JSON into normalized text while preserving structure metadata."""

    source_type = "json"
    supported_extensions = (".json",)

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"JSON file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ParserError(f"Invalid JSON in {path}: {exc}") from exc

        lines: list[str] = []
        paths: list[str] = []
        self._flatten(payload, "root", lines, paths)
        normalized = "\n".join(line for line in lines if line.strip()).strip()
        if not normalized:
            raise ParserError(f"JSON file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "root_type": type(payload).__name__,
            "top_level_keys": list(payload.keys()) if isinstance(payload, dict) else [],
            "leaf_count": len(paths),
            "paths": paths,
        }
        return self._build_document(
            text=normalized,
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _flatten(
        self,
        value: Any,
        path: str,
        lines: list[str],
        paths: list[str],
    ) -> None:
        if isinstance(value, dict):
            if not value:
                lines.append(f"{path}: {{}}")
                paths.append(path)
                return

            for key, item in value.items():
                child_path = f"{path}.{key}"
                self._flatten(item, child_path, lines, paths)
            return

        if isinstance(value, list):
            if not value:
                lines.append(f"{path}: []")
                paths.append(path)
                return

            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                self._flatten(item, child_path, lines, paths)
            return

        lines.append(f"{path}: {json.dumps(value, ensure_ascii=False)}")
        paths.append(path)
