"""XML parser that extracts hierarchical text and attribute metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from lxml import etree
except ImportError:  # pragma: no cover - optional dependency
    etree = None

from .base import BaseParser, ParserError


class XmlParser(BaseParser):
    """Parse XML files while preserving tag hierarchy and attributes."""

    source_type = "xml"
    supported_extensions = (".xml",)

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"XML file not found: {path}")
        if etree is None:
            raise ParserError("lxml is required to parse .xml files")

        try:
            parser = etree.XMLParser(recover=True, remove_comments=True)
            tree = etree.parse(str(path), parser)
            root = tree.getroot()
        except Exception as exc:  # pragma: no cover - backend exceptions vary
            raise ParserError(f"Unable to parse XML file {path}: {exc}") from exc

        output_lines: list[str] = []
        counters = {"elements": 0, "attributes": 0}
        self._walk(root, f"/{self._tag_name(root.tag)}", output_lines, counters)

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"XML file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "root_tag": self._tag_name(root.tag),
            "element_count": counters["elements"],
            "attribute_count": counters["attributes"],
            "namespaces": {
                key or "default": value for key, value in (root.nsmap or {}).items()
            },
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _walk(
        self,
        element: Any,
        element_path: str,
        output_lines: list[str],
        counters: dict[str, int],
    ) -> None:
        counters["elements"] += 1

        for attr_name, attr_value in element.attrib.items():
            counters["attributes"] += 1
            output_lines.append(
                f"{element_path}@{self._tag_name(attr_name)}: {attr_value}"
            )

        if element.text and element.text.strip():
            output_lines.append(f"{element_path}: {element.text.strip()}")

        sibling_counts: dict[str, int] = {}
        for child in element:
            child_name = self._tag_name(child.tag)
            sibling_counts[child_name] = sibling_counts.get(child_name, 0) + 1
            child_path = f"{element_path}/{child_name}[{sibling_counts[child_name]}]"
            self._walk(child, child_path, output_lines, counters)

            if child.tail and child.tail.strip():
                output_lines.append(f"{element_path}#tail: {child.tail.strip()}")

    def _tag_name(self, tag: Any) -> str:
        if etree is None:
            return str(tag)
        try:
            return etree.QName(tag).localname
        except Exception:
            return str(tag)
