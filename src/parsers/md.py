"""Markdown parser that preserves heading structure in metadata."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseParser, ParserError

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BOLD_RE = re.compile(r"(\*\*|__)(.+?)(\1)")
ITALIC_RE = re.compile(r"(\*|_)(.+?)(\1)")
BLOCKQUOTE_RE = re.compile(r"^>\s?")
LIST_RE = re.compile(r"^[-*+]\s+")


class MarkdownParser(BaseParser):
    """Parse markdown into readable text and heading metadata."""

    source_type = "md"
    supported_extensions = (".md", ".markdown", ".mdown")

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"Markdown file not found: {path}")

        text = path.read_text(encoding="utf-8")
        normalized = self._normalize_newlines(text)
        lines = normalized.split("\n")

        headings: list[dict[str, object]] = []
        output_lines: list[str] = []
        in_code_block = False

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()

            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                output_lines.append(line)
                continue

            heading_match = HEADING_RE.match(stripped)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = self._clean_inline_markdown(heading_match.group(2))
                headings.append({"level": level, "text": heading_text})
                output_lines.append(heading_text)
                continue

            cleaned = self._clean_inline_markdown(stripped)
            cleaned = BLOCKQUOTE_RE.sub("", cleaned)
            cleaned = LIST_RE.sub("", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                output_lines.append(cleaned)

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"Markdown file is empty: {path}")

        metadata = {
            "format": self.source_type,
            "heading_count": len(headings),
            "headings": headings,
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _clean_inline_markdown(self, text: str) -> str:
        text = LINK_RE.sub(r"\1", text)
        text = BOLD_RE.sub(r"\2", text)
        text = ITALIC_RE.sub(r"\2", text)
        text = INLINE_CODE_RE.sub(r"\1", text)
        text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
        return text.strip()
