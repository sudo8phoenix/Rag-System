"""EPUB parser that extracts chapter-level text with chapter metadata."""

from __future__ import annotations

from pathlib import Path

try:
    from ebooklib import ITEM_DOCUMENT, epub
except ImportError:  # pragma: no cover - optional dependency
    ITEM_DOCUMENT = None
    epub = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

from .base import BaseParser, ParserError


class EpubParser(BaseParser):
    """Parse .epub books into chapter-separated text."""

    source_type = "epub"
    supported_extensions = (".epub",)

    def parse(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"EPUB file not found: {path}")
        if epub is None or ITEM_DOCUMENT is None:
            raise ParserError("ebooklib is required to parse .epub files")

        try:
            book = epub.read_epub(str(path))
        except Exception as exc:  # pragma: no cover - backend exceptions vary
            raise ParserError(f"Unable to read EPUB file {path}: {exc}") from exc

        output_lines: list[str] = []
        chapters: list[dict[str, object]] = []

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            raw_content = item.get_body_content().decode("utf-8", errors="ignore")
            chapter_text, title = self._extract_chapter_text(raw_content, fallback_title=item.get_name())
            if not chapter_text:
                continue

            chapter_index = len(chapters) + 1
            output_lines.append(f"Chapter {chapter_index}: {title}")
            output_lines.append(chapter_text)
            output_lines.append("")

            chapters.append(
                {
                    "index": chapter_index,
                    "title": title,
                    "item_id": item.get_id(),
                    "text_length": len(chapter_text),
                }
            )

        output_lines = self._strip_outer_whitespace(output_lines)
        if not output_lines:
            raise ParserError(f"EPUB file has no readable chapter content: {path}")

        metadata = {
            "format": self.source_type,
            "chapter_count": len(chapters),
            "chapters": chapters,
        }
        return self._build_document(
            text="\n".join(output_lines),
            file_path=path,
            source_type=self.source_type,
            metadata=metadata,
        )

    def _extract_chapter_text(self, html: str, *, fallback_title: str) -> tuple[str, str]:
        if BeautifulSoup is None:
            stripped = " ".join(html.replace("<", " ").replace(">", " ").split())
            return stripped, fallback_title

        soup = BeautifulSoup(html, "html.parser")
        title_node = soup.find(["h1", "h2", "h3", "title"])
        title = title_node.get_text(" ", strip=True) if title_node else fallback_title
        chapter_text = "\n".join(part.strip() for part in soup.stripped_strings if part.strip())
        return chapter_text, title
