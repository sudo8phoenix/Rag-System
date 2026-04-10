"""Tag-based chunker for XML/HTML-like documents."""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

from src.config.settings import ChunkingConfig
from src.models.chunk import Chunk
from src.models.document import Document

from .base import BaseChunker


class TagBasedChunker(BaseChunker):
    """Split documents at configurable target tags and preserve attributes."""

    strategy_name = "tag_based"

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        target_tags = [
            tag.lower() for tag in (config.target_tags or ["record", "article", "item"])
        ]
        chunks = self._chunk_with_xml(document, config, target_tags)
        if chunks:
            return chunks
        return self._chunk_with_regex(document, config, target_tags)

    def _chunk_with_xml(
        self,
        document: Document,
        config: ChunkingConfig,
        target_tags: list[str],
    ) -> list[Chunk]:
        try:
            root = ET.fromstring(document.text)
        except ET.ParseError:
            return []

        chunks: list[Chunk] = []
        for element in root.iter():
            tag_name = self._normalize_tag(element.tag)
            if tag_name not in target_tags:
                continue

            element_text = ET.tostring(
                element, encoding="unicode", method="xml"
            ).strip()
            if not element_text:
                continue

            metadata = self._base_metadata(
                document, config, self.strategy_name, len(chunks)
            )
            metadata.update(
                {
                    "tag": tag_name,
                    "attributes": dict(element.attrib),
                    "path": self._element_path(element),
                }
            )
            chunks.append(
                self._build_chunk(
                    text=element_text,
                    document=document,
                    chunk_index=len(chunks),
                    strategy_used=self.strategy_name,
                    metadata=metadata,
                )
            )

        return chunks

    def _chunk_with_regex(
        self,
        document: Document,
        config: ChunkingConfig,
        target_tags: list[str],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for tag in target_tags:
            pattern = re.compile(
                rf"<(?P<tag>{tag})(?P<attrs>\s+[^>]*)?>(?P<body>.*?)</{tag}>",
                flags=re.IGNORECASE | re.DOTALL,
            )
            for match in pattern.finditer(document.text):
                full_text = match.group(0).strip()
                attrs = self._parse_attrs(match.group("attrs") or "")
                metadata = self._base_metadata(
                    document, config, self.strategy_name, len(chunks)
                )
                metadata.update(
                    {
                        "tag": tag,
                        "attributes": attrs,
                        "path": f"/{tag}[{len(chunks) + 1}]",
                    }
                )
                chunks.append(
                    self._build_chunk(
                        text=full_text,
                        document=document,
                        chunk_index=len(chunks),
                        strategy_used=self.strategy_name,
                        metadata=metadata,
                    )
                )
        return chunks

    def _normalize_tag(self, tag: str) -> str:
        if "}" in tag:
            return tag.split("}", maxsplit=1)[1].lower()
        return tag.lower()

    def _element_path(self, element: ET.Element) -> str:
        tag = self._normalize_tag(element.tag)
        return f"/{tag}"

    def _parse_attrs(self, attrs_raw: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for key, value in re.findall(r'(\w+)\s*=\s*"(.*?)"', attrs_raw):
            attrs[key] = value
        return attrs
