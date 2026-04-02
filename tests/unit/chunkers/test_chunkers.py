from __future__ import annotations

from src.chunkers import (
    CharacterBasedChunker,
    ChunkingRegistry,
    LineBasedChunker,
    ParagraphBasedChunker,
)
from src.config.settings import ChunkingConfig, FormatChunkingConfig
from src.models.document import Document


def _make_document(text: str, source_type: str = "txt", filename: str = "sample.txt") -> Document:
    return Document(
        text=text,
        filename=filename,
        source_type=source_type,
        original_metadata={"format": source_type},
    )


def test_line_chunker_respects_line_boundaries_and_overlap() -> None:
    document = _make_document("line1\nline2\nline3\nline4\nline5\nline6")
    config = ChunkingConfig(strategy="line", chunk_size=3, chunk_overlap=1)

    chunks = LineBasedChunker().chunk(document, config)

    assert [chunk.text for chunk in chunks] == [
        "line1\nline2\nline3",
        "line3\nline4\nline5",
        "line5\nline6",
    ]
    assert chunks[0].metadata["line_start"] == 1
    assert chunks[0].metadata["line_end"] == 3
    assert chunks[1].metadata["line_start"] == 3
    assert chunks[1].metadata["line_end"] == 5


def test_character_chunker_respects_word_boundaries() -> None:
    document = _make_document("alpha beta gamma delta epsilon")
    config = ChunkingConfig(strategy="character", chunk_size=11, chunk_overlap=0)

    chunks = CharacterBasedChunker().chunk(document, config)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta", "epsilon"]
    assert chunks[0].metadata["char_start"] == 1
    assert chunks[0].metadata["char_end"] <= 11
    assert chunks[1].metadata["char_start"] > chunks[0].metadata["char_start"]


def test_paragraph_chunker_merges_small_paragraphs_and_uses_overlap_context() -> None:
    document = _make_document(
        "Intro.\n\nShort.\n\nThis is a longer paragraph that should trigger the first chunk boundary.\n\nFinal note.",
    )
    config = ChunkingConfig(strategy="paragraph", chunk_size=70, chunk_overlap=10, min_chunk_size=20)

    chunks = ParagraphBasedChunker().chunk(document, config)

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("Intro.")
    assert "Short." in chunks[0].text
    assert chunks[1].text.startswith("This is a longer paragraph")
    assert chunks[1].metadata["overlap_paragraphs"] == 1


def test_chunking_registry_applies_per_format_overrides() -> None:
    config = ChunkingConfig(
        strategy="paragraph",
        chunk_size=50,
        chunk_overlap=5,
        per_format={
            "txt": FormatChunkingConfig(strategy="line", chunk_size=2, chunk_overlap=1),
        },
    )
    registry = ChunkingRegistry(config)
    document = _make_document("a\nb\nc\nd", filename="notes.txt")

    chunker = registry.get_chunker(document)
    chunks = registry.chunk_document(document)

    assert isinstance(chunker, LineBasedChunker)
    assert [chunk.text for chunk in chunks] == ["a\nb", "b\nc", "c\nd"]
    assert chunks[0].metadata["chunk_size"] == 2
    assert chunks[0].metadata["chunk_overlap"] == 1


def test_chunking_registry_falls_back_to_global_strategy_for_unsupported_override() -> None:
    config = ChunkingConfig(
        strategy="paragraph",
        per_format={
            "json": FormatChunkingConfig(strategy="array_item", chunk_size=1),
        },
    )
    registry = ChunkingRegistry(config)
    document = _make_document("one\n\ntwo", source_type="json", filename="payload.json")

    chunker = registry.get_chunker(document)

    assert isinstance(chunker, ParagraphBasedChunker)