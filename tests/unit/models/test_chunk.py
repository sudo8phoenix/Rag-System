from __future__ import annotations

import pytest

from src.models.chunk import Chunk
from src.models.document import Document


def _make_document() -> Document:
    return Document(
        text="Document content",
        filename="report.pdf",
        source_type="pdf",
        original_metadata={"pages": 2},
    )


def test_chunk_creation_success() -> None:
    chunk = Chunk(
        text="Chunk 1 text",
        chunk_id="report.pdf:0",
        source_doc=_make_document(),
        chunk_index=0,
        strategy_used="paragraph",
        metadata={"page": 1, "section": "Intro"},
    )

    assert chunk.chunk_id == "report.pdf:0"
    assert chunk.chunk_index == 0
    assert chunk.strategy_used == "paragraph"
    assert chunk.metadata["page"] == 1


def test_chunk_to_dict_preserves_metadata_and_source_document() -> None:
    source = _make_document()
    chunk = Chunk(
        text="Chunk text",
        chunk_id="chunk-42",
        source_doc=source,
        chunk_index=42,
        strategy_used="line",
        metadata={"row": 10},
    )

    payload = chunk.to_dict()

    assert payload == {
        "text": "Chunk text",
        "chunk_id": "chunk-42",
        "source_doc": source.to_dict(),
        "chunk_index": 42,
        "strategy_used": "line",
        "metadata": {"row": 10},
    }
    assert payload["metadata"] is not chunk.metadata


def test_chunk_equality_checks_use_dataclass_value_semantics() -> None:
    source = _make_document()
    left = Chunk(
        text="Same",
        chunk_id="id-1",
        source_doc=source,
        chunk_index=1,
        strategy_used="character",
        metadata={"page": 1},
    )
    right = Chunk(
        text="Same",
        chunk_id="id-1",
        source_doc=source,
        chunk_index=1,
        strategy_used="character",
        metadata={"page": 1},
    )
    different = Chunk(
        text="Different",
        chunk_id="id-2",
        source_doc=source,
        chunk_index=2,
        strategy_used="character",
        metadata={"page": 2},
    )

    assert left == right
    assert left != different


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        (
            {
                "text": "",
                "chunk_id": "id",
                "chunk_index": 0,
                "strategy_used": "line",
                "source_doc": _make_document(),
            },
            ValueError,
            "text must be a non-empty string",
        ),
        (
            {
                "text": "ok",
                "chunk_id": "",
                "chunk_index": 0,
                "strategy_used": "line",
                "source_doc": _make_document(),
            },
            ValueError,
            "chunk_id must be a non-empty string",
        ),
        (
            {
                "text": "ok",
                "chunk_id": "id",
                "chunk_index": -1,
                "strategy_used": "line",
                "source_doc": _make_document(),
            },
            ValueError,
            "chunk_index must be greater than or equal to 0",
        ),
        (
            {
                "text": "ok",
                "chunk_id": "id",
                "chunk_index": 0,
                "strategy_used": "",
                "source_doc": _make_document(),
            },
            ValueError,
            "strategy_used must be a non-empty string",
        ),
        (
            {
                "text": "ok",
                "chunk_id": "id",
                "chunk_index": 0,
                "strategy_used": "line",
                "source_doc": "not-a-document",
            },
            TypeError,
            "source_doc must be an instance of Document",
        ),
        (
            {
                "text": "ok",
                "chunk_id": "id",
                "chunk_index": 0,
                "strategy_used": "line",
                "source_doc": _make_document(),
                "metadata": ["invalid"],
            },
            TypeError,
            "metadata must be a dictionary",
        ),
    ],
)
def test_chunk_validation_errors(kwargs: dict, exc_type: type[Exception], message: str) -> None:
    with pytest.raises(exc_type, match=message):
        Chunk(**kwargs)
