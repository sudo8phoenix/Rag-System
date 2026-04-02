from __future__ import annotations

import pytest

from src.models.document import Document


def test_document_creation_success() -> None:
    document = Document(
        text="Hello world",
        filename="example.txt",
        source_type="txt",
        original_metadata={"author": "system", "lines": 1},
    )

    assert document.text == "Hello world"
    assert document.filename == "example.txt"
    assert document.source_type == "txt"
    assert document.original_metadata["author"] == "system"


def test_document_to_dict_serialization() -> None:
    document = Document(
        text="Sample",
        filename="notes.md",
        source_type="md",
        original_metadata={"headings": ["H1"]},
    )

    payload = document.to_dict()

    assert payload == {
        "text": "Sample",
        "filename": "notes.md",
        "source_type": "md",
        "original_metadata": {"headings": ["H1"]},
    }
    assert payload["original_metadata"] is not document.original_metadata


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("text", "", "text must be a non-empty string"),
        ("filename", "", "filename must be a non-empty string"),
        ("source_type", "", "source_type must be a non-empty string"),
    ],
)
def test_document_rejects_empty_required_fields(
    field_name: str,
    value: str,
    message: str,
) -> None:
    kwargs = {
        "text": "ok",
        "filename": "file.txt",
        "source_type": "txt",
        "original_metadata": {},
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        Document(**kwargs)


def test_document_rejects_non_dict_metadata() -> None:
    with pytest.raises(TypeError, match="original_metadata must be a dictionary"):
        Document(
            text="content",
            filename="file.txt",
            source_type="txt",
            original_metadata=["not", "a", "dict"],
        )
