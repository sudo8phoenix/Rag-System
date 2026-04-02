from __future__ import annotations

from pathlib import Path

from src.parsers.md import MarkdownParser
from src.parsers.registry import ParserRegistry, get_parser_for_path, parse_file
from src.parsers.txt import TxtParser


def test_registry_routes_by_extension(tmp_path: Path) -> None:
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Title\n\nBody", encoding="utf-8")

    registry = ParserRegistry()

    assert isinstance(registry.get_parser(md_file), MarkdownParser)
    assert isinstance(get_parser_for_path(md_file), MarkdownParser)


def test_registry_falls_back_to_plain_text_on_parse_failure(tmp_path: Path) -> None:
    broken_json = tmp_path / "broken.json"
    broken_json.write_text("not valid json", encoding="utf-8")

    document = parse_file(broken_json)

    assert document.source_type == "txt"
    assert document.text == "not valid json"
    assert document.original_metadata["parser_used"] == "txt"
    assert "fallback_reason" in document.original_metadata


def test_registry_uses_plain_text_parser_for_unknown_extensions(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.unknown"
    file_path.write_text("plain content", encoding="utf-8")

    registry = ParserRegistry()
    parser = registry.get_parser(file_path)

    assert isinstance(parser, TxtParser)
    document = registry.parse_file(file_path)
    assert document.source_type == "txt"
    assert document.text == "plain content"