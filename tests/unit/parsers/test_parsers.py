from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.parsers.csv import CsvParser
from src.parsers.docx import DocxParser
from src.parsers.json import JsonParser
from src.parsers.md import MarkdownParser
from src.parsers.pdf import PdfParser
from src.parsers.txt import TxtParser


def test_txt_parser_detects_encoding_and_normalizes_line_endings(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes("Café\r\nLine two\rLine three".encode("cp1252"))

    document = TxtParser().parse(file_path)

    assert document.source_type == "txt"
    assert document.text == "Café\nLine two\nLine three"
    assert document.original_metadata["format"] == "txt"
    assert document.original_metadata["line_count"] == 3


def test_markdown_parser_extracts_headings_and_content(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        """
        # Title

        Intro paragraph with **bold** and [link](https://example.com).

        ## Details

        ```python
        print("hello")
        ```
        """.strip(),
        encoding="utf-8",
    )

    document = MarkdownParser().parse(file_path)

    assert document.source_type == "md"
    assert document.text.splitlines()[0] == "Title"
    assert "Intro paragraph with bold and link." in document.text
    assert 'print("hello")' in document.text
    assert document.original_metadata["heading_count"] == 2
    assert document.original_metadata["headings"][0]["text"] == "Title"


def test_docx_parser_extracts_paragraphs_and_tables(tmp_path: Path) -> None:
    file_path = tmp_path / "report.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p>
          <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
          <w:r><w:t>Report Title</w:t></w:r>
        </w:p>
        <w:p>
          <w:r><w:t>First paragraph.</w:t></w:r>
        </w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Header A</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Header B</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Value 1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Value 2</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>
    """.strip()

    with zipfile.ZipFile(file_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    document = DocxParser().parse(file_path)

    assert document.source_type == "docx"
    assert document.text.splitlines()[0] == "Report Title"
    assert "First paragraph." in document.text
    assert "Table 1:" in document.text
    assert "Header A | Header B" in document.text
    assert document.original_metadata["table_count"] == 1
    assert document.original_metadata["headings"][0]["level"] == 1


def _build_sample_pdf(page_texts: list[str]) -> bytes:
    objects: list[bytes] = []
    contents_object_ids: list[int] = []

    object_number = 1
    catalog_number = object_number
    object_number += 1
    pages_number = object_number
    object_number += 1

    page_object_ids: list[int] = []
    content_streams: list[bytes] = []

    for page_text in page_texts:
        page_object_ids.append(object_number)
        object_number += 1
        contents_object_ids.append(object_number)
        content_streams.append(
            f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode("latin-1")
        )
        object_number += 1

    font_number = object_number
    object_number += 1

    objects.append(
        f"{catalog_number} 0 obj\n<< /Type /Catalog /Pages {pages_number} 0 R >>\nendobj\n".encode(
            "latin-1"
        )
    )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects.append(
        f"{pages_number} 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>\nendobj\n".encode(
            "latin-1"
        )
    )

    for index, page_id in enumerate(page_object_ids):
        content_id = contents_object_ids[index]
        page_object = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent {pages_number} 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_number} 0 R >> >> >>\n"
            f"endobj\n"
        ).encode("latin-1")
        objects.append(page_object)

        content = content_streams[index]
        objects.append(
            (
                f"{content_id} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("latin-1")
                + content
                + b"\nendstream\nendobj\n"
            )
        )

    objects.append(
        f"{font_number} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode(
            "latin-1"
        )
    )

    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets: list[int] = [0]
    for obj in objects:
        offsets.append(len(header) + len(body))
        body.extend(obj)

    xref_offset = len(header) + len(body)
    xref = [b"xref\n", f"0 {len(objects) + 1}\n".encode("latin-1"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_number} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return header + bytes(body) + b"".join(xref) + trailer


def test_pdf_parser_extracts_text_and_page_numbers(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(_build_sample_pdf(["First page", "Second page"]))

    document = PdfParser().parse(file_path)

    assert document.source_type == "pdf"
    assert "First page" in document.text
    assert "Second page" in document.text
    assert document.original_metadata["page_count"] == 2
    assert document.original_metadata["pages"][0]["page"] == 1


def test_csv_parser_uses_detected_delimiter_and_headers(tmp_path: Path) -> None:
    file_path = tmp_path / "data.tsv"
    file_path.write_text("name\tage\nAlice\t30\nBob\t41\n", encoding="utf-8")

    document = CsvParser().parse(file_path)

    assert document.source_type == "csv"
    assert document.text.splitlines()[0] == "name | age"
    assert "Alice | 30" in document.text
    assert document.original_metadata["delimiter"] == "\t"
    assert document.original_metadata["row_count"] == 2


def test_json_parser_flattens_nested_structures(tmp_path: Path) -> None:
    file_path = tmp_path / "payload.json"
    file_path.write_text(
        json.dumps(
            {
                "user": {"name": "Ada", "roles": ["admin", "editor"]},
                "active": True,
            }
        ),
        encoding="utf-8",
    )

    document = JsonParser().parse(file_path)

    assert document.source_type == "json"
    assert 'root.user.name: "Ada"' in document.text
    assert 'root.user.roles[1]: "editor"' in document.text
    assert "root.active: true" in document.text
    assert document.original_metadata["leaf_count"] == 4
    assert document.original_metadata["top_level_keys"] == ["user", "active"]