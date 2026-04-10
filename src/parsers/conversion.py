"""Helpers for parser-level document conversion and fallback extraction."""

from __future__ import annotations

from contextlib import contextmanager
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from .base import ParserError

LEGACY_CONVERSION_TARGETS: dict[str, str] = {
    ".doc": "docx",
    ".ppt": "pptx",
    ".xls": "xlsx",
    ".odt": "docx",
}


def find_libreoffice_binary() -> str | None:
    """Return the available LibreOffice CLI executable, if any."""

    for candidate in ("soffice", "libreoffice"):
        binary = shutil.which(candidate)
        if binary:
            return binary
    return None


def convert_file_with_libreoffice(
    source_path: Path,
    *,
    output_dir: Path,
    target_extension: str,
) -> Path:
    """Convert a file with LibreOffice and return the converted output path."""

    binary = find_libreoffice_binary()
    if binary is None:
        raise ParserError("LibreOffice CLI is not available on PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_ext = target_extension.lstrip(".").lower()
    command = [
        binary,
        "--headless",
        "--convert-to",
        normalized_ext,
        "--outdir",
        str(output_dir),
        str(source_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise ParserError(
            f"LibreOffice conversion failed for {source_path.name}: {details or 'unknown error'}"
        )

    expected_output = output_dir / f"{source_path.stem}.{normalized_ext}"
    if expected_output.exists():
        return expected_output

    candidates = sorted(output_dir.glob(f"*.{normalized_ext}"))
    if not candidates:
        raise ParserError(
            f"LibreOffice conversion produced no .{normalized_ext} output for {source_path.name}"
        )
    return candidates[0]


def try_textract_extract(file_path: Path) -> str | None:
    """Return extracted text via textract when available, otherwise None."""

    try:
        import textract  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        payload = textract.process(str(file_path))
    except Exception:  # pragma: no cover - backend specific failures
        return None

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore").strip()


def get_legacy_conversion_target(extension: str) -> str | None:
    """Return the conversion target extension for legacy formats."""

    return LEGACY_CONVERSION_TARGETS.get(extension.lower())


@contextmanager
def temporary_converted_file(
    source_path: Path,
    *,
    target_extension: str,
    temp_dir_prefix: str = "rag-convert-",
) -> Iterator[Path]:
    """Convert a source file into a temporary directory and yield the converted path."""

    with tempfile.TemporaryDirectory(prefix=temp_dir_prefix) as temp_dir:
        yield convert_file_with_libreoffice(
            source_path,
            output_dir=Path(temp_dir),
            target_extension=target_extension,
        )
