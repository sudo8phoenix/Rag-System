from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.config.loader import ConfigLoadError, ConfigValidationError, load_config


def test_load_config_with_partial_values_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            voice:
              stt_engine: whisper
            retrieval:
              top_k: 9
            """
        ).strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.voice.stt_engine == "whisper"
    assert config.retrieval.top_k == 9
    assert config.chunking.strategy == "paragraph"
    assert config.llm.provider == "ollama"


def test_load_config_missing_file_raises_clear_error() -> None:
    with pytest.raises(ConfigLoadError, match="Config file not found"):
        load_config("/tmp/does-not-exist-config.yaml")


def test_load_config_invalid_yaml_raises_clear_error(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("voice: [", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Invalid YAML"):
        load_config(config_file)


def test_load_config_validation_error_contains_path_and_field(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            retrieval:
              top_k: 0
            """
        ).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        load_config(config_file)

    message = str(exc_info.value)
    assert "Config validation failed" in message
    assert "retrieval.top_k" in message
